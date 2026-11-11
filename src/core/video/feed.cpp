#include "feed.h"

#include <chrono>
#include <sstream>
#include <thread>
#include <vector>

#include "core/video/pattern.h"
#include "core/video/v4l2_writer.h"
#include "core/video/v4l2loopback.h"

namespace studiocast::video {
namespace {

std::string ChooseWritableLoopbackDevice(std::string *error) {
  const auto rep = ProbeLoopback();
  for (const auto &d : rep.devices) {
    if (d.is_loopback && d.can_write)
      return d.dev_node;
  }

  if (error) {
    std::ostringstream oss;
    oss << "No writable v4l2loopback device found.\n"
        << "Run the suggested command from studiocast-video status, e.g.:\n"
        << "  " << rep.suggested_modprobe_cmd << "\n"
        << "Then re-run and ensure /dev/video* is writable (video group).";
    *error = oss.str();
  }
  return {};
}

} // namespace

VideoFeed::~VideoFeed() { Stop(); }

FeedStatus VideoFeed::Status() const {
  std::lock_guard<std::mutex> lock(mu_);
  FeedStatus s;
  s.running = running_;
  s.starting = starting_;
  s.device = device_;
  s.actual = actual_;
  s.frame_index = frame_index_;
  s.last_error = last_error_;
  return s;
}

bool VideoFeed::OpenWriterWithMode(V4l2Writer *writer,
                                   const std::string &device, int width,
                                   int height, int fps,
                                   FeedPixelFormatMode mode,
                                   std::string *error) {
  if (!writer) {
    if (error)
      *error = "writer is null";
    return false;
  }

  if (mode == FeedPixelFormatMode::auto_select) {
    {
      std::string err;
      if (writer->Open(device, width, height, fps, PixelFormat::yuyv, &err))
        return true;
      if (error)
        *error = "Tried yuyv: " + err;
    }
    {
      std::string err;
      if (writer->Open(device, width, height, fps, PixelFormat::rgb24, &err))
        return true;
      if (error)
        *error += "\nTried rgb24: " + err;
    }
    return false;
  }

  const PixelFormat fmt = (mode == FeedPixelFormatMode::rgb24)
                              ? PixelFormat::rgb24
                              : PixelFormat::yuyv;
  return writer->Open(device, width, height, fps, fmt, error);
}

bool VideoFeed::StartTestPattern(const FeedConfig &cfg, std::string *error) {
  // Validate early
  if (cfg.width <= 0 || cfg.height <= 0) {
    if (error)
      *error = "Invalid width/height.";
    return false;
  }
  if (cfg.fps <= 0 || cfg.fps > 240) {
    if (error)
      *error = "Invalid fps (1..240).";
    return false;
  }

  std::unique_lock<std::mutex> lock(mu_);
  if (running_ || starting_) {
    if (error)
      *error = "Video feed already running.";
    return false;
  }

  // If a previous thread is still joinable for any reason, join it.
  if (th_.joinable()) {
    lock.unlock();
    Stop();
    lock.lock();
  }

  stop_.store(false);
  last_error_.clear();
  device_.clear();
  actual_ = ActualFormat{};
  frame_index_ = 0;

  starting_ = true;
  running_ = false;
  start_notified_ = false;

  th_ = std::thread(&VideoFeed::ThreadMain, this, cfg);

  // Wait until the thread has either started successfully or failed.
  cv_.wait(lock, [&]() { return start_notified_; });

  starting_ = false;

  if (!running_) {
    const std::string err =
        last_error_.empty() ? "Failed to start video feed." : last_error_;
    lock.unlock();
    if (th_.joinable())
      th_.join();
    if (error)
      *error = err;
    return false;
  }

  return true;
}

void VideoFeed::Stop() {
  stop_.store(true);
  cv_.notify_all();

  std::thread toJoin;
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (th_.joinable()) {
      toJoin = std::move(th_);
    } else {
      running_ = false;
      starting_ = false;
      return;
    }
  }

  toJoin.join();

  std::lock_guard<std::mutex> lock(mu_);
  running_ = false;
  starting_ = false;
}

void VideoFeed::ThreadMain(FeedConfig cfg) {
  std::string device = cfg.device;
  if (device.empty()) {
    std::string derr;
    device = ChooseWritableLoopbackDevice(&derr);
    if (device.empty()) {
      std::lock_guard<std::mutex> lock(mu_);
      last_error_ =
          derr.empty() ? "No writable v4l2loopback device found." : derr;
      running_ = false;
      start_notified_ = true;
      cv_.notify_all();
      return;
    }
  }

  V4l2Writer writer;
  std::string oerr;
  if (!OpenWriterWithMode(&writer, device, cfg.width, cfg.height, cfg.fps,
                          cfg.format, &oerr)) {
    std::lock_guard<std::mutex> lock(mu_);
    last_error_ = "Failed to open " + device + ":\n" + oerr;
    running_ = false;
    start_notified_ = true;
    cv_.notify_all();
    return;
  }

  const auto a = writer.Actual();

  {
    std::lock_guard<std::mutex> lock(mu_);
    device_ = device;
    actual_ = a;
    frame_index_ = 0;
    last_error_.clear();
    running_ = true;

    start_notified_ = true;
    cv_.notify_all();
  }

  std::vector<std::uint8_t> frame(a.size_image);

  FrameLayout layout;
  layout.width = a.width;
  layout.height = a.height;
  layout.format = a.format;
  layout.bytes_per_line = a.bytes_per_line;
  layout.size_image = a.size_image;

  const int fps = (a.fps > 0) ? a.fps : cfg.fps;
  const auto period =
      std::chrono::nanoseconds(1000000000LL / (fps > 0 ? fps : 30));
  auto next = std::chrono::steady_clock::now();

  int frameIndex = 0;

  while (!stop_.load()) {
    next += period;

    std::string perr;
    if (!FillMovingColorBars(frame.data(), frame.size(), layout, frameIndex,
                             &perr)) {
      std::lock_guard<std::mutex> lock(mu_);
      last_error_ = "Pattern fill failed: " + perr;
      break;
    }

    const FrameWriteResult write_result =
        writer.WriteFrameDetailed(frame.data(), frame.size(), &stop_);
    if (write_result.status == FrameWriteStatus::stopped)
      break;
    if (write_result.status == FrameWriteStatus::fatal) {
      const std::string werr = DescribeFrameWriteResult(write_result);
      std::lock_guard<std::mutex> lock(mu_);
      last_error_ = "Write failed: " + werr;
      break;
    }

    // Written and backpressured frames both advance the generated latest
    // frame. EAGAIN is a zero-queue drop, not a transport failure.
    ++frameIndex;

    // Update frame count without locking every single frame (cheap but
    // cleaner).
    if ((frameIndex % 10) == 0) {
      std::lock_guard<std::mutex> lock(mu_);
      frame_index_ = frameIndex;
    }

    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait_until(lock, next, [this] { return stop_.load(); });
  }

  // Final state update
  {
    std::lock_guard<std::mutex> lock(mu_);
    frame_index_ = frameIndex;
    running_ = false;

    if (!start_notified_) {
      start_notified_ = true;
      cv_.notify_all();
    }
  }

  // writer closes in destructor
}

} // namespace studiocast::video
