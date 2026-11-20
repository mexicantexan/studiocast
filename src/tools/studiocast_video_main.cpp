#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <string>
#include <thread>

#include "core/video/pattern.h"
#include "core/video/v4l2_writer.h"
#include "core/video/v4l2loopback.h"

namespace {

std::atomic_bool g_running{true};

void HandleSigInt(int) { g_running.store(false); }

void Usage(const char *argv0) {
  std::cout << "StudioCast Video Tool\n\n"
            << "Usage:\n"
            << "  " << argv0 << " status\n"
            << "  " << argv0
            << " feed [--device /dev/videoX] [--width W] [--height H] [--fps "
               "N] [--format auto|yuyv|rgb24]\n\n"
            << "Examples:\n"
            << "  " << argv0 << " status\n"
            << "  sudo modprobe v4l2loopback devices=1 video_nr=10 "
               "card_label=\"StudioCast Camera\" exclusive_caps=1\n"
            << "  " << argv0
            << " feed --device /dev/video10 --width 1280 --height 720 --fps 30 "
               "--format auto\n";
}

std::string GetArgValue(int argc, char **argv, const std::string &key) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] && key == argv[i]) {
      return argv[i + 1] ? std::string(argv[i + 1]) : std::string();
    }
  }
  return {};
}

int GetArgInt(int argc, char **argv, const std::string &key, int fallback) {
  const auto v = GetArgValue(argc, argv, key);
  if (v.empty())
    return fallback;
  return std::atoi(v.c_str());
}

std::string ToLowerAscii(std::string s) {
  for (char &c : s) {
    if (c >= 'A' && c <= 'Z')
      c = static_cast<char>(c - 'A' + 'a');
  }
  return s;
}

std::string
ChooseWritableLoopbackDevice(const studiocast::video::LoopbackReport &rep) {
  for (const auto &d : rep.devices) {
    if (d.is_loopback && d.can_write)
      return d.dev_node;
  }
  return {};
}

bool OpenWithFormatFallback(studiocast::video::V4l2Writer *w,
                            const std::string &dev, int width, int height,
                            int fps, const std::string &formatArg,
                            std::string *error) {
  if (!w)
    return false;

  const std::string fmt = ToLowerAscii(formatArg);

  if (fmt == "auto" || fmt.empty()) {
    {
      std::string err;
      if (w->Open(dev, width, height, fps, studiocast::video::PixelFormat::yuyv,
                  &err))
        return true;
      if (error)
        *error = "Tried yuyv: " + err;
    }
    {
      std::string err;
      if (w->Open(dev, width, height, fps,
                  studiocast::video::PixelFormat::rgb24, &err))
        return true;
      if (error)
        *error += "\nTried rgb24: " + err;
    }
    return false;
  }

  auto parsed = studiocast::video::ParsePixelFormat(fmt);
  if (!parsed) {
    if (error)
      *error = "Unknown --format '" + formatArg + "'. Use auto|yuyv|rgb24.";
    return false;
  }

  return w->Open(dev, width, height, fps, *parsed, error);
}

} // namespace

int main(int argc, char **argv) {
  std::signal(SIGINT, HandleSigInt);

  if (argc < 2) {
    Usage(argv[0]);
    return 1;
  }

  const std::string cmd = argv[1];

  if (cmd == "status") {
    const auto rep = studiocast::video::ProbeLoopbackDiagnostics();
    std::cout << rep.ToText() << "\n";
    return rep.ReadyForVirtualCamera() ? 0 : 3;
  }

  if (cmd == "feed") {
    const std::string requestedDev = GetArgValue(argc, argv, "--device");
    const int width = GetArgInt(argc, argv, "--width", 1280);
    const int height = GetArgInt(argc, argv, "--height", 720);
    const int fps = GetArgInt(argc, argv, "--fps", 30);
    const std::string formatArg = GetArgValue(argc, argv, "--format").empty()
                                      ? std::string("auto")
                                      : GetArgValue(argc, argv, "--format");

    // Find a loopback device if not specified.
    std::string dev = requestedDev;
    const auto rep = studiocast::video::ProbeLoopback();

    if (dev.empty()) {
      dev = ChooseWritableLoopbackDevice(rep);
    }

    if (dev.empty()) {
      std::cerr << "ERROR: No writable v4l2loopback device found.\n\n";
      std::cerr << rep.ToText() << "\n";
      std::cerr << "\nRun the suggested modprobe command above, then retry.\n";
      return 4;
    }

    studiocast::video::V4l2Writer writer;
    std::string err;
    if (!OpenWithFormatFallback(&writer, dev, width, height, fps, formatArg,
                                &err)) {
      std::cerr << "ERROR: Failed to open " << dev << " for output.\n"
                << err << "\n";
      return 5;
    }

    const auto a = writer.Actual();
    std::cout << "Feeding test pattern to " << dev << "\n"
              << "  format: " << studiocast::video::PixelFormatName(a.format)
              << "\n"
              << "  size:   " << a.width << "x" << a.height << "\n"
              << "  fps:    " << a.fps << "\n"
              << "  bpl:    " << a.bytes_per_line << "\n"
              << "  image:  " << a.size_image << " bytes\n\n"
              << "Press Ctrl+C to stop.\n";

    std::vector<std::uint8_t> frame(a.size_image);

    studiocast::video::FrameLayout layout;
    layout.width = a.width;
    layout.height = a.height;
    layout.format = a.format;
    layout.bytes_per_line = a.bytes_per_line;
    layout.size_image = a.size_image;

    const auto period =
        std::chrono::nanoseconds(1000000000LL / (a.fps > 0 ? a.fps : 30));
    auto next = std::chrono::steady_clock::now();

    int frameIndex = 0;
    while (g_running.load()) {
      next += period;

      std::string perr;
      if (!studiocast::video::FillMovingColorBars(frame.data(), frame.size(),
                                                  layout, frameIndex, &perr)) {
        std::cerr << "ERROR: pattern fill failed: " << perr << "\n";
        return 6;
      }

      const auto write_result =
          writer.WriteFrameDetailed(frame.data(), frame.size());
      if (write_result.status ==
          studiocast::video::FrameWriteStatus::stopped) {
        break;
      }
      if (write_result.status ==
          studiocast::video::FrameWriteStatus::fatal) {
        const std::string werr =
            studiocast::video::DescribeFrameWriteResult(write_result);
        std::cerr << "ERROR: write failed: " << werr << "\n";
        return 7;
      }

      // EAGAIN follows the same zero-queue latest-frame policy as the service:
      // advance the pattern and try the newest frame at the next cadence.
      ++frameIndex;
      std::this_thread::sleep_until(next);
    }

    std::cout << "\nStopped.\n";
    return 0;
  }

  Usage(argv[0]);
  return 1;
}
