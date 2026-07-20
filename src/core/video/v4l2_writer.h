#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include <sys/types.h>

namespace studiocast::video {

enum class PixelFormat {
  yuyv,
  rgb24,
};

std::string PixelFormatName(PixelFormat fmt);
std::optional<PixelFormat> ParsePixelFormat(const std::string &s);

struct ActualFormat {
  int width = 0;
  int height = 0;
  int fps = 0;
  int fps_num = 0;
  int fps_den = 0;
  PixelFormat format = PixelFormat::yuyv;

  // V4L2 negotiated pixel format, as FourCC.
  // Example: V4L2_PIX_FMT_YUYV -> "YUYV".
  std::uint32_t pixfmt_fourcc = 0;
  std::string pixfmt;

  std::size_t bytes_per_line = 0;
  std::size_t size_image = 0;
};

// Output writes have no userspace queue: each call either commits the current
// frame, drops it immediately when the nonblocking device would block, observes
// stop, or reports a fatal transport failure.
inline constexpr std::size_t kV4l2WriterMaxQueuedFrames = 0;
inline constexpr std::uint32_t kV4l2WriterMaxEintrRetries = 4;

enum class FrameWriteStatus {
  written,
  would_block_dropped,
  stopped,
  fatal,
};

enum class FrameWriteFailure {
  none,
  writer_not_open,
  null_data,
  undersized_input,
  partial_frame,
  zero_write,
  interrupted_retry_limit,
  disconnected,
  format_mismatch,
  io_error,
};

struct FrameWriteResult {
  FrameWriteStatus status = FrameWriteStatus::fatal;
  FrameWriteFailure failure = FrameWriteFailure::io_error;
  int error_number = 0;
  std::size_t bytes_written = 0;
  std::uint32_t write_syscalls = 0;

  bool FrameCommitted() const { return status == FrameWriteStatus::written; }
  bool ShouldRefreshFormat() const {
    return failure == FrameWriteFailure::format_mismatch;
  }
  bool RequiresReopen() const {
    return failure == FrameWriteFailure::partial_frame ||
           failure == FrameWriteFailure::zero_write;
  }
};

struct FrameWriteCounters {
  std::uint64_t write_syscalls = 0;
  std::uint64_t frames_written = 0;
  std::uint64_t would_block_drops = 0;
  std::uint64_t stopped_writes = 0;
  std::uint64_t eintr_retries = 0;
  std::uint64_t partial_frame_failures = 0;
  std::uint64_t fatal_failures = 0;
};

// Narrow syscall seam for deterministic transport tests and benchmarks.
// Production's V4l2Writer path invokes ::write directly; this hook is not in
// its successful-frame call path.
struct FrameWriteTransport {
  using WriteFn = ssize_t (*)(void *context, int fd, const void *data,
                              std::size_t bytes);
  void *context = nullptr;
  WriteFn write = nullptr;
};

FrameWriteResult WriteFrameToTransport(
    const FrameWriteTransport &transport, int fd, const std::uint8_t *data,
    std::size_t bytes, std::size_t size_image,
    const std::atomic_bool *stop = nullptr,
    FrameWriteCounters *counters = nullptr) noexcept;

// Direct-syscall form shared by production and successful-path benchmarks.
FrameWriteResult WriteFrameToFd(int fd, const std::uint8_t *data,
                                std::size_t bytes, std::size_t size_image,
                                const std::atomic_bool *stop = nullptr,
                                FrameWriteCounters *counters = nullptr) noexcept;

// Open flags are part of the transport contract and are exposed as a pure
// helper so hermetic tests can prove O_NONBLOCK is configured once at open.
int V4l2WriterOpenFlags(bool read_write) noexcept;

std::string DescribeFrameWriteResult(const FrameWriteResult &result);

class V4l2Writer final {
public:
  V4l2Writer() = default;
  ~V4l2Writer();

  V4l2Writer(const V4l2Writer &) = delete;
  V4l2Writer &operator=(const V4l2Writer &) = delete;

  bool Open(const std::string &device, int width, int height, int fps,
            PixelFormat fmt, std::string *error);

  void Close();

  // Refresh cached negotiated format from the kernel.
  //
  // This is useful for v4l2loopback: some consumers may renegotiate
  // global caps via VIDIOC_S_FMT. When that happens, the writer must
  // update its cached size_image/bytes_per_line to avoid write() failures.
  bool RefreshActual(std::string *error);

  // Re-negotiate output format/fps on the currently open writer FD.
  //
  // This avoids closing/re-opening the producer FD, which can destabilize
  // certain v4l2loopback configurations (especially with exclusive_caps=1).
  bool Renegotiate(const std::string &device, int width, int height, int fps,
                   PixelFormat fmt, std::string *error);

  bool WriteFrame(const std::uint8_t *data, std::size_t bytes,
                  std::string *error);

  FrameWriteResult
  WriteFrameDetailed(const std::uint8_t *data, std::size_t bytes,
                     const std::atomic_bool *stop = nullptr) noexcept;

  bool IsOpen() const { return fd_ >= 0; }
  const ActualFormat &Actual() const { return actual_; }
  const FrameWriteCounters &WriteCounters() const { return write_counters_; }

private:
  int fd_ = -1;
  ActualFormat actual_{};
  FrameWriteCounters write_counters_{};
};

} // namespace studiocast::video
