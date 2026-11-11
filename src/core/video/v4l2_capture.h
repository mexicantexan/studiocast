#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace studiocast::video {

enum class CapturePixelFormat {
  yuyv,
  mjpeg,
  rgb24,
};

// Heuristic used by `V4l2Capture::Open()` to decide whether to prefer MJPEG at
// a given requested resolution (many webcams cap uncompressed YUYV at ~720p,
// but can do 1080p+ in MJPEG).
bool ShouldPreferMjpegForResolution(int width, int height);

struct CaptureFormatSupport {
  bool yuyv = false;

  // True when either MJPEG or JPEG compressed capture is supported.
  bool mjpeg = false;
};

// Pure negotiation helper used by `V4l2Capture::Open()` and tests. It returns
// the pixel-format attempt order after applying MJPEG preference and removing
// unsupported YUYV/MJPEG attempts.
std::vector<CapturePixelFormat>
CaptureFormatTryOrderForRequest(CapturePixelFormat requested,
                                bool prefer_mjpeg, int width, int height,
                                CaptureFormatSupport support);

struct CaptureFormat {
  int width = 0;
  int height = 0;
  int fps = 0;
  int fps_num = 0;
  int fps_den = 0;
  CapturePixelFormat format = CapturePixelFormat::yuyv;

  // V4L2 negotiated pixel format, as FourCC.
  // Example: V4L2_PIX_FMT_YUYV -> "YUYV".
  std::uint32_t pixfmt_fourcc = 0;
  std::string pixfmt;

  std::size_t bytes_per_line = 0;
  std::size_t size_image = 0;
};

struct CapturedFrameView {
  const std::uint8_t *data = nullptr;
  std::size_t bytes = 0;
  int index = -1;
  std::uint64_t sequence = 0;

  // Timestamp from the V4L2 driver (v4l2_buffer.timestamp).
  // Units: nanoseconds.
  // Epoch: depends on the driver; check `timestamp_monotonic`.
  std::uint64_t timestamp_ns = 0;
  bool timestamp_monotonic = false;
};

enum class CaptureAcquireResult {
  frame,
  no_frame,
  failure,
};

// Pure classification used at the poll boundary and by hermetic tests.
CaptureAcquireResult ClassifyCapturePollResult(int poll_result) noexcept;

class V4l2Capture final {
public:
  V4l2Capture() = default;
  ~V4l2Capture();

  V4l2Capture(const V4l2Capture &) = delete;
  V4l2Capture &operator=(const V4l2Capture &) = delete;

  bool Open(const std::string &device, int width, int height, int fps,
            CapturePixelFormat fmt, bool prefer_mjpeg, std::string *error);

  // Enumerate supported capture modes and open the best-scoring one.
  // Intended for `CaptureMode::auto_best`.
  bool OpenBest(const std::string &device, int target_fps, bool prefer_mjpeg,
                std::string *error);

  void Close();

  bool IsOpen() const { return fd_ >= 0; }
  const CaptureFormat &Actual() const { return actual_; }

  // Acquire a frame (DQBUF). Caller MUST call ReleaseFrame() with the returned
  // view.
  bool AcquireFrame(CapturedFrameView *out, int timeout_ms, std::string *error);

  // Typed variant used by zero-timeout latest-frame draining. An expected
  // timeout/EAGAIN returns no_frame without formatting an error; actual poll or
  // DQBUF failures remain distinguishable and populate error when requested.
  CaptureAcquireResult AcquireFrameDetailed(CapturedFrameView *out,
                                            int timeout_ms,
                                            std::string *error);

  // Release a frame back to driver (QBUF).
  bool ReleaseFrame(const CapturedFrameView &f, std::string *error);

private:
  struct Buffer {
    void *start = nullptr;
    std::size_t length = 0;
  };

  bool StreamOn(std::string *error);
  bool StreamOff(std::string *error);

  // chosen buffer type (single-plane capture or mplane capture)
  unsigned int buf_type_ = 0;
  bool mplane_ = false;

  int fd_ = -1;
  bool streaming_ = false;

  CaptureFormat actual_{};
  std::vector<Buffer> buffers_;
};

} // namespace studiocast::video
