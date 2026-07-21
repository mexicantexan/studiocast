#pragma once

#include <optional>
#include <string>

namespace studiocast::video {

enum class CaptureMode {
  // Use the requested width and height; both must be positive.
  requested,

  // Select a good capture mode automatically. Width/height may be sentinels.
  auto_best,
};

enum class PixelFormat {
  yuyv,
  rgb24,
};

std::string PixelFormatName(PixelFormat format);
std::optional<PixelFormat> ParsePixelFormat(const std::string &value);

} // namespace studiocast::video
