#include "core/video/video_config_types.h"

namespace studiocast::video {

std::string PixelFormatName(PixelFormat format) {
  switch (format) {
  case PixelFormat::yuyv:
    return "yuyv";
  case PixelFormat::rgb24:
    return "rgb24";
  }
  return "yuyv";
}

std::optional<PixelFormat> ParsePixelFormat(const std::string &value) {
  std::string normalized = value;
  for (char &c : normalized) {
    if (c >= 'A' && c <= 'Z')
      c = static_cast<char>(c - 'A' + 'a');
  }
  if (normalized == "yuyv" || normalized == "yuy2")
    return PixelFormat::yuyv;
  if (normalized == "rgb24" || normalized == "rgb")
    return PixelFormat::rgb24;
  return std::nullopt;
}

} // namespace studiocast::video
