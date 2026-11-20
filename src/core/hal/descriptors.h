#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace studiocast::hal {

inline constexpr std::size_t kMaxImagePlanes = 4;
inline constexpr std::size_t kMaxTensorDimensions = 8;

enum class PixelFormat : std::uint16_t {
  invalid = 0,
  rgb_u8,
  bgr_u8,
  rgba_u8,
  bgra_u8,
  yuyv_u8,
  nv12_u8,
  alpha_u8,
  alpha_f32,
};

enum class TensorElementType : std::uint16_t {
  invalid = 0,
  uint8,
  int8,
  float16,
  float32,
  int32,
  int64,
};

enum class TensorLayout : std::uint16_t {
  invalid = 0,
  linear,
  nchw,
  nhwc,
  chw,
  hwc,
};

enum class ChannelOrder : std::uint16_t {
  unspecified = 0,
  rgb,
  bgr,
  rgba,
  bgra,
  alpha,
};

enum class ColorSpace : std::uint16_t {
  unspecified = 0,
  srgb,
  rec601,
  rec709,
  linear_rgb,
};

enum class ColorRange : std::uint16_t {
  unspecified = 0,
  full,
  limited,
  normalized_zero_to_one,
  normalized_minus_one_to_one,
};

enum class AlphaMeaning : std::uint16_t {
  none = 0,
  straight,
  premultiplied,
  foreground_coverage,
};

enum class Orientation : std::uint16_t {
  top_left = 1,
  top_right,
  bottom_left,
  bottom_right,
};

enum class HostCoherence : std::uint16_t {
  not_host_visible = 0,
  coherent,
  noncoherent_explicit_flush,
  externally_synchronized,
};

enum class Mutability : std::uint16_t {
  read_only = 1,
  write_only,
  read_write,
};

enum class AliasingPermission : std::uint16_t {
  forbidden = 0,
  exact_alias_only,
  overlapping_allowed,
};

struct PlaneDescriptor {
  std::uint64_t offset_bytes = 0;
  std::uint64_t row_stride_bytes = 0;
  std::uint64_t size_bytes = 0;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t alignment_bytes = 1;
};

struct ImageDescriptor {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint16_t plane_count = 0;
  PixelFormat format = PixelFormat::invalid;
  ChannelOrder channel_order = ChannelOrder::unspecified;
  ColorSpace color_space = ColorSpace::unspecified;
  ColorRange color_range = ColorRange::unspecified;
  AlphaMeaning alpha_meaning = AlphaMeaning::none;
  Orientation orientation = Orientation::top_left;
  HostCoherence host_coherence = HostCoherence::not_host_visible;
  Mutability mutability = Mutability::read_only;
  AliasingPermission aliasing = AliasingPermission::forbidden;
  bool in_place_permitted = false;
  std::uint32_t base_alignment_bytes = 1;
  std::array<PlaneDescriptor, kMaxImagePlanes> planes{};
};

struct TensorDescriptor {
  TensorElementType element_type = TensorElementType::invalid;
  TensorLayout layout = TensorLayout::invalid;
  ChannelOrder channel_order = ChannelOrder::unspecified;
  ColorSpace color_space = ColorSpace::unspecified;
  ColorRange color_range = ColorRange::unspecified;
  AlphaMeaning alpha_meaning = AlphaMeaning::none;
  Orientation orientation = Orientation::top_left;
  HostCoherence host_coherence = HostCoherence::not_host_visible;
  Mutability mutability = Mutability::read_only;
  AliasingPermission aliasing = AliasingPermission::forbidden;
  bool in_place_permitted = false;
  std::uint8_t rank = 0;
  std::uint32_t base_alignment_bytes = 1;
  std::array<std::uint64_t, kMaxTensorDimensions> dimensions{};
  std::array<std::uint64_t, kMaxTensorDimensions> strides_bytes{};
};

[[nodiscard]] bool
ValidateImageDescriptor(const ImageDescriptor &descriptor) noexcept;
[[nodiscard]] bool
ValidateTensorDescriptor(const TensorDescriptor &descriptor) noexcept;

} // namespace studiocast::hal
