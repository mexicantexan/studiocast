#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

#include "core/open_video/vulkan_matting_runtime.h"
#include "core/video/replace_background_cache_policy.h"
#include "core/vulkan/kernels/utility_kernels.h"

namespace studiocast::video {

inline constexpr std::string_view
    kOpenVulkanVirtualBackgroundReplaceInitializationReason =
        "vulkan_virtual_background_replace_initialization_failed";
inline constexpr std::string_view
    kOpenVulkanVirtualBackgroundReplaceRuntimeReason =
        "vulkan_virtual_background_replace_runtime_failed";
inline constexpr std::string_view
    kOpenVulkanVirtualBackgroundReplaceAssetInvalidReason =
        "vulkan_virtual_background_replace_asset_invalid";
inline constexpr std::string_view
    kOpenVulkanVirtualBackgroundReplaceAssetUploadReason =
        "vulkan_virtual_background_replace_asset_upload_failed";

std::string OpenVulkanVirtualBackgroundReplaceInitializationFailure(
    std::string_view detail);
std::string
OpenVulkanVirtualBackgroundReplaceRuntimeFailure(std::string_view detail);
std::string
OpenVulkanVirtualBackgroundReplaceAssetInvalidFailure(std::string_view detail);
std::string
OpenVulkanVirtualBackgroundReplaceAssetUploadFailure(std::string_view detail);

struct OpenVulkanVirtualBackgroundReplaceParameters {
  int alpha_feather_radius = 0;
};

// Open CUDA compatibility: replace strength affects only alpha feathering.
OpenVulkanVirtualBackgroundReplaceParameters
ResolveOpenVulkanVirtualBackgroundReplaceParameters(int strength);

struct OpenVulkanVirtualBackgroundReplaceReadiness {
  bool production_ready = false;
  std::string reason_code;
  std::string detail;
};

OpenVulkanVirtualBackgroundReplaceReadiness
EvaluateOpenVulkanVirtualBackgroundReplaceReadiness(
    const studiocast::vulkan::OpenVulkanDiagnostics &diagnostics,
    const studiocast::open_vulkan::VulkanMattingReadiness &matting_readiness);

struct OpenVulkanVirtualBackgroundReplaceInput {
  const studiocast::vulkan::VulkanImage *foreground = nullptr;
  const studiocast::vulkan::VulkanImage *alpha = nullptr;
  const studiocast::vulkan::VulkanImage *alpha_tmp = nullptr;
  const studiocast::vulkan::VulkanImage *alpha_feathered = nullptr;
  const studiocast::vulkan::VulkanImage *output = nullptr;
  std::uint64_t capture_sequence = 0;
  std::uint64_t resident_alpha_sequence = 0;
  std::uint64_t alpha_resize_completion_count = 0;
  const studiocast::open_vulkan::VulkanMattingReadiness *matting_readiness =
      nullptr;
};

struct OpenVulkanVirtualBackgroundReplaceCounters {
  // Setup/reconfiguration-only work. These remain unchanged in Apply().
  std::uint64_t asset_allocation_calls = 0;
  std::uint64_t asset_decode_calls = 0;
  std::uint64_t asset_upload_calls = 0;
  std::uint64_t asset_resize_dispatch_calls = 0;
  // Per-frame resident GPU work.
  std::uint64_t alpha_feather_dispatch_calls = 0;
  std::uint64_t replacement_composite_dispatch_calls = 0;
  std::uint64_t runtime_failure_frames = 0;
  std::uint64_t device_loss_frames = 0;
  // This effect has no selectable CPU implementation and owns no readback.
  std::uint64_t alpha_readback_calls = 0;
  std::uint64_t cpu_fallback_calls = 0;
};

// Canonical live-boundary wrapper for virtual_background.replace. Image stat
// is prepared by CameraPipeline at configuration time; decode, staging upload,
// and replacement resize happen only in EnsureInitialized. Apply performs no
// allocation, filesystem access, upload, readback, or CPU fallback.
class OpenVulkanVirtualBackgroundReplace {
public:
  bool EnsureInitialized(
      studiocast::vulkan::kernels::UtilityKernels *kernels, int width,
      int height, int strength,
      const detail::PreparedReplaceBackgroundSource &prepared_source,
      const studiocast::open_vulkan::VulkanMattingReadiness &matting_readiness,
      OpenVulkanVirtualBackgroundReplaceCounters *counters,
      std::string *error_out);
  bool Apply(const OpenVulkanVirtualBackgroundReplaceInput &input,
             OpenVulkanVirtualBackgroundReplaceCounters *counters,
             std::string *error_out);
  void Shutdown() noexcept;

  bool initialized() const { return initialized_; }
  bool asset_valid() const { return asset_valid_; }
  const OpenVulkanVirtualBackgroundReplaceParameters &parameters() const {
    return parameters_;
  }
  const studiocast::vulkan::VulkanImage &replacement_image() const {
    return replacement_rgb_;
  }
  const studiocast::vulkan::VulkanImage &source_image() const {
    return source_rgb_;
  }
  const studiocast::vulkan::VulkanImage &upload_staging_image() const {
    return upload_staging_;
  }

private:
  bool EnsureAssetImage(studiocast::vulkan::VulkanImage *image, int width,
                        int height, bool map_memory,
                        OpenVulkanVirtualBackgroundReplaceCounters *counters,
                        std::string *error_out);
  void InvalidateAsset() noexcept;

  studiocast::vulkan::kernels::UtilityKernels *kernels_ = nullptr;
  studiocast::vulkan::VulkanContextIdentity context_identity_{};
  studiocast::vulkan::VulkanImage upload_staging_;
  studiocast::vulkan::VulkanImage source_rgb_;
  studiocast::vulkan::VulkanImage replacement_rgb_;
  OpenVulkanVirtualBackgroundReplaceParameters parameters_{};
  std::filesystem::path source_path_;
  std::filesystem::file_time_type source_mtime_{};
  int width_ = 0;
  int height_ = 0;
  bool source_valid_ = false;
  bool asset_valid_ = false;
  bool initialized_ = false;
};

} // namespace studiocast::video
