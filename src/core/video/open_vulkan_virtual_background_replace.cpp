#include "core/video/open_vulkan_virtual_background_replace.h"

#include <algorithm>
#include <array>
#include <vector>

#include "core/video/effects/broadcast_effect_contract.h"
#include "core/video/image_ppm.h"

namespace studiocast::video {
namespace {

std::string StableFailure(std::string_view code, std::string_view summary,
                          std::string_view detail) {
  std::string out = "[";
  out += code;
  out += "] ";
  out += summary;
  if (!detail.empty()) {
    out += ": ";
    out += detail;
  }
  return out;
}

OpenVulkanVirtualBackgroundReplaceReadiness NotReady(std::string reason_code,
                                                     std::string detail) {
  OpenVulkanVirtualBackgroundReplaceReadiness result;
  result.reason_code = std::move(reason_code);
  result.detail = std::move(detail);
  return result;
}

bool Aliases(const studiocast::vulkan::VulkanImage *left,
             const studiocast::vulkan::VulkanImage *right) {
  return left == right || left->buffer() == right->buffer();
}

} // namespace

std::string OpenVulkanVirtualBackgroundReplaceInitializationFailure(
    std::string_view detail) {
  return StableFailure(
      kOpenVulkanVirtualBackgroundReplaceInitializationReason,
      "Open Vulkan virtual background replace initialization failed", detail);
}

std::string
OpenVulkanVirtualBackgroundReplaceRuntimeFailure(std::string_view detail) {
  return StableFailure(kOpenVulkanVirtualBackgroundReplaceRuntimeReason,
                       "Open Vulkan virtual background replace runtime failed",
                       detail);
}

std::string OpenVulkanVirtualBackgroundReplaceAssetInvalidFailure(
    std::string_view detail) {
  return StableFailure(kOpenVulkanVirtualBackgroundReplaceAssetInvalidReason,
                       "Open Vulkan replacement asset is invalid", detail);
}

std::string OpenVulkanVirtualBackgroundReplaceAssetUploadFailure(
    std::string_view detail) {
  return StableFailure(kOpenVulkanVirtualBackgroundReplaceAssetUploadReason,
                       "Open Vulkan replacement asset upload failed", detail);
}

OpenVulkanVirtualBackgroundReplaceParameters
ResolveOpenVulkanVirtualBackgroundReplaceParameters(int strength) {
  OpenVulkanVirtualBackgroundReplaceParameters parameters;
  const int clamped_strength =
      std::clamp(strength, effects::contract::kVbStrengthMin,
                 effects::contract::kVbStrengthMax);
  parameters.alpha_feather_radius = std::min(4, clamped_strength / 16);
  return parameters;
}

OpenVulkanVirtualBackgroundReplaceReadiness
EvaluateOpenVulkanVirtualBackgroundReplaceReadiness(
    const studiocast::vulkan::OpenVulkanDiagnostics &diagnostics,
    const studiocast::open_vulkan::VulkanMattingReadiness &matting_readiness) {
  if (!diagnostics.compiled_enabled) {
    return NotReady("vulkan_backend_disabled_in_build",
                    "Open Vulkan support is disabled in this build");
  }
  if (!diagnostics.runtime_library_found) {
    return NotReady("vulkan_runtime_not_found",
                    "the Vulkan runtime library is unavailable");
  }
  if (!diagnostics.physical_device_found) {
    return NotReady("vulkan_no_physical_device",
                    "no Vulkan physical device was found");
  }
  if (diagnostics.cpu_device_selected || !diagnostics.non_cpu_device_selected) {
    return NotReady(
        "vulkan_only_cpu_devices_available",
        "the selected Vulkan device is a CPU/software implementation");
  }
  if (!diagnostics.compute_queue_available) {
    return NotReady("vulkan_no_compute_queue",
                    "the selected Vulkan device has no compute queue");
  }
  if (!diagnostics.logical_device_created) {
    return NotReady("vulkan_device_create_failed",
                    "the Vulkan logical device was not created");
  }
  if (!diagnostics.context_created || !diagnostics.context_healthy) {
    return NotReady(diagnostics.context_failure_reason.empty()
                        ? "vulkan_context_uninitialized"
                        : diagnostics.context_failure_reason,
                    diagnostics.error.empty()
                        ? "the shared Vulkan context is not healthy"
                        : diagnostics.error);
  }
  if (!diagnostics.production_hardware_ready) {
    return NotReady("vulkan_production_hardware_not_ready",
                    "shared Vulkan diagnostics did not prove production "
                    "hardware readiness");
  }
  if (!diagnostics.shader_pipeline_created || !diagnostics.ok) {
    return NotReady("open_vulkan_utility_kernels_unavailable",
                    "the shared Vulkan utility shader pipeline is unavailable");
  }
  if (!matting_readiness.production_ready) {
    return NotReady(
        matting_readiness.reason_code.empty()
            ? studiocast::open_vulkan::kOpenVulkanMattingUnavailableReason
            : matting_readiness.reason_code,
        studiocast::open_vulkan::FormatVulkanMattingReadiness(
            matting_readiness));
  }

  const auto &runtime = matting_readiness.runtime;
  if (!runtime.runtime_created || !runtime.graph_loaded ||
      !runtime.warmup_complete || runtime.cpu_layers_used ||
      !runtime.device_identity_matches || !runtime.input_device_resident ||
      !runtime.alpha_device_resident || !runtime.output_device_resident ||
      !runtime.shared_device_imported || !runtime.queue_ownership_explicit ||
      !runtime.synchronous_completion ||
      !runtime.bounded_reusable_allocations ||
      runtime.dynamic_allocation_count != 0 ||
      runtime.cpu_readback_count != 0) {
    return NotReady(
        studiocast::open_vulkan::kOpenVulkanMattingUnavailableReason,
        "shared matting readiness omitted required residency, synchronization, "
        "or bounded-allocation evidence");
  }

  OpenVulkanVirtualBackgroundReplaceReadiness result;
  result.production_ready = true;
  return result;
}

void OpenVulkanVirtualBackgroundReplace::InvalidateAsset() noexcept {
  source_valid_ = false;
  asset_valid_ = false;
  initialized_ = false;
  source_path_.clear();
  source_mtime_ = {};
}

void OpenVulkanVirtualBackgroundReplace::Shutdown() noexcept {
  if (kernels_ && (upload_staging_.Valid() || source_rgb_.Valid() ||
                   replacement_rgb_.Valid())) {
    kernels_->InvalidateDescriptorBindingCacheForSetup();
  }
  upload_staging_.Free();
  source_rgb_.Free();
  replacement_rgb_.Free();
  kernels_ = nullptr;
  context_identity_ = {};
  parameters_ = {};
  width_ = 0;
  height_ = 0;
  InvalidateAsset();
}

bool OpenVulkanVirtualBackgroundReplace::EnsureAssetImage(
    studiocast::vulkan::VulkanImage *image, int width, int height,
    bool map_memory, OpenVulkanVirtualBackgroundReplaceCounters *counters,
    std::string *error_out) {
  if (!image || !kernels_ || !kernels_->device()) {
    if (error_out)
      *error_out = "replacement asset image has no shared context";
    return false;
  }
  if (image->Valid() && image->BelongsTo(*kernels_->device()) &&
      image->context_identity() == context_identity_ &&
      image->width() == width && image->height() == height &&
      image->format() == studiocast::vulkan::VulkanPixelFormat::rgb_u8 &&
      (map_memory ? image->mapped() != nullptr
                  : image->mapped() == nullptr && image->device_local())) {
    return true;
  }

  // Reallocation may reuse an opaque VkBuffer handle. Invalidate descriptor
  // tuples before destroying the old resource so the next setup dispatch
  // cannot retain a descriptor for the previous allocation.
  kernels_->InvalidateDescriptorBindingCacheForSetup();
  std::string detail;
  if (!image->Allocate(kernels_->device(), width, height,
                       studiocast::vulkan::VulkanPixelFormat::rgb_u8,
                       map_memory, &detail)) {
    if (error_out)
      *error_out = detail;
    return false;
  }
  ++counters->asset_allocation_calls;
  return true;
}

bool OpenVulkanVirtualBackgroundReplace::EnsureInitialized(
    studiocast::vulkan::kernels::UtilityKernels *kernels, int width, int height,
    int strength, const detail::PreparedReplaceBackgroundSource &prepared_source,
    const studiocast::open_vulkan::VulkanMattingReadiness &matting_readiness,
    OpenVulkanVirtualBackgroundReplaceCounters *counters,
    std::string *error_out) {
  if (error_out)
    error_out->clear();
  auto fail = [&](std::string nested) {
    InvalidateAsset();
    if (error_out) {
      *error_out = OpenVulkanVirtualBackgroundReplaceInitializationFailure(
          nested);
    }
    return false;
  };
  if (!counters)
    return fail("counter output is null");
  if (!kernels || width <= 0 || height <= 0)
    return fail("invalid shared context or frame dimensions");

  std::string detail;
  if (!kernels->Initialized() && !kernels->Initialize(&detail))
    return fail(detail);
  const auto readiness = EvaluateOpenVulkanVirtualBackgroundReplaceReadiness(
      kernels->Diagnostics(), matting_readiness);
  if (!readiness.production_ready) {
    return fail(StableFailure(readiness.reason_code, readiness.detail, {}));
  }
  if (!kernels->device() || !kernels->device()->context_identity().Valid())
    return fail("shared Vulkan context identity is invalid");

  const auto identity = kernels->device()->context_identity();
  const auto &runtime_device = matting_readiness.runtime.active_device;
  if (runtime_device.ownership_domain != kernels->device() ||
      runtime_device.logical_device !=
          reinterpret_cast<std::uintptr_t>(kernels->device()->device()) ||
      runtime_device.queue !=
          reinterpret_cast<std::uintptr_t>(kernels->device()->queue()) ||
      runtime_device.context_id != identity.context_id ||
      runtime_device.context_generation != identity.generation) {
    return fail(StableFailure(
        studiocast::open_vulkan::kOpenVulkanMattingDeviceMismatchReason,
        "matting readiness does not belong to the shared utility context", {}));
  }

  if (!prepared_source.valid || prepared_source.path.empty()) {
    const std::string asset_detail =
        prepared_source.error.empty()
            ? "virtual_background.replace_path not set."
            : prepared_source.error;
    return fail(OpenVulkanVirtualBackgroundReplaceAssetInvalidFailure(
        asset_detail));
  }

  const bool source_cache_hit =
      source_valid_ && kernels_ == kernels && context_identity_ == identity &&
      source_path_ == prepared_source.path &&
      source_mtime_ == prepared_source.mtime && upload_staging_.Valid() &&
      source_rgb_.Valid() &&
      upload_staging_.BelongsTo(*kernels->device()) &&
      source_rgb_.BelongsTo(*kernels->device());
  const bool exact_asset_cache_hit =
      source_cache_hit && initialized_ && asset_valid_ && width_ == width &&
      height_ == height && replacement_rgb_.Valid() &&
      replacement_rgb_.BelongsTo(*kernels->device());
  parameters_ = ResolveOpenVulkanVirtualBackgroundReplaceParameters(strength);
  if (exact_asset_cache_hit)
    return true;

  // A failed refresh must never leave the previous frame-size asset eligible
  // for Apply. Geometry-only refresh keeps the already uploaded source; path,
  // mtime, device, or context changes invalidate the full source cache.
  if (!source_cache_hit) {
    InvalidateAsset();
  } else {
    asset_valid_ = false;
    initialized_ = false;
  }
  kernels_ = kernels;
  context_identity_ = identity;
  width_ = width;
  height_ = height;

  if (!source_cache_hit) {
    int source_width = 0;
    int source_height = 0;
    std::vector<std::uint8_t> source_rgb;
    if (!LoadImageRgb24(prepared_source.path, &source_width, &source_height,
                        &source_rgb, &detail) ||
        source_width <= 0 || source_height <= 0 || source_rgb.empty()) {
      if (detail.empty())
        detail = "decoded replacement image is empty";
      return fail(
          OpenVulkanVirtualBackgroundReplaceAssetInvalidFailure(detail));
    }
    ++counters->asset_decode_calls;

    if (!EnsureAssetImage(&upload_staging_, source_width, source_height,
                          /*map_memory=*/true, counters, &detail) ||
        !EnsureAssetImage(&source_rgb_, source_width, source_height,
                          /*map_memory=*/false, counters, &detail)) {
      return fail(
          OpenVulkanVirtualBackgroundReplaceAssetUploadFailure(detail));
    }
    if (!kernels_->UploadRgb24ToDeviceLocal(
            source_rgb.data(), static_cast<std::size_t>(source_width) * 3u,
            upload_staging_, source_rgb_, &detail)) {
      return fail(
          OpenVulkanVirtualBackgroundReplaceAssetUploadFailure(detail));
    }
    ++counters->asset_upload_calls;
    source_path_ = prepared_source.path;
    source_mtime_ = prepared_source.mtime;
    source_valid_ = true;
  }

  if (!EnsureAssetImage(&replacement_rgb_, width, height,
                        /*map_memory=*/false, counters, &detail)) {
    return fail(OpenVulkanVirtualBackgroundReplaceAssetUploadFailure(detail));
  }

  if (!kernels_->ResizeBilinear(source_rgb_, replacement_rgb_, &detail)) {
    return fail(OpenVulkanVirtualBackgroundReplaceAssetUploadFailure(detail));
  }
  ++counters->asset_resize_dispatch_calls;

  asset_valid_ = true;
  initialized_ = true;
  return true;
}

bool OpenVulkanVirtualBackgroundReplace::Apply(
    const OpenVulkanVirtualBackgroundReplaceInput &input,
    OpenVulkanVirtualBackgroundReplaceCounters *counters,
    std::string *error_out) {
  if (error_out)
    error_out->clear();
  auto fail = [&](std::string detail) {
    if (counters) {
      ++counters->runtime_failure_frames;
      if (kernels_ && kernels_->device() &&
          kernels_->device()->health().health ==
              studiocast::vulkan::VulkanContextHealth::device_lost) {
        ++counters->device_loss_frames;
      }
    }
    if (error_out)
      *error_out = OpenVulkanVirtualBackgroundReplaceRuntimeFailure(detail);
    return false;
  };
  if (!counters)
    return fail("counter output is null");
  if (!initialized_ || !asset_valid_ || !kernels_ || !kernels_->device())
    return fail("wrapper or replacement asset is not initialized");
  if (kernels_->device()->context_identity() != context_identity_) {
    return fail("[vulkan_context_generation_changed] shared Vulkan context "
                "generation changed");
  }
  if (!input.foreground || !input.alpha || !input.alpha_tmp ||
      !input.alpha_feathered || !input.output || !input.matting_readiness) {
    return fail("one or more resident effect resources are null");
  }

  const auto &matting = *input.matting_readiness;
  const auto &runtime = matting.runtime;
  const auto current_readiness =
      EvaluateOpenVulkanVirtualBackgroundReplaceReadiness(
          kernels_->Diagnostics(), matting);
  if (!current_readiness.production_ready) {
    return fail(StableFailure(current_readiness.reason_code,
                              current_readiness.detail, {}));
  }
  if (runtime.cpu_layers_used || runtime.cpu_readback_count != 0 ||
      !runtime.input_device_resident || !runtime.alpha_device_resident ||
      !runtime.output_device_resident || !runtime.queue_ownership_explicit ||
      !runtime.synchronous_completion || runtime.inference_count == 0 ||
      runtime.completion_count <
          runtime.warmup_inference_count + runtime.inference_count ||
      input.capture_sequence == 0 ||
      input.resident_alpha_sequence != input.capture_sequence ||
      input.alpha_resize_completion_count == 0) {
    return fail("[open_vulkan_matting_unavailable] current-frame resident "
                "alpha inference/residency/synchronization evidence is "
                "incomplete");
  }
  if (runtime.active_device.ownership_domain != kernels_->device() ||
      runtime.active_device.logical_device !=
          reinterpret_cast<std::uintptr_t>(kernels_->device()->device()) ||
      runtime.active_device.queue !=
          reinterpret_cast<std::uintptr_t>(kernels_->device()->queue()) ||
      runtime.active_device.context_id != context_identity_.context_id ||
      runtime.active_device.context_generation !=
          context_identity_.generation) {
    return fail("[open_vulkan_matting_device_mismatch] current-frame matte "
                "does not belong to the initialized shared context");
  }

  const std::array<const studiocast::vulkan::VulkanImage *, 6> images = {
      input.foreground, input.alpha, input.alpha_tmp, input.alpha_feathered,
      &replacement_rgb_, input.output};
  for (const auto *image : images) {
    if (!image->Valid())
      return fail("one or more resident effect resources are invalid");
    if (!image->BelongsTo(*kernels_->device()) ||
        image->context_identity() != context_identity_) {
      return fail("[vulkan_foreign_context] replace resources do not belong "
                  "to the initialized shared context");
    }
    if (image->width() != width_ || image->height() != height_)
      return fail("frame dimensions changed");
  }
  for (std::size_t left = 0; left < images.size(); ++left) {
    for (std::size_t right = left + 1; right < images.size(); ++right) {
      if (Aliases(images[left], images[right])) {
        return fail("foreground, replacement, alpha scratch, and output "
                    "resources must be distinct");
      }
    }
  }
  if (!upload_staging_.Valid() || !source_rgb_.Valid() ||
      !upload_staging_.BelongsTo(*kernels_->device()) ||
      !source_rgb_.BelongsTo(*kernels_->device()) ||
      upload_staging_.context_identity() != context_identity_ ||
      source_rgb_.context_identity() != context_identity_) {
    return fail("replacement upload/source cache is stale or foreign");
  }
  if (input.foreground->format() !=
          studiocast::vulkan::VulkanPixelFormat::rgb_u8 ||
      replacement_rgb_.format() !=
          studiocast::vulkan::VulkanPixelFormat::rgb_u8 ||
      source_rgb_.format() != studiocast::vulkan::VulkanPixelFormat::rgb_u8 ||
      upload_staging_.format() !=
          studiocast::vulkan::VulkanPixelFormat::rgb_u8 ||
      input.output->format() != studiocast::vulkan::VulkanPixelFormat::rgb_u8) {
    return fail("canonical foreground, replacement, and output formats must "
                "be rgb_u8");
  }
  if (input.alpha->format() != studiocast::vulkan::VulkanPixelFormat::f32_1 ||
      input.alpha_tmp->format() !=
          studiocast::vulkan::VulkanPixelFormat::f32_1 ||
      input.alpha_feathered->format() !=
          studiocast::vulkan::VulkanPixelFormat::f32_1) {
    return fail("alpha and alpha scratch formats must be f32_1");
  }
  if (input.alpha->mapped() || input.alpha_tmp->mapped() ||
      input.alpha_feathered->mapped() || !input.alpha->device_local() ||
      !input.alpha_tmp->device_local() ||
      !input.alpha_feathered->device_local() || source_rgb_.mapped() ||
      replacement_rgb_.mapped() || !source_rgb_.device_local() ||
      !replacement_rgb_.device_local()) {
    return fail("alpha scratch and replacement images must be non-mapped "
                "DEVICE_LOCAL resources");
  }
  if (!upload_staging_.mapped() || !upload_staging_.host_visible())
    return fail("replacement setup upload staging is not host-visible");
  // The pipeline's mapped output is the one final RGB transport. Replace only
  // writes it from Vulkan and never invalidates, maps, or host-reads it.
  if (!input.output->mapped())
    return fail("output must be the mapped final-frame transport resource");

  const studiocast::vulkan::VulkanImage *alpha_use = input.alpha;
  std::string detail;
  if (parameters_.alpha_feather_radius > 0) {
    if (!kernels_->BoxBlurSeparableF32_1(
            *input.alpha, *input.alpha_tmp, *input.alpha_feathered,
            parameters_.alpha_feather_radius, &detail)) {
      return fail(detail);
    }
    ++counters->alpha_feather_dispatch_calls;
    alpha_use = input.alpha_feathered;
  }
  if (!kernels_->CompositeAlphaU8x3(*input.foreground, replacement_rgb_,
                                    *alpha_use, *input.output, &detail)) {
    return fail(detail);
  }
  ++counters->replacement_composite_dispatch_calls;
  return true;
}

} // namespace studiocast::video
