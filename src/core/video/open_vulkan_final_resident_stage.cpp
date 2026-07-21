#include "core/video/open_vulkan_final_resident_stage.h"

#include <utility>

namespace studiocast::video {
namespace {

using studiocast::vulkan::VulkanImage;
using studiocast::vulkan::VulkanPixelFormat;

bool EnsureReusableImage(studiocast::vulkan::kernels::UtilityKernels *kernels,
                         VulkanImage *image, int width, int height,
                         VulkanPixelFormat format, const char *label,
                         OpenVulkanFinalResidentStageCounters *counters,
                         std::string *error_out) {
  if (!kernels || !kernels->device() || !image) {
    if (error_out)
      *error_out = std::string("Open Vulkan final stage: missing ") + label;
    return false;
  }
  if (image->Valid() && image->BelongsTo(*kernels->device()) &&
      image->width() == width && image->height() == height &&
      image->format() == format && !image->mapped() && image->device_local()) {
    return true;
  }

  std::string detail;
  kernels->InvalidateDescriptorBindingCacheForSetup();
  if (!image->Allocate(kernels->device(), width, height, format,
                       /*map_memory=*/false, &detail)) {
    if (error_out) {
      *error_out = std::string("Open Vulkan final stage: failed to allocate ") +
                   label + ": " + detail;
    }
    return false;
  }
  ++counters->resource_allocation_calls;
  ++counters->device_local_allocation_calls;
  return true;
}

std::string ForeignSourceFailure() {
  return "[vulkan_foreign_context] final resident source does not belong to "
         "the shared Vulkan context";
}

std::string ResidencyFailure(std::string_view detail) {
  std::string out = "[vulkan_effect_residency_contract_failed] Open Vulkan "
                    "final resident stage requires non-mapped DEVICE_LOCAL "
                    "resources";
  if (!detail.empty()) {
    out += ": ";
    out += detail;
  }
  return out;
}

} // namespace

bool ExecuteOpenVulkanResidentAutoFrameStage(
    const OpenVulkanResidentAutoFrameInput &input,
    OpenVulkanAutoFrame *auto_frame, OpenVulkanAutoFrameCounters *counters,
    OpenVulkanResidentFrameState *state, std::string *error_out) {
  if (error_out)
    error_out->clear();
  if (state)
    *state = OpenVulkanResidentFrameState{};
  auto fail = [&](std::string detail) {
    if (error_out)
      *error_out = OpenVulkanAutoFrameRuntimeFailure(detail);
    return false;
  };
  if (!auto_frame || !counters || !state || !input.source ||
      !input.source->Valid()) {
    return fail("resident orchestration input is invalid");
  }
  if (input.source->mapped() || !input.source->device_local()) {
    ++counters->residency_rejection_frames;
    return fail("[vulkan_effect_residency_contract_failed] Auto Frame input "
                "must be non-mapped DEVICE_LOCAL");
  }

  const VulkanImage *output = input.source;
  if (input.request_crop) {
    OpenVulkanAutoFrameCropInput crop;
    crop.src = input.source;
    crop.dst = input.crop_output;
    crop.crop_x = input.crop_x;
    crop.crop_y = input.crop_y;
    crop.crop_w = input.crop_width;
    crop.crop_h = input.crop_height;
    crop.host_analysis_complete = input.host_analysis_complete;
    crop.cpu_crop_plan_complete = input.cpu_crop_plan_complete;
    if (!auto_frame->ApplyCrop(crop, counters, error_out))
      return false;
    output = input.crop_output;
    state->auto_frame_crop_applied = true;
  } else {
    ++counters->identity_resident_output_frames;
  }

  if (!output || !output->Valid() || output->mapped() ||
      !output->device_local()) {
    ++counters->residency_rejection_frames;
    return fail("[vulkan_effect_residency_contract_failed] Auto Frame output "
                "must be non-mapped DEVICE_LOCAL");
  }
  state->current = output;
  state->auto_frame_applied = true;
  ++counters->resident_output_frames;
  return true;
}

bool ExecuteOpenVulkanResidentFrameFinalStage(
    const OpenVulkanResidentFrameState &state,
    const OpenVulkanFinalResidentStageInput &input_without_source,
    const OpenVulkanFinalResidentStageResources &resources,
    OpenVulkanFinalResidentStageCounters *counters,
    OpenVulkanFinalResidentStageResult *result) {
  OpenVulkanFinalResidentStageInput input = input_without_source;
  input.source = state.current;
  return ExecuteOpenVulkanFinalResidentStage(input, resources, counters,
                                             result);
}

bool ExecuteOpenVulkanFinalResidentStage(
    const OpenVulkanFinalResidentStageInput &input,
    const OpenVulkanFinalResidentStageResources &resources,
    OpenVulkanFinalResidentStageCounters *counters,
    OpenVulkanFinalResidentStageResult *result) {
  if (!result)
    return false;
  *result = OpenVulkanFinalResidentStageResult{};
  if (!counters) {
    result->fatal_error =
        "Open Vulkan final resident stage counter output is null.";
    return false;
  }
  ++counters->execution_calls;

  auto *kernels = resources.kernels;
  if (!kernels || !kernels->Initialized() || !kernels->device() ||
      !input.source || !input.source->Valid() || input.output_width <= 0 ||
      input.output_height <= 0) {
    result->fatal_error =
        "Open Vulkan final resident stage input or shared context is invalid.";
    return false;
  }
  if (input.source->format() != VulkanPixelFormat::rgb_u8 &&
      input.source->format() != VulkanPixelFormat::bgr_u8) {
    result->fatal_error =
        "Open Vulkan final resident source must be RGB or BGR u8.";
    return false;
  }
  if (input.source->mapped() || !input.source->device_local()) {
    const std::string detail = ResidencyFailure(
        "the final-stage source is mapped or not DEVICE_LOCAL");
    if (input.request_fixed_center_vignette) {
      result->vignette_failed = true;
      result->vignette_error = OpenVulkanVignetteRuntimeFailure(detail);
    }
    if (input.request_mirror) {
      result->mirror_failed = true;
      result->mirror_error = OpenVulkanMirrorRuntimeFailure(detail);
    }
    result->fatal_error = detail;
    ++counters->residency_rejection_calls;
    return false;
  }

  const auto submissions_before = kernels->synchronous_submission_count();
  const bool resize_needed = input.source->width() != input.output_width ||
                             input.source->height() != input.output_height;
  const VulkanPixelFormat format = input.source->format();

  bool vignette_eligible = input.request_fixed_center_vignette;
  bool mirror_eligible = input.request_mirror;

  if (vignette_eligible && !input.preceding_effects_complete) {
    vignette_eligible = false;
    result->vignette_failed = true;
    result->vignette_error = OpenVulkanVignetteRuntimeFailure(
        "final-stage ordering boundary is not satisfied");
  }
  if (mirror_eligible && !input.unmirrored_analysis_complete) {
    mirror_eligible = false;
    result->mirror_failed = true;
    result->mirror_error = OpenVulkanMirrorRuntimeFailure(
        "final-stage ordering boundary is not satisfied");
  }

  if (!input.source->BelongsTo(*kernels->device())) {
    const std::string detail = ForeignSourceFailure();
    if (input.request_fixed_center_vignette) {
      result->vignette_failed = true;
      result->vignette_error = OpenVulkanVignetteRuntimeFailure(detail);
    }
    if (input.request_mirror) {
      result->mirror_failed = true;
      result->mirror_error = OpenVulkanMirrorRuntimeFailure(detail);
    }
    result->fatal_error = detail;
    return false;
  }

  bool resize_ready = !resize_needed;
  if (resize_needed &&
      (vignette_eligible || mirror_eligible ||
       (!input.request_fixed_center_vignette && !input.request_mirror))) {
    std::string resize_error;
    resize_ready = EnsureReusableImage(
        kernels, resources.resize_scratch, input.output_width,
        input.output_height, format, "resize scratch", counters, &resize_error);
    if (!resize_ready) {
      result->fatal_error = resize_error;
      if (vignette_eligible) {
        result->vignette_failed = true;
        result->vignette_initialization_failed = true;
        result->vignette_error =
            OpenVulkanVignetteInitializationFailure(resize_error);
      }
      if (mirror_eligible) {
        result->mirror_failed = true;
        result->mirror_initialization_failed = true;
        result->mirror_error = OpenVulkanMirrorRuntimeFailure(resize_error);
      }
    }
  }

  bool vignette_ready = false;
  if (vignette_eligible && resize_ready) {
    std::string detail;
    vignette_ready = resources.vignette && resources.vignette_counters &&
                     resources.vignette_output &&
                     resources.vignette->EnsureInitialized(
                         kernels, input.output_width, input.output_height,
                         input.vignette_intensity_percent,
                         resources.vignette_counters, &detail);
    if (vignette_ready) {
      vignette_ready = EnsureReusableImage(
          kernels, resources.vignette_output, input.output_width,
          input.output_height, format, "vignette output", counters, &detail);
    }
    if (!vignette_ready) {
      result->vignette_failed = true;
      result->vignette_initialization_failed = true;
      if (detail.find("[vulkan_effect_initialization_failed]") ==
          std::string::npos) {
        detail = OpenVulkanVignetteInitializationFailure(detail);
      }
      result->vignette_error = std::move(detail);
    }
  }

  bool mirror_ready = false;
  if (mirror_eligible && resize_ready) {
    std::string detail;
    mirror_ready =
        resources.mirror && resources.mirror_counters &&
        resources.mirror_output &&
        resources.mirror->EnsureInitialized(kernels, input.output_width,
                                            input.output_height, &detail);
    if (mirror_ready) {
      mirror_ready = EnsureReusableImage(
          kernels, resources.mirror_output, input.output_width,
          input.output_height, format, "mirror output", counters, &detail);
    }
    if (!mirror_ready) {
      result->mirror_failed = true;
      result->mirror_initialization_failed = true;
      result->mirror_error = OpenVulkanMirrorRuntimeFailure(detail);
    }
  }

  if (vignette_ready) {
    OpenVulkanVignetteFinalStageInput stage;
    stage.src = input.source;
    stage.resize_scratch = resize_needed ? resources.resize_scratch : nullptr;
    stage.dst = resources.vignette_output;
    stage.mirrored_dst = mirror_ready ? resources.mirror_output : nullptr;
    stage.preceding_effects_complete = input.preceding_effects_complete;
    stage.intensity_percent = input.vignette_intensity_percent;
    std::string detail;
    if (resources.vignette->ApplyFinal(stage, resources.vignette_counters,
                                       &detail)) {
      result->resize_applied = resize_needed;
      result->vignette_applied = true;
      result->vignette_backend = "open_vulkan";
      result->output = resources.vignette_output;
      if (mirror_ready) {
        // ApplyFinalVignetteU8x3 recorded mirror as the last stage in the same
        // resident batch; attribute that dispatch to the canonical wrapper's
        // public counter exactly as CameraPipeline did before extraction.
        ++resources.mirror_counters->dispatch_calls;
        result->mirror_applied = true;
        result->mirror_backend = "open_vulkan";
        result->output = resources.mirror_output;
      }
    } else {
      result->vignette_failed = true;
      result->vignette_error = std::move(detail);
      vignette_ready = false;
    }
  }

  // Vignette failure is isolated: a requested mirror retries independently
  // from the original unmirrored resident source.
  if (!result->mirror_applied && mirror_ready) {
    std::string detail;
    bool applied = false;
    if (resize_needed) {
      OpenVulkanMirrorResizeFinalStageInput stage;
      stage.src = input.source;
      stage.resized = resources.resize_scratch;
      stage.dst = resources.mirror_output;
      stage.unmirrored_analysis_complete = input.unmirrored_analysis_complete;
      applied = resources.mirror->ApplyResizeFinal(
          stage, resources.mirror_counters, &detail);
    } else {
      OpenVulkanMirrorFinalStageInput stage;
      stage.src = input.source;
      stage.dst = resources.mirror_output;
      stage.unmirrored_analysis_complete = input.unmirrored_analysis_complete;
      stage.output_geometry_ready = true;
      applied = resources.mirror->ApplyFinal(stage, resources.mirror_counters,
                                             &detail);
    }
    if (applied) {
      result->resize_applied = resize_needed;
      result->mirror_applied = true;
      result->mirror_backend = "open_vulkan";
      result->output = resources.mirror_output;
    } else {
      result->mirror_failed = true;
      result->mirror_error = std::move(detail);
    }
  }

  if (!input.request_fixed_center_vignette && !input.request_mirror) {
    if (!resize_needed) {
      result->output = input.source;
    } else if (resize_ready) {
      std::string detail;
      if (kernels->ResizeBilinear(*input.source, *resources.resize_scratch,
                                  &detail)) {
        result->resize_applied = true;
        result->output = resources.resize_scratch;
      } else {
        result->fatal_error = std::move(detail);
      }
    }
  }

  // Same-geometry optional-stage failure remains a valid pass-through frame.
  // A resize-needed frame cannot use an unresized source as final output.
  if (!result->output && !resize_needed)
    result->output = input.source;

  result->output_valid = result->output != nullptr;
  result->synchronous_completion_count =
      kernels->synchronous_submission_count() - submissions_before;
  result->resident_stage_count = (result->resize_applied ? 1u : 0u) +
                                 (result->vignette_applied ? 1u : 0u) +
                                 (result->mirror_applied ? 1u : 0u);

  if (!result->output_valid && result->fatal_error.empty()) {
    result->fatal_error =
        "Open Vulkan final resident stage could not produce final geometry.";
  }
  if (result->output_valid)
    ++counters->successful_output_frames;
  return result->output_valid;
}

} // namespace studiocast::video
