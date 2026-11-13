#include "core/vulkan/vulkan_device.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>

int main() {
  const auto *required =
      std::getenv("STUDIOCAST_REQUIRE_VULKAN_MODEL_DIAGNOSTICS");
  if (!required || std::string(required) != "1") {
    std::cout
        << "SKIP: opt-in prepared model diagnostics; use isolated HOME/XDG\n";
    return 77;
  }
  const auto d = studiocast::vulkan::DiagnoseOpenVulkanDefault();
  const auto available = [&](const char *name) {
    return std::find(d.available_effects.begin(), d.available_effects.end(),
                     name) != d.available_effects.end();
  };
  const bool ready =
      d.ok && d.non_cpu_device_selected && d.context_healthy &&
      d.matting_production_ready && d.matting_parity_validated &&
      d.eye_contact_production_ready && d.eye_contact_parity_validated &&
      d.eye_contact_device_resident_analysis &&
      d.eye_contact_device_resident_tensor_io &&
      d.eye_contact_dispatch_count > 0 &&
      d.eye_contact_cpu_readback_count == 0 &&
      d.eye_contact_cpu_fallback_count == 0 &&
      d.video_noise_removal_production_ready &&
      d.video_noise_removal_parity_validated &&
      d.video_noise_removal_temporal_history_device_resident &&
      d.video_noise_removal_dispatch_count > 0 &&
      d.video_noise_removal_cpu_readback_count == 0 &&
      d.video_noise_removal_cpu_fallback_count == 0 &&
      d.auto_frame_production_ready && d.auto_frame_gpu_tracking_ready &&
      d.auto_frame_model_contract_validated && d.auto_frame_warmup_complete &&
      d.auto_frame_parity_validated && !d.auto_frame_cpu_tail &&
      d.auto_frame_shared_device_imported &&
      d.auto_frame_queue_ownership_explicit &&
      d.auto_frame_bounded_reusable_allocations &&
      d.auto_frame_synchronization_contract_validated &&
      !d.auto_frame_cpu_layers_used &&
      d.auto_frame_dynamic_allocation_count == 0 &&
      d.auto_frame_cpu_readback_count == 0 &&
      d.auto_frame_cpu_fallback_count == 0 && d.auto_frame_dispatch_count > 0 &&
      d.vignette_tracked_center_production_ready;
  if (!ready) {
    std::cerr << "FAIL incomplete actual prepared model diagnostics\n"
              << d.ToJson() << '\n';
    return 1;
  }
  for (const auto *name :
       {"video_noise_removal", "eye_contact", "auto_frame",
        "virtual_background.blur", "virtual_background.remove",
        "virtual_background.replace", "virtual_key_light", "vignette",
        "mirror"})
    if (!available(name)) {
      std::cerr << "FAIL missing available effect: " << name << '\n';
      return 1;
    }
  if (d.eye_contact_reason_code != "open_vulkan_eye_contact_production_ready" ||
      d.video_noise_removal_reason_code !=
          "open_vulkan_video_noise_removal_production_ready" ||
      d.auto_frame_readiness_code !=
          "open_vulkan_auto_frame_production_ready") {
    std::cerr << "FAIL exact success tokens missing\n";
    return 1;
  }
  std::cout << "PASS actual same-owner prepared diagnostics: all nine effects, "
               "exact reviewed profiles, "
               "resident analysis/history and zero CPU tails; setup dispatches="
            << d.eye_contact_dispatch_count << '/'
            << d.video_noise_removal_dispatch_count << '/'
            << d.auto_frame_dispatch_count << '\n';
  return 0;
}
