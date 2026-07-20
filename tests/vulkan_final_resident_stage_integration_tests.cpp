#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "core/video/open_vulkan_final_resident_stage.h"

namespace {

using studiocast::video::ExecuteOpenVulkanFinalResidentStage;
using studiocast::video::OpenVulkanFinalResidentStageCounters;
using studiocast::video::OpenVulkanFinalResidentStageInput;
using studiocast::video::OpenVulkanFinalResidentStageResources;
using studiocast::video::OpenVulkanFinalResidentStageResult;
using studiocast::video::OpenVulkanMirror;
using studiocast::video::OpenVulkanMirrorCounters;
using studiocast::video::OpenVulkanVignette;
using studiocast::video::OpenVulkanVignetteCounters;
using studiocast::vulkan::VulkanImage;
using studiocast::vulkan::VulkanPixelFormat;
using studiocast::vulkan::kernels::UtilityKernels;

bool Require(bool condition, const std::string &message) {
  if (!condition)
    std::cerr << message << '\n';
  return condition;
}

bool RequireVulkanRuntime() {
  const char *value = std::getenv("STUDIOCAST_REQUIRE_VULKAN_RUNTIME");
  return value && std::string(value) == "1";
}

int OptionalSkip(const std::string &reason) {
  std::cout << "[SKIP] Open Vulkan final resident stage: " << reason << '\n';
  return RequireVulkanRuntime() ? 1 : 0;
}

std::size_t ChannelIndex(int x, int y, int width, int channel) {
  return (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
          static_cast<std::size_t>(x)) *
             3u +
         static_cast<std::size_t>(channel);
}

std::uint8_t RoundClampByte(float value) {
  if (value <= 0.0f)
    return 0;
  if (value >= 255.0f)
    return 255;
  return static_cast<std::uint8_t>(static_cast<int>(value + 0.5f));
}

struct AxisSample {
  int first = 0;
  int second = 0;
  float fraction = 0.0f;
};

AxisSample SampleAxis(int source_length, int destination_length,
                      int destination_index) {
  const float scale = static_cast<float>(source_length) /
                      static_cast<float>(destination_length);
  const float source =
      (static_cast<float>(destination_index) + 0.5f) * scale - 0.5f;
  const int first = std::clamp(static_cast<int>(source), 0, source_length - 1);
  return {first, std::clamp(first + 1, 0, source_length - 1),
          source - static_cast<float>(first)};
}

std::vector<std::uint8_t>
ResizeReference(const std::vector<std::uint8_t> &source, int source_width,
                int source_height, int output_width, int output_height) {
  std::vector<std::uint8_t> output(static_cast<std::size_t>(output_width) *
                                   static_cast<std::size_t>(output_height) *
                                   3u);
  for (int y = 0; y < output_height; ++y) {
    const AxisSample sy = SampleAxis(source_height, output_height, y);
    for (int x = 0; x < output_width; ++x) {
      const AxisSample sx = SampleAxis(source_width, output_width, x);
      for (int channel = 0; channel < 3; ++channel) {
        const auto at = [&](int px, int py) {
          return static_cast<float>(
              source[ChannelIndex(px, py, source_width, channel)]);
        };
        const float top =
            at(sx.first, sy.first) +
            sx.fraction * (at(sx.second, sy.first) - at(sx.first, sy.first));
        const float bottom =
            at(sx.first, sy.second) +
            sx.fraction * (at(sx.second, sy.second) - at(sx.first, sy.second));
        output[ChannelIndex(x, y, output_width, channel)] =
            RoundClampByte(top + sy.fraction * (bottom - top));
      }
    }
  }
  return output;
}

std::vector<std::uint8_t>
MirrorReference(const std::vector<std::uint8_t> &source, int width,
                int height) {
  std::vector<std::uint8_t> output(source.size());
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int channel = 0; channel < 3; ++channel) {
        output[ChannelIndex(x, y, width, channel)] =
            source[ChannelIndex(width - 1 - x, y, width, channel)];
      }
    }
  }
  return output;
}

std::vector<std::uint8_t>
VignetteReference(const std::vector<std::uint8_t> &source, int width,
                  int height, int intensity_percent) {
  std::vector<std::uint8_t> output(source);
  const float intensity =
      static_cast<float>(std::clamp(intensity_percent, 0, 100)) / 100.0f;
  const float center_x = static_cast<float>(width) * 0.5f;
  const float center_y = static_cast<float>(height) * 0.5f;
  const float inv_half_w = 2.0f / static_cast<float>(width);
  const float inv_half_h = 2.0f / static_cast<float>(height);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const float fx = (static_cast<float>(x) + 0.5f - center_x) * inv_half_w;
      const float fy = (static_cast<float>(y) + 0.5f - center_y) * inv_half_h;
      float radius = std::sqrt(fx * fx + fy * fy) * 0.70710677f;
      radius = std::clamp(radius, 0.0f, 1.0f);
      const float factor = std::max(0.0f, 1.0f - intensity * radius * radius);
      for (int channel = 0; channel < 3; ++channel) {
        const std::size_t index = ChannelIndex(x, y, width, channel);
        output[index] =
            RoundClampByte(static_cast<float>(source[index]) * factor);
      }
    }
  }
  return output;
}

bool AllocateAndUpload(UtilityKernels *kernels, VulkanImage *upload_staging,
                       VulkanImage *image, int width, int height,
                       VulkanPixelFormat format,
                       const std::vector<std::uint8_t> &bytes,
                       std::string *error) {
  if (!upload_staging->Allocate(kernels->device(), width, height, format,
                                /*map_memory=*/true, error) ||
      !image->Allocate(kernels->device(), width, height, format,
                       /*map_memory=*/false, error)) {
    return false;
  }
  return kernels->UploadU8x3ToDeviceLocal(
      bytes.data(), static_cast<std::size_t>(width) * 3u, *upload_staging,
      *image, error);
}

std::vector<std::uint8_t> ReadFinalOutput(UtilityKernels *kernels,
                                          const VulkanImage &image,
                                          const VulkanImage &readback_staging,
                                          std::uint64_t *readback_count,
                                          std::string *error) {
  std::vector<std::uint8_t> output(static_cast<std::size_t>(image.width()) *
                                   static_cast<std::size_t>(image.height()) *
                                   3u);
  if (!kernels->ReadbackU8x3(
          image, readback_staging, output.data(),
          static_cast<std::size_t>(image.width()) * 3u, error)) {
    return {};
  }
  ++*readback_count;
  return output;
}

bool CompareBytes(const std::vector<std::uint8_t> &actual,
                  const std::vector<std::uint8_t> &expected, bool exact,
                  const std::string &label) {
  if (!Require(actual.size() == expected.size(), label + ": size mismatch"))
    return false;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const int difference = std::abs(static_cast<int>(actual[index]) -
                                    static_cast<int>(expected[index]));
    if ((exact && difference != 0) || (!exact && difference > 1)) {
      std::cerr << label << ": byte " << index << " got "
                << static_cast<int>(actual[index]) << " expected "
                << static_cast<int>(expected[index]) << '\n';
      return false;
    }
  }
  return true;
}

struct FinalStageFixture {
  VulkanImage resize_scratch;
  VulkanImage vignette_output;
  VulkanImage mirror_output;
  VulkanImage final_readback_staging;
  OpenVulkanVignette vignette;
  OpenVulkanVignetteCounters vignette_counters;
  OpenVulkanMirror mirror;
  OpenVulkanMirrorCounters mirror_counters;
  OpenVulkanFinalResidentStageCounters counters;

  OpenVulkanFinalResidentStageResources Resources(UtilityKernels *kernels) {
    OpenVulkanFinalResidentStageResources resources;
    resources.kernels = kernels;
    resources.resize_scratch = &resize_scratch;
    resources.vignette_output = &vignette_output;
    resources.mirror_output = &mirror_output;
    resources.vignette = &vignette;
    resources.vignette_counters = &vignette_counters;
    resources.mirror = &mirror;
    resources.mirror_counters = &mirror_counters;
    return resources;
  }
};

struct Case {
  const char *label;
  int source_width;
  int source_height;
  int output_width;
  int output_height;
  VulkanPixelFormat format;
  bool vignette;
  bool mirror;
  int intensity;
};

bool RunProductionCases(UtilityKernels *kernels) {
  const Case cases[] = {
      {"mirror RGB no resize", 5, 3, 5, 3, VulkanPixelFormat::rgb_u8, false,
       true, 0},
      {"mirror BGR resize", 3, 2, 5, 3, VulkanPixelFormat::bgr_u8, false, true,
       0},
      {"vignette RGB no resize", 6, 4, 6, 4, VulkanPixelFormat::rgb_u8, true,
       false, 73},
      {"vignette BGR resize", 3, 2, 5, 3, VulkanPixelFormat::bgr_u8, true,
       false, 35},
      {"vignette mirror RGB no resize", 5, 3, 5, 3, VulkanPixelFormat::rgb_u8,
       true, true, 73},
      {"vignette mirror BGR resize", 3, 2, 5, 3, VulkanPixelFormat::bgr_u8,
       true, true, 73},
  };

  bool ok = true;
  for (const Case &test : cases) {
    kernels->InvalidateDescriptorBindingCacheForSetup();
    std::vector<std::uint8_t> source(
        static_cast<std::size_t>(test.source_width) *
        static_cast<std::size_t>(test.source_height) * 3u);
    for (std::size_t index = 0; index < source.size(); ++index) {
      source[index] = static_cast<std::uint8_t>((index * 67u + 29u) & 0xffu);
    }
    std::string error;
    VulkanImage input_upload_staging;
    VulkanImage input_image;
    const std::uint64_t input_upload_submissions_before =
        kernels->synchronous_submission_count();
    if (!AllocateAndUpload(kernels, &input_upload_staging, &input_image,
                           test.source_width,
                           test.source_height, test.format, source, &error)) {
      std::cerr << test.label << ": source setup failed: " << error << '\n';
      return false;
    }
    ok &= Require(kernels->synchronous_submission_count() ==
                      input_upload_submissions_before + 1,
                  std::string(test.label) +
                      ": source preparation must use exactly one synchronized "
                      "host-staging to DEVICE_LOCAL upload");

    std::vector<std::uint8_t> expected = source;
    if (test.source_width != test.output_width ||
        test.source_height != test.output_height) {
      expected =
          ResizeReference(expected, test.source_width, test.source_height,
                          test.output_width, test.output_height);
    }
    if (test.vignette) {
      expected = VignetteReference(expected, test.output_width,
                                   test.output_height, test.intensity);
    }
    if (test.mirror) {
      expected =
          MirrorReference(expected, test.output_width, test.output_height);
    }

    FinalStageFixture fixture;
    if (!fixture.final_readback_staging.Allocate(
            kernels->device(), test.output_width, test.output_height,
            test.format, /*map_memory=*/true, &error)) {
      return Require(false, std::string(test.label) +
                                ": final readback staging failed: " + error);
    }
    if (test.vignette) {
      const std::uint64_t setup_submissions_before =
          kernels->synchronous_submission_count();
      if (!fixture.vignette.EnsureInitialized(
              kernels, test.output_width, test.output_height, test.intensity,
              &fixture.vignette_counters, &error)) {
        return Require(false, std::string(test.label) +
                                  ": vignette setup failed: " + error);
      }
      ok &= Require(
          kernels->synchronous_submission_count() ==
                  setup_submissions_before + 1 &&
              fixture.vignette.attenuation_factor_mask().device_local() &&
              !fixture.vignette.attenuation_factor_mask().mapped() &&
              fixture.vignette_counters.factor_allocation_calls == 1 &&
              fixture.vignette_counters.factor_staging_allocation_calls == 1 &&
              fixture.vignette_counters.factor_upload_calls == 1 &&
              fixture.vignette_counters.factor_upload_completion_calls == 1,
          std::string(test.label) +
              ": setup must perform one bounded staging upload into one "
              "persistent DEVICE_LOCAL attenuation mask");
    }
    auto resources = fixture.Resources(kernels);
    OpenVulkanFinalResidentStageInput input;
    input.source = &input_image;
    input.output_width = test.output_width;
    input.output_height = test.output_height;
    input.request_fixed_center_vignette = test.vignette;
    input.request_mirror = test.mirror;
    input.vignette_intensity_percent = test.intensity;
    input.preceding_effects_complete = true;
    input.unmirrored_analysis_complete = true;

    std::uint64_t final_readbacks = 0;
    for (int frame = 0; frame < 2; ++frame) {
      const auto allocations_before = kernels->device()->allocation_stats();
      const std::uint64_t helper_allocations_before =
          fixture.counters.resource_allocation_calls;
      OpenVulkanFinalResidentStageResult result;
      ok &= Require(ExecuteOpenVulkanFinalResidentStage(
                        input, resources, &fixture.counters, &result),
                    std::string(test.label) +
                        ": production helper failed: " + result.fatal_error);
      ok &= Require(result.output_valid && result.output &&
                        result.vignette_applied == test.vignette &&
                        result.mirror_applied == test.mirror &&
                        result.resize_applied ==
                            (test.source_width != test.output_width ||
                             test.source_height != test.output_height),
                    std::string(test.label) +
                        ": canonical stage attribution/order is wrong");
      if (!result.output_valid || !result.output)
        return false;
      ok &= Require(
          result.output->device_local() && !result.output->mapped() &&
              (!fixture.resize_scratch.Valid() ||
               (fixture.resize_scratch.device_local() &&
                !fixture.resize_scratch.mapped())) &&
              (!fixture.vignette_output.Valid() ||
               (fixture.vignette_output.device_local() &&
                !fixture.vignette_output.mapped())) &&
              (!fixture.mirror_output.Valid() ||
               (fixture.mirror_output.device_local() &&
                !fixture.mirror_output.mapped())) &&
              input_image.device_local() && !input_image.mapped() &&
              input_upload_staging.host_visible() &&
              input_upload_staging.mapped() &&
              fixture.final_readback_staging.host_visible() &&
              fixture.final_readback_staging.mapped(),
          std::string(test.label) +
              ": only upload/readback transports may be mapped; every "
              "source/intermediate/effect output must be DEVICE_LOCAL");
      ok &= Require(
          (!test.vignette || result.vignette_backend == "open_vulkan") &&
              (!test.mirror || result.mirror_backend == "open_vulkan"),
          std::string(test.label) + ": applied backend attribution is missing");
      const std::size_t expected_stage_count =
          (test.source_width != test.output_width ||
                   test.source_height != test.output_height
               ? 1u
               : 0u) +
          (test.vignette ? 1u : 0u) + (test.mirror ? 1u : 0u);
      ok &= Require(result.synchronous_completion_count == 1 &&
                        result.resident_stage_count == expected_stage_count,
                    std::string(test.label) +
                        ": resident stages must share one completion");

      const std::uint64_t readback_submissions_before =
          kernels->synchronous_submission_count();
      const auto actual = ReadFinalOutput(
          kernels, *result.output, fixture.final_readback_staging,
          &final_readbacks, &error);
      ok &= Require(!actual.empty(), std::string(test.label) +
                                         ": final readback failed: " + error);
      ok &= Require(kernels->synchronous_submission_count() ==
                        readback_submissions_before + 1,
                    std::string(test.label) +
                        ": the final host transport must use exactly one "
                        "synchronized readback submission");
      ok &= CompareBytes(actual, expected,
                         /*exact=*/!test.vignette &&
                             test.source_width == test.output_width &&
                             test.source_height == test.output_height,
                         test.label);

      if (frame == 1) {
        const auto allocations_after = kernels->device()->allocation_stats();
        ok &= Require(
            allocations_after.current_bytes ==
                    allocations_before.current_bytes &&
                allocations_after.high_water_bytes ==
                    allocations_before.high_water_bytes &&
                allocations_after.allocation_count ==
                    allocations_before.allocation_count &&
                fixture.counters.resource_allocation_calls ==
                    helper_allocations_before,
            std::string(test.label) +
                ": repeated frame must reuse bounded setup allocations");
      }
    }
    ok &= Require(final_readbacks == 2 &&
                      fixture.counters.successful_output_frames == 2 &&
                      fixture.counters.intermediate_readback_calls == 0 &&
                      fixture.counters.cpu_fallback_calls == 0 &&
                      fixture.counters.host_visible_intermediate_allocation_calls ==
                          0 &&
                      fixture.counters.residency_rejection_calls == 0 &&
                      fixture.counters.device_local_allocation_calls ==
                          fixture.counters.resource_allocation_calls &&
                      fixture.vignette_counters.runtime_failure_frames == 0 &&
                      fixture.mirror_counters.runtime_failure_frames == 0,
                  std::string(test.label) +
                      ": each frame needs exactly one boundary readback and "
                      "zero intermediate readback/CPU fallback");
  }
  return ok;
}

bool RunComputeProducedSameSizeMirrorCase(UtilityKernels *kernels) {
  constexpr int width = 5;
  constexpr int height = 3;
  // Prior production fixtures released their images. Prevent recycled opaque
  // handles from matching stale descriptor tuples before this new setup.
  kernels->InvalidateDescriptorBindingCacheForSetup();
  std::vector<std::uint8_t> source(static_cast<std::size_t>(width) *
                                   static_cast<std::size_t>(height) * 3u);
  for (std::size_t index = 0; index < source.size(); ++index)
    source[index] = static_cast<std::uint8_t>((index * 53u + 31u) & 0xffu);

  std::string error;
  VulkanImage upload_staging;
  VulkanImage uploaded;
  VulkanImage compute_produced;
  if (!AllocateAndUpload(kernels, &upload_staging, &uploaded, width, height,
                         VulkanPixelFormat::rgb_u8, source, &error) ||
      !compute_produced.Allocate(kernels->device(), width, height,
                                 VulkanPixelFormat::rgb_u8,
                                 /*map_memory=*/false, &error)) {
    return Require(false,
                   "compute-produced final-stage setup failed: " + error);
  }

  const auto submissions_before = kernels->synchronous_submission_count();
  if (!Require(kernels->ResizeBilinear(uploaded, compute_produced, &error),
               "preceding real Vulkan producer failed: " + error)) {
    return false;
  }

  FinalStageFixture fixture;
  if (!fixture.final_readback_staging.Allocate(
          kernels->device(), width, height, VulkanPixelFormat::rgb_u8,
          /*map_memory=*/true, &error)) {
    return Require(false,
                   "compute-produced final readback setup failed: " + error);
  }
  auto resources = fixture.Resources(kernels);
  OpenVulkanFinalResidentStageInput input;
  input.source = &compute_produced;
  input.output_width = width;
  input.output_height = height;
  input.request_mirror = true;
  input.preceding_effects_complete = true;
  input.unmirrored_analysis_complete = true;
  OpenVulkanFinalResidentStageResult result;
  bool ok = Require(ExecuteOpenVulkanFinalResidentStage(
                        input, resources, &fixture.counters, &result),
                    "canonical same-size mirror after GPU producer failed: " +
                        result.fatal_error);
  ok &= Require(
      result.output_valid && result.output && result.mirror_applied &&
          !result.resize_applied && result.synchronous_completion_count == 1 &&
          result.resident_stage_count == 1 &&
          kernels->synchronous_submission_count() == submissions_before + 2,
      "GPU producer and exact canonical same-size mirror must execute as two "
      "ordered resident submissions");
  std::uint64_t final_readbacks = 0;
  if (!result.output)
    return false;
  const auto actual = ReadFinalOutput(
      kernels, *result.output, fixture.final_readback_staging,
      &final_readbacks, &error);
  ok &= CompareBytes(actual, MirrorReference(source, width, height), true,
                     "compute-produced canonical same-size mirror");
  ok &= Require(final_readbacks == 1 &&
                    fixture.counters.intermediate_readback_calls == 0 &&
                    fixture.counters.cpu_fallback_calls == 0,
                "compute-produced final path must retain one final-only "
                "readback and no CPU/intermediate fallback");
  return ok;
}

bool RunOrderingAndIsolationCases(UtilityKernels *kernels) {
  constexpr int width = 5;
  constexpr int height = 3;
  std::vector<std::uint8_t> source(static_cast<std::size_t>(width) *
                                   static_cast<std::size_t>(height) * 3u);
  for (std::size_t index = 0; index < source.size(); ++index)
    source[index] = static_cast<std::uint8_t>((index * 41u + 17u) & 0xffu);

  std::string error;
  VulkanImage input_upload_staging;
  VulkanImage input_image;
  if (!AllocateAndUpload(kernels, &input_upload_staging, &input_image, width,
                         height,
                         VulkanPixelFormat::rgb_u8, source, &error)) {
    return Require(false, "isolation source setup failed: " + error);
  }
  FinalStageFixture fixture;
  if (!fixture.final_readback_staging.Allocate(
          kernels->device(), width, height, VulkanPixelFormat::rgb_u8,
          /*map_memory=*/true, &error)) {
    return Require(false, "isolation final readback setup failed: " + error);
  }
  auto resources = fixture.Resources(kernels);
  OpenVulkanFinalResidentStageInput input;
  input.source = &input_image;
  input.output_width = width;
  input.output_height = height;
  input.request_fixed_center_vignette = true;
  input.request_mirror = true;
  input.vignette_intensity_percent = 73;
  input.preceding_effects_complete = false;
  input.unmirrored_analysis_complete = true;

  const auto submissions_before = kernels->synchronous_submission_count();
  OpenVulkanFinalResidentStageResult result;
  bool ok = Require(ExecuteOpenVulkanFinalResidentStage(
                        input, resources, &fixture.counters, &result),
                    "isolated ordering failure must preserve mirror output");
  ok &= Require(
      result.vignette_failed && !result.vignette_applied &&
          result.vignette_error.find("ordering boundary") !=
              std::string::npos &&
          result.mirror_applied && result.mirror_backend == "open_vulkan" &&
          kernels->synchronous_submission_count() == submissions_before + 1,
      "vignette ordering failure must not disable unrelated mirror");
  std::uint64_t final_readbacks = 0;
  if (!result.output)
    return false;
  const auto actual = ReadFinalOutput(
      kernels, *result.output, fixture.final_readback_staging,
      &final_readbacks, &error);
  ok &= CompareBytes(actual, MirrorReference(source, width, height), true,
                     "isolated mirror output");
  ok &= Require(final_readbacks == 1 &&
                    fixture.counters.intermediate_readback_calls == 0 &&
                    fixture.counters.cpu_fallback_calls == 0,
                "isolated failure path must retain one final-only readback");

  // A rejected optional stage is not latched inside the helper. The next
  // valid frame can execute it with the same reusable resources.
  input.request_mirror = false;
  input.preceding_effects_complete = true;
  ok &= Require(ExecuteOpenVulkanFinalResidentStage(
                    input, resources, &fixture.counters, &result) &&
                    result.vignette_applied && !result.mirror_applied,
                "a prior isolated ordering failure must not disable vignette");

  UtilityKernels foreign_kernels;
  if (!foreign_kernels.Initialize(&error))
    return Require(false, "foreign context setup failed: " + error);
  VulkanImage foreign_upload_staging;
  VulkanImage foreign_source;
  if (!AllocateAndUpload(&foreign_kernels, &foreign_upload_staging,
                         &foreign_source, width, height,
                         VulkanPixelFormat::rgb_u8, source, &error)) {
    return Require(false, "foreign source setup failed: " + error);
  }
  input.source = &foreign_source;
  input.request_fixed_center_vignette = true;
  input.request_mirror = true;
  const auto main_submissions = kernels->synchronous_submission_count();
  const auto foreign_submissions =
      foreign_kernels.synchronous_submission_count();
  ok &= Require(
      !ExecuteOpenVulkanFinalResidentStage(input, resources, &fixture.counters,
                                           &result) &&
          result.vignette_failed && result.mirror_failed &&
          result.vignette_error.find("[vulkan_foreign_context]") !=
              std::string::npos &&
          result.mirror_error.find("[vulkan_foreign_context]") !=
              std::string::npos &&
          kernels->synchronous_submission_count() == main_submissions &&
          foreign_kernels.synchronous_submission_count() == foreign_submissions,
      "foreign-context input must fail closed before any dispatch");

  input.source = &input_image;
  input.request_fixed_center_vignette = false;
  input.request_mirror = true;
  ok &= Require(ExecuteOpenVulkanFinalResidentStage(
                    input, resources, &fixture.counters, &result) &&
                    result.mirror_applied,
                "context mismatch must not latch or disable a later mirror");

  VulkanImage mapped_source;
  if (!mapped_source.Allocate(kernels->device(), width, height,
                              VulkanPixelFormat::rgb_u8,
                              /*map_memory=*/true, &error)) {
    return Require(false, "mapped-source rejection setup failed: " + error);
  }
  input.source = &mapped_source;
  const auto submissions_before_residency_rejection =
      kernels->synchronous_submission_count();
  ok &= Require(
      !ExecuteOpenVulkanFinalResidentStage(input, resources, &fixture.counters,
                                           &result) &&
          result.mirror_error.find(
              "[vulkan_effect_residency_contract_failed]") !=
              std::string::npos &&
          fixture.counters.residency_rejection_calls == 1 &&
          kernels->synchronous_submission_count() ==
              submissions_before_residency_rejection,
      "mapped source must fail the production residency contract before any "
      "dispatch or fallback");
  return ok;
}

} // namespace

int main() {
  UtilityKernels kernels;
  std::string error;
  if (!kernels.Initialize(&error))
    return OptionalSkip(error);
  const auto diagnostics = kernels.Diagnostics();
  if (!diagnostics.non_cpu_device_selected || diagnostics.cpu_device_selected)
    return OptionalSkip("selected Vulkan device is CPU/software, not hardware");

  bool ok = Require(diagnostics.compute_queue_available &&
                        diagnostics.context_healthy &&
                        diagnostics.production_hardware_ready,
                    "real-device integration requires a healthy non-CPU "
                    "compute context");
  ok &= RunProductionCases(&kernels);
  ok &= RunComputeProducedSameSizeMirrorCase(&kernels);
  ok &= RunOrderingAndIsolationCases(&kernels);
  if (ok)
    std::cout << "Open Vulkan final resident production-path tests passed\n";
  return ok ? 0 : 1;
}
