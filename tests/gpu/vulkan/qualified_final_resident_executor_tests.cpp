#include "core/vulkan/qualified_final_resident_executor.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <new>
#include <string>
#include <vector>

namespace {
std::atomic<std::uint64_t> g_allocations{0};
}

void *operator new(std::size_t size) {
  ++g_allocations;
  if (void *pointer = std::malloc(size))
    return pointer;
  throw std::bad_alloc();
}
void *operator new[](std::size_t size) {
  ++g_allocations;
  if (void *pointer = std::malloc(size))
    return pointer;
  throw std::bad_alloc();
}
void operator delete(void *pointer) noexcept { std::free(pointer); }
void operator delete[](void *pointer) noexcept { std::free(pointer); }
void operator delete(void *pointer, std::size_t) noexcept {
  std::free(pointer);
}
void operator delete[](void *pointer, std::size_t) noexcept {
  std::free(pointer);
}

namespace {

using namespace studiocast;
static_assert(sizeof(vulkan::VkPhysicalDeviceIDProperties) == 64);
static_assert(alignof(vulkan::VkPhysicalDeviceProperties) >= 8);
static_assert(sizeof(vulkan::VkPhysicalDeviceProperties) >= 824);
static_assert(offsetof(vulkan::VkPhysicalDeviceProperties2, properties) == 16);
using hal::AliasingPermission;
using hal::BackendId;
using hal::ChannelOrder;
using hal::ColorRange;
using hal::ColorSpace;
using hal::FailureAction;
using hal::HostCoherence;
using hal::ImageDescriptor;
using hal::Mutability;
using hal::PixelFormat;
using hal::PreparedProgram;
using hal::ReasonId;
using hal::SemanticEffectId;
using hal::SemanticEffectProviderId;
using hal::StatusId;
using vulkan::QualifiedFinalResidentExecutor;
using vulkan::VulkanExecutorPreparedProgram;
using vulkan::VulkanHostFrameBinding;
using vulkan::VulkanResidentProgramOptions;
using vulkan::VulkanVignetteCenterMode;
using vulkan::kernels::UtilityKernels;

constexpr std::uint64_t kAdapterId = 77;
constexpr int kWidth = 5;
constexpr int kHeight = 3;
constexpr std::size_t kStride = kWidth * 3u;
constexpr std::size_t kBytes = kStride * kHeight;

bool Require(bool condition, const std::string &message) {
  if (!condition)
    std::cerr << message << '\n';
  return condition;
}

int OptionalSkip(const std::string &reason) {
  std::cout << "[SKIP] qualified Vulkan executor: " << reason << '\n';
  const char *required = std::getenv("STUDIOCAST_REQUIRE_VULKAN_RUNTIME");
  return required && std::string(required) == "1" ? 1 : 0;
}

ImageDescriptor Descriptor(Mutability mutability, bool host) {
  ImageDescriptor descriptor;
  descriptor.width = kWidth;
  descriptor.height = kHeight;
  descriptor.plane_count = 1;
  descriptor.format = PixelFormat::rgb_u8;
  descriptor.channel_order = ChannelOrder::rgb;
  descriptor.color_space = ColorSpace::srgb;
  descriptor.color_range = ColorRange::full;
  descriptor.host_coherence =
      host ? HostCoherence::coherent : HostCoherence::not_host_visible;
  descriptor.mutability = mutability;
  descriptor.aliasing = AliasingPermission::forbidden;
  descriptor.base_alignment_bytes = 1;
  descriptor.planes[0].row_stride_bytes = kStride;
  descriptor.planes[0].size_bytes = kBytes;
  descriptor.planes[0].width = kWidth;
  descriptor.planes[0].height = kHeight;
  descriptor.planes[0].alignment_bytes = 1;
  return descriptor;
}

PreparedProgram Program(const hal::ExecutionDomainId &domain,
                        std::uint64_t id = 101) {
  PreparedProgram program;
  program.program_id = id;
  program.owner_adapter_instance_id = kAdapterId;
  program.execution_domain = domain;
  program.host_input = Descriptor(Mutability::read_only, true);
  program.host_output = Descriptor(Mutability::write_only, true);
  program.requested_backend = BackendId::vulkan;
  program.invalidation_key.canonical_plan_hash = 0xa500 + id;
  program.invalidation_key.generations.configuration = 1;
  program.invalidation_key.generations.geometry = 1;
  program.invalidation_key.generations.provider = 1;
  program.invalidation_key.generations.device = 1;
  program.invalidation_key.generations.context =
      domain.logical_context_generation;
  program.invalidation_key.execution_domain_hash = 0xb500 + id;
  program.budget.persistent_bytes = 65536;
  program.budget.scratch_bytes = 65536;
  program.budget.upload_staging_bytes = 65536;
  program.budget.readback_staging_bytes = 65536;
  program.budget.resource_count = 16;
  program.budget.in_flight_slots = 1;
  program.requires_upload = true;
  program.requires_final_readback = true;

  const auto resident_input = Descriptor(Mutability::read_only, false);
  const auto resident_output = Descriptor(Mutability::read_write, false);
  auto &vignette = program.stages[program.stage_count++];
  vignette.stage_id = 1;
  vignette.semantic = SemanticEffectId::vignette;
  vignette.provider = SemanticEffectProviderId::open_vulkan_production_kernels;
  vignette.backend = BackendId::vulkan;
  vignette.execution_domain = domain;
  vignette.input.image = resident_input;
  vignette.output.image = resident_output;
  vignette.failure_action = FailureAction::fail_open_preserve_input;

  auto &mirror = program.stages[program.stage_count++];
  mirror.stage_id = 2;
  mirror.semantic = SemanticEffectId::mirror;
  mirror.provider = SemanticEffectProviderId::open_vulkan_production_kernels;
  mirror.backend = BackendId::vulkan;
  mirror.execution_domain = domain;
  mirror.input.image = resident_input;
  mirror.output.image = resident_output;
  mirror.failure_action = FailureAction::fail_open_preserve_input;
  return program;
}

VulkanResidentProgramOptions Options(const PreparedProgram &program) {
  VulkanResidentProgramOptions options;
  options.program_id = program.program_id;
  options.canonical_plan_hash = program.invalidation_key.canonical_plan_hash;
  options.request_vignette = true;
  options.request_mirror = true;
  options.vignette_center = VulkanVignetteCenterMode::fixed_frame_center;
  options.vignette_intensity_percent = 70;
  return options;
}

std::vector<std::uint8_t>
VignetteThenMirror(const std::vector<std::uint8_t> &input, int intensity) {
  std::vector<std::uint8_t> vignette(input.size());
  const float scale = static_cast<float>(intensity) / 100.0f;
  const float cx = static_cast<float>(kWidth) * 0.5f;
  const float cy = static_cast<float>(kHeight) * 0.5f;
  for (int y = 0; y < kHeight; ++y) {
    for (int x = 0; x < kWidth; ++x) {
      const float fx = (static_cast<float>(x) + 0.5f - cx) * 2.0f / kWidth;
      const float fy = (static_cast<float>(y) + 0.5f - cy) * 2.0f / kHeight;
      const float radius =
          std::clamp(std::sqrt(fx * fx + fy * fy) * 0.70710677f, 0.0f, 1.0f);
      const float factor = std::max(0.0f, 1.0f - scale * radius * radius);
      for (int c = 0; c < 3; ++c) {
        const auto index = (static_cast<std::size_t>(y) * kWidth + x) * 3 + c;
        vignette[index] = static_cast<std::uint8_t>(
            std::clamp(static_cast<int>(input[index] * factor + 0.5f), 0, 255));
      }
    }
  }
  std::vector<std::uint8_t> output(input.size());
  for (int y = 0; y < kHeight; ++y)
    for (int x = 0; x < kWidth; ++x)
      for (int c = 0; c < 3; ++c)
        output[(static_cast<std::size_t>(y) * kWidth + x) * 3 + c] =
            vignette[(static_cast<std::size_t>(y) * kWidth + (kWidth - 1 - x)) *
                         3 +
                     c];
  return output;
}

bool Close(const std::vector<std::uint8_t> &actual,
           const std::vector<std::uint8_t> &expected) {
  if (actual.size() != expected.size())
    return false;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (std::abs(static_cast<int>(actual[i]) - static_cast<int>(expected[i])) >
        1)
      return false;
  }
  return true;
}

} // namespace

int main() {
  UtilityKernels kernels;
  std::string error;
  if (!kernels.Initialize(&error))
    return OptionalSkip(error);
  const auto diagnostics = kernels.Diagnostics();
  if (!diagnostics.non_cpu_device_selected || diagnostics.cpu_device_selected)
    return OptionalSkip("selected device is CPU/software");

  QualifiedFinalResidentExecutor executor(kAdapterId, kernels);
  const auto evidence = executor.CapabilityEvidence();
  if (!kernels.device()->identity().device_uuid_valid)
    return OptionalSkip("Vulkan deviceUUID identity evidence is unavailable");
  bool ok = Require(evidence.blocker == ReasonId::none &&
                        evidence.mirror_production_ready &&
                        evidence.vignette_fixed_center_production_ready,
                    "healthy hardware must produce exact qualified evidence");

  auto program = Program(evidence.execution_domain);
  auto options = Options(program);
  VulkanExecutorPreparedProgram prepared;
  ReasonId reason = ReasonId::none;

  auto exhausted = program;
  exhausted.program_id = 102;
  exhausted.invalidation_key.canonical_plan_hash = 0xa566;
  exhausted.budget.persistent_bytes = 1;
  auto exhausted_options = Options(exhausted);
  exhausted_options.program_id = exhausted.program_id;
  exhausted_options.canonical_plan_hash =
      exhausted.invalidation_key.canonical_plan_hash;
  VulkanExecutorPreparedProgram rejected;
  ok &= Require(executor.Prepare(exhausted, exhausted_options, &rejected,
                                 &reason) == StatusId::resource_exhausted &&
                    rejected.state_id == 0,
                "resource exhaustion must roll back the candidate");

  const auto prepare_status = executor.Prepare(program, options, &prepared,
                                               &reason);
  ok &= Require(prepare_status == StatusId::ready && prepared.state_id != 0,
                "qualified program must prepare");

  std::vector<std::uint8_t> input(kBytes);
  for (std::size_t i = 0; i < input.size(); ++i)
    input[i] = static_cast<std::uint8_t>((i * 47u + 13u) & 0xffu);
  std::vector<std::uint8_t> output(kBytes, 0);
  VulkanHostFrameBinding binding;
  binding.input = input.data();
  binding.input_size_bytes = input.size();
  binding.input_row_stride_bytes = kStride;
  binding.output = output.data();
  binding.output_size_bytes = output.size();
  binding.output_row_stride_bytes = kStride;

  auto foreign = program;
  foreign.execution_domain.logical_context_generation++;
  auto foreign_result =
      executor.Execute(prepared, foreign, options, binding, 1);
  ok &= Require(foreign_result.status == StatusId::blocked &&
                    foreign_result.reason == ReasonId::wrong_execution_domain &&
                    foreign_result.transfers.uploads == 0,
                "foreign execution domain must fail before transfer");

  auto zero_program = program;
  zero_program.stages = {};
  zero_program.stage_count = 0;
  auto zero_options = options;
  zero_options.request_vignette = false;
  zero_options.request_mirror = false;
  const auto allocations_before = g_allocations.load();
  const auto device_allocations_before = kernels.device()->allocation_stats();
  auto zero_result =
      executor.Execute(prepared, zero_program, zero_options, binding, 2);
  const auto allocations_after = g_allocations.load();
  const auto device_allocations_after = kernels.device()->allocation_stats();
  if (allocations_after != allocations_before)
    std::cerr << "zero-stage host allocations: "
              << (allocations_after - allocations_before) << '\n';
  ok &= Require(
      zero_result.status == StatusId::executed && zero_result.output_written &&
          zero_result.transfers.uploads == 1 &&
          zero_result.transfers.final_readbacks == 1 &&
          zero_result.transfers.synchronizations == 2 && output == input,
      "zero-stage subset must preserve bytes with one upload/readback");
  ok &= Require(allocations_after == allocations_before &&
                    device_allocations_after.allocation_count ==
                        device_allocations_before.allocation_count,
                "frame execution must not allocate host or Vulkan resources");

  std::fill(output.begin(), output.end(), 0);
  const auto full_allocations_before = g_allocations.load();
  const auto full_device_allocations_before =
      kernels.device()->allocation_stats();
  auto full_result = executor.Execute(prepared, program, options, binding, 3);
  const auto full_allocations_after = g_allocations.load();
  const auto full_device_allocations_after =
      kernels.device()->allocation_stats();
  if (full_allocations_after != full_allocations_before)
    std::cerr << "full-stage host allocations: "
              << (full_allocations_after - full_allocations_before) << '\n';
  ok &= Require(
      full_result.status == StatusId::executed && full_result.output_written &&
          full_result.stage_count == 2 && full_result.stages[0].succeeded &&
          full_result.stages[1].succeeded &&
          full_result.stages[0].elapsed_ns > 0 &&
          full_result.stages[1].elapsed_ns > 0 &&
          full_result.transfers.uploads == 1 &&
          full_result.transfers.final_readbacks == 1 &&
          full_result.transfers.synchronizations == 3 &&
          Close(output, VignetteThenMirror(input, 70)),
      "qualified island must execute vignette before mirror in one "
      "resident submission");
  ok &= Require(full_allocations_after == full_allocations_before &&
                    full_device_allocations_after.allocation_count ==
                        full_device_allocations_before.allocation_count,
                "full prepared execution must not allocate");

  VulkanExecutorPreparedProgram stale = prepared;
  executor.DestroyPrepared(&prepared);
  auto stale_result = executor.Execute(stale, program, options, binding, 4);
  ok &= Require(stale_result.status == StatusId::stale &&
                    stale_result.reason == ReasonId::stale_program,
                "destroyed prepared state must be stale");

  vulkan::VulkanHalResidentAdapter adapter(kAdapterId, executor);
  auto adapter_program = Program(evidence.execution_domain, 104);
  auto adapter_options = Options(adapter_program);
  hal::PreparedProgramHandle adapter_prepared;
  hal::ReadinessSnapshot readiness;
  const bool options_set = adapter.SetProgramOptions(adapter_options);
  const auto adapter_prepare_status = adapter.Prepare(
      adapter_program, &adapter_prepared, &readiness, &reason);
  ok &= Require(options_set && adapter_prepare_status == StatusId::ready,
                "common Vulkan adapter must prepare the production executor");
  hal::ResourceHandle adapter_input;
  ok &= Require(adapter.BindFrame(adapter_program.program_id, 5, binding,
                                  &adapter_input, &reason) == StatusId::ready,
                "common Vulkan adapter must bind exact host boundaries");
  hal::ExecuteRequest adapter_request;
  adapter_request.invalidation_key = adapter_program.invalidation_key;
  adapter_request.execution_domain = adapter_program.execution_domain;
  adapter_request.capture_sequence = 5;
  adapter_request.input = adapter_input;
  const auto adapter_result =
      adapter.Execute(adapter_prepared, adapter_request);
  ok &= Require(adapter_result.status == StatusId::executed &&
                    adapter_result.output_valid &&
                    adapter_result.stage_count == 2 &&
                    adapter_result.stages[0].elapsed_ns > 0 &&
                    adapter_result.stages[1].elapsed_ns > 0 &&
                    adapter_result.transfers.uploads == 1 &&
                    adapter_result.transfers.final_readbacks == 1,
                "common adapter must preserve production output, transfers, "
                "and per-stage timing");
  adapter.DestroyPrepared(&adapter_prepared);

  auto loss_program = Program(evidence.execution_domain, 103);
  auto loss_options = Options(loss_program);
  VulkanExecutorPreparedProgram loss_prepared;
  ok &= Require(executor.Prepare(loss_program, loss_options, &loss_prepared,
                                 &reason) == StatusId::ready,
                "device-loss fixture must prepare");
  kernels.device()->InjectNextSubmissionResultForTesting(
      vulkan::VulkanSubmissionPhase::queue_submit,
      vulkan::VK_ERROR_DEVICE_LOST);
  auto loss_result =
      executor.Execute(loss_prepared, loss_program, loss_options, binding, 6);
  ok &= Require(loss_result.status == StatusId::quarantined &&
                    loss_result.reason == ReasonId::device_lost &&
                    loss_result.device_lost && !loss_result.output_written,
                "driver device loss must quarantine the prepared island");
  auto quarantined_result =
      executor.Execute(loss_prepared, loss_program, loss_options, binding, 7);
  ok &= Require(quarantined_result.status == StatusId::quarantined &&
                    quarantined_result.reason == ReasonId::device_lost &&
                    quarantined_result.transfers.uploads == 0,
                "quarantined island must reject later frames before transfer");
  executor.DestroyPrepared(&loss_prepared);

  if (ok)
    std::cout << "Qualified Vulkan final resident executor tests passed\n";
  return ok ? 0 : 1;
}
