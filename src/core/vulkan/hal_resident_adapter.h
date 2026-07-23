#pragma once

#include "core/hal/prepared_program.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace studiocast::vulkan {

inline constexpr std::size_t kMaxVulkanPreparedPrograms = 4;

enum class VulkanVignetteCenterMode : std::uint8_t {
  fixed_frame_center = 1,
  tracked_face,
};

// Adapter-specific setup values are bound to the neutral canonical-plan hash.
// They are supplied only while building a prepared program; the frame path
// never reparses configuration or infers vignette semantics.
struct VulkanResidentProgramOptions {
  std::uint64_t program_id = 0;
  std::uint64_t canonical_plan_hash = 0;
  bool request_mirror = false;
  bool request_vignette = false;
  VulkanVignetteCenterMode vignette_center =
      VulkanVignetteCenterMode::fixed_frame_center;
  int vignette_intensity_percent = 0;
};

// Typed production evidence corresponding to
// docs/INSTALLER_VULKAN_CAPABILITIES.md. Every common fact is additive;
// loader or device presence alone can never qualify a program.
struct VulkanResidentCapabilityEvidence {
  hal::ExecutionDomainId execution_domain{};
  bool build_enabled = false;
  bool runtime_loaded = false;
  bool non_cpu_physical_device = false;
  bool compute_queue_available = false;
  bool owned_context_healthy = false;
  bool utility_kernels_ready = false;
  bool exact_device_and_queue_shared = false;
  bool persistent_device_local_resources = false;
  bool bounded_reusable_allocations = false;
  bool no_host_intermediate_storage = false;
  bool one_upload_transport = false;
  bool one_final_readback_transport = false;
  bool mirror_production_ready = false;
  bool vignette_fixed_center_production_ready = false;
  hal::ReasonId blocker = hal::ReasonId::none;
};

struct VulkanExecutorPreparedProgram {
  std::uint64_t state_id = 0;
  hal::ResourceBudget reserved_budget{};
  std::array<std::uint64_t, 4> opaque{};
};

struct VulkanHostFrameBinding {
  const std::uint8_t *input = nullptr;
  std::size_t input_size_bytes = 0;
  std::size_t input_row_stride_bytes = 0;
  std::uint8_t *output = nullptr;
  std::size_t output_size_bytes = 0;
  std::size_t output_row_stride_bytes = 0;
};

struct VulkanExecutorStageResult {
  bool succeeded = false;
  hal::ReasonId reason = hal::ReasonId::none;
  std::uint64_t elapsed_ns = 0;
};

struct VulkanExecutorResult {
  hal::StatusId status = hal::StatusId::failed;
  hal::ReasonId reason = hal::ReasonId::none;
  bool output_written = false;
  bool device_lost = false;
  std::array<VulkanExecutorStageResult, hal::kMaxPreparedStages> stages{};
  std::size_t stage_count = 0;
  hal::TransferCounters transfers{};
  std::uint64_t frame_elapsed_ns = 0;
};

// Implementations must be synchronous and must not retain a frame binding.
// Production implementations are expected to wrap the existing canonical
// OpenVulkan final resident stage and report actual queue submissions and
// transport crossings. This interface deliberately provides no import/export
// hook for another adapter or runtime.
class IQualifiedVulkanResidentExecutor {
public:
  virtual ~IQualifiedVulkanResidentExecutor() = default;

  [[nodiscard]] virtual VulkanResidentCapabilityEvidence
  CapabilityEvidence() const noexcept = 0;
  virtual hal::StatusId
  Prepare(const hal::PreparedProgram &program,
          const VulkanResidentProgramOptions &options,
          VulkanExecutorPreparedProgram *prepared,
          hal::ReasonId *reason) noexcept = 0;
  virtual VulkanExecutorResult
  Execute(const VulkanExecutorPreparedProgram &prepared,
          const hal::PreparedProgram &program,
          const VulkanResidentProgramOptions &options,
          const VulkanHostFrameBinding &binding,
          std::uint64_t capture_sequence) noexcept = 0;
  virtual void
  DestroyPrepared(VulkanExecutorPreparedProgram *prepared) noexcept = 0;
  virtual void Shutdown() noexcept = 0;
};

class VulkanHalResidentAdapter final : public hal::IResidentAdapter {
public:
  VulkanHalResidentAdapter(std::uint64_t adapter_instance_id,
                           IQualifiedVulkanResidentExecutor &executor) noexcept;
  ~VulkanHalResidentAdapter() override;
  VulkanHalResidentAdapter(const VulkanHalResidentAdapter &) = delete;
  VulkanHalResidentAdapter &
  operator=(const VulkanHalResidentAdapter &) = delete;

  [[nodiscard]] std::uint64_t adapter_instance_id() const noexcept override {
    return adapter_instance_id_;
  }
  [[nodiscard]] hal::BackendId backend_id() const noexcept override {
    return hal::BackendId::vulkan;
  }

  // Supplies exact setup-time options for the next matching Prepare call.
  // Existing prepared slots are immutable and remain valid on setup failure.
  [[nodiscard]] bool
  SetProgramOptions(const VulkanResidentProgramOptions &options) noexcept;

  // Creates an adapter-owned, one-in-flight frame binding without allocation.
  // The returned resource is the explicit upload boundary consumed by
  // PreparedProgramSession::Execute.
  [[nodiscard]] hal::StatusId
  BindFrame(std::uint64_t program_id, std::uint64_t capture_sequence,
            const VulkanHostFrameBinding &binding, hal::ResourceHandle *input,
            hal::ReasonId *reason) noexcept;

  hal::StatusId Prepare(const hal::PreparedProgram &program,
                        hal::PreparedProgramHandle *candidate,
                        hal::ReadinessSnapshot *readiness,
                        hal::ReasonId *reason) noexcept override;
  hal::ExecuteResult
  Execute(const hal::PreparedProgramHandle &program,
          const hal::ExecuteRequest &request) noexcept override;
  hal::StatusId
  ResetTemporalState(const hal::PreparedProgramHandle &program,
                     hal::ReasonId reason) noexcept override;
  void
  DestroyPrepared(hal::PreparedProgramHandle *program) noexcept override;
  void Shutdown() noexcept override;

private:
  struct PreparedSlot {
    bool occupied = false;
    bool quarantined = false;
    bool binding_armed = false;
    bool executing = false;
    hal::PreparedProgram program{};
    hal::PreparedProgramHandle handle{};
    VulkanResidentProgramOptions options{};
    VulkanExecutorPreparedProgram executor_program{};
    VulkanHostFrameBinding binding{};
    hal::ResourceHandle binding_resource{};
    std::uint64_t binding_capture_sequence = 0;
  };

  [[nodiscard]] PreparedSlot *
  FindSlot(const hal::PreparedProgramHandle &program) noexcept;
  [[nodiscard]] PreparedSlot *FindSlot(std::uint64_t program_id) noexcept;
  void ReleaseSlot(PreparedSlot *slot) noexcept;

  std::uint64_t adapter_instance_id_ = 0;
  IQualifiedVulkanResidentExecutor *executor_ = nullptr;
  std::array<PreparedSlot, kMaxVulkanPreparedPrograms> slots_{};
  VulkanResidentProgramOptions pending_options_{};
  bool pending_options_valid_ = false;
  bool shutdown_ = false;
  std::uint64_t next_handle_id_ = 1;
  std::uint64_t next_resource_id_ = 1;
  std::uint64_t next_completion_id_ = 1;
};

} // namespace studiocast::vulkan
