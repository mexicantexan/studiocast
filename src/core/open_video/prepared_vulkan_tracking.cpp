#include "core/open_video/prepared_vulkan_tracking.h"

#include "core/util/process_exit.h"

#include "core/vulkan/kernels/tracking_kernels.h"
#include "core/vulkan/resident_command_batch.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace studiocast::open_vulkan {
namespace {
using vulkan::VkCommandBuffer;
using vulkan::kernels::TrackingBufferBinding;
using vulkan::kernels::TrackingDispatch;
using vulkan::kernels::TrackingShader;
constexpr std::size_t kStateWords = 5512;
constexpr std::size_t kPhases = 8;
enum Phase : unsigned {
  preprocess,
  decode,
  landmarks,
  eye_inputs,
  eye_tail,
  auto_frame,
  vignette,
  reset
};
TrackingBufferBinding Bind(const vulkan::VulkanBuffer &b) {
  return {b.buffer(), b.size(), b.context_identity()};
}
TrackingBufferBinding Bind(const vulkan::VulkanTensor &b) {
  return {b.buffer(), b.byte_size(), b.context_identity()};
}
TrackingBufferBinding Bind(const vulkan::VulkanImage &b) {
  return {b.buffer(), b.byte_size(), b.context_identity()};
}
std::uint32_t Bits(float f) { return std::bit_cast<std::uint32_t>(f); }
bool Fail(std::string *e, const char *message) {
  if (e)
    *e = message;
  return false;
}
bool Same(const TrackingBufferBinding &a, const TrackingBufferBinding &b) {
  return a.buffer == b.buffer && a.bytes == b.bytes && a.context == b.context;
}
} // namespace

struct PreparedVulkanTracking::Impl {
  struct Recorded {
    VkCommandBuffer command = nullptr;
    std::array<TrackingBufferBinding, 3> inputs{};
    vulkan::ResidentCommandTicket ticket{};
    bool pending = false, unsafe = false;
    std::uint64_t last_sequence = 0;
  };
  PreparedVulkanTrackingSetup setup{};
  vulkan::VulkanContextIdentity context{};
  vulkan::kernels::TrackingKernels kernels;
  vulkan::VulkanBuffer state, candidates, masks, partials, forest,
      predictor_scratch;
  vulkan::VulkanBuffer dummy, frame_parameters, upload_staging;
  vulkan::VulkanTensor face_input;
  std::array<vulkan::VulkanTensor, 12> face_outputs;
  vulkan::VulkanTensor points;
  std::array<vulkan::VulkanTensor, 2> eye_rgb, anchors, angles, flow, lcm,
      corrected;
  std::array<std::uint32_t, 16> predictor_parameters{};
  std::array<Recorded, kPhases> phases{};
  VkCommandBuffer setup_upload_command = nullptr;
  PreparedVulkanTrackingCounters counters{};
  hal::ResourceBudget used{};
  std::uint64_t parameter_sequence = 0;
  bool ready = false, sealed = false;

  bool HasPending() const noexcept {
    for (const auto &p : phases)
      if (p.pending)
        return true;
    return false;
  }
  static void
  Completion(void *owner, const vulkan::ResidentCommandTicket &ticket,
             vulkan::ResidentCommandCompletion completion) noexcept {
    auto &self = *static_cast<Impl *>(owner);
    for (auto &phase : self.phases) {
      if (!phase.pending || phase.ticket != ticket)
        continue;
      switch (completion) {
      case vulkan::ResidentCommandCompletion::completed:
        ++self.counters.committed_command_groups;
        phase.pending = false;
        break;
      case vulkan::ResidentCommandCompletion::aborted_unsubmitted:
        ++self.counters.aborted_command_groups;
        phase.pending = false;
        phase.last_sequence = 0;
        break;
      case vulkan::ResidentCommandCompletion::unsafe_pending:
        if (!phase.unsafe)
          ++self.counters.unsafe_command_groups;
        phase.unsafe = true;
        break;
      case vulkan::ResidentCommandCompletion::retired_after_quiescence:
        ++self.counters.retired_command_groups;
        phase.pending = phase.unsafe = false;
        phase.last_sequence = 0;
        break;
      }
      return;
    }
  }
  bool Frame(vulkan::ResidentCommandBatch &batch, std::string *error) {
    if (!ready || !setup.device || !setup.device->health_state().healthy() ||
        batch.state() != vulkan::ResidentCommandBatchState::recording ||
        batch.context_identity() != context || !batch.capture_sequence())
      return Fail(error, "stale or unavailable resident tracking frame");
    for (const auto &phase : phases)
      if (phase.unsafe || (phase.pending && phase.ticket.capture_sequence !=
                                                batch.capture_sequence()))
        return Fail(error, "resident tracking frame still requires retirement");
    if (parameter_sequence != batch.capture_sequence()) {
      if (HasPending())
        return Fail(error, "tracking parameter overwrite while pending");
      auto *p = static_cast<std::uint32_t *>(frame_parameters.mapped());
      p[0] = static_cast<std::uint32_t>(batch.capture_sequence());
      p[1] = static_cast<std::uint32_t>(batch.capture_sequence() >> 32);
      p[2] = 1;
      if (!frame_parameters.Flush(error))
        return false;
      parameter_sequence = batch.capture_sequence();
    }
    return true;
  }
  bool Image(const vulkan::VulkanImage &image, int w, int h,
             std::string *error) const {
    if (!image.Valid() || !image.BelongsTo(*setup.device) ||
        image.context_identity() != context || image.mapped() ||
        !image.device_local() ||
        image.format() != vulkan::VulkanPixelFormat::rgb_u8 ||
        image.width() != w || image.height() != h ||
        image.pitch_pixels() > std::numeric_limits<std::uint32_t>::max())
      return Fail(error, "foreign, mapped or incompatible tracking image");
    return true;
  }
  TrackingDispatch Dispatch(std::uint32_t op) const {
    TrackingDispatch d;
    d.buffers.fill(Bind(dummy));
    d.buffers[4] = Bind(candidates);
    d.buffers[5] = Bind(state);
    d.buffers[7] = Bind(frame_parameters);
    d.parameters[0] = op;
    d.parameters[1] = setup.frame_width;
    d.parameters[2] = setup.frame_height;
    return d;
  }
  bool Replay(Phase which, vulkan::ResidentCommandBatch &batch,
              const std::array<TrackingBufferBinding, 3> &inputs,
              std::string *error, bool *cached) {
    *cached = false;
    if (!Frame(batch, error))
      return false;
    auto &phase = phases[which];
    const auto mark = batch.Mark();
    if (phase.pending || (phase.last_sequence == batch.capture_sequence() &&
                          phase.ticket.batch_id == mark.batch_id &&
                          phase.ticket.epoch == mark.epoch))
      return Fail(error,
                  "duplicate resident tracking phase in capture sequence");
    if (phase.command) {
      for (std::size_t i = 0; i < inputs.size(); ++i)
        if (!Same(inputs[i], phase.inputs[i]))
          return Fail(
              error, "resident tracking input binding changed after recording");
      if (!Append(which, batch, error))
        return false;
      *cached = true;
      return true;
    }
    if (sealed)
      return Fail(error,
                  "resident tracking phase was not warmed before sealing");
    phase.inputs = inputs;
    return true;
  }
  bool Append(Phase which, vulkan::ResidentCommandBatch &batch,
              std::string *error) {
    auto &phase = phases[which];
    const vulkan::ResidentCommandGroup group{
        context, &phase.command, 1, this, setup.owner_generation, Completion};
    if (!batch.Append(group, &phase.ticket, error))
      return false;
    phase.pending = true;
    phase.last_sequence = batch.capture_sequence();
    return true;
  }
  bool Record(Phase which, vulkan::ResidentCommandBatch &batch,
              std::span<const TrackingDispatch> dispatches,
              std::string *error) {
    return kernels.Record(which, dispatches, &phases[which].command, error) &&
           Append(which, batch, error);
  }
};

PreparedVulkanTracking::PreparedVulkanTracking()
    : impl_(std::make_unique<Impl>()) {}
PreparedVulkanTracking::~PreparedVulkanTracking() {
  if (!Shutdown(nullptr)) {
    util::MarkUnsafeNativeResourcesRetained();
    (void)impl_.release();
  }
}

bool PreparedVulkanTracking::Prepare(const PreparedVulkanTrackingSetup &s,
                                     std::string *error) {
  if (error)
    error->clear();
  if (!s.device || !s.device->Initialized() ||
      !hal::ValidateExecutionDomain(s.domain) || !s.owner_generation ||
      !std::isfinite(s.auto_frame_headroom) || s.frame_width < 2 ||
      s.frame_height < 2 || s.final_width < 2 || s.final_height < 2 ||
      s.frame_width > 8192 || s.frame_height > 8192 || s.final_width > 8192 ||
      s.final_height > 8192 || s.reservation.in_flight_slots != 1 ||
      s.reservation.resource_count < 84 || !s.reservation.persistent_bytes ||
      !s.reservation.scratch_bytes || !s.reservation.upload_staging_bytes ||
      !Shutdown(error))
    return Fail(error, "invalid or under-reserved resident tracking setup");
  if (s.domain.backend != hal::BackendId::vulkan ||
      s.domain.api_family != hal::ApiFamily::vulkan ||
      s.domain.logical_context_generation !=
          s.device->context_identity().generation ||
      s.domain.queue_or_stream_identity !=
          reinterpret_cast<std::uintptr_t>(s.device->queue()) ||
      s.domain.physical_device.vendor_id != s.device->identity().vendor_id ||
      s.domain.physical_device.device_id != s.device->identity().device_id ||
      s.domain.physical_device.stable_id != s.device->identity().device_uuid ||
      s.domain.allocator_family != hal::AllocatorFamily::vulkan ||
      s.domain.memory_handle_family != hal::MemoryHandleFamily::vulkan_buffer ||
      s.domain.ownership_model != hal::OwnershipModel::adapter_exclusive ||
      s.domain.synchronization_protocol !=
          hal::SynchronizationProtocol::vulkan_queue_fence)
    return Fail(
        error,
        "tracking execution domain does not match its exact native owner");
  // Zero strength is a live-plan omission. Setup still validates/warms the
  // selected graph and records exact bindings so canonical model failures and
  // uniform readiness remain observable without changing the saved strength.
  const bool eye = s.eye_contact;
  if ((eye || s.auto_frame) && !s.face_analysis)
    return Fail(
        error, "tracking consumers require their exact resident face producer");
  if ((s.face_analysis && !s.face_artifact.Valid()) ||
      (eye && (!s.predictor || !s.landmark_artifact.Valid() ||
               !s.eye_left_artifact.Valid() || !s.eye_right_artifact.Valid() ||
               !s.device->shader_float64_enabled())))
    return Fail(error,
                "resident tracking model identity or enabled FP64 is missing");
  auto &p = *impl_;
  p.setup = s;
  p.setup.eye_contact = eye;
  p.context = s.device->context_identity();
  p.used.in_flight_slots = 1;
  p.used.resource_count = 84;
  constexpr auto usage = vulkan::VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                         vulkan::VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                         vulkan::VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  auto allocate = [&](vulkan::VulkanBuffer &b, std::size_t bytes, bool scratch,
                      bool mapped = false) {
    auto &used = scratch ? p.used.scratch_bytes : p.used.persistent_bytes;
    const auto budget =
        scratch ? s.reservation.scratch_bytes : s.reservation.persistent_bytes;
    if (!bytes || used > budget || bytes > budget - used)
      return Fail(
          error, "resident tracking buffer budget exhausted before allocation");
    const auto required = mapped ? vulkan::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                 : vulkan::VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    const auto preferred = mapped ? vulkan::VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                                  : vulkan::VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    if (!b.Allocate(s.device, bytes, usage, required, preferred, mapped, error,
                    budget - used))
      return false;
    used += b.allocation_size();
    return true;
  };
  auto tensor = [&](vulkan::VulkanTensor &t, int c, int h, int w) {
    const auto bytes = std::uint64_t(c) * h * w * sizeof(float);
    if (p.used.persistent_bytes > s.reservation.persistent_bytes ||
        bytes > s.reservation.persistent_bytes - p.used.persistent_bytes)
      return Fail(
          error, "resident tracking tensor budget exhausted before allocation");
    const auto before = s.device->allocation_stats().current_bytes;
    if (!t.AllocateNchwF32(s.device, 1, c, h, w, false, error,
                           s.reservation.persistent_bytes -
                               p.used.persistent_bytes))
      return false;
    const auto after = s.device->allocation_stats().current_bytes;
    if (after < before || after - before > s.reservation.persistent_bytes -
                                               p.used.persistent_bytes)
      return Fail(error,
                  "resident tracking aligned tensor exceeds reservation");
    p.used.persistent_bytes += after - before;
    return true;
  };
  if (!allocate(p.dummy, 16, false) ||
      !allocate(p.frame_parameters, 16, false, true) ||
      !allocate(p.state, kStateWords * 4, false) ||
      !allocate(p.candidates, s.face_analysis ? 16384 * 32 : 16, true) ||
      !allocate(p.masks, s.face_analysis ? 5000 * 157 * 4 : 16, true) ||
      !allocate(
          p.partials,
          s.auto_frame
              ? ((std::size_t(s.frame_width) * s.frame_height + 255) / 256) * 20
              : 20,
          true))
    return false;
  std::memset(p.frame_parameters.mapped(), 0, 16);
  if (!p.frame_parameters.Flush(error))
    return false;
  if (s.face_analysis) {
    if (!tensor(p.face_input, 3, 640, 640))
      return false;
    for (int i = 0; i < 12; ++i) {
      const int stride = 8 << (i % 3), cells = (640 / stride) * (640 / stride);
      const int channels = i < 6 ? 1 : i < 9 ? 4 : 10;
      if (!tensor(p.face_outputs[i], 1, cells, channels))
        return false;
    }
  }
  std::vector<std::uint32_t> forest_words;
  if (eye) {
    const auto &data = *s.predictor;
    if (!data.maximum_features || data.cascade_trees.size() < 2 ||
        data.resident_bytes > s.reservation.persistent_bytes)
      return Fail(error, "invalid bounded dlib predictor data");
    const auto add = [&](const auto &values, std::uint32_t parameter) {
      p.predictor_parameters[parameter] = forest_words.size();
      const auto bytes = values.size() * sizeof(values[0]);
      const auto old = forest_words.size();
      forest_words.resize(old + (bytes + 3) / 4);
      std::memcpy(forest_words.data() + old, values.data(), bytes);
    };
    p.predictor_parameters[4] = data.cascade_trees.size() - 1;
    add(data.initial_shape, 5);
    add(data.cascade_trees, 6);
    add(data.tree_splits, 7);
    add(data.tree_leaves, 8);
    add(data.splits, 9);
    add(data.leaves, 10);
    add(data.cascade_features, 11);
    add(data.feature_anchors, 12);
    add(data.feature_deltas, 13);
    if (forest_words.size() & 1)
      forest_words.push_back(0);
    add(data.rectangle_basis_pseudoinverse, 14);
    p.predictor_parameters[15] = data.maximum_features;
    if (!allocate(p.forest, forest_words.size() * 4, false) ||
        !allocate(p.predictor_scratch,
                  (136 + data.maximum_features + data.maximum_trees) * 4,
                  true) ||
        !tensor(p.points, 1, 1, 136))
      return false;
    for (unsigned e = 0; e < 2; ++e)
      if (!tensor(p.eye_rgb[e], 3, 48, 64) ||
          !tensor(p.anchors[e], 12, 48, 64) || !tensor(p.angles[e], 2, 1, 1) ||
          !tensor(p.flow[e], 2, 48, 64) || !tensor(p.lcm[e], 2, 48, 64) ||
          !tensor(p.corrected[e], 3, 48, 64))
        return false;
  }
  // One bounded setup staging buffer is retained until this upload completes.
  const auto upload_bytes =
      std::max<std::size_t>(kStateWords * 4, forest_words.size() * 4);
  auto &staging = p.upload_staging;
  if (upload_bytes > s.reservation.upload_staging_bytes ||
      !staging.Allocate(s.device, upload_bytes, usage,
                        vulkan::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                        vulkan::VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true,
                        error, s.reservation.upload_staging_bytes))
    return false;
  p.used.upload_staging_bytes = staging.allocation_size();
  std::memset(staging.mapped(), 0, upload_bytes);
  if (!staging.Flush(error))
    return false;
  auto &command = p.setup_upload_command;
  vulkan::VkCommandBufferAllocateInfo alloc{};
  alloc.commandPool = s.device->command_pool();
  alloc.level = vulkan::VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc.commandBufferCount = 1;
  if (!s.device->CheckDriverResult(s.device->f().vkAllocateCommandBuffers(
                                       s.device->device(), &alloc, &command),
                                   "tracking setup upload allocation", false,
                                   error))
    return false;
  const auto upload = [&](const vulkan::VulkanBuffer &destination,
                          std::size_t bytes) {
    vulkan::VkCommandBufferBeginInfo begin{};
    if (!s.device->CheckDriverResult(
            s.device->f().vkBeginCommandBuffer(command, &begin),
            "tracking upload begin", false, error))
      return false;
    if (!s.device->RecordBufferBarrier(
            command, staging.buffer(), bytes, p.context,
            vulkan::VulkanBufferAccess::host_write,
            vulkan::VulkanBufferAccess::transfer_read, error))
      return false;
    const vulkan::VkBufferCopy copy{0, 0, bytes};
    s.device->f().vkCmdCopyBuffer(command, staging.buffer(),
                                  destination.buffer(), 1, &copy);
    if (!s.device->RecordBufferBarrier(
            command, destination.buffer(), bytes, p.context,
            vulkan::VulkanBufferAccess::transfer_write,
            vulkan::VulkanBufferAccess::compute_read, error) ||
        !s.device->CheckDriverResult(s.device->f().vkEndCommandBuffer(command),
                                     "tracking upload end", false, error))
      return false;
    return s.device->SubmitAndWait(command, error);
  };
  if (!upload(p.state, kStateWords * 4)) {
    // The context retains unsafe child memory; preserve the mapped upload owner
    // as well because submitted commands may still read its bytes.
    return false;
  }
  if (eye) {
    if (!s.device->CheckDriverResult(
            s.device->f().vkResetCommandBuffer(command, 0),
            "tracking upload reset", false, error))
      return false;
    std::memcpy(staging.mapped(), forest_words.data(), forest_words.size() * 4);
    if (!staging.Flush(error) || !upload(p.forest, forest_words.size() * 4)) {
      return false;
    }
  }
  s.device->f().vkFreeCommandBuffers(s.device->device(),
                                     s.device->command_pool(), 1, &command);
  command = nullptr;
  staging.Free();
  if (!p.kernels.Prepare(s.device, eye, kPhases, 32, error))
    return false;
  auto clear = p.Dispatch(11);
  clear.invocations = kStateWords;
  if (!p.kernels.Record(reset, std::span(&clear, 1), &p.phases[reset].command,
                        error))
    return false;
  p.ready = true;
  p.setup.predictor = nullptr;
  return true;
}

const vulkan::VulkanTensor &PreparedVulkanTracking::FaceInput() const noexcept {
  return impl_->face_input;
}
std::array<const vulkan::VulkanTensor *, 12>
PreparedVulkanTracking::FaceOutputs() const noexcept {
  std::array<const vulkan::VulkanTensor *, 12> out{};
  for (unsigned i = 0; i < 12; ++i)
    out[i] = &impl_->face_outputs[i];
  return out;
}
VulkanEyeInputSlots
PreparedVulkanTracking::EyeInputs(unsigned eye) const noexcept {
  if (eye > 1)
    return {};
  return {&impl_->eye_rgb[eye], &impl_->anchors[eye], &impl_->angles[eye]};
}
VulkanEyePrefixOutputSlots
PreparedVulkanTracking::EyePrefixOutputs(unsigned eye) const noexcept {
  if (eye > 1)
    return {};
  return {&impl_->flow[eye], &impl_->lcm[eye]};
}

bool PreparedVulkanTracking::AppendFacePreprocess(
    vulkan::ResidentCommandBatch &batch, const vulkan::VulkanImage &image,
    std::string *error) {
  auto &p = *impl_;
  if (!p.setup.face_analysis ||
      !p.Image(image, p.setup.frame_width, p.setup.frame_height, error))
    return false;
  bool cached = false;
  if (!p.Replay(preprocess, batch, {Bind(image), {}, {}}, error, &cached) ||
      cached)
    return cached;
  std::array<TrackingDispatch, 2> commands{p.Dispatch(0), p.Dispatch(1)};
  for (auto &d : commands) {
    d.buffers[0] = Bind(image);
    d.buffers[1] = Bind(p.face_input);
    d.parameters[3] = image.pitch_pixels();
    d.parameters[25] = Bits(1.F / 255.F);
    d.parameters[26] = Bits(1.F / (1.F / 255.F));
  }
  commands[1].invocations = 3 * 640 * 640;
  return p.Record(preprocess, batch, commands, error);
}
bool PreparedVulkanTracking::AppendFaceDecode(
    vulkan::ResidentCommandBatch &batch, std::string *error) {
  auto &p = *impl_;
  bool cached = false;
  if (!p.setup.face_analysis || !p.Replay(decode, batch, {}, error, &cached) ||
      cached)
    return cached;
  std::vector<TrackingDispatch> commands;
  commands.reserve(112);
  auto clear = p.Dispatch(2);
  clear.invocations = 16384;
  commands.push_back(clear);
  unsigned offset = 0;
  for (unsigned s = 0; s < 3; s++) {
    auto d = p.Dispatch(3);
    d.buffers[1] = Bind(p.face_outputs[s]);
    d.buffers[2] = Bind(p.face_outputs[3 + s]);
    d.buffers[3] = Bind(p.face_outputs[6 + s]);
    d.parameters[5] = 8u << s;
    d.parameters[6] = offset;
    d.invocations = (640 / (8u << s)) * (640 / (8u << s));
    offset += d.invocations;
    commands.push_back(d);
  }
  for (unsigned k = 2; k <= 16384; k *= 2)
    for (unsigned j = k / 2; j; j /= 2) {
      auto d = p.Dispatch(4);
      d.parameters[11] = k;
      d.parameters[12] = j;
      d.invocations = 16384;
      commands.push_back(d);
    }
  auto mask = p.Dispatch(5);
  mask.buffers[6] = Bind(p.masks);
  mask.invocations = 5000 * 157;
  commands.push_back(mask);
  auto nms = p.Dispatch(6);
  nms.buffers[6] = Bind(p.masks);
  commands.push_back(nms);
  return p.Record(decode, batch, commands, error);
}
bool PreparedVulkanTracking::AppendLandmarks(
    vulkan::ResidentCommandBatch &batch, const vulkan::VulkanImage &image,
    std::string *error) {
  auto &p = *impl_;
  bool cached = false;
  if (!p.setup.eye_contact ||
      !p.Image(image, p.setup.frame_width, p.setup.frame_height, error))
    return false;
  if (!p.Replay(landmarks, batch, {Bind(image), {}, {}}, error, &cached) ||
      cached)
    return cached;
  auto d = p.Dispatch(0);
  d.shader = TrackingShader::dlib_predictor;
  std::copy(p.predictor_parameters.begin(), p.predictor_parameters.end(),
            d.parameters.begin());
  d.parameters[1] = image.width();
  d.parameters[2] = image.height();
  d.parameters[3] = image.pitch_pixels();
  d.buffers[0] = Bind(image);
  d.buffers[1] = Bind(p.forest);
  d.buffers[2] = Bind(p.predictor_scratch);
  d.buffers[3] = Bind(p.points);
  return p.Record(landmarks, batch, std::span(&d, 1), error);
}
bool PreparedVulkanTracking::AppendEyeInputs(
    vulkan::ResidentCommandBatch &batch, const vulkan::VulkanImage &image,
    std::string *error) {
  auto &p = *impl_;
  bool cached = false;
  if (!p.setup.eye_contact ||
      !p.Image(image, p.setup.frame_width, p.setup.frame_height, error))
    return false;
  if (!p.Replay(eye_inputs, batch, {Bind(image), {}, {}}, error, &cached) ||
      cached)
    return cached;
  std::array<TrackingDispatch, 2> commands;
  for (unsigned e = 0; e < 2; e++) {
    auto &d = commands[e];
    d = p.Dispatch(0);
    d.shader = TrackingShader::eye_contact;
    d.parameters[3] = image.pitch_pixels();
    d.parameters[5] = e;
    d.parameters[6] = Bits(std::clamp(p.setup.eye_strength, 0, 100) * .01F);
    d.buffers[0] = Bind(image);
    d.buffers[1] = Bind(p.points);
    d.buffers[2] = Bind(p.eye_rgb[e]);
    d.buffers[3] = Bind(p.anchors[e]);
    d.buffers[4] = Bind(p.angles[e]);
    d.invocations = 36864;
  }
  return p.Record(eye_inputs, batch, commands, error);
}
bool PreparedVulkanTracking::AppendEyeTailAndComposite(
    vulkan::ResidentCommandBatch &batch, const vulkan::VulkanImage &image,
    const vulkan::VulkanImage &output, std::string *error) {
  auto &p = *impl_;
  bool cached = false;
  if (!p.setup.eye_contact ||
      !p.Image(image, p.setup.frame_width, p.setup.frame_height, error) ||
      !p.Image(output, p.setup.frame_width, p.setup.frame_height, error) ||
      image.buffer() == output.buffer())
    return false;
  if (!p.Replay(eye_tail, batch, {Bind(image), Bind(output), {}}, error,
                &cached) ||
      cached)
    return cached;
  std::array<TrackingDispatch, 3> commands;
  for (unsigned e = 0; e < 2; e++) {
    auto &d = commands[e];
    d = p.Dispatch(1);
    d.shader = TrackingShader::eye_contact;
    d.buffers[0] = Bind(p.eye_rgb[e]);
    d.buffers[1] = Bind(p.flow[e]);
    d.buffers[2] = Bind(p.lcm[e]);
    d.buffers[3] = Bind(p.corrected[e]);
    d.invocations = 9216;
  }
  auto &d = commands[2];
  d = p.Dispatch(2);
  d.shader = TrackingShader::eye_contact;
  d.buffers[0] = Bind(image);
  d.buffers[1] = Bind(p.corrected[0]);
  d.buffers[2] = Bind(p.corrected[1]);
  d.buffers[3] = Bind(output);
  d.parameters[3] = image.pitch_pixels();
  d.parameters[4] = output.pitch_pixels();
  d.parameters[6] = Bits(std::clamp(p.setup.eye_strength, 0, 100) * .01F);
  d.invocations = image.width() * image.height();
  return p.Record(eye_tail, batch, commands, error);
}
bool PreparedVulkanTracking::AppendAutoFrame(
    vulkan::ResidentCommandBatch &batch, const vulkan::VulkanImage &image,
    const vulkan::VulkanImage *alpha, const vulkan::VulkanImage &output,
    std::string *error, bool current_matte_available) {
  auto &p = *impl_;
  bool cached = false;
  if (!p.setup.auto_frame ||
      !p.Image(image, p.setup.frame_width, p.setup.frame_height, error) ||
      !p.Image(output, p.setup.frame_width, p.setup.frame_height, error) ||
      image.buffer() == output.buffer())
    return false;
  if (alpha && (!alpha->Valid() || !alpha->BelongsTo(*p.setup.device) ||
                alpha->mapped() || !alpha->device_local() ||
                alpha->format() != vulkan::VulkanPixelFormat::f32_1 ||
                alpha->width() > p.setup.frame_width ||
                alpha->height() > p.setup.frame_height))
    return Fail(error, "invalid resident Auto Frame matte");
  if (!p.Frame(batch, error))
    return false;
  static_cast<std::uint32_t *>(p.frame_parameters.mapped())[2] =
      current_matte_available ? 1u : 0u;
  if (!p.frame_parameters.Flush(error))
    return false;
  if (!p.Replay(auto_frame, batch,
                {Bind(image), Bind(output),
                 alpha ? Bind(*alpha) : TrackingBufferBinding{}},
                error, &cached) ||
      cached)
    return cached;
  std::array<TrackingDispatch, 3> commands;
  unsigned count = 0;
  if (alpha) {
    auto &d = commands[count++];
    d = p.Dispatch(7);
    d.buffers[0] = Bind(*alpha);
    d.buffers[6] = Bind(p.partials);
    d.parameters[20] = alpha->width();
    d.parameters[21] = alpha->height();
    d.parameters[22] = alpha->pitch_pixels();
    d.invocations = (alpha->width() * alpha->height() + 255) / 256;
  }
  auto &plan = commands[count++];
  plan = p.Dispatch(8);
  plan.buffers[6] = Bind(p.partials);
  plan.parameters[13] = std::clamp(p.setup.auto_frame_strength, 0, 100);
  plan.parameters[14] = std::clamp(p.setup.auto_frame_smoothing, 0, 100);
  plan.parameters[15] = Bits(p.setup.auto_frame_headroom);
  plan.parameters[20] = alpha ? alpha->width() : 0;
  plan.parameters[21] = alpha ? alpha->height() : 0;
  plan.parameters[23] = alpha ? 1 : 0;
  plan.parameters[24] = alpha && p.setup.matte_shared_current_frame ? 1 : 0;
  auto &crop = commands[count++];
  crop = p.Dispatch(9);
  crop.buffers[0] = Bind(image);
  crop.buffers[1] = Bind(output);
  crop.parameters[3] = image.pitch_pixels();
  crop.parameters[4] = output.pitch_pixels();
  crop.invocations = image.width() * image.height();
  return p.Record(auto_frame, batch, std::span(commands.data(), count), error);
}
bool PreparedVulkanTracking::AppendVignette(vulkan::ResidentCommandBatch &batch,
                                            const vulkan::VulkanImage &image,
                                            const vulkan::VulkanImage &output,
                                            std::string *error) {
  auto &p = *impl_;
  bool cached = false;
  if (!p.setup.vignette ||
      !p.Image(image, p.setup.final_width, p.setup.final_height, error) ||
      !p.Image(output, p.setup.final_width, p.setup.final_height, error) ||
      image.buffer() == output.buffer())
    return false;
  if (!p.Replay(vignette, batch, {Bind(image), Bind(output), {}}, error,
                &cached) ||
      cached)
    return cached;
  std::array<TrackingDispatch, 2> commands{p.Dispatch(0), p.Dispatch(10)};
  auto &d = commands[1];
  d.parameters[1] = image.width();
  d.parameters[2] = image.height();
  d.parameters[3] = image.pitch_pixels();
  d.parameters[4] = output.pitch_pixels();
  d.parameters[16] =
      Bits(std::clamp(p.setup.vignette_intensity, 0, 100) * .01F);
  d.parameters[17] = p.setup.vignette_tracked_center && p.setup.auto_frame;
  d.buffers[0] = Bind(image);
  d.buffers[1] = Bind(output);
  d.invocations = image.width() * image.height();
  return p.Record(vignette, batch, commands, error);
}
bool PreparedVulkanTracking::AppendReset(vulkan::ResidentCommandBatch &batch,
                                         std::string *error) {
  auto &p = *impl_;
  bool cached = false;
  return p.Replay(reset, batch, {}, error, &cached) && cached;
}
bool PreparedVulkanTracking::SealBindingsForExecution(std::string *error) {
  auto &p = *impl_;
  if (!p.ready || p.HasPending())
    return Fail(error, "tracking warmup has not completed");
  if ((p.setup.face_analysis &&
       (!p.phases[preprocess].command || !p.phases[decode].command)) ||
      (p.setup.eye_contact &&
       (!p.phases[landmarks].command || !p.phases[eye_inputs].command ||
        !p.phases[eye_tail].command)) ||
      (p.setup.auto_frame && !p.phases[auto_frame].command) ||
      (p.setup.vignette && !p.phases[vignette].command))
    return Fail(error, "tracking phase bindings are incomplete");
  p.sealed = true;
  return true;
}
bool PreparedVulkanTracking::Shutdown(std::string *error) noexcept {
  if (!impl_ || !impl_->setup.device)
    return true;
  auto &p = *impl_;
  if (p.HasPending() || !p.setup.device->SafeToDestroyResources())
    return Fail(error, "tracking resources still require proven retirement");
  if (!p.kernels.Shutdown(error))
    return false;
  if (p.setup_upload_command) {
    p.setup.device->f().vkFreeCommandBuffers(p.setup.device->device(),
                                             p.setup.device->command_pool(), 1,
                                             &p.setup_upload_command);
    p.setup_upload_command = nullptr;
  }
  for (auto *a :
       {&p.eye_rgb, &p.anchors, &p.angles, &p.flow, &p.lcm, &p.corrected})
    for (auto &t : *a)
      t.Free();
  for (auto &t : p.face_outputs)
    t.Free();
  p.face_input.Free();
  p.points.Free();
  for (auto *b :
       {&p.state, &p.candidates, &p.masks, &p.partials, &p.forest,
        &p.predictor_scratch, &p.dummy, &p.frame_parameters, &p.upload_staging})
    b->Free();
  p.setup = {};
  p.phases = {};
  p.used = {};
  p.parameter_sequence = 0;
  p.ready = p.sealed = false;
  return true;
}
bool PreparedVulkanTracking::ready() const noexcept {
  return impl_ && impl_->ready;
}
PreparedVulkanTrackingCounters
PreparedVulkanTracking::counters() const noexcept {
  return impl_ ? impl_->counters : PreparedVulkanTrackingCounters{};
}
hal::ResourceBudget PreparedVulkanTracking::resources() const noexcept {
  return impl_ ? impl_->used : hal::ResourceBudget{};
}
} // namespace studiocast::open_vulkan
