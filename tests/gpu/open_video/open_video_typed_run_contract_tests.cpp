#include "core/onnx/ort_session.h"
#include "core/open_video/matting_session.h"

#include <cstdint>
#include <iostream>
#include <iterator>
#include <string>
#include <type_traits>
#include <utility>

namespace {

bool Expect(bool condition, const std::string &message) {
  if (!condition)
    std::cerr << "FAIL: " << message << "\n";
  return condition;
}

studiocast::onnx::OrtPreparedCudaRunEvidence OrtEvidence() {
  studiocast::onnx::OrtPreparedCudaRunEvidence evidence;
  evidence.session_generation = 17;
  evidence.provider.provider =
      studiocast::onnx::OrtExecutionProviderId::cuda;
  evidence.provider.cuda_device_id = 2;
  evidence.provider.cuda_context_generation = 9;
  evidence.provider.requested_stream_identity = 0x1234;
  evidence.provider.user_stream_bound = true;
  evidence.provider.explicit_stream_synchronization = false;
  evidence.provider.cpu_ep_fallback_disabled = true;
  return evidence;
}

studiocast::open_cuda::MattingPreparedRunEvidence MattingEvidence() {
  studiocast::open_cuda::MattingPreparedRunEvidence evidence;
  evidence.preparation_generation = 41;
  evidence.cuda_device_id = 2;
  evidence.strict_cuda_only = true;
  evidence.context.physical_device.uuid[0] = 1;
  evidence.context.retained_context_id = 7;
  evidence.context.logical_generation = 9;
  evidence.stream_identity = 0x1234;
  evidence.input_tensor_ptr = 0x1000;
  evidence.output_tensor_ptr = 0x2000;
  evidence.frame_width = 1920;
  evidence.frame_height = 1080;
  evidence.artifact.registry_generation = 5;
  evidence.artifact.model_id_sha256[0] = 1;
  evidence.artifact.manifest_sha256[0] = 2;
  evidence.artifact.artifact_sha256[0] = 3;
  evidence.artifact.artifact_set_sha256[0] = 4;
  evidence.ort = OrtEvidence();
  return evidence;
}

bool TestStableIdsAndSchemas() {
  using O = studiocast::onnx::OrtPreparedRunStatus;
  using OQ = studiocast::onnx::OrtPreparedRunQuarantine;
  using M = studiocast::open_cuda::MattingPreparedRunStatus;
  using MQ = studiocast::open_cuda::MattingPreparedRunQuarantine;
  const O ort_ids[] = {
      O::success,
      O::not_prepared,
      O::invalid_input_output_contract,
      O::foreign_provider_evidence,
      O::stale_session_generation,
      O::wrong_device,
      O::stale_context_generation,
      O::stale_stream,
      O::stale_binding,
      O::provider_unavailable,
      O::provider_mismatch,
      O::cpu_fallback_forbidden,
      O::tensorrt_forbidden,
      O::resource_exhausted,
      O::inference_failure,
      O::provider_failure_quarantine,
      O::cancelled_or_shutdown,
      O::setup_resource_mutation,
  };
  const M matting_ids[] = {
      M::success,
      M::not_prepared,
      M::invalid_input_contract,
      M::invalid_output_contract,
      M::foreign_evidence,
      M::stale_preparation_generation,
      M::wrong_device,
      M::stale_context_generation,
      M::stale_stream,
      M::stale_artifact,
      M::stale_ort_session,
      M::stale_binding,
      M::provider_unavailable,
      M::provider_mismatch,
      M::cpu_fallback_forbidden,
      M::tensorrt_forbidden,
      M::resource_exhausted,
      M::preprocess_failure,
      M::inference_failure,
      M::provider_failure_quarantine,
      M::cancelled_or_shutdown,
      M::setup_resource_mutation,
  };
  bool exact_ids = true;
  for (std::size_t i = 0; i < std::size(ort_ids); ++i)
    exact_ids = exact_ids &&
                static_cast<std::uint16_t>(ort_ids[i]) == i;
  for (std::size_t i = 0; i < std::size(matting_ids); ++i)
    exact_ids = exact_ids &&
                static_cast<std::uint16_t>(matting_ids[i]) == i;
  return Expect(exact_ids,
                "ORT status IDs must remain fixed") &&
         Expect(static_cast<std::uint8_t>(OQ::session) == 1 &&
                    static_cast<std::uint8_t>(OQ::provider) == 2,
                "ORT quarantine IDs must remain fixed") &&
         Expect(static_cast<std::uint16_t>(M::success) == 0 &&
                    static_cast<std::uint16_t>(M::setup_resource_mutation) ==
                        21,
                "matting status IDs must remain fixed") &&
         Expect(static_cast<std::uint8_t>(MQ::session) == 1 &&
                    static_cast<std::uint8_t>(MQ::provider) == 2,
                "matting quarantine IDs must remain fixed") &&
         Expect(studiocast::onnx::OrtPreparedRunResult::kSchemaVersion == 1 &&
                    studiocast::open_cuda::MattingPreparedRunResult::
                            kSchemaVersion ==
                        1,
                "typed run schemas must be explicitly versioned") &&
         Expect(std::is_trivially_copyable_v<
                    studiocast::onnx::OrtPreparedRunResult> &&
                    std::is_trivially_copyable_v<
                        studiocast::open_cuda::MattingPreparedRunEvidence>,
                "typed evidence/results must stay fixed-layout values");
}

bool TestOrtContractClassification() {
  using namespace studiocast::onnx;
  internal::OrtPreparedRunContract contract;
  contract.session_present = true;
  contract.binding_present = true;
  contract.binding_contract_matches = true;
  contract.strict_cuda_only = true;
  contract.active = OrtEvidence();
  contract.requested = contract.active;
  if (!Expect(internal::ClassifyOrtPreparedRunContract(contract).ok(),
              "exact ORT contract should pass"))
    return false;

  auto check = [&](OrtPreparedRunStatus status, auto mutate,
                   const char *message) {
    auto candidate = contract;
    mutate(candidate);
    return Expect(
        internal::ClassifyOrtPreparedRunContract(candidate).status == status,
        message);
  };
  return check(OrtPreparedRunStatus::cancelled_or_shutdown,
               [](auto &c) { c.shutdown_requested = true; }, "shutdown") &&
         check(OrtPreparedRunStatus::not_prepared,
               [](auto &c) { c.session_present = false; }, "not prepared") &&
         check(OrtPreparedRunStatus::foreign_provider_evidence,
               [](auto &c) { c.requested.schema_version = 0; },
               "foreign evidence") &&
         check(OrtPreparedRunStatus::foreign_provider_evidence,
               [](auto &c) {
                 c.requested.provider.schema_version = 0;
               },
               "foreign provider schema") &&
         check(OrtPreparedRunStatus::provider_unavailable,
               [](auto &c) { c.active.schema_version = 0; },
               "provider unavailable") &&
         check(OrtPreparedRunStatus::stale_session_generation,
               [](auto &c) { ++c.requested.session_generation; },
               "stale session") &&
         check(OrtPreparedRunStatus::wrong_device,
               [](auto &c) { ++c.requested.provider.cuda_device_id; },
               "wrong device") &&
         check(OrtPreparedRunStatus::stale_context_generation,
               [](auto &c) {
                 ++c.requested.provider.cuda_context_generation;
               },
               "stale context") &&
         check(OrtPreparedRunStatus::stale_stream,
               [](auto &c) {
                 ++c.requested.provider.requested_stream_identity;
               },
               "stale stream") &&
         check(OrtPreparedRunStatus::provider_mismatch,
               [](auto &c) {
                 c.requested.provider.user_stream_bound = false;
               },
               "user-stream binding evidence") &&
         check(OrtPreparedRunStatus::provider_mismatch,
               [](auto &c) {
                 c.requested.provider.explicit_stream_synchronization = true;
               },
               "stream synchronization evidence") &&
         check(OrtPreparedRunStatus::provider_mismatch,
               [](auto &c) {
                 c.requested.provider.cpu_ep_fallback_disabled = false;
               },
               "CPU fallback-disable evidence") &&
         check(OrtPreparedRunStatus::stale_binding,
               [](auto &c) { c.binding_contract_matches = false; },
               "stale binding") &&
         check(OrtPreparedRunStatus::provider_mismatch,
               [](auto &c) {
                 c.requested.provider.provider =
                     OrtExecutionProviderId::tensorrt;
               },
               "provider mismatch") &&
         check(OrtPreparedRunStatus::cpu_fallback_forbidden,
               [](auto &c) {
                 c.active.provider.provider = OrtExecutionProviderId::cpu;
                 c.requested.provider.provider =
                     OrtExecutionProviderId::cpu;
               },
               "CPU fallback") &&
         check(OrtPreparedRunStatus::tensorrt_forbidden,
               [](auto &c) {
                 c.active.provider.provider =
                     OrtExecutionProviderId::tensorrt;
                 c.requested.provider.provider =
                     OrtExecutionProviderId::tensorrt;
               },
               "TensorRT forbidden");
}

bool TestOrtFailureClassification() {
  using namespace studiocast::onnx;
  const auto allocation = internal::ClassifyOrtPreparedRunFailure(
      OrtStableErrorCode::allocation_failure, true);
  const auto provider = internal::ClassifyOrtPreparedRunFailure(
      OrtStableErrorCode::execution_provider_failure, true);
  const auto cuda_runtime = internal::ClassifyOrtPreparedRunFailure(
      OrtStableErrorCode::runtime_exception, true);
  const auto generic_runtime = internal::ClassifyOrtPreparedRunFailure(
      OrtStableErrorCode::runtime_exception, false);
  const auto cancelled = internal::ClassifyOrtPreparedRunFailure(
      OrtStableErrorCode::model_load_cancelled, true);
  const auto humanized = HumanizeOrtError(
      "CUDA failure 2: out of memory", "/models/matting.onnx");
  return Expect(
             allocation.status == OrtPreparedRunStatus::resource_exhausted &&
                 allocation.quarantine ==
                     OrtPreparedRunQuarantine::session,
             "allocation failure should be typed resource exhaustion") &&
         Expect(provider.status ==
                        OrtPreparedRunStatus::provider_failure_quarantine &&
                    provider.quarantine ==
                        OrtPreparedRunQuarantine::provider,
                "EP failure should quarantine provider") &&
         Expect(cuda_runtime.status ==
                        OrtPreparedRunStatus::provider_failure_quarantine &&
                    generic_runtime.status ==
                        OrtPreparedRunStatus::inference_failure,
                "runtime failures must conservatively use provider evidence") &&
         Expect(cancelled.status ==
                    OrtPreparedRunStatus::cancelled_or_shutdown,
                "ORT cancellation should remain typed") &&
         Expect(humanized.find("GPU is likely out of VRAM") !=
                        std::string::npos &&
                    humanized.find("matting.onnx") != std::string::npos &&
                    humanized.find("CUDA failure 2: out of memory") !=
                        std::string::npos,
                "legacy ORT failure projection must remain humanized");
}

bool TestMattingContractAndProjection() {
  using namespace studiocast::open_cuda;
  internal::MattingPreparedRunContract contract;
  contract.buffers_ready = true;
  contract.execution_prepared = true;
  contract.input_valid = true;
  contract.output_valid = true;
  contract.binding_matches = true;
  contract.strict_cuda_only = true;
  contract.active = MattingEvidence();
  contract.requested = contract.active;
  if (!Expect(internal::ClassifyMattingPreparedRunContract(contract).ok(),
              "exact matting contract should pass"))
    return false;

  auto check = [&](MattingPreparedRunStatus status, auto mutate,
                   const char *message) {
    auto candidate = contract;
    mutate(candidate);
    return Expect(
        internal::ClassifyMattingPreparedRunContract(candidate).status ==
            status,
        message);
  };
  if (!check(MattingPreparedRunStatus::not_prepared,
             [](auto &c) { c.execution_prepared = false; }, "not prepared") ||
      !check(MattingPreparedRunStatus::invalid_input_contract,
             [](auto &c) { c.input_valid = false; }, "invalid input") ||
      !check(MattingPreparedRunStatus::invalid_output_contract,
             [](auto &c) { c.output_valid = false; }, "invalid output") ||
      !check(MattingPreparedRunStatus::foreign_evidence,
             [](auto &c) { c.requested.schema_version = 0; },
             "foreign evidence") ||
      !check(MattingPreparedRunStatus::provider_unavailable,
             [](auto &c) { c.active.schema_version = 0; },
             "provider unavailable") ||
      !check(MattingPreparedRunStatus::stale_preparation_generation,
             [](auto &c) { ++c.requested.preparation_generation; },
             "stale preparation") ||
      !check(MattingPreparedRunStatus::wrong_device,
             [](auto &c) { ++c.requested.cuda_device_id; }, "wrong device") ||
      !check(MattingPreparedRunStatus::stale_context_generation,
             [](auto &c) { ++c.requested.context.logical_generation; },
             "stale context") ||
      !check(MattingPreparedRunStatus::stale_stream,
             [](auto &c) { ++c.requested.stream_identity; }, "stale stream") ||
      !check(MattingPreparedRunStatus::stale_artifact,
             [](auto &c) { ++c.requested.artifact.artifact_sha256[0]; },
             "stale artifact") ||
      !check(MattingPreparedRunStatus::stale_ort_session,
             [](auto &c) { ++c.requested.ort.session_generation; },
             "stale ORT session") ||
      !check(MattingPreparedRunStatus::provider_mismatch,
             [](auto &c) {
               c.requested.ort.provider.provider =
                   studiocast::onnx::OrtExecutionProviderId::tensorrt;
             },
             "provider mismatch") ||
      !check(MattingPreparedRunStatus::stale_binding,
             [](auto &c) { c.binding_matches = false; }, "stale binding"))
    return false;

  if (!check(MattingPreparedRunStatus::cpu_fallback_forbidden,
             [](auto &c) {
               c.active.ort.provider.provider =
                   studiocast::onnx::OrtExecutionProviderId::cpu;
               c.requested.ort.provider.provider =
                   studiocast::onnx::OrtExecutionProviderId::cpu;
             },
             "CPU fallback forbidden") ||
      !check(MattingPreparedRunStatus::tensorrt_forbidden,
             [](auto &c) {
               c.active.ort.provider.provider =
                   studiocast::onnx::OrtExecutionProviderId::tensorrt;
               c.requested.ort.provider.provider =
                   studiocast::onnx::OrtExecutionProviderId::tensorrt;
             },
             "TensorRT forbidden") ||
      !check(MattingPreparedRunStatus::cancelled_or_shutdown,
             [](auto &c) { c.shutdown_requested = true; }, "shutdown"))
    return false;

  studiocast::onnx::OrtPreparedRunResult ort;
  ort.status =
      studiocast::onnx::OrtPreparedRunStatus::provider_failure_quarantine;
  ort.quarantine = studiocast::onnx::OrtPreparedRunQuarantine::provider;
  const auto projected = internal::ProjectOrtPreparedRunResult(ort);
  if (!Expect(projected.status ==
                      MattingPreparedRunStatus::provider_failure_quarantine &&
                  projected.quarantine ==
                      MattingPreparedRunQuarantine::provider,
              "matting should preserve typed ORT provider quarantine"))
    return false;

  struct Projection {
    studiocast::onnx::OrtPreparedRunStatus ort;
    MattingPreparedRunStatus matting;
  };
  const Projection projections[] = {
      {studiocast::onnx::OrtPreparedRunStatus::success,
       MattingPreparedRunStatus::success},
      {studiocast::onnx::OrtPreparedRunStatus::not_prepared,
       MattingPreparedRunStatus::not_prepared},
      {studiocast::onnx::OrtPreparedRunStatus::
           invalid_input_output_contract,
       MattingPreparedRunStatus::invalid_output_contract},
      {studiocast::onnx::OrtPreparedRunStatus::foreign_provider_evidence,
       MattingPreparedRunStatus::foreign_evidence},
      {studiocast::onnx::OrtPreparedRunStatus::stale_session_generation,
       MattingPreparedRunStatus::stale_ort_session},
      {studiocast::onnx::OrtPreparedRunStatus::wrong_device,
       MattingPreparedRunStatus::wrong_device},
      {studiocast::onnx::OrtPreparedRunStatus::stale_context_generation,
       MattingPreparedRunStatus::stale_context_generation},
      {studiocast::onnx::OrtPreparedRunStatus::stale_stream,
       MattingPreparedRunStatus::stale_stream},
      {studiocast::onnx::OrtPreparedRunStatus::stale_binding,
       MattingPreparedRunStatus::stale_binding},
      {studiocast::onnx::OrtPreparedRunStatus::provider_unavailable,
       MattingPreparedRunStatus::provider_unavailable},
      {studiocast::onnx::OrtPreparedRunStatus::provider_mismatch,
       MattingPreparedRunStatus::provider_mismatch},
      {studiocast::onnx::OrtPreparedRunStatus::cpu_fallback_forbidden,
       MattingPreparedRunStatus::cpu_fallback_forbidden},
      {studiocast::onnx::OrtPreparedRunStatus::tensorrt_forbidden,
       MattingPreparedRunStatus::tensorrt_forbidden},
      {studiocast::onnx::OrtPreparedRunStatus::resource_exhausted,
       MattingPreparedRunStatus::resource_exhausted},
      {studiocast::onnx::OrtPreparedRunStatus::inference_failure,
       MattingPreparedRunStatus::inference_failure},
      {studiocast::onnx::OrtPreparedRunStatus::provider_failure_quarantine,
       MattingPreparedRunStatus::provider_failure_quarantine},
      {studiocast::onnx::OrtPreparedRunStatus::cancelled_or_shutdown,
       MattingPreparedRunStatus::cancelled_or_shutdown},
      {studiocast::onnx::OrtPreparedRunStatus::setup_resource_mutation,
       MattingPreparedRunStatus::setup_resource_mutation},
  };
  for (const auto &projection : projections) {
    studiocast::onnx::OrtPreparedRunResult source;
    source.status = projection.ort;
    if (!Expect(internal::ProjectOrtPreparedRunResult(source).status ==
                    projection.matting,
                "every ORT status must have a stable matting projection"))
      return false;
  }
  return true;
}

bool TestRepeatedClassificationAndInferenceOnlyPreflight() {
  using namespace studiocast::open_cuda;
  internal::MattingPreparedRunContract contract;
  contract.buffers_ready = true;
  contract.execution_prepared = true;
  contract.input_valid = true;
  contract.output_valid = true;
  contract.binding_matches = true;
  contract.strict_cuda_only = true;
  contract.active = MattingEvidence();
  contract.requested = contract.active;
  for (int i = 0; i < 10000; ++i) {
    if (!internal::ClassifyMattingPreparedRunContract(contract).ok())
      return Expect(false, "repeated exact classifications must be stable");
  }
  auto replacement = MattingEvidence();
  replacement.preparation_generation = 42;
  const auto retained = internal::ResolveMattingReplacementEvidence(
      contract.active, replacement, false);
  const auto committed = internal::ResolveMattingReplacementEvidence(
      contract.active, replacement, true);
  if (!Expect(retained.preparation_generation ==
                  contract.active.preparation_generation,
              "failed replacement must retain the old prepared generation") ||
      !Expect(committed.preparation_generation ==
                  replacement.preparation_generation,
              "complete replacement must atomically commit its generation"))
    return false;

  studiocast::cuda::CudaDriverApi cuda;
  studiocast::open_video::ModelPack pack;
  OpenCudaMattingSession session(&cuda, std::move(pack));
  const auto before = session.prepared_run_stats();
  const auto view = session.prepared_input_tensor_view();
  const auto result = session.RunPreparedInferenceTyped(
      nullptr, nullptr, MattingPreparedRunEvidence{});
  const auto after = session.prepared_run_stats();
  if (!(Expect(result.status == MattingPreparedRunStatus::not_prepared,
               "inference-only call must fail closed before preparation") &&
        Expect(!view.Valid(),
               "prepared input view must fail closed before setup") &&
        Expect(after.preprocess_attempts == before.preprocess_attempts,
               "inference-only API must never invoke preprocessing") &&
        Expect(after.inference_attempts == before.inference_attempts,
               "failed preflight must not invoke or rebuild inference") &&
        Expect(after.setup_commits == before.setup_commits &&
                   after.preparation_generation ==
                       before.preparation_generation,
               "run path must not construct or commit a session") &&
        Expect(after.run_attempts == before.run_attempts + 1 &&
                   after.run_failures == before.run_failures + 1,
               "typed run counters must be monotonic")))
    return false;

  studiocast::cuda::CudaImage invalid_input;
  std::string compatibility_error;
  if (!Expect(!session.RunPrepared(nullptr, invalid_input, nullptr,
                                   &compatibility_error),
              "compatibility bool API should project typed failure") ||
      !Expect(!compatibility_error.empty(),
              "compatibility projection should retain a diagnostic"))
    return false;
  session.RequestShutdown();
  const auto shutdown = session.RunPreparedTyped(
      nullptr, invalid_input, nullptr, MattingPreparedRunEvidence{});
  return Expect(
      shutdown.status == MattingPreparedRunStatus::cancelled_or_shutdown,
      "shutdown must be observed before any preprocessing or inference");
}

} // namespace

int main() {
  const bool ok =
      TestStableIdsAndSchemas() && TestOrtContractClassification() &&
      TestOrtFailureClassification() && TestMattingContractAndProjection() &&
      TestRepeatedClassificationAndInferenceOnlyPreflight();
  if (ok)
    std::cout << "open video typed run contract tests passed\n";
  return ok ? 0 : 1;
}
