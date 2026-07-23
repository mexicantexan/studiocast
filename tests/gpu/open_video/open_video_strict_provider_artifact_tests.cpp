#include "core/onnx/ort_session.h"
#include "core/open_video/model_pack_registry.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using studiocast::onnx::OrtProviderPolicy;
using studiocast::onnx::OrtSessionInfo;
using studiocast::onnx::OrtSessionOptions;
using studiocast::onnx::internal::OrtProviderAppendHooks;
using studiocast::onnx::internal::OrtProviderAppendResult;
using studiocast::onnx::internal::OrtSessionCreateAttempt;
using studiocast::onnx::internal::OrtSessionCreateHooks;
using studiocast::onnx::internal::OrtSessionCreateResult;

bool Expect(bool condition, const std::string &message) {
  if (!condition)
    std::cerr << "FAIL: " << message << "\n";
  return condition;
}

struct TempTree {
  std::filesystem::path root;

  TempTree() {
    static std::uint64_t sequence = 0;
    root = std::filesystem::temp_directory_path() /
           ("studiocast-strict-artifact-" +
            std::to_string(++sequence) + "-" +
            std::to_string(
                reinterpret_cast<std::uintptr_t>(this)));
    std::filesystem::create_directories(root / "matting" / "strict-pack");
  }

  ~TempTree() {
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
  }

  std::filesystem::path Pack() const {
    return root / "matting" / "strict-pack";
  }
};

bool WriteFile(const std::filesystem::path &path, const std::string &bytes) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  return stream.good();
}

std::string Manifest(const std::string &declared_sha256 = {}) {
  std::string checksum;
  if (!declared_sha256.empty())
    checksum = ",\"sha256\":\"" + declared_sha256 + "\"";
  return std::string{
             R"({"schema_version":2,"id":"strict-matting","display_name":"Strict Matting","task":"matting","files":[{"name":"model.onnx","kind":"onnx","role":"main")"} +
         checksum +
         R"(}],"input":{"name":"input","layout":"nchw","dtype":"float32","width":8,"height":8,"channels":3},"output":{"name":"alpha","kind":"alpha","dtype":"float32"},"preprocess":{"mean":[0,0,0],"std":[1,1,1],"color":"rgb","range":"0..1"}})";
}

const studiocast::open_video::ModelPack *
OnlyPack(const studiocast::open_video::ModelPackRegistry &registry) {
  if (registry.ListModels().size() != 1) {
    for (const auto &[key, problem] : registry.Problems())
      std::cerr << "registry problem " << key << ": " << problem << "\n";
  }
  return registry.ListModels().size() == 1 ? &registry.ListModels()[0]
                                           : nullptr;
}

bool TestArtifactIdentityAndCurrentness() {
  TempTree tree;
  if (!WriteFile(tree.Pack() / "model.onnx", "artifact-a") ||
      !WriteFile(tree.Pack() / "model.json", Manifest()))
    return Expect(false, "fixture files should be writable");

  const auto first =
      studiocast::open_video::ModelPackRegistry::Scan(tree.root);
  const auto *first_pack = OnlyPack(first);
  if (!Expect(first_pack != nullptr, "valid model pack should scan") ||
      !Expect(first.generation() != 0,
              "registry scan must have a non-zero generation"))
    return false;
  const auto identity_a =
      studiocast::open_video::ValidateModelArtifactIdentity(
          *first_pack, first_pack->files[0]);
  if (!Expect(identity_a.ok(),
              "registered artifact should produce exact identity: " +
                  identity_a.detail))
    return false;

  const auto second =
      studiocast::open_video::ModelPackRegistry::Scan(tree.root);
  const auto *second_pack = OnlyPack(second);
  if (!Expect(second_pack != nullptr, "second model scan should succeed"))
    return false;
  const auto identity_new_generation =
      studiocast::open_video::ValidateModelArtifactIdentity(
          *second_pack, second_pack->files[0]);
  if (!Expect(identity_new_generation.ok(),
              "rescanned artifact should validate"))
    return false;
  if (!Expect(identity_a.identity.artifact_sha256 ==
                  identity_new_generation.identity.artifact_sha256,
              "unchanged artifact bytes should keep exact content hash") ||
      !Expect(identity_a.identity.registry_generation !=
                  identity_new_generation.identity.registry_generation,
              "registry refresh must advance generation") ||
      !Expect(!studiocast::open_video::ModelArtifactIdentityMatches(
                  identity_a.identity, identity_new_generation.identity),
              "old identity must be stale after registry refresh"))
    return false;

  if (!WriteFile(tree.Pack() / "model.onnx", "artifact-b"))
    return Expect(false, "changed artifact should be writable");
  const auto changed =
      studiocast::open_video::ModelPackRegistry::Scan(tree.root);
  const auto *changed_pack = OnlyPack(changed);
  if (!Expect(changed_pack != nullptr, "changed pack should scan"))
    return false;
  const auto identity_b =
      studiocast::open_video::ValidateModelArtifactIdentity(
          *changed_pack, changed_pack->files[0]);
  return Expect(identity_b.ok(), "changed artifact should validate") &&
         Expect(identity_a.identity.artifact_sha256 !=
                    identity_b.identity.artifact_sha256,
                "artifact byte change must change actual SHA-256") &&
         Expect(identity_a.identity.manifest_sha256 ==
                    identity_b.identity.manifest_sha256,
                "unchanged manifest must keep exact SHA-256");
}

bool TestArtifactFailuresAndManifestIdentity() {
  TempTree tree;
  if (!WriteFile(tree.Pack() / "model.onnx", "artifact") ||
      !WriteFile(tree.Pack() / "model.json",
                 Manifest(std::string(64, '0'))))
    return Expect(false, "bad-checksum fixture should be writable");

  const auto bad = studiocast::open_video::ModelPackRegistry::Scan(tree.root);
  const auto *bad_pack = OnlyPack(bad);
  if (!Expect(bad_pack != nullptr, "checksum mismatch pack should parse"))
    return false;
  const auto mismatch =
      studiocast::open_video::ValidateModelArtifactIdentity(
          *bad_pack, bad_pack->files[0]);
  if (!Expect(
          mismatch.status ==
              studiocast::open_video::ModelArtifactValidationStatus::
                  checksum_mismatch,
          "declared checksum mismatch must fail closed"))
    return false;

  auto foreign = *bad_pack;
  foreign.registry_generation = 0;
  const auto foreign_result =
      studiocast::open_video::ValidateModelArtifactIdentity(
          foreign, foreign.files[0]);
  if (!Expect(
          foreign_result.status ==
              studiocast::open_video::ModelArtifactValidationStatus::
                  foreign_pack,
          "pack without registry generation must be rejected"))
    return false;

  if (!WriteFile(tree.Pack() / "model.json", Manifest()))
    return Expect(false, "good manifest should be writable");
  const auto good = studiocast::open_video::ModelPackRegistry::Scan(tree.root);
  const auto *good_pack = OnlyPack(good);
  if (!Expect(good_pack != nullptr, "good pack should parse"))
    return false;
  const auto before =
      studiocast::open_video::ValidateModelArtifactIdentity(
          *good_pack, good_pack->files[0]);
  if (!Expect(before.ok(), "good artifact should validate"))
    return false;

  if (!WriteFile(tree.Pack() / "model.json",
                 Manifest() + std::string{"\n"}))
    return Expect(false, "manifest-only change should be writable");
  const auto stale_manifest =
      studiocast::open_video::ValidateModelArtifactIdentity(
          *good_pack, good_pack->files[0]);
  if (!Expect(
          stale_manifest.status ==
              studiocast::open_video::ModelArtifactValidationStatus::
                  stale_registry,
          "manifest change after registry parsing must fail closed"))
    return false;
  const auto manifest_changed =
      studiocast::open_video::ModelPackRegistry::Scan(tree.root);
  const auto *manifest_changed_pack = OnlyPack(manifest_changed);
  if (!Expect(manifest_changed_pack != nullptr,
              "whitespace-changed manifest should parse"))
    return false;
  const auto after =
      studiocast::open_video::ValidateModelArtifactIdentity(
          *manifest_changed_pack, manifest_changed_pack->files[0]);
  if (!Expect(after.ok(), "manifest-changed artifact should validate") ||
      !Expect(before.identity.manifest_sha256 !=
                  after.identity.manifest_sha256,
              "manifest byte change must change exact SHA-256"))
    return false;

  std::filesystem::remove(tree.Pack() / "model.onnx");
  if (!Expect(studiocast::open_video::ModelArtifactIdentityMatches(
                  after.identity, after.identity),
              "execution-time identity matching must use cached POD evidence "
              "without filesystem discovery"))
    return false;
  const auto missing =
      studiocast::open_video::ValidateModelArtifactIdentity(
          *manifest_changed_pack, manifest_changed_pack->files[0]);
  return Expect(
      missing.status ==
          studiocast::open_video::ModelArtifactValidationStatus::missing,
      "missing artifact must fail closed");
}

struct FakeProvider {
  studiocast::onnx::OrtRuntimeInfo runtime;
  bool append_cuda = false;
  bool create = true;
  std::vector<OrtSessionCreateAttempt> attempts;
};

OrtProviderAppendResult AppendCuda(void *context,
                                   const OrtSessionOptions &) {
  const auto *fake = static_cast<FakeProvider *>(context);
  OrtProviderAppendResult result;
  result.appended = fake->append_cuda;
  result.status = result.appended ? "appended" : "unavailable";
  return result;
}

OrtSessionCreateResult CreateSession(void *context,
                                     OrtSessionCreateAttempt attempt,
                                     const OrtSessionOptions &options,
                                     OrtSessionInfo *info) {
  auto *fake = static_cast<FakeProvider *>(context);
  fake->attempts.push_back(attempt);
  OrtProviderAppendHooks append;
  append.context = fake;
  append.append_cuda = AppendCuda;
  if (info) {
    *info = studiocast::onnx::internal::PlanOrtProviderAttempt(
        fake->runtime, options, false, append);
  }
  OrtSessionCreateResult result;
  result.created = fake->create;
  if (!result.created)
    result.error = "fake session creation failure";
  return result;
}

studiocast::onnx::internal::OrtSessionCreatePlanResult
RunPlan(FakeProvider *fake, OrtSessionOptions options) {
  OrtSessionCreateHooks hooks;
  hooks.context = fake;
  hooks.create_session = CreateSession;
  return studiocast::onnx::internal::CreateOrtSessionWithProviderFallbacks(
      options, hooks);
}

OrtSessionOptions StrictOptions() {
  OrtSessionOptions options;
  options.provider_policy = OrtProviderPolicy::strict_cuda_only;
  options.prefer_cuda = true;
  options.cuda_device_id = 3;
  options.cuda_context_generation = 41;
  options.user_compute_stream = reinterpret_cast<void *>(0x1234);
  return options;
}

bool TestStrictProviderPolicy() {
  FakeProvider fake;
  fake.runtime.providers = {"CPUExecutionProvider",
                            "CUDAExecutionProvider"};
  fake.runtime.cpu_provider_present = true;
  fake.runtime.cuda_provider_present = true;
  fake.append_cuda = true;

  const auto strict = RunPlan(&fake, StrictOptions());
  if (!Expect(strict.created, "exact strict CUDA plan should create") ||
      !Expect(strict.info.strict_provider_validation_passed,
              "strict plan should commit typed evidence") ||
      !Expect(strict.info.cpu_ep_fallback_disabled,
              "strict plan must disable CPU graph/provider fallback") ||
      !Expect(strict.info.prepared_provider_evidence.ValidStrictCuda(),
              "strict provider evidence should be valid"))
    return false;

  const auto &evidence = strict.info.prepared_provider_evidence;
  if (!Expect(studiocast::onnx::OrtPreparedProviderEvidenceMatches(
                  evidence, 3, 41, 0x1234),
              "exact provider/device/context/stream should match") ||
      !Expect(!studiocast::onnx::OrtPreparedProviderEvidenceMatches(
                  evidence, 4, 41, 0x1234),
              "wrong CUDA device must fail") ||
      !Expect(!studiocast::onnx::OrtPreparedProviderEvidenceMatches(
                  evidence, 3, 42, 0x1234),
              "wrong context generation must fail") ||
      !Expect(!studiocast::onnx::OrtPreparedProviderEvidenceMatches(
                  evidence, 3, 41, 0x9999),
              "wrong stream identity must fail"))
    return false;

  FakeProvider unavailable = fake;
  unavailable.append_cuda = false;
  unavailable.attempts.clear();
  const auto rejected = RunPlan(&unavailable, StrictOptions());
  if (!Expect(!rejected.created,
              "strict mode must reject unavailable CUDA EP") ||
      !Expect(unavailable.attempts.size() == 1,
              "strict mode must not retry a CPU session"))
    return false;

  FakeProvider failed = fake;
  failed.create = false;
  failed.attempts.clear();
  const auto create_failed = RunPlan(&failed, StrictOptions());
  if (!Expect(!create_failed.created,
              "strict CUDA session failure must fail closed") ||
      !Expect(failed.attempts.size() == 1,
              "strict session failure must not retry CPU"))
    return false;

  FakeProvider tensorrt = fake;
  tensorrt.attempts.clear();
  auto mismatched = StrictOptions();
  mismatched.enable_tensorrt = true;
  const auto trt_result = RunPlan(&tensorrt, mismatched);
  return Expect(!trt_result.created,
                "strict CUDA mode must reject TensorRT identity") &&
         Expect(tensorrt.attempts.empty(),
                "TensorRT mismatch should be rejected before creation");
}

bool TestCompatibilityProviderPolicy() {
  FakeProvider fake;
  fake.runtime.providers = {"CPUExecutionProvider"};
  fake.runtime.cpu_provider_present = true;
  OrtSessionOptions options;
  options.prefer_cuda = true;
  const auto compatibility = RunPlan(&fake, options);
  return Expect(compatibility.created,
                "compatibility mode should retain CPU fallback") &&
         Expect(compatibility.info.cpu_provider_usable,
                "compatibility CPU session should be usable") &&
         Expect(!compatibility.info.strict_provider_validation_passed,
                "compatibility mode must not claim strict evidence");
}

} // namespace

int main() {
  static_assert(std::is_trivially_copyable_v<
                studiocast::onnx::OrtPreparedProviderEvidence>);
  static_assert(std::is_trivially_copyable_v<
                studiocast::open_video::ValidatedModelArtifactIdentity>);

  bool ok = true;
  ok = TestArtifactIdentityAndCurrentness() && ok;
  ok = TestArtifactFailuresAndManifestIdentity() && ok;
  ok = TestStrictProviderPolicy() && ok;
  ok = TestCompatibilityProviderPolicy() && ok;
  if (ok)
    std::cout << "open video strict provider/artifact tests passed\n";
  return ok ? 0 : 1;
}
