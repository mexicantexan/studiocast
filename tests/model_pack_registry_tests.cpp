#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

#include "core/onnx/ort_session.h"
#include "core/open_audio/model_pack_registry.h"
#include "core/open_audio/open_audio_diagnostics.h"
#include "core/open_audio/open_audio_onnx_session.h"
#include "core/open_video/diagnose.h"
#include "core/open_video/diagnostics.h"
#include "core/open_video/model_pack_registry.h"

namespace {

bool Expect(bool condition, const std::string &message) {
  if (!condition) {
    std::cerr << message << "\n";
    return false;
  }
  return true;
}

bool ExpectEq(const std::string &name, const std::string &got,
              const std::string &want) {
  if (got == want)
    return true;
  std::cerr << name << "\n  got:  " << got << "\n  want: " << want << "\n";
  return false;
}

bool ExpectContains(const std::string &name, const std::string &haystack,
                    const std::string &needle) {
  if (haystack.find(needle) != std::string::npos)
    return true;
  std::cerr << name << "\n  expected substring: " << needle
            << "\n  in: " << haystack << "\n";
  return false;
}

[[maybe_unused]] bool
VectorContainsSubstring(const std::vector<std::string> &items,
                        const std::string &needle) {
  for (const auto &item : items) {
    if (item.find(needle) != std::string::npos)
      return true;
  }
  return false;
}

bool ShouldRunTest(const char *name) {
  const char *only = std::getenv("STUDIOCAST_TEST_ONLY");
  if (!only || !*only)
    return true;
  const std::string filter(only);
  std::size_t pos = 0;
  while (pos <= filter.size()) {
    const std::size_t comma = filter.find(',', pos);
    const std::string token = filter.substr(
        pos, comma == std::string::npos ? std::string::npos : comma - pos);
    if (token == name)
      return true;
    if (comma == std::string::npos)
      break;
    pos = comma + 1;
  }
  return false;
}

bool RunTestInChild(const char *name, bool (*fn)()) {
  const pid_t pid = fork();
  if (pid < 0) {
    std::cerr << "fork failed for " << name << "; running in-process\n";
    return fn();
  }
  if (pid == 0) {
    const bool ok = fn();
    std::cout.flush();
    std::cerr.flush();
    _exit(ok ? 0 : 1);
  }

  int status = 0;
  if (waitpid(pid, &status, 0) < 0) {
    std::cerr << "waitpid failed for " << name << "\n";
    return false;
  }
  if (WIFEXITED(status) && WEXITSTATUS(status) == 0)
    return true;
  if (WIFSIGNALED(status)) {
    std::cerr << name << " terminated by signal " << WTERMSIG(status) << "\n";
  } else {
    std::cerr << name << " failed with status " << status << "\n";
  }
  return false;
}

bool RunNamedTest(const char *name, bool (*fn)(), bool isolate = false) {
  if (!ShouldRunTest(name))
    return true;
  return isolate ? RunTestInChild(name, fn) : fn();
}

template <typename Result>
const Result *FindResult(const std::vector<Result> &results,
                         const std::string &id) {
  for (const auto &r : results) {
    if (r.id == id)
      return &r;
  }
  return nullptr;
}

class ScopedDataHome {
public:
  explicit ScopedDataHome(const std::string &name) {
    had_old_ = std::getenv("XDG_DATA_HOME") != nullptr;
    if (had_old_)
      old_ = std::getenv("XDG_DATA_HOME");

    std::error_code ec;
    const auto tmp = std::filesystem::temp_directory_path(ec);
    if (ec) {
      error_ = "failed to resolve temp directory: " + ec.message();
      return;
    }

    dir_ =
        tmp / (name + "-" + std::to_string(static_cast<long long>(::getpid())));
    std::filesystem::remove_all(dir_, ec);
    std::filesystem::create_directories(dir_, ec);
    if (ec) {
      error_ = "failed to create temp data dir: " + ec.message();
      return;
    }

    if (::setenv("XDG_DATA_HOME", dir_.string().c_str(), 1) != 0) {
      error_ =
          std::string("setenv(XDG_DATA_HOME) failed: ") + std::strerror(errno);
    }
  }

  ~ScopedDataHome() {
    if (had_old_) {
      (void)::setenv("XDG_DATA_HOME", old_.c_str(), 1);
    } else {
      (void)::unsetenv("XDG_DATA_HOME");
    }

    std::error_code ec;
    std::filesystem::remove_all(dir_, ec);
  }

  bool ok() const { return error_.empty(); }
  const std::string &error() const { return error_; }
  const std::filesystem::path &path() const { return dir_; }

private:
  std::filesystem::path dir_;
  bool had_old_ = false;
  std::string old_;
  std::string error_;
};

bool WriteTextFile(const std::filesystem::path &path, const std::string &text) {
  std::error_code ec;
  std::filesystem::create_directories(path.parent_path(), ec);
  if (ec) {
    std::cerr << "failed to create parent directory for " << path << ": "
              << ec.message() << "\n";
    return false;
  }

  std::ofstream out(path, std::ios::out | std::ios::trunc);
  if (!out) {
    std::cerr << "failed to open " << path << " for write\n";
    return false;
  }
  out << text;
  return static_cast<bool>(out);
}

bool WriteOpenVideoMattingPack(const std::filesystem::path &root,
                               const std::string &dir_name,
                               const std::string &id) {
  const auto dir = root / "matting" / dir_name;
  const std::string manifest =
      std::string("{\n") + "  \"id\":\"" + id + "\",\n" +
      "  \"display_name\":\"" + id + "\",\n" + "  \"task\":\"matting\",\n" +
      "  \"onnx_filename\":\"model.onnx\",\n" +
      "  \"input\":{\"name\":\"input\",\"layout\":\"nchw\","
      "\"dtype\":\"float32\",\"width\":256,\"height\":256,"
      "\"channels\":3},\n" +
      "  \"output\":{\"name\":\"alpha\",\"kind\":\"alpha\","
      "\"dtype\":\"float32\"},\n" +
      "  \"preprocess\":{\"mean\":[0.5,0.5,0.5],"
      "\"std\":[0.5,0.5,0.5],\"color\":\"rgb\","
      "\"range\":\"0..1\"}\n" +
      "}\n";
  return WriteTextFile(dir / "model.onnx", "synthetic matting model\n") &&
         WriteTextFile(dir / "LICENSE.txt", "synthetic test license\n") &&
         WriteTextFile(dir / "model.json", manifest);
}

bool WriteOpenVideoV2Pack(const std::filesystem::path &root,
                          const std::string &task, const std::string &id,
                          const std::string &file_name = "model.onnx") {
  const auto dir = root / task / id;
  const std::string manifest =
      std::string("{\n") + "  \"schema_version\":2,\n" + "  \"id\":\"" + id +
      "\",\n" + "  \"display_name\":\"" + id + "\",\n" + "  \"task\":\"" +
      task + "\",\n" + "  \"files\":[{\"name\":\"" + file_name +
      "\",\"kind\":\"onnx\",\"role\":\"main\",\"sha256\":\"\"}]\n" + "}\n";
  return WriteTextFile(dir / file_name, "synthetic model file\n") &&
         WriteTextFile(dir / "LICENSE.txt", "synthetic test license\n") &&
         WriteTextFile(dir / "model.json", manifest);
}

bool WriteNcnnVulkanMattingPack(
    const std::filesystem::path &root, const std::string &id,
    const std::string &param_sha256, const std::string &bin_sha256,
    const std::string &ncnn_section, bool declare_bin = true,
    bool exact_output = true, int schema_version = 2,
    const std::string &param_role = "vulkan_matting") {
  const auto dir = root / "matting" / id;
  const std::string bin_entry =
      declare_bin
          ? std::string(
                ",\n    {\"name\":\"model.ncnn.bin\",\"kind\":\"ncnn_bin\","
                "\"role\":\"vulkan_matting\",\"sha256\":\"") +
                bin_sha256 + "\"}"
          : "";
  const std::string manifest =
      std::string("{\n") +
      "  \"schema_version\":" + std::to_string(schema_version) + ",\n" +
      "  \"id\":\"" + id + "\",\n" + "  \"display_name\":\"" + id + "\",\n" +
      "  \"task\":\"matting\",\n" +
      (schema_version == 1 ? "  \"onnx_filename\":\"model.onnx\",\n" : "") +
      "  \"input\":{\"name\":\"input\",\"layout\":\"nchw\","
      "\"dtype\":\"float32\",\"width\":256,\"height\":256,"
      "\"channels\":3},\n" +
      "  \"output\":{\"name\":\"alpha\",\"kind\":\"alpha\","
      "\"dtype\":\"float32\"" +
      (exact_output ? ",\"layout\":\"nchw\",\"width\":160,\"height\":144,"
                      "\"channels\":1"
                    : "") +
      "},\n" +
      "  \"preprocess\":{\"mean\":[0.5,0.5,0.5],"
      "\"std\":[0.5,0.5,0.5],\"color\":\"rgb\","
      "\"range\":\"0..1\"},\n" +
      "  \"files\":[\n"
      "    {\"name\":\"model.onnx\",\"kind\":\"onnx\","
      "\"role\":\"main\",\"sha256\":\"\"},\n"
      "    {\"name\":\"model.ncnn.param\",\"kind\":\"ncnn_param\","
      "\"role\":\"" +
      param_role + "\",\"sha256\":\"" + param_sha256 + "\"}" + bin_entry +
      "\n  ],\n" + ncnn_section + "\n}\n";
  return WriteTextFile(dir / "model.onnx", "synthetic matting model\n") &&
         WriteTextFile(dir / "model.ncnn.param", "synthetic ncnn param\n") &&
         WriteTextFile(dir / "model.ncnn.bin", "synthetic ncnn weights\n") &&
         WriteTextFile(dir / "model.json", manifest);
}

bool DiagnosticsContainsModel(
    const studiocast::open_cuda::OpenCudaDiagnostics &d, const std::string &id,
    const std::string &task) {
  return std::any_of(d.models.begin(), d.models.end(), [&](const auto &m) {
    return m.id == id && m.task == task;
  });
}

bool TestOpenVideoIntegrity() {
  namespace ov = studiocast::open_video;
  const auto root = std::filesystem::path("tests") / "data" / "models" /
                    "model_integrity" / "open_video";

  const auto reg = ov::ModelPackRegistry::Scan(root);
  if (!Expect(reg.ResolveModel("video_good_sha").has_value(),
              "video_good_sha should scan as usable"))
    return false;
  if (!Expect(reg.ResolveModel("video_placeholder_pack") == std::nullopt,
              "placeholder Open Video pack should not be usable"))
    return false;
  {
    const auto &problems = reg.Problems();
    auto it = problems.find("video_placeholder_pack");
    if (!Expect(it != problems.end(),
                "placeholder Open Video pack should be reported"))
      return false;
    if (!Expect(it->second.find("placeholder") != std::string::npos,
                "placeholder Open Video problem should be classified"))
      return false;
  }

  const auto results = ov::ModelPackRegistry::Verify(root);
  const auto *good = FindResult(results, "video_good_sha");
  if (!Expect(good && good->ok, "video_good_sha should verify"))
    return false;
  if (!Expect(!good->files.empty(),
              "video_good_sha should include file result"))
    return false;
  if (!ExpectEq("video_good_sha status", good->status, "ok"))
    return false;
  if (!ExpectEq(
          "video_good_sha file actual", good->files.front().actual_sha256,
          "cbc724d038823203ec77f127b74e6d84341e15f2befee18af5d6f5739f00492b"))
    return false;
  if (!ExpectEq("video_good_sha checksum kind",
                good->files.front().checksum_kind, "installed_sha256"))
    return false;

  const auto *bad = FindResult(results, "video_bad_sha");
  if (!Expect(bad && !bad->ok, "video_bad_sha should fail verification"))
    return false;
  if (!ExpectEq("video_bad_sha status", bad->status, "checksum_mismatch"))
    return false;
  if (!Expect(!bad->files.empty() &&
                  bad->files.front().status == "checksum_mismatch",
              "video_bad_sha file should report checksum_mismatch"))
    return false;

  const auto *missing = FindResult(results, "video_missing_file");
  if (!Expect(missing && !missing->ok,
              "video_missing_file should fail verification"))
    return false;
  if (!ExpectEq("video_missing_file status", missing->status, "missing"))
    return false;

  const auto *invalid = FindResult(results, "video_invalid_manifest");
  if (!Expect(invalid && !invalid->ok,
              "video_invalid_manifest should fail verification"))
    return false;
  if (!ExpectEq("video_invalid_manifest status", invalid->status,
                "invalid_manifest"))
    return false;

  const auto *placeholder = FindResult(results, "video_placeholder_pack");
  if (!Expect(placeholder && !placeholder->ok,
              "video_placeholder_pack should fail verification"))
    return false;
  return ExpectEq("video_placeholder_pack status", placeholder->status,
                  "placeholder");
}

bool TestProductionNcnnVulkanModelContract() {
  namespace ov = studiocast::open_video;
  ScopedDataHome data_home("studiocast-ncnn-vulkan-model-contract");
  if (!Expect(data_home.ok(), data_home.error()))
    return false;
  const auto root = data_home.path() / "studiocast" / "models" / "open_video";
  constexpr const char *kParamSha =
      "596bb1f4737715a50855892043c1b3cce8128a17c660d434f79ad6e2bd43b474";
  constexpr const char *kBinSha =
      "01a2da95e2c0d77e052f6696f6897a27f29ef2f489d3ce4801fdf7f4276c5e4a";
  const std::string valid_section =
      "  \"ncnn_vulkan\":{\n"
      "    \"param_file\":\"model.ncnn.param\",\n"
      "    \"bin_file\":\"model.ncnn.bin\",\n"
      "    \"input_blob\":\"input\",\n"
      "    \"output_blob\":\"alpha\",\n"
      "    \"converter\":{\"name\":\"pnnx\",\"version\":\"20250503\"},\n"
      "    \"precision\":\"fp32\"\n"
      "  }";
  const std::string missing_blob_section =
      "  \"ncnn_vulkan\":{\n"
      "    \"param_file\":\"model.ncnn.param\",\n"
      "    \"bin_file\":\"model.ncnn.bin\",\n"
      "    \"input_blob\":\"input\",\n"
      "    \"converter\":{\"name\":\"pnnx\",\"version\":\"20250503\"},\n"
      "    \"precision\":\"fp32\"\n"
      "  }";
  const std::string unsafe_path_section =
      "  \"ncnn_vulkan\":{\n"
      "    \"param_file\":\"../model.ncnn.param\",\n"
      "    \"bin_file\":\"model.ncnn.bin\",\n"
      "    \"input_blob\":\"input\",\n"
      "    \"output_blob\":\"alpha\",\n"
      "    \"converter\":{\"name\":\"pnnx\",\"version\":\"20250503\"},\n"
      "    \"precision\":\"fp32\"\n"
      "  }";
  const std::string fp16_artifact_section =
      "  \"ncnn_vulkan\":{\n"
      "    \"param_file\":\"model.ncnn.param\",\n"
      "    \"bin_file\":\"model.ncnn.bin\",\n"
      "    \"input_blob\":\"input\",\n"
      "    \"output_blob\":\"alpha\",\n"
      "    \"converter\":{\"name\":\"pnnx\",\"version\":\"20250503\"},\n"
      "    \"precision\":\"fp16\"\n"
      "  }";

  if (!WriteNcnnVulkanMattingPack(root, "ncnn_valid", kParamSha, kBinSha,
                                  valid_section) ||
      !WriteNcnnVulkanMattingPack(
          root, "ncnn_bad_checksum",
          "696bb1f4737715a50855892043c1b3cce8128a17c660d434f79ad6e2bd43b474",
          kBinSha, valid_section) ||
      !WriteNcnnVulkanMattingPack(root, "ncnn_missing_artifact", kParamSha,
                                  kBinSha, valid_section, false) ||
      !WriteNcnnVulkanMattingPack(root, "ncnn_missing_blob", kParamSha, kBinSha,
                                  missing_blob_section) ||
      !WriteNcnnVulkanMattingPack(root, "ncnn_unsafe_path", kParamSha, kBinSha,
                                  unsafe_path_section) ||
      !WriteNcnnVulkanMattingPack(root, "ncnn_missing_output_geometry",
                                  kParamSha, kBinSha, valid_section,
                                  /*declare_bin=*/true,
                                  /*exact_output=*/false) ||
      !WriteNcnnVulkanMattingPack(root, "ncnn_schema_v1", kParamSha, kBinSha,
                                  valid_section, /*declare_bin=*/true,
                                  /*exact_output=*/true,
                                  /*schema_version=*/1) ||
      !WriteNcnnVulkanMattingPack(root, "ncnn_wrong_role", kParamSha, kBinSha,
                                  valid_section, /*declare_bin=*/true,
                                  /*exact_output=*/true,
                                  /*schema_version=*/2, "main") ||
      !WriteNcnnVulkanMattingPack(root, "ncnn_fp16_artifacts", kParamSha,
                                  kBinSha, fp16_artifact_section) ||
      !WriteOpenVideoMattingPack(root, "onnx-only", "onnx_only")) {
    return false;
  }

  const auto registry = ov::ModelPackRegistry::Scan(root);
  const auto valid = registry.ResolveModel("ncnn_valid");
  if (!Expect(valid.has_value(), "complete ncnn Vulkan pack should scan") ||
      !Expect(valid->ncnn_vulkan.has_value(),
              "complete pack should expose ncnn Vulkan metadata")) {
    return false;
  }
  const auto &spec = *valid->ncnn_vulkan;
  if (!Expect(spec.param_path.is_absolute() && spec.bin_path.is_absolute(),
              "ncnn artifact paths should resolve absolutely") ||
      !ExpectEq("ncnn input blob", spec.input_blob, "input") ||
      !ExpectEq("ncnn output blob", spec.output_blob, "alpha") ||
      !ExpectEq("ncnn converter", spec.converter_name, "pnnx") ||
      !ExpectEq("ncnn converter version", spec.converter_version, "20250503") ||
      !ExpectEq("ncnn precision", spec.precision, "fp32")) {
    return false;
  }
  if (!ExpectEq("ncnn param relative path", spec.param_file,
                "model.ncnn.param") ||
      !Expect(valid->matting->output.width == 160 &&
                  valid->matting->output.height == 144 &&
                  valid->matting->output.channels == 1,
              "ncnn output geometry must remain exact and independent of "
              "input geometry")) {
    return false;
  }

  std::string error;
  if (!Expect(ov::ValidateProductionNcnnVulkanMattingPack(*valid, &error),
              "complete ncnn Vulkan pack should validate: " + error)) {
    return false;
  }
  const auto fp16 = registry.ResolveModel("ncnn_fp16_artifacts");
  if (!Expect(fp16.has_value(),
              "fp16 artifact pack with float32 bindings should scan") ||
      !Expect(
          ov::ValidateProductionNcnnVulkanMattingPack(*fp16, &error),
          "fp16 artifact pack should retain float32 binding compatibility: " +
              error)) {
    return false;
  }
  const auto schema_v1 = registry.ResolveModel("ncnn_schema_v1");
  if (!Expect(schema_v1.has_value(),
              "legacy schema pack should remain usable outside Vulkan") ||
      !Expect(!ov::ValidateProductionNcnnVulkanMattingPack(*schema_v1, &error),
              "legacy schema must fail the production Vulkan gate") ||
      !ExpectContains("production Vulkan schema rejection", error,
                      "requires schema_version=2")) {
    return false;
  }

  const auto bad_checksum = registry.ResolveModel("ncnn_bad_checksum");
  if (!Expect(bad_checksum.has_value(),
              "checksum mismatch is a session validation failure") ||
      !Expect(
          !ov::ValidateProductionNcnnVulkanMattingPack(*bad_checksum, &error),
          "checksum mismatch must fail closed") ||
      !ExpectContains("ncnn checksum failure", error, "checksum_mismatch")) {
    return false;
  }

  const auto onnx_only = registry.ResolveModel("onnx_only");
  if (!Expect(onnx_only.has_value(),
              "ONNX-only matting pack should remain valid") ||
      !Expect(!ov::ValidateProductionNcnnVulkanMattingPack(*onnx_only, &error),
              "ONNX-only pack must not pass the production ncnn gate") ||
      !ExpectContains("legacy ONNX-only schema", error,
                      "requires schema_version=2")) {
    return false;
  }

  const auto &problems = registry.Problems();
  const auto expect_problem = [&](const char *id, const char *needle) {
    const auto it = problems.find(id);
    return Expect(it != problems.end(),
                  std::string(id) + " should be rejected") &&
           ExpectContains(id, it->second, needle);
  };
  if (!expect_problem("ncnn_missing_artifact", "must be declared") ||
      !expect_problem("ncnn_missing_blob", "output_blob") ||
      !expect_problem("ncnn_unsafe_path", "safe relative paths") ||
      !expect_problem("ncnn_missing_output_geometry", "exact NCHW") ||
      !expect_problem("ncnn_wrong_role", "role='vulkan_matting'")) {
    return false;
  }

  const auto verification = ov::ModelPackRegistry::Verify(root);
  const auto *bad_result = FindResult(verification, "ncnn_bad_checksum");
  return Expect(bad_result && !bad_result->ok,
                "explicit pack verification should reject bad ncnn checksum") &&
         ExpectEq("ncnn bad checksum status", bad_result->status,
                  "checksum_mismatch");
}

bool TestOpenCudaDiagnosticsReportAllOpenVideoTasks() {
  ScopedDataHome dataHome("studiocast-open-video-diagnostics-test");
  if (!Expect(dataHome.ok(), dataHome.error()))
    return false;

  const auto root = dataHome.path() / "studiocast" / "models" / "open_video";
  if (!WriteOpenVideoMattingPack(root, "modnet-webnn-256-fp32",
                                 "modnet-webnn-256-fp32") ||
      !WriteOpenVideoMattingPack(root, "duplicate-modnet",
                                 "modnet-webnn-256-fp32") ||
      !WriteOpenVideoV2Pack(root, "face_detection",
                            "yunet_opencv_zoo_2023mar_fp32") ||
      !WriteOpenVideoV2Pack(root, "face_landmarks", "dlib_68_ibug_300w",
                            "shape_predictor_68_face_landmarks.dat") ||
      !WriteOpenVideoV2Pack(root, "eye_contact",
                            "gaze_correction_cam_flx_v0_1_1") ||
      !WriteOpenVideoV2Pack(root, "video_denoise", "fastdvdnet_sigma15")) {
    return false;
  }

  const auto diag = studiocast::open_cuda::DiagnoseOpenCudaDefault();

  const auto installedContains = [&](const std::string &id) {
    return std::find(diag.installed_models.begin(), diag.installed_models.end(),
                     id) != diag.installed_models.end();
  };

  return ExpectEq("Open CUDA default model remains the matting default",
                  diag.default_model_id, "modnet-webnn-256-fp32") &&
         Expect(
             DiagnosticsContainsModel(diag, "modnet-webnn-256-fp32", "matting"),
             "Open CUDA diagnostics should include matting packs") &&
         Expect(DiagnosticsContainsModel(diag, "yunet_opencv_zoo_2023mar_fp32",
                                         "face_detection"),
                "Open CUDA diagnostics should include face_detection packs") &&
         Expect(DiagnosticsContainsModel(diag, "dlib_68_ibug_300w",
                                         "face_landmarks"),
                "Open CUDA diagnostics should include face_landmarks packs") &&
         Expect(DiagnosticsContainsModel(diag, "gaze_correction_cam_flx_v0_1_1",
                                         "eye_contact"),
                "Open CUDA diagnostics should include eye_contact packs") &&
         Expect(DiagnosticsContainsModel(diag, "fastdvdnet_sigma15",
                                         "video_denoise"),
                "Open CUDA diagnostics should include video_denoise packs") &&
         Expect(installedContains("yunet_opencv_zoo_2023mar_fp32"),
                "installed_models should include non-matting model IDs") &&
         Expect(diag.missing_models.find("modnet-webnn-256-fp32") ==
                    diag.missing_models.end(),
                "duplicate installed model IDs should not be reported as "
                "missing model packs");
}

bool TestOpenAudioIntegrity() {
  namespace oa = studiocast::open_audio;
  const auto root = std::filesystem::path("tests") / "data" / "models" /
                    "model_integrity" / "open_audio";

  const auto reg = oa::ModelPackRegistry::Scan(root);
  const auto goodPack = reg.ResolveModel("audio_good_sha");
  if (!Expect(goodPack.has_value(), "audio_good_sha should scan as usable"))
    return false;
  if (!ExpectEq(
          "audio_good_sha origin_sha256", goodPack->origin_sha256,
          "0fb0fbc0fc5b6c4ddff835be62dfaa50e204ad86b993a4c4695eee5fba8e5c2a"))
    return false;
  if (!Expect(reg.ResolveModel("audio_placeholder_pack") == std::nullopt,
              "placeholder Open Audio pack should not be usable"))
    return false;

  const auto results = oa::ModelPackRegistry::Verify(root);
  const auto *good = FindResult(results, "audio_good_sha");
  if (!Expect(good && good->ok, "audio_good_sha should verify"))
    return false;
  if (!Expect(!good->files.empty(),
              "audio_good_sha should include file result"))
    return false;
  if (!ExpectEq(
          "audio_good_sha expected", good->files.front().expected_sha256,
          "0fb0fbc0fc5b6c4ddff835be62dfaa50e204ad86b993a4c4695eee5fba8e5c2a"))
    return false;
  if (!ExpectEq(
          "audio_good_sha actual", good->files.front().actual_sha256,
          "0fb0fbc0fc5b6c4ddff835be62dfaa50e204ad86b993a4c4695eee5fba8e5c2a"))
    return false;
  if (!ExpectEq("audio_good_sha checksum kind",
                good->files.front().checksum_kind, "installed_sha256"))
    return false;

  const auto *bad = FindResult(results, "audio_bad_sha");
  if (!Expect(bad && !bad->ok, "audio_bad_sha should fail verification"))
    return false;
  if (!ExpectEq("audio_bad_sha status", bad->status, "checksum_mismatch"))
    return false;

  const auto *missing = FindResult(results, "audio_missing_file");
  if (!Expect(missing && !missing->ok,
              "audio_missing_file should fail verification"))
    return false;
  if (!ExpectEq("audio_missing_file status", missing->status, "missing"))
    return false;

  const auto *invalid = FindResult(results, "audio_invalid_manifest");
  if (!Expect(invalid && !invalid->ok,
              "audio_invalid_manifest should fail verification"))
    return false;
  if (!ExpectEq("audio_invalid_manifest status", invalid->status,
                "invalid_manifest"))
    return false;

  const auto *placeholder = FindResult(results, "audio_placeholder_pack");
  if (!Expect(placeholder && !placeholder->ok,
              "audio_placeholder_pack should fail verification"))
    return false;
  return ExpectEq("audio_placeholder_pack status", placeholder->status,
                  "placeholder");
}

studiocast::onnx::OrtRuntimeInfo
MakeRuntime(std::vector<std::string> providers,
            std::vector<std::string> warnings = {}) {
  studiocast::onnx::OrtRuntimeInfo runtime;
  runtime.providers = std::move(providers);
  runtime.warnings = std::move(warnings);
  runtime.cuda_provider_present =
      std::find(runtime.providers.begin(), runtime.providers.end(),
                "CUDAExecutionProvider") != runtime.providers.end();
  runtime.tensorrt_provider_present =
      std::find(runtime.providers.begin(), runtime.providers.end(),
                "TensorrtExecutionProvider") != runtime.providers.end();
  runtime.cpu_provider_present =
      std::find(runtime.providers.begin(), runtime.providers.end(),
                "CPUExecutionProvider") != runtime.providers.end();
  return runtime;
}

struct FakeOnnxProviderBackend {
  studiocast::onnx::OrtRuntimeInfo runtime;
  bool tensorrt_ep_v2_build = false;
  studiocast::onnx::internal::OrtProviderAppendResult tensorrt_append;
  studiocast::onnx::internal::OrtProviderAppendResult cuda_append;
  bool fail_initial_create = false;
  bool fail_tensorrt_disabled_create = false;
  bool fail_cpu_create = false;
  std::string initial_error = "initial session create failed";
  std::string tensorrt_disabled_error = "cuda session create failed";
  std::string cpu_error = "cpu session create failed";
  int tensorrt_append_calls = 0;
  int cuda_append_calls = 0;
  std::vector<studiocast::onnx::internal::OrtSessionCreateAttempt> attempts;
};

studiocast::onnx::internal::OrtProviderAppendResult
FakeAppendTensorRt(void *context, const studiocast::onnx::OrtSessionOptions &) {
  auto *fake = static_cast<FakeOnnxProviderBackend *>(context);
  ++fake->tensorrt_append_calls;
  return fake->tensorrt_append;
}

studiocast::onnx::internal::OrtProviderAppendResult
FakeAppendCuda(void *context, const studiocast::onnx::OrtSessionOptions &) {
  auto *fake = static_cast<FakeOnnxProviderBackend *>(context);
  ++fake->cuda_append_calls;
  return fake->cuda_append;
}

bool FakeShouldFail(const FakeOnnxProviderBackend &fake,
                    studiocast::onnx::internal::OrtSessionCreateAttempt attempt,
                    std::string *error) {
  switch (attempt) {
  case studiocast::onnx::internal::OrtSessionCreateAttempt::Initial:
    if (fake.fail_initial_create) {
      if (error)
        *error = fake.initial_error;
      return true;
    }
    return false;
  case studiocast::onnx::internal::OrtSessionCreateAttempt::TensorRtDisabled:
    if (fake.fail_tensorrt_disabled_create) {
      if (error)
        *error = fake.tensorrt_disabled_error;
      return true;
    }
    return false;
  case studiocast::onnx::internal::OrtSessionCreateAttempt::CpuOnly:
    if (fake.fail_cpu_create) {
      if (error)
        *error = fake.cpu_error;
      return true;
    }
    return false;
  }
  return false;
}

studiocast::onnx::internal::OrtSessionCreateResult
FakeCreateSession(void *context,
                  studiocast::onnx::internal::OrtSessionCreateAttempt attempt,
                  const studiocast::onnx::OrtSessionOptions &opts,
                  studiocast::onnx::OrtSessionInfo *info_out) {
  auto *fake = static_cast<FakeOnnxProviderBackend *>(context);
  fake->attempts.push_back(attempt);

  studiocast::onnx::internal::OrtProviderAppendHooks append_hooks;
  append_hooks.context = fake;
  append_hooks.append_tensorrt = FakeAppendTensorRt;
  append_hooks.append_cuda = FakeAppendCuda;
  if (info_out) {
    *info_out = studiocast::onnx::internal::PlanOrtProviderAttempt(
        fake->runtime, opts, fake->tensorrt_ep_v2_build, append_hooks);
  }

  studiocast::onnx::internal::OrtSessionCreateResult result;
  std::string error;
  if (FakeShouldFail(*fake, attempt, &error)) {
    result.error = error;
    return result;
  }
  result.created = true;
  return result;
}

studiocast::onnx::internal::OrtSessionCreatePlanResult
RunFakeOnnxProviderPlan(studiocast::onnx::OrtSessionOptions opts,
                        FakeOnnxProviderBackend *fake) {
  studiocast::onnx::internal::OrtSessionCreateHooks create_hooks;
  create_hooks.context = fake;
  create_hooks.create_session = FakeCreateSession;
  return studiocast::onnx::internal::CreateOrtSessionWithProviderFallbacks(
      opts, create_hooks);
}

bool TestOnnxCpuOnlySessionDiagnostics() {
  FakeOnnxProviderBackend fake;
  fake.runtime = MakeRuntime({"CPUExecutionProvider"});
  studiocast::onnx::OrtSessionOptions opts;
  opts.prefer_cuda = false;

  const auto plan = RunFakeOnnxProviderPlan(opts, &fake);
  if (!Expect(plan.created, "CPU-only provider plan should create"))
    return false;
  const auto &info = plan.info;
  return Expect(!info.using_cuda, "CPU-only plan must not report CUDA") &&
         Expect(!info.using_tensorrt,
                "CPU-only plan must not report TensorRT") &&
         Expect(info.cpu_provider_advertised,
                "CPU-only plan should preserve advertised CPU provider") &&
         Expect(info.cpu_provider_usable,
                "CPU-only plan should mark CPU provider usable") &&
         ExpectEq("CPU-only active provider", info.active_provider, "cpu") &&
         Expect(fake.cuda_append_calls == 0,
                "CPU-only plan should not append CUDA");
}

bool TestOnnxCudaAppendFailureFallsBackToCpu() {
  FakeOnnxProviderBackend fake;
  fake.runtime = MakeRuntime({"CPUExecutionProvider", "CUDAExecutionProvider"});
  fake.cuda_append.status = "unavailable";
  fake.cuda_append.warnings.push_back(
      "cuda_ep_unavailable: fake CUDA runtime missing");

  studiocast::onnx::OrtSessionOptions opts;
  opts.prefer_cuda = true;
  const auto plan = RunFakeOnnxProviderPlan(opts, &fake);
  if (!Expect(plan.created,
              "CUDA append failure should still create CPU provider plan"))
    return false;

  const auto &info = plan.info;
  return Expect(info.cuda_provider_advertised,
                "CUDA append failure should preserve advertised CUDA state") &&
         Expect(!info.cuda_provider_appended,
                "CUDA append failure should not mark CUDA appended") &&
         Expect(!info.using_cuda,
                "CUDA append failure should not report CUDA active") &&
         Expect(info.cpu_provider_usable,
                "CUDA append failure should leave CPU usable") &&
         ExpectEq("CUDA append failure active provider", info.active_provider,
                  "cpu") &&
         Expect(VectorContainsSubstring(info.warnings,
                                        "cuda_ep_unavailable: fake"),
                "CUDA append failure warning should propagate");
}

bool TestOnnxCudaSessionFailureFallsBackToCpu() {
  FakeOnnxProviderBackend fake;
  fake.runtime = MakeRuntime({"CPUExecutionProvider", "CUDAExecutionProvider"});
  fake.cuda_append.appended = true;
  fake.cuda_append.status = "appended";
  fake.cuda_append.needs_stream_sync = true;
  fake.fail_initial_create = true;
  fake.initial_error = "fake CUDA session constructor failed";

  studiocast::onnx::OrtSessionOptions opts;
  opts.prefer_cuda = true;
  const auto plan = RunFakeOnnxProviderPlan(opts, &fake);
  if (!Expect(plan.created,
              "CUDA session create failure should fall back to CPU"))
    return false;

  const auto &info = plan.info;
  return Expect(fake.attempts.size() == 2,
                "CUDA session failure should try initial and CPU plans") &&
         Expect(
             fake.attempts[1] ==
                 studiocast::onnx::internal::OrtSessionCreateAttempt::CpuOnly,
             "CUDA session failure should retry CPU only") &&
         Expect(!info.using_cuda,
                "CPU fallback should not report CUDA active") &&
         Expect(info.cpu_provider_usable,
                "CPU fallback should mark CPU provider usable") &&
         Expect(info.cuda_session_create_failed_fell_back_to_cpu,
                "CPU fallback should record CUDA session create fallback") &&
         Expect(VectorContainsSubstring(
                    info.warnings,
                    "cuda_session_create_failed_fell_back_to_cpu: fake CUDA"),
                "CPU fallback should propagate CUDA create warning");
}

bool TestOnnxTensorRtRequestedFallsBackToCuda() {
  FakeOnnxProviderBackend fake;
  fake.runtime = MakeRuntime({"CPUExecutionProvider", "CUDAExecutionProvider",
                              "TensorrtExecutionProvider"});
  fake.tensorrt_ep_v2_build = true;
  fake.tensorrt_append.appended = true;
  fake.tensorrt_append.status = "appended";
  fake.tensorrt_append.cache_path = "/tmp/studiocast-test-trt-cache";
  fake.tensorrt_append.warnings.push_back(
      "tensorrt_builder_optimization_level_unavailable: fake");
  fake.cuda_append.appended = true;
  fake.cuda_append.status = "appended";
  fake.fail_initial_create = true;
  fake.initial_error = "fake TensorRT engine build failed";

  studiocast::onnx::OrtSessionOptions opts;
  opts.prefer_cuda = true;
  opts.enable_tensorrt = true;
  opts.tensorrt_enable_cuda_fallback = true;
  const auto plan = RunFakeOnnxProviderPlan(opts, &fake);
  if (!Expect(plan.created,
              "TensorRT session create failure should fall back to CUDA"))
    return false;

  const auto &info = plan.info;
  return Expect(fake.attempts.size() == 2,
                "TensorRT failure should try initial and CUDA plans") &&
         Expect(fake.attempts[1] ==
                    studiocast::onnx::internal::OrtSessionCreateAttempt::
                        TensorRtDisabled,
                "TensorRT failure should retry without TensorRT") &&
         Expect(!info.using_tensorrt,
                "TensorRT fallback should not report TensorRT active") &&
         Expect(info.using_cuda,
                "TensorRT fallback should report CUDA active") &&
         Expect(info.cuda_provider_usable,
                "TensorRT fallback should mark CUDA usable") &&
         Expect(info.tensorrt_session_create_failed_fell_back_to_cuda,
                "TensorRT fallback should record CUDA fallback") &&
         ExpectEq("TensorRT fallback status", info.tensorrt_status,
                  "session_create_failed_fell_back_to_cuda") &&
         ExpectEq("TensorRT fallback active provider", info.active_provider,
                  "cuda") &&
         Expect(info.tensorrt_engine_cache_path ==
                    std::filesystem::path("/tmp/studiocast-test-trt-cache"),
                "TensorRT fallback should preserve cache path") &&
         Expect(VectorContainsSubstring(info.warnings,
                                        "tensorrt_builder_optimization"),
                "TensorRT append warning should propagate") &&
         Expect(VectorContainsSubstring(
                    info.warnings,
                    "tensorrt_session_create_failed_fell_back_to_cuda: fake"),
                "TensorRT create warning should propagate");
}

bool TestOnnxProviderWarningPropagation() {
  FakeOnnxProviderBackend fake;
  fake.runtime =
      MakeRuntime({"CPUExecutionProvider", "CUDAExecutionProvider"},
                  {"onnxruntime_provider_query_failed: fake query warning"});
  fake.cuda_append.status = "unavailable";
  fake.cuda_append.warnings.push_back(
      "cuda_ep_unavailable: fake append warning");

  studiocast::onnx::OrtSessionOptions opts;
  opts.prefer_cuda = true;
  const auto plan = RunFakeOnnxProviderPlan(opts, &fake);
  if (!Expect(plan.created, "warning propagation plan should create"))
    return false;
  return Expect(
             VectorContainsSubstring(plan.info.warnings,
                                     "onnxruntime_provider_query_failed: fake"),
             "runtime warning should propagate to session info") &&
         Expect(VectorContainsSubstring(plan.info.warnings,
                                        "cuda_ep_unavailable: fake append"),
                "append warning should propagate to session info");
}

bool TestOpenAudioUsesSharedOnnxProviderInfo() {
#if !STUDIOCAST_HAVE_ONNXRUNTIME
  return true;
#else
  const auto model = std::filesystem::path("tests") / "data" / "models" /
                     "open_cuda" / "mock_model" / "model.onnx";
  studiocast::open_audio::OrtSessionOptions opts;
  opts.prefer_cuda = false;
  studiocast::open_audio::OrtSessionInfo info;
  std::string err;
  auto session = studiocast::open_audio::OpenAudioOrtSession::Create(
      model, opts, &info, &err);
  if (!Expect(session != nullptr,
              "Open Audio ORT adapter should create CPU session: " + err))
    return false;
  return Expect(!info.using_cuda,
                "Open Audio CPU session must not report CUDA") &&
         Expect(info.cpu_provider_usable,
                "Open Audio adapter should map shared CPU usable flag") &&
         ExpectEq("Open Audio active provider", info.active_provider, "cpu");
#endif
}

bool TestOnnxWarningsPropagateToDiagnosticsJson() {
  studiocast::open_audio::OpenAudioDiagnostics audio;
  audio.onnxruntime_warnings.push_back(
      "cuda_session_create_failed_fell_back_to_cpu: test");
  const auto audio_json = audio.ToJson();

  studiocast::open_cuda::OpenCudaDiagnostics video;
  video.onnxruntime_warnings.push_back(
      "tensorrt_session_create_failed_fell_back_to_cuda: test");
  const auto video_json = video.ToJson();

  return ExpectContains("Open Audio diagnostics warnings", audio_json,
                        "\"onnxruntime_warnings\"") &&
         ExpectContains("Open Audio diagnostics warning value", audio_json,
                        "cuda_session_create_failed_fell_back_to_cpu") &&
         ExpectContains("Open CUDA diagnostics warnings", video_json,
                        "\"onnxruntime_warnings\"") &&
         ExpectContains("Open CUDA diagnostics warning value", video_json,
                        "tensorrt_session_create_failed_fell_back_to_cuda");
}

} // namespace

int main() {
  bool ok = true;
  ok = RunNamedTest("open_video_integrity", TestOpenVideoIntegrity) && ok;
  ok = RunNamedTest("production_ncnn_vulkan_model_contract",
                    TestProductionNcnnVulkanModelContract) &&
       ok;
  ok = RunNamedTest("open_cuda_diagnostics_open_video_tasks",
                    TestOpenCudaDiagnosticsReportAllOpenVideoTasks, true) &&
       ok;
  ok = RunNamedTest("open_audio_integrity", TestOpenAudioIntegrity) && ok;
  ok = RunNamedTest("onnx_cpu_only", TestOnnxCpuOnlySessionDiagnostics, true) &&
       ok;
  ok = RunNamedTest("onnx_cuda_append_fallback",
                    TestOnnxCudaAppendFailureFallsBackToCpu) &&
       ok;
  ok = RunNamedTest("onnx_cuda_session_fallback",
                    TestOnnxCudaSessionFailureFallsBackToCpu) &&
       ok;
  ok = RunNamedTest("onnx_tensorrt_fallback",
                    TestOnnxTensorRtRequestedFallsBackToCuda) &&
       ok;
  ok = RunNamedTest("onnx_provider_warning_propagation",
                    TestOnnxProviderWarningPropagation) &&
       ok;
  ok = RunNamedTest("open_audio_shared_onnx",
                    TestOpenAudioUsesSharedOnnxProviderInfo, true) &&
       ok;
  ok = RunNamedTest("onnx_diagnostics_warnings",
                    TestOnnxWarningsPropagateToDiagnosticsJson) &&
       ok;
  if (!ok)
    return 1;
  std::cout << "MODEL REGISTRY TESTS OK\n";
  return 0;
}
