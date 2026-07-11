#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <csetjmp>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <jpeglib.h>
#include <png.h>

#include "../core/open_video/diagnose.h"
#include "../core/open_video/matting_session.h"
#include "core/audio/effects/broadcast_audio_effect_contract.h"
#include "core/audio/effects/broadcast_audio_effects_json.h"
#include "core/audio/effects/broadcast_audio_effects_plan.h"
#include "core/audio/pulse/pactl.h"
#include "core/config/daemon_config.h"
#include "core/cuda/cuda_image.h"
#include "core/cuda/cuda_tensor.h"
#include "core/cuda/kernels/open_cuda_vb_kernels.h"
#include "core/cuda/kernels/preprocess_to_nchw.h"
#include "core/cuda/kernels/resize_bilinear.h"
#include "core/maxine/afx/afx_effect.h"
#include "core/maxine/afx_api.h"
#include "core/maxine/ar_api.h"
#include "core/maxine/availability.h"
#include "core/maxine/effects/ar_auto_frame_tracker.h"
#include "core/maxine/gpu_selection.h"
#include "core/maxine/maxine_manager.h"
#include "core/maxine/nvcv_api.h"
#include "core/maxine/paths.h"
#include "core/maxine/reason_codes.h"
#include "core/maxine/vfx_api.h"
#include "core/onnx/ort_session.h"
#include "core/open_audio/model_pack_registry.h"
#include "core/open_audio/open_audio_diagnostics.h"
#include "core/open_video/model_pack_registry.h"
#include "core/probe/probe.h"
#include "core/util/json.h"
#include "core/util/strings.h"
#include "core/util/ttl_cache.h"
#include "core/util/xdg.h"
#include "core/video/broadcast_camera_effects_json.h"
#include "core/video/broadcast_camera_effects_legacy_adapter.h"
#include "core/video/camera_effects_json.h"
#include "core/video/camera_pipeline.h"
#include "core/video/capture_error_policy.h"
#include "core/video/effects/broadcast_effect_contract.h"
#include "core/video/effects/broadcast_effect_maxine_gate.h"
#include "core/video/effects/broadcast_effect_open_cuda_gate.h"
#include "core/video/effects/broadcast_effect_rules.h"
#include "core/video/effects/broadcast_effects_json.h"
#include "core/video/effects/effect_types.h"
#include "core/video/image_ppm.h"
#include "core/video/legacy_camera_effects.h"
#include "core/video/mjpeg_decode.h"
#include "core/video/scaling_policy.h"
#include "core/video/v4l2_capture.h"
#include "studiocast/version.h"

#if STUDIOCAST_ENABLE_OPEN_VULKAN
#include "core/vulkan/kernels/resize_bilinear.h"
#include "core/vulkan/vulkan_device.h"
#endif

namespace {
bool hasArg(int argc, char **argv, std::string_view flag) {
  for (int i = 1; i < argc; ++i) {
    if (argv[i] && std::string_view(argv[i]) == flag)
      return true;
  }
  return false;
}

bool EnvFlagValueEnabled(std::string_view value) {
  std::string normalized;
  normalized.reserve(value.size());
  for (char c : value) {
    if (!std::isspace(static_cast<unsigned char>(c))) {
      normalized.push_back(
          static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
  }
  return !normalized.empty() && normalized != "0" && normalized != "false" &&
         normalized != "off" && normalized != "no";
}

bool EnvFlagEnabled(const char *name) {
  const char *value = std::getenv(name);
  return value && EnvFlagValueEnabled(value);
}

struct SelfTestOptions {
  bool require_cuda_runtime = false;
};

SelfTestOptions ParseSelfTestOptions(int argc, char **argv) {
  SelfTestOptions options;
  options.require_cuda_runtime =
      hasArg(argc, argv, "--require-cuda-runtime") ||
      EnvFlagEnabled("STUDIOCAST_REQUIRE_CUDA_RUNTIME");
  return options;
}

struct VerifyFileResult {
  std::string name;
  std::string kind;
  std::string role;
  std::filesystem::path path;
  std::string expected_sha256;
  std::string actual_sha256;
  std::string checksum_kind;
  std::string status;
  std::string message;
  bool ok = false;
};

struct VerifyModelResult {
  std::string engine;
  std::string id;
  std::string display_name;
  std::string task;
  std::filesystem::path root_dir;
  std::filesystem::path manifest_path;
  std::string status;
  std::string message;
  bool ok = false;
  std::vector<VerifyFileResult> files;
};

struct VerifyEngineResult {
  std::string engine;
  std::filesystem::path root;
  std::vector<VerifyModelResult> models;

  bool Ok() const {
    for (const auto &m : models) {
      if (!m.ok)
        return false;
    }
    return true;
  }
};

std::string JsonEscape(const std::string &s) {
  return studiocast::util::json::EscapeString(s);
}

std::string BoolJson(bool v) { return v ? "true" : "false"; }

std::filesystem::path DefaultModelsRoot() {
  const auto root = studiocast::util::StudioCastModelsDir();
  if (!root.empty())
    return root;
  return std::filesystem::path{"~/.local/share/studiocast/models"};
}

VerifyFileResult
ConvertVerifyFile(const studiocast::open_video::ModelFileVerification &f) {
  VerifyFileResult out;
  out.name = f.name;
  out.kind = f.kind;
  out.role = f.role;
  out.path = f.path;
  out.expected_sha256 = f.expected_sha256;
  out.actual_sha256 = f.actual_sha256;
  out.checksum_kind = f.checksum_kind;
  out.status = f.status;
  out.message = f.message;
  out.ok = f.ok;
  return out;
}

VerifyFileResult
ConvertVerifyFile(const studiocast::open_audio::ModelFileVerification &f) {
  VerifyFileResult out;
  out.name = f.name;
  out.kind = f.kind;
  out.path = f.path;
  out.expected_sha256 = f.expected_sha256;
  out.actual_sha256 = f.actual_sha256;
  out.checksum_kind = f.checksum_kind;
  out.status = f.status;
  out.message = f.message;
  out.ok = f.ok;
  return out;
}

VerifyModelResult
ConvertVerifyModel(const std::string &engine,
                   const studiocast::open_video::ModelPackVerification &m) {
  VerifyModelResult out;
  out.engine = engine;
  out.id = m.id;
  out.display_name = m.display_name;
  out.task = m.task;
  out.root_dir = m.root_dir;
  out.manifest_path = m.manifest_path;
  out.status = m.status;
  out.message = m.message;
  out.ok = m.ok;
  out.files.reserve(m.files.size());
  for (const auto &f : m.files)
    out.files.push_back(ConvertVerifyFile(f));
  return out;
}

VerifyModelResult
ConvertVerifyModel(const std::string &engine,
                   const studiocast::open_audio::ModelPackVerification &m) {
  VerifyModelResult out;
  out.engine = engine;
  out.id = m.id;
  out.display_name = m.display_name;
  out.root_dir = m.root_dir;
  out.manifest_path = m.manifest_path;
  out.status = m.status;
  out.message = m.message;
  out.ok = m.ok;
  out.files.reserve(m.files.size());
  for (const auto &f : m.files)
    out.files.push_back(ConvertVerifyFile(f));
  return out;
}

std::vector<VerifyEngineResult> VerifyDefaultModels() {
  const auto modelsRoot = DefaultModelsRoot();
  VerifyEngineResult video;
  video.engine = "open_video";
  video.root = modelsRoot / "open_video";
  for (const auto &m :
       studiocast::open_video::ModelPackRegistry::Verify(video.root)) {
    video.models.push_back(ConvertVerifyModel(video.engine, m));
  }

  VerifyEngineResult audio;
  audio.engine = "open_audio";
  audio.root = modelsRoot / "open_audio";
  for (const auto &m :
       studiocast::open_audio::ModelPackRegistry::Verify(audio.root)) {
    audio.models.push_back(ConvertVerifyModel(audio.engine, m));
  }

  std::vector<VerifyEngineResult> engines;
  engines.push_back(std::move(video));
  engines.push_back(std::move(audio));
  return engines;
}

bool VerifyEnginesOk(const std::vector<VerifyEngineResult> &engines) {
  for (const auto &engine : engines) {
    if (!engine.Ok())
      return false;
  }
  return true;
}

std::string StatusText(const std::string &status, const std::string &message) {
  if (message.empty() || message == "ok" || message == status)
    return status;
  if (message.starts_with(status + ":"))
    return message;
  return status + ": " + message;
}

std::string VerifyModelsToText(const std::vector<VerifyEngineResult> &engines) {
  std::ostringstream oss;
  oss << "StudioCast model verification\n";
  for (const auto &engine : engines) {
    oss << "\n" << engine.engine << " (" << engine.root.string() << ")\n";
    if (engine.models.empty()) {
      oss << "  no model packs found\n";
      continue;
    }
    for (const auto &m : engine.models) {
      oss << "  " << (m.ok ? "OK" : "FAIL") << " " << m.id;
      if (!m.task.empty())
        oss << " [" << m.task << "]";
      oss << " - " << StatusText(m.status, m.message) << "\n";
      for (const auto &f : m.files) {
        oss << "    " << (f.ok ? "OK" : "FAIL") << " " << f.name;
        if (!f.kind.empty())
          oss << " (" << f.kind << ")";
        oss << " - " << StatusText(f.status, f.message) << "\n";
      }
    }
  }
  return oss.str();
}

void AppendJsonFile(std::ostringstream *oss, const VerifyFileResult &f) {
  *oss << "{";
  *oss << "\"name\":\"" << JsonEscape(f.name) << "\",";
  *oss << "\"kind\":\"" << JsonEscape(f.kind) << "\",";
  *oss << "\"role\":\"" << JsonEscape(f.role) << "\",";
  *oss << "\"path\":\"" << JsonEscape(f.path.string()) << "\",";
  *oss << "\"expected_sha256\":\"" << JsonEscape(f.expected_sha256) << "\",";
  *oss << "\"actual_sha256\":\"" << JsonEscape(f.actual_sha256) << "\",";
  *oss << "\"checksum_kind\":\"" << JsonEscape(f.checksum_kind) << "\",";
  *oss << "\"status\":\"" << JsonEscape(f.status) << "\",";
  *oss << "\"message\":\"" << JsonEscape(f.message) << "\",";
  *oss << "\"ok\":" << BoolJson(f.ok);
  *oss << "}";
}

void AppendJsonModel(std::ostringstream *oss, const VerifyModelResult &m) {
  *oss << "{";
  *oss << "\"engine\":\"" << JsonEscape(m.engine) << "\",";
  *oss << "\"id\":\"" << JsonEscape(m.id) << "\",";
  *oss << "\"display_name\":\"" << JsonEscape(m.display_name) << "\",";
  *oss << "\"task\":\"" << JsonEscape(m.task) << "\",";
  *oss << "\"root_dir\":\"" << JsonEscape(m.root_dir.string()) << "\",";
  *oss << "\"manifest_path\":\"" << JsonEscape(m.manifest_path.string())
       << "\",";
  *oss << "\"status\":\"" << JsonEscape(m.status) << "\",";
  *oss << "\"message\":\"" << JsonEscape(m.message) << "\",";
  *oss << "\"ok\":" << BoolJson(m.ok) << ",";
  *oss << "\"files\":[";
  for (std::size_t i = 0; i < m.files.size(); ++i) {
    if (i)
      *oss << ",";
    AppendJsonFile(oss, m.files[i]);
  }
  *oss << "]";
  *oss << "}";
}

std::string VerifyModelsToJson(const std::vector<VerifyEngineResult> &engines) {
  std::ostringstream oss;
  oss << "{";
  oss << "\"ok\":" << BoolJson(VerifyEnginesOk(engines)) << ",";
  oss << "\"engines\":[";
  for (std::size_t i = 0; i < engines.size(); ++i) {
    if (i)
      oss << ",";
    const auto &engine = engines[i];
    oss << "{";
    oss << "\"engine\":\"" << JsonEscape(engine.engine) << "\",";
    oss << "\"root\":\"" << JsonEscape(engine.root.string()) << "\",";
    oss << "\"ok\":" << BoolJson(engine.Ok()) << ",";
    oss << "\"models\":[";
    for (std::size_t j = 0; j < engine.models.size(); ++j) {
      if (j)
        oss << ",";
      AppendJsonModel(&oss, engine.models[j]);
    }
    oss << "]";
    oss << "}";
  }
  oss << "]";
  oss << "}";
  return oss.str();
}

int RunVerifyModels(bool json, bool strict) {
  const auto engines = VerifyDefaultModels();
  if (json) {
    std::printf("%s\n", VerifyModelsToJson(engines).c_str());
  } else {
    std::printf("%s", VerifyModelsToText(engines).c_str());
  }
  if (!strict)
    return 0;
  return VerifyEnginesOk(engines) ? 0 : 1;
}

bool WritePngRgb24File(const std::filesystem::path &path, int w, int h,
                       const std::uint8_t *rgb, std::string *error) {
  if (error)
    error->clear();
  if (!rgb || w <= 0 || h <= 0) {
    if (error)
      *error = "invalid dimensions";
    return false;
  }

  std::FILE *fp = std::fopen(path.c_str(), "wb");
  if (!fp) {
    if (error)
      *error = "failed to open for write";
    return false;
  }

  png_structp png_ptr =
      png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (!png_ptr) {
    std::fclose(fp);
    if (error)
      *error = "png_create_write_struct failed";
    return false;
  }
  png_infop info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr) {
    png_destroy_write_struct(&png_ptr, nullptr);
    std::fclose(fp);
    if (error)
      *error = "png_create_info_struct failed";
    return false;
  }

  if (setjmp(png_jmpbuf(png_ptr))) {
    png_destroy_write_struct(&png_ptr, &info_ptr);
    std::fclose(fp);
    if (error && error->empty())
      *error = "libpng write failed";
    return false;
  }

  png_init_io(png_ptr, fp);
  png_set_IHDR(png_ptr, info_ptr, static_cast<png_uint_32>(w),
               static_cast<png_uint_32>(h), 8, PNG_COLOR_TYPE_RGB,
               PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
               PNG_FILTER_TYPE_DEFAULT);
  png_write_info(png_ptr, info_ptr);

  std::vector<png_bytep> rows(static_cast<std::size_t>(h));
  for (int y = 0; y < h; ++y) {
    rows[static_cast<std::size_t>(y)] =
        const_cast<png_bytep>(reinterpret_cast<const png_byte *>(
            rgb +
            static_cast<std::size_t>(y) * static_cast<std::size_t>(w) * 3u));
  }
  png_write_image(png_ptr, rows.data());
  png_write_end(png_ptr, nullptr);

  png_destroy_write_struct(&png_ptr, &info_ptr);
  std::fclose(fp);
  return true;
}

int RunSelfTest(const SelfTestOptions &self_test_options) {
  int failures = 0;

  auto expectEq = [&](const char *name, const std::string &got,
                      const std::string &want) {
    if (got == want)
      return;
    ++failures;
    std::printf("[FAIL] %s\n  got:  '%s'\n  want: '%s'\n", name, got.c_str(),
                want.c_str());
  };

  auto expectContains = [&](const char *name, const std::string &got,
                            const std::string &needle) {
    if (got.find(needle) != std::string::npos)
      return;
    ++failures;
    std::printf("[FAIL] %s\n  did not find: '%s'\n  in: '%s'\n", name,
                needle.c_str(), got.c_str());
  };

  auto expectVecEq = [&](const char *name, const std::vector<std::string> &got,
                         const std::vector<std::string> &want) {
    if (got == want)
      return;
    ++failures;
    std::printf("[FAIL] %s\n", name);
    std::printf("  got (%zu):\n", got.size());
    for (const auto &s : got)
      std::printf("    '%s'\n", s.c_str());
    std::printf("  want (%zu):\n", want.size());
    for (const auto &s : want)
      std::printf("    '%s'\n", s.c_str());
  };

  using studiocast::util::FirstNonEmptyLine;
  using studiocast::util::Split;
  using studiocast::util::SplitLines;
  using studiocast::util::TrimCopy;

  auto expectTrue = [&](const char *name, bool v) {
    if (v)
      return;
    ++failures;
    std::printf("[FAIL] %s\n", name);
  };

  auto expectIntEq = [&](const char *name, int got, int want) {
    if (got == want)
      return;
    ++failures;
    std::printf("[FAIL] %s\n  got:  %d\n  want: %d\n", name, got, want);
  };

  auto reportCudaSmokeSkip = [&](const char *name, const char *reason,
                                 const std::string &detail) {
    if (self_test_options.require_cuda_runtime) {
      ++failures;
      std::printf("[FAIL] %s\n  required CUDA runtime smoke skipped: %s",
                  name, reason);
      if (!detail.empty())
        std::printf(": %s", detail.c_str());
      std::printf("\n");
      return;
    }

    std::printf("[SKIP] %s (%s)", name, reason);
    if (!detail.empty())
      std::printf(": %s", detail.c_str());
    std::printf("\n");
  };

  auto reportOptionalSkip = [&](const char *name, const char *reason,
                                const std::string &detail) {
    std::printf("[SKIP] %s (%s)", name, reason);
    if (!detail.empty())
      std::printf(": %s", detail.c_str());
    std::printf("\n");
  };

  expectEq("TrimCopy", TrimCopy("  hi \n"), "hi");
  expectVecEq("Split", Split("a,b,,c", ','), {"a", "b", "", "c"});
  expectVecEq("SplitLines", SplitLines("a\r\nb\n\nc"), {"a", "b", "", "c"});
  expectEq("FirstNonEmptyLine", FirstNonEmptyLine("\n  \n x \n"), "x");

  {
    char arg0[] = "studiocast-probe";
    char arg1[] = "--self-test";
    char arg2[] = "--require-cuda-runtime";
    char *args[] = {arg0, arg1, arg2};
    expectTrue("hasArg --require-cuda-runtime",
               hasArg(3, args, "--require-cuda-runtime"));
    expectTrue("hasArg missing option", !hasArg(3, args, "--missing"));
    expectTrue("EnvFlagValueEnabled true",
               EnvFlagValueEnabled(" true "));
    expectTrue("EnvFlagValueEnabled one", EnvFlagValueEnabled("1"));
    expectTrue("EnvFlagValueEnabled false",
               !EnvFlagValueEnabled("false"));
    expectTrue("EnvFlagValueEnabled zero", !EnvFlagValueEnabled("0"));
  }

  // Capture error policy (pure logic; used by the camera pipeline).
  {
    using studiocast::video::IsRecoverableCaptureAcquireFailure;
    expectTrue("IsRecoverableCaptureAcquireFailure(timeout)",
               IsRecoverableCaptureAcquireFailure(
                   "Timed out waiting for camera frame."));
    expectTrue("IsRecoverableCaptureAcquireFailure(empty)",
               IsRecoverableCaptureAcquireFailure(""));
    expectTrue("IsRecoverableCaptureAcquireFailure(fatal) == false",
               !IsRecoverableCaptureAcquireFailure("poll failed: EIO"));
  }

  // Scaling policy guard (pure logic; used by the camera pipeline).
  {
    using studiocast::video::CheckOutputResizeAllowed;
    std::string err;

    err.clear();
    expectTrue("CheckOutputResizeAllowed(no resize)",
               CheckOutputResizeAllowed(640, 480, 640, 480,
                                        /*gpu_resize_available=*/false,
                                        /*allow_cpu_resize=*/false, &err));

    err.clear();
    expectTrue("CheckOutputResizeAllowed(gpu resize available)",
               CheckOutputResizeAllowed(640, 480, 1280, 720,
                                        /*gpu_resize_available=*/true,
                                        /*allow_cpu_resize=*/false, &err));

    err.clear();
    expectTrue("CheckOutputResizeAllowed(cpu resize allowed)",
               CheckOutputResizeAllowed(640, 480, 1280, 720,
                                        /*gpu_resize_available=*/false,
                                        /*allow_cpu_resize=*/true, &err));

    err.clear();
    expectTrue("CheckOutputResizeAllowed(cpu disabled) == false",
               !CheckOutputResizeAllowed(640, 480, 1280, 720,
                                         /*gpu_resize_available=*/false,
                                         /*allow_cpu_resize=*/false, &err));
    expectContains("CheckOutputResizeAllowed(cpu disabled message)", err,
                   "CPU resize is disabled");
  }

  // Standalone Open CUDA scaler gating (pure logic; used by the camera
  // pipeline).
  {
    using studiocast::video::ShouldRunStandaloneGpuScaler;
    using studiocast::video::ShouldRunStandaloneOpenCudaScaler;

    expectTrue("ShouldRunStandaloneGpuScaler(skip when cpu allowed + no effects)",
               !ShouldRunStandaloneGpuScaler(
                   /*scaling_needed=*/true,
                   /*gpu_backend_active=*/true,
                   /*have_deferred_gpu_out=*/false,
                   /*allow_cpu_resize=*/true,
                   /*same_backend_effects_ran=*/false));

    expectTrue("ShouldRunStandaloneGpuScaler(run when backend effect ran)",
               ShouldRunStandaloneGpuScaler(
                   /*scaling_needed=*/true,
                   /*gpu_backend_active=*/true,
                   /*have_deferred_gpu_out=*/false,
                   /*allow_cpu_resize=*/true,
                   /*same_backend_effects_ran=*/true));

    expectTrue("ShouldRunStandaloneOpenCudaScaler(no scaling) == false",
               !ShouldRunStandaloneOpenCudaScaler(
                   /*scaling_needed=*/false,
                   /*gpu_backend_is_open_cuda_or_maxine=*/true,
                   /*have_deferred_gpu_out=*/false,
                   /*allow_cpu_resize=*/true,
                   /*open_cuda_effects_ran=*/false));

    expectTrue("ShouldRunStandaloneOpenCudaScaler(no gpu backend) == false",
               !ShouldRunStandaloneOpenCudaScaler(
                   /*scaling_needed=*/true,
                   /*gpu_backend_is_open_cuda_or_maxine=*/false,
                   /*have_deferred_gpu_out=*/false,
                   /*allow_cpu_resize=*/true,
                   /*open_cuda_effects_ran=*/false));

    // Key requirement: if no deferred GPU output exists, CPU resize is allowed,
    // and no Open CUDA effects ran, do not run the standalone GPU scaler (avoid
    // unnecessary GPU transfers).
    expectTrue(
        "ShouldRunStandaloneOpenCudaScaler(skip when cpu allowed + no effects)",
        !ShouldRunStandaloneOpenCudaScaler(
            /*scaling_needed=*/true,
            /*gpu_backend_is_open_cuda_or_maxine=*/true,
            /*have_deferred_gpu_out=*/false,
            /*allow_cpu_resize=*/true,
            /*open_cuda_effects_ran=*/false));

    // If an Open CUDA effect ran, we allow the GPU scaler so resize can stay in
    // the GPU section.
    expectTrue("ShouldRunStandaloneOpenCudaScaler(run when effects ran)",
               ShouldRunStandaloneOpenCudaScaler(
                   /*scaling_needed=*/true,
                   /*gpu_backend_is_open_cuda_or_maxine=*/true,
                   /*have_deferred_gpu_out=*/false,
                   /*allow_cpu_resize=*/true,
                   /*open_cuda_effects_ran=*/true));

    // Strict policy fallback: if CPU resize is disallowed and scaling is
    // needed, allow GPU scaling as last resort.
    expectTrue(
        "ShouldRunStandaloneOpenCudaScaler(run when cpu resize disallowed)",
        ShouldRunStandaloneOpenCudaScaler(
            /*scaling_needed=*/true,
            /*gpu_backend_is_open_cuda_or_maxine=*/true,
            /*have_deferred_gpu_out=*/false,
            /*allow_cpu_resize=*/false,
            /*open_cuda_effects_ran=*/false));
  }

  // Backend resolver policy (pure logic).
  // Mirrors CameraPipeline behavior for maxine vs open_cuda selection.
  {
    using studiocast::video::effects::EffectsEnginePreference;

    auto resolveVb = [&](EffectsEnginePreference engine, bool maxine_runnable,
                         bool open_cuda_runnable) -> std::string {
      switch (engine) {
      case EffectsEnginePreference::maxine:
        return maxine_runnable ? "maxine" : "";
      case EffectsEnginePreference::open_cuda:
        return open_cuda_runnable ? "open_cuda" : "";
      case EffectsEnginePreference::auto_select:
      default:
        if (maxine_runnable)
          return "maxine";
        if (open_cuda_runnable)
          return "open_cuda";
        return "";
      }
    };

    expectEq("ResolveVB(auto_select prefers maxine)",
             resolveVb(EffectsEnginePreference::auto_select, /*maxine=*/true,
                       /*open_cuda=*/true),
             "maxine");
    expectEq("ResolveVB(auto_select falls back to open_cuda)",
             resolveVb(EffectsEnginePreference::auto_select, /*maxine=*/false,
                       /*open_cuda=*/true),
             "open_cuda");
    expectEq("ResolveVB(auto_select none)",
             resolveVb(EffectsEnginePreference::auto_select, /*maxine=*/false,
                       /*open_cuda=*/false),
             "");
    expectEq("ResolveVB(maxine strict; no fallback)",
             resolveVb(EffectsEnginePreference::maxine, /*maxine=*/false,
                       /*open_cuda=*/true),
             "");
    expectEq("ResolveVB(open_cuda strict; no fallback)",
             resolveVb(EffectsEnginePreference::open_cuda, /*maxine=*/true,
                       /*open_cuda=*/false),
             "");
  }

  // GPU resize backend selection policy (pure logic).
  // Mirrors the CameraPipeline scaling fallback order:
  //   Maxine/NVCV -> OpenCUDA (pure CUDA) -> CPU/none
  {
    using studiocast::video::ScalingBackendPreference;

    auto resolveScaling = [&](ScalingBackendPreference pref,
                              bool maxine_runnable,
                              bool open_cuda_runnable) -> std::string {
      if (pref == ScalingBackendPreference::cpu)
        return "cpu";
      if (maxine_runnable)
        return "gpu:maxine";
      if (open_cuda_runnable)
        return "gpu:open_cuda";
      return "cpu";
    };

    expectEq("ResolveScaling(auto prefers maxine)",
             resolveScaling(ScalingBackendPreference::auto_select,
                            /*maxine=*/true, /*open_cuda=*/true),
             "gpu:maxine");
    expectEq("ResolveScaling(auto falls back to open_cuda)",
             resolveScaling(ScalingBackendPreference::auto_select,
                            /*maxine=*/false, /*open_cuda=*/true),
             "gpu:open_cuda");
    expectEq("ResolveScaling(auto none)",
             resolveScaling(ScalingBackendPreference::auto_select,
                            /*maxine=*/false, /*open_cuda=*/false),
             "cpu");

    expectEq("ResolveScaling(gpu prefers maxine)",
             resolveScaling(ScalingBackendPreference::gpu, /*maxine=*/true,
                            /*open_cuda=*/true),
             "gpu:maxine");
    expectEq("ResolveScaling(gpu falls back to open_cuda)",
             resolveScaling(ScalingBackendPreference::gpu, /*maxine=*/false,
                            /*open_cuda=*/true),
             "gpu:open_cuda");
    expectEq("ResolveScaling(gpu none)",
             resolveScaling(ScalingBackendPreference::gpu, /*maxine=*/false,
                            /*open_cuda=*/false),
             "cpu");

    expectEq("ResolveScaling(cpu strict)",
             resolveScaling(ScalingBackendPreference::cpu, /*maxine=*/true,
                            /*open_cuda=*/true),
             "cpu");
  }

  // Safety-net policy: Open CUDA VB apply failures should not abort the camera
  // pipeline.
  {
    expectTrue("OpenCudaVbApplyFailureIsNonFatal",
               !studiocast::video::effects::
                   ShouldAbortPipelineOnOpenCudaVbApplyFailure());
  }

  // Open CUDA model pack registry (pure filesystem + JSON; no GPU/ORT
  // required).
  {
    const auto reg = studiocast::open_video::ModelPackRegistry::Scan(
        std::filesystem::path("tests") / "data" / "models" / "open_cuda");

    std::vector<std::string> ids;
    for (const auto &m : reg.ListModels()) {
      if (m.task != "matting")
        continue;
      ids.push_back(m.id);
    }
    expectVecEq("OpenCudaModelRegistry.ListModels", ids, {"mock_model"});

    const auto packOpt = reg.Find("matting", "mock_model");
    expectTrue("OpenCudaModelRegistry.ResolveModel(mock_model)",
               packOpt.has_value());
    if (packOpt) {
      expectEq("OpenCudaModelRegistry.mock_model.display_name",
               packOpt->display_name, "Mock Matting Model");
      expectEq("OpenCudaModelRegistry.mock_model.task", packOpt->task,
               "matting");
      expectTrue("OpenCudaModelRegistry.mock_model.matting",
                 packOpt->matting.has_value());
      if (packOpt->matting) {
        expectEq("OpenCudaModelRegistry.mock_model.input.layout",
                 packOpt->matting->input.layout, "nchw");
        expectIntEq("OpenCudaModelRegistry.mock_model.input.width",
                    packOpt->matting->input.width, 256);
        expectIntEq("OpenCudaModelRegistry.mock_model.input.height",
                    packOpt->matting->input.height, 256);
        expectIntEq("OpenCudaModelRegistry.mock_model.input.channels",
                    packOpt->matting->input.channels, 3);
        expectEq("OpenCudaModelRegistry.mock_model.output.kind",
                 packOpt->matting->output.kind, "alpha");
        expectEq("OpenCudaModelRegistry.mock_model.preprocess.color",
                 packOpt->matting->preprocess.color, "rgb");
        expectEq("OpenCudaModelRegistry.mock_model.preprocess.range",
                 packOpt->matting->preprocess.range, "0..1");
      }
    }

    const auto &problems = reg.Problems();
    {
      auto it = problems.find("missing_onnx");
      expectTrue("OpenCudaModelRegistry.Problems(missing_onnx)",
                 it != problems.end());
      if (it != problems.end()) {
        expectContains("OpenCudaModelRegistry.Problems(missing_onnx).reason",
                       it->second, "missing model file");
      }
    }
    {
      bool found = false;
      for (const auto &[k, v] : problems) {
        if (k.find("invalid_json") == std::string::npos)
          continue;
        found = true;
        expectContains("OpenCudaModelRegistry.Problems(invalid_json).reason", v,
                       "model.json");
        break;
      }
      expectTrue("OpenCudaModelRegistry.Problems(invalid_json)", found);
    }

    // Open CUDA diagnostics JSON includes model metadata for GUI dropdowns.
    {
      studiocast::open_cuda::OpenCudaDiagnostics od;
      od.ok = true;
      od.onnxruntime_version = "1.24.1-test";
      od.onnxruntime_providers = {"TensorrtExecutionProvider",
                                  "CUDAExecutionProvider",
                                  "CPUExecutionProvider"};
      od.onnxruntime_cuda_provider_present = true;
      od.onnxruntime_tensorrt_provider_present = true;
      od.onnxruntime_cpu_provider_present = true;
      od.onnxruntime_cuda_ep_v2_build = true;
      od.onnxruntime_library_path = "/opt/ort/lib/libonnxruntime.so";
      od.cuda_driver_api_available = true;
      od.cuda_context_available = true;
      od.cuda_device_count = 1;
      od.cuda_driver_version = 12040;
      od.default_model_id = reg.DefaultModelIdForTask("matting");
      for (const auto &m : reg.ListModels()) {
        if (m.task != "matting")
          continue;
        od.installed_models.push_back(m.id);
        studiocast::open_cuda::OpenCudaDiagnostics::ModelInfo mi;
        mi.id = m.id;
        mi.display_name = m.display_name;
        mi.task = m.task;
        if (m.matting) {
          mi.width = m.matting->input.width;
          mi.height = m.matting->input.height;
        }
        od.models.push_back(std::move(mi));
      }
      od.missing_models = reg.Problems();
      od.tensorrt_supported = true;
      od.tensorrt_available = true;
      od.tensorrt_requested = true;
      od.tensorrt_cache_path = "/tmp/studiocast/trt_cache/gpu0";
      od.tensorrt_status = "available";

      const std::string j = od.ToJson();
      expectContains("OpenCudaDiagnosticsJson.onnxruntime_version", j,
                     "\"onnxruntime_version\":\"1.24.1-test\"");
      expectContains("OpenCudaDiagnosticsJson.onnxruntime_providers", j,
                     "\"onnxruntime_providers\":[\"TensorrtExecutionProvider\","
                     "\"CUDAExecutionProvider\","
                     "\"CPUExecutionProvider\"]");
      expectContains("OpenCudaDiagnosticsJson.cuda_provider_present", j,
                     "\"onnxruntime_cuda_provider_present\":true");
      expectContains("OpenCudaDiagnosticsJson.tensorrt_provider_present", j,
                     "\"onnxruntime_tensorrt_provider_present\":true");
      expectContains("OpenCudaDiagnosticsJson.cpu_provider_present", j,
                     "\"onnxruntime_cpu_provider_present\":true");
      expectContains("OpenCudaDiagnosticsJson.cuda_ep_v2_build", j,
                     "\"onnxruntime_cuda_ep_v2_build\":true");
      expectContains("OpenCudaDiagnosticsJson.ort_library_path", j,
                     "\"onnxruntime_library_path\":\"/opt/ort/lib/"
                     "libonnxruntime.so\"");
      expectContains("OpenCudaDiagnosticsJson.cuda_driver_api_available", j,
                     "\"cuda_driver_api_available\":true");
      expectContains("OpenCudaDiagnosticsJson.cuda_context_available", j,
                     "\"cuda_context_available\":true");
      expectContains("OpenCudaDiagnosticsJson.cuda_device_count", j,
                     "\"cuda_device_count\":1");
      expectContains("OpenCudaDiagnosticsJson.cuda_driver_version", j,
                     "\"cuda_driver_version\":12040");
      expectContains("OpenCudaDiagnosticsJson.tensorrt_supported", j,
                     "\"tensorrt_supported\":true");
      expectContains("OpenCudaDiagnosticsJson.tensorrt_available", j,
                     "\"tensorrt_available\":true");
      expectContains("OpenCudaDiagnosticsJson.tensorrt_requested", j,
                     "\"tensorrt_requested\":true");
      expectContains("OpenCudaDiagnosticsJson.tensorrt_cache_path", j,
                     "\"tensorrt_cache_path\":\"/tmp/studiocast/trt_cache/"
                     "gpu0\"");
      expectContains("OpenCudaDiagnosticsJson.tensorrt_status", j,
                     "\"tensorrt_status\":\"available\"");
      expectContains("OpenCudaDiagnosticsJson.default_model_id", j,
                     "\"default_model_id\":\"mock_model\"");
      expectContains("OpenCudaDiagnosticsJson.models", j, "\"models\":[");
      expectContains("OpenCudaDiagnosticsJson.models.id", j,
                     "\"id\":\"mock_model\"");
      expectContains("OpenCudaDiagnosticsJson.models.display_name", j,
                     "\"display_name\":\"Mock Matting Model\"");
      expectContains("OpenCudaDiagnosticsJson.models.task", j,
                     "\"task\":\"matting\"");
      expectContains("OpenCudaDiagnosticsJson.models.width", j,
                     "\"width\":256");
      expectContains("OpenCudaDiagnosticsJson.models.height", j,
                     "\"height\":256");
      // Backward compatibility field.
      expectContains("OpenCudaDiagnosticsJson.installed_models", j,
                     "\"installed_models\":[");

      const auto trt_cache =
          studiocast::onnx::DefaultTensorRtCachePath(/*cuda_device_id=*/2);
      expectEq("DefaultTensorRtCachePath.filename",
               trt_cache.filename().string(), "gpu2");
      expectEq("DefaultTensorRtCachePath.parent",
               trt_cache.parent_path().filename().string(), "trt_cache");

      if (packOpt) {
        studiocast::open_cuda::OpenCudaMattingSession::Options trt_opts;
        trt_opts.device_id = 2;
        trt_opts.enable_tensorrt = true;
        studiocast::open_cuda::OpenCudaMattingSession sess(nullptr, *packOpt,
                                                           trt_opts);
        expectTrue("OpenCudaMattingSession options propagate TensorRT",
                   sess.options().enable_tensorrt);
        expectIntEq("OpenCudaMattingSession options propagate device_id",
                    sess.options().device_id, 2);
      }
    }

#if STUDIOCAST_ENABLE_OPEN_VULKAN
    {
      studiocast::vulkan::OpenVulkanDiagnostics vd;
      vd.compiled_enabled = true;
      vd.ok = true;
      vd.runtime_library_found = true;
      vd.runtime_library_path = "libvulkan.so.1";
      vd.instance_created = true;
      vd.physical_device_found = true;
      vd.compute_queue_available = true;
      vd.logical_device_created = true;
      vd.shader_pipeline_created = true;
      vd.vendor_id = 0x10de;
      vd.device_id = 123;
      vd.vendor_name = "NVIDIA";
      vd.device_name = "Test Vulkan Device";
      vd.compute_queue_family_index = 2;
      const std::string j = vd.ToJson();
      expectContains("OpenVulkanDiagnosticsJson.compiled_enabled", j,
                     "\"compiled_enabled\":true");
      expectContains("OpenVulkanDiagnosticsJson.runtime_library_found", j,
                     "\"runtime_library_found\":true");
      expectContains("OpenVulkanDiagnosticsJson.compute_queue_available", j,
                     "\"compute_queue_available\":true");
      expectContains("OpenVulkanDiagnosticsJson.shader_pipeline_created", j,
                     "\"shader_pipeline_created\":true");
      expectContains("OpenVulkanDiagnosticsJson.device_name", j,
                     "\"device_name\":\"Test Vulkan Device\"");
    }
#endif

    // Open Audio diagnostics JSON keeps legacy ORT fields and includes
    // provider/runtime details used by daemon/GUI/CLI diagnostics.
    {
      studiocast::open_audio::OpenAudioDiagnostics od;
      od.ok = true;
      od.onnxruntime_version = "1.20.0";
      od.onnxruntime_providers = {"CPUExecutionProvider"};
      od.onnxruntime_cuda_provider_present = false;
      od.onnxruntime_tensorrt_provider_present = false;
      od.onnxruntime_cpu_provider_present = true;
      od.onnxruntime_cuda_ep_v2_build = false;
      od.onnxruntime_library_path = "/opt/ort/lib/libonnxruntime.so";
      od.acceleration_likely = "cpu_fallback";
      od.installed_models = {"fastenhancer_s_vd_v1"};
      od.default_model_id = "fastenhancer_s_vd_v1";

      const std::string j = od.ToJson();
      expectContains("OpenAudioDiagnosticsJson.onnxruntime_version", j,
                     "\"onnxruntime_version\":\"1.20.0\"");
      expectContains("OpenAudioDiagnosticsJson.onnxruntime_providers", j,
                     "\"onnxruntime_providers\":[\"CPUExecutionProvider\"]");
      expectContains("OpenAudioDiagnosticsJson.cuda_provider_present", j,
                     "\"onnxruntime_cuda_provider_present\":false");
      expectContains("OpenAudioDiagnosticsJson.tensorrt_provider_present", j,
                     "\"onnxruntime_tensorrt_provider_present\":false");
      expectContains("OpenAudioDiagnosticsJson.cpu_provider_present", j,
                     "\"onnxruntime_cpu_provider_present\":true");
      expectContains("OpenAudioDiagnosticsJson.cuda_ep_v2_build", j,
                     "\"onnxruntime_cuda_ep_v2_build\":false");
      expectContains("OpenAudioDiagnosticsJson.ort_library_path", j,
                     "\"onnxruntime_library_path\":\"/opt/ort/lib/"
                     "libonnxruntime.so\"");
      expectContains("OpenAudioDiagnosticsJson.acceleration_likely", j,
                     "\"acceleration_likely\":\"cpu_fallback\"");
      expectContains("OpenAudioDiagnosticsJson.installed_models", j,
                     "\"installed_models\":[\"fastenhancer_s_vd_v1\"]");
    }
  }

  // Image loader: PNG support for virtual background replace images.
  {
    const std::filesystem::path tmp = std::filesystem::temp_directory_path() /
                                      "studiocast_selftest_vb_replace.png";

    const std::vector<std::uint8_t> src_rgb = {
        // Row 0: red, green
        255,
        0,
        0,
        0,
        255,
        0,
        // Row 1: blue, white
        0,
        0,
        255,
        255,
        255,
        255,
    };

    std::string werr;
    expectTrue("WritePngRgb24File",
               WritePngRgb24File(tmp, /*w=*/2, /*h=*/2, src_rgb.data(), &werr));
    if (!werr.empty()) {
      // Keep visibility for flaky FS issues.
      std::printf("[INFO] WritePngRgb24File err: %s\n", werr.c_str());
    }

    int w = 0, h = 0;
    std::vector<std::uint8_t> out;
    std::string err;
    expectTrue("LoadImageRgb24(png)",
               studiocast::video::LoadImageRgb24(tmp, &w, &h, &out, &err));
    if (!err.empty()) {
      std::printf("[INFO] LoadImageRgb24(png) err: %s\n", err.c_str());
    }
    expectIntEq("LoadImageRgb24(png) width", w, 2);
    expectIntEq("LoadImageRgb24(png) height", h, 2);
    expectIntEq("LoadImageRgb24(png) size", static_cast<int>(out.size()),
                2 * 2 * 3);

    auto at = [&](int x, int y, int c) -> int {
      return static_cast<int>(
          out[static_cast<std::size_t>((y * w + x) * 3 + c)]);
    };
    expectIntEq("LoadImageRgb24(png) (0,0) R", at(0, 0, 0), 255);
    expectIntEq("LoadImageRgb24(png) (0,0) G", at(0, 0, 1), 0);
    expectIntEq("LoadImageRgb24(png) (0,0) B", at(0, 0, 2), 0);
    expectIntEq("LoadImageRgb24(png) (1,0) R", at(1, 0, 0), 0);
    expectIntEq("LoadImageRgb24(png) (1,0) G", at(1, 0, 1), 255);
    expectIntEq("LoadImageRgb24(png) (1,0) B", at(1, 0, 2), 0);
    expectIntEq("LoadImageRgb24(png) (0,1) R", at(0, 1, 0), 0);
    expectIntEq("LoadImageRgb24(png) (0,1) G", at(0, 1, 1), 0);
    expectIntEq("LoadImageRgb24(png) (0,1) B", at(0, 1, 2), 255);
    expectIntEq("LoadImageRgb24(png) (1,1) R", at(1, 1, 0), 255);
    expectIntEq("LoadImageRgb24(png) (1,1) G", at(1, 1, 1), 255);
    expectIntEq("LoadImageRgb24(png) (1,1) B", at(1, 1, 2), 255);

    std::error_code ec;
    (void)std::filesystem::remove(tmp, ec);
  }

  // GPU buffer roundtrip (CUDA driver only; no Maxine/NvCV dependencies).
  {
    studiocast::maxine::CudaDriverApi cuda;
    std::string err;
    if (!cuda.Initialize(&err)) {
      reportCudaSmokeSkip("CudaImageRoundtrip", "CUDA unavailable", err);
    } else if (!cuda.EnsureContext(&err)) {
      reportCudaSmokeSkip("CudaImageRoundtrip", "no CUDA context/device",
                          err);
    } else {
      studiocast::maxine::CUstream stream = nullptr;
      if (!cuda.CreateStream(&stream, &err)) {
        ++failures;
        std::printf("[FAIL] CudaImageRoundtrip\n  CreateStream failed: %s\n",
                    err.c_str());
      } else {
        studiocast::cuda::CudaImage img;
        if (!img.Allocate(&cuda, 37, 19,
                          studiocast::cuda::PixelFormatGpu::rgb_u8, &err)) {
          ++failures;
          std::printf("[FAIL] CudaImageRoundtrip\n  Allocate failed: %s\n",
                      err.c_str());
        } else {
          const std::size_t stride = static_cast<std::size_t>(img.w) * 3u;
          std::vector<std::uint8_t> src(stride *
                                        static_cast<std::size_t>(img.h));
          std::vector<std::uint8_t> dst(
              stride * static_cast<std::size_t>(img.h), 0xCD);

          for (int y = 0; y < img.h; ++y) {
            for (int x = 0; x < img.w; ++x) {
              const std::size_t i = static_cast<std::size_t>(y) * stride +
                                    static_cast<std::size_t>(x) * 3u;
              src[i + 0] = static_cast<std::uint8_t>((x * 3 + y * 7) & 0xFF);
              src[i + 1] = static_cast<std::uint8_t>((x * 5 + y * 11) & 0xFF);
              src[i + 2] = static_cast<std::uint8_t>((x * 13 + y * 17) & 0xFF);
            }
          }

          if (!img.UploadFromCpuRgb24(&cuda, src.data(), stride, stream,
                                      &err)) {
            ++failures;
            std::printf("[FAIL] CudaImageRoundtrip\n  Upload failed: %s\n",
                        err.c_str());
          } else if (!img.DownloadToCpuRgb24(&cuda, dst.data(), stride, stream,
                                             &err)) {
            ++failures;
            std::printf("[FAIL] CudaImageRoundtrip\n  Download failed: %s\n",
                        err.c_str());
          } else if (!cuda.StreamSynchronize(stream, &err)) {
            ++failures;
            std::printf(
                "[FAIL] CudaImageRoundtrip\n  StreamSynchronize failed: %s\n",
                err.c_str());
          } else if (src != dst) {
            ++failures;
            std::size_t mismatch_i = 0;
            for (; mismatch_i < src.size(); ++mismatch_i) {
              if (src[mismatch_i] != dst[mismatch_i])
                break;
            }
            std::printf("[FAIL] CudaImageRoundtrip\n  Byte mismatch at index "
                        "%zu: got=%u want=%u\n",
                        mismatch_i, static_cast<unsigned int>(dst[mismatch_i]),
                        static_cast<unsigned int>(src[mismatch_i]));
          }
        }

        (void)img.Free(&cuda, nullptr);
        (void)cuda.DestroyStream(stream, nullptr);
      }
    }
  }

  // Open CUDA VB kernels smoke test (CUDA driver only; no ORT required).
  {
    studiocast::maxine::CudaDriverApi cuda;
    std::string err;
    if (!cuda.Initialize(&err)) {
      reportCudaSmokeSkip("OpenCudaVbKernelsSmoke", "CUDA unavailable", err);
    } else if (!cuda.EnsureContext(&err)) {
      reportCudaSmokeSkip("OpenCudaVbKernelsSmoke", "no CUDA context/device",
                          err);
    } else {
      studiocast::maxine::CUstream stream = nullptr;
      if (!cuda.CreateStream(&stream, &err)) {
        ++failures;
        std::printf(
            "[FAIL] OpenCudaVbKernelsSmoke\n  CreateStream failed: %s\n",
            err.c_str());
      } else {
        const int w = 13;
        const int h = 7;
        const std::size_t stride = static_cast<std::size_t>(w) * 3u;

        std::vector<std::uint8_t> fg_cpu(stride * static_cast<std::size_t>(h));
        std::vector<std::uint8_t> bg_cpu(stride * static_cast<std::size_t>(h));
        std::vector<std::uint8_t> out_cpu(stride * static_cast<std::size_t>(h),
                                          0xCC);

        for (int y = 0; y < h; ++y) {
          for (int x = 0; x < w; ++x) {
            const std::size_t i = static_cast<std::size_t>(y) * stride +
                                  static_cast<std::size_t>(x) * 3u;
            fg_cpu[i + 0] = static_cast<std::uint8_t>((x * 9 + y * 3) & 0xFF);
            fg_cpu[i + 1] = static_cast<std::uint8_t>((x * 7 + y * 11) & 0xFF);
            fg_cpu[i + 2] = static_cast<std::uint8_t>((x * 5 + y * 13) & 0xFF);

            bg_cpu[i + 0] =
                static_cast<std::uint8_t>((200 + x * 2 + y * 1) & 0xFF);
            bg_cpu[i + 1] =
                static_cast<std::uint8_t>((100 + x * 3 + y * 4) & 0xFF);
            bg_cpu[i + 2] =
                static_cast<std::uint8_t>((50 + x * 5 + y * 6) & 0xFF);
          }
        }

        // Alpha: left half 0, right half 1.
        std::vector<float> alpha_cpu(static_cast<std::size_t>(w) *
                                     static_cast<std::size_t>(h));
        for (int y = 0; y < h; ++y) {
          for (int x = 0; x < w; ++x) {
            alpha_cpu[static_cast<std::size_t>(y) *
                          static_cast<std::size_t>(w) +
                      static_cast<std::size_t>(x)] =
                (x < (w / 2)) ? 0.0f : 1.0f;
          }
        }

        studiocast::cuda::CudaImage fg;
        studiocast::cuda::CudaImage bg;
        studiocast::cuda::CudaImage tmp;
        studiocast::cuda::CudaImage blurred;
        studiocast::cuda::CudaImage out;
        studiocast::cuda::CudaTensor alpha_tensor;

        bool ok = true;
        if (!fg.Allocate(&cuda, w, h, studiocast::cuda::PixelFormatGpu::rgb_u8,
                         &err) ||
            !bg.Allocate(&cuda, w, h, studiocast::cuda::PixelFormatGpu::rgb_u8,
                         &err) ||
            !tmp.Allocate(&cuda, w, h, studiocast::cuda::PixelFormatGpu::rgb_u8,
                          &err) ||
            !blurred.Allocate(&cuda, w, h,
                              studiocast::cuda::PixelFormatGpu::rgb_u8, &err) ||
            !out.Allocate(&cuda, w, h, studiocast::cuda::PixelFormatGpu::rgb_u8,
                          &err) ||
            !alpha_tensor.AllocateNchwF32(&cuda, 1, 1, h, w, &err)) {
          ++failures;
          std::printf("[FAIL] OpenCudaVbKernelsSmoke\n  Allocate failed: %s\n",
                      err.c_str());
          ok = false;
        }

        studiocast::cuda::CudaImage alpha_img;
        if (ok) {
          // Upload buffers.
          if (!fg.UploadFromCpuRgb24(&cuda, fg_cpu.data(), stride, stream,
                                     &err) ||
              !bg.UploadFromCpuRgb24(&cuda, bg_cpu.data(), stride, stream,
                                     &err)) {
            ++failures;
            std::printf("[FAIL] OpenCudaVbKernelsSmoke\n  Upload failed: %s\n",
                        err.c_str());
            ok = false;
          }
        }
        if (ok) {
          const std::size_t alpha_bytes = alpha_cpu.size() * sizeof(float);
          if (!cuda.MemcpyHtoD2DAsync(alpha_tensor.ptr, alpha_tensor.pitch,
                                      alpha_cpu.data(), alpha_bytes,
                                      alpha_bytes, 1, stream, &err)) {
            ++failures;
            std::printf(
                "[FAIL] OpenCudaVbKernelsSmoke\n  Alpha upload failed: %s\n",
                err.c_str());
            ok = false;
          }

          alpha_img.ptr = alpha_tensor.ptr;
          alpha_img.pitch = static_cast<std::size_t>(w) * 4u;
          alpha_img.w = w;
          alpha_img.h = h;
          alpha_img.format = studiocast::cuda::PixelFormatGpu::f32_1;
          alpha_img.owns_memory = false;
        }

        // Blur radius=0 should be identity.
        if (ok) {
          std::string kerr;
          if (!studiocast::cuda::kernels::BoxBlurSeparableU8x3(
                  fg, tmp, blurred, /*radius=*/0, stream, &kerr)) {
            ++failures;
            std::printf("[FAIL] OpenCudaVbKernelsSmoke\n  BoxBlurSeparableU8x3 "
                        "failed: %s\n",
                        kerr.c_str());
            ok = false;
          } else if (!blurred.DownloadToCpuRgb24(&cuda, out_cpu.data(), stride,
                                                 stream, &err) ||
                     !cuda.StreamSynchronize(stream, &err)) {
            ++failures;
            std::printf("[FAIL] OpenCudaVbKernelsSmoke\n  Blur download/sync "
                        "failed: %s\n",
                        err.c_str());
            ok = false;
          } else if (out_cpu != fg_cpu) {
            ++failures;
            std::printf(
                "[FAIL] OpenCudaVbKernelsSmoke\n  BoxBlur radius=0 mismatch\n");
            ok = false;
          }
        }

        // Composite alpha against bg image / solid.
        // Validate hard 0/1 cases first (should be exact), then validate the
        // half-mask.
        if (ok) {
          const std::size_t alpha_bytes = alpha_cpu.size() * sizeof(float);
          auto upload_alpha = [&](const std::vector<float> &a) -> bool {
            if (!cuda.MemcpyHtoD2DAsync(alpha_tensor.ptr, alpha_tensor.pitch,
                                        a.data(), alpha_bytes, alpha_bytes, 1,
                                        stream, &err)) {
              return false;
            }
            return true;
          };

          auto first_mismatch_rgb = [&](const std::vector<std::uint8_t> &got,
                                        const std::vector<std::uint8_t> &want,
                                        int *out_x, int *out_y) -> bool {
            for (int y = 0; y < h; ++y) {
              for (int x = 0; x < w; ++x) {
                const std::size_t i = static_cast<std::size_t>(y) * stride +
                                      static_cast<std::size_t>(x) * 3u;
                if (got[i + 0] != want[i + 0] || got[i + 1] != want[i + 1] ||
                    got[i + 2] != want[i + 2]) {
                  *out_x = x;
                  *out_y = y;
                  return true;
                }
              }
            }
            return false;
          };

          std::vector<float> alpha0(alpha_cpu.size(), 0.0f);
          std::vector<float> alpha1(alpha_cpu.size(), 1.0f);

          auto run_comp_bg_expect =
              [&](const std::vector<float> &a,
                  const std::vector<std::uint8_t> &want) -> bool {
            std::string kerr;
            if (!upload_alpha(a)) {
              std::printf(
                  "[FAIL] OpenCudaVbKernelsSmoke\n  Alpha upload failed: %s\n",
                  err.c_str());
              return false;
            }
            if (!studiocast::cuda::kernels::CompositeAlphaU8x3(
                    fg, bg, alpha_img, out, stream, &kerr)) {
              std::printf("[FAIL] OpenCudaVbKernelsSmoke\n  CompositeAlphaU8x3 "
                          "failed: %s\n",
                          kerr.c_str());
              return false;
            }
            if (!out.DownloadToCpuRgb24(&cuda, out_cpu.data(), stride, stream,
                                        &err) ||
                !cuda.StreamSynchronize(stream, &err)) {
              std::printf("[FAIL] OpenCudaVbKernelsSmoke\n  Composite "
                          "download/sync failed: %s\n",
                          err.c_str());
              return false;
            }
            int mx = 0, my = 0;
            if (first_mismatch_rgb(out_cpu, want, &mx, &my)) {
              const std::size_t i = static_cast<std::size_t>(my) * stride +
                                    static_cast<std::size_t>(mx) * 3u;
              std::printf(
                  "[FAIL] OpenCudaVbKernelsSmoke\n  Composite result mismatch "
                  "at (%d,%d): got=(%u,%u,%u) want=(%u,%u,%u)\n",
                  mx, my, static_cast<unsigned>(out_cpu[i + 0]),
                  static_cast<unsigned>(out_cpu[i + 1]),
                  static_cast<unsigned>(out_cpu[i + 2]),
                  static_cast<unsigned>(want[i + 0]),
                  static_cast<unsigned>(want[i + 1]),
                  static_cast<unsigned>(want[i + 2]));
              return false;
            }
            return true;
          };

          // Expect exact bg for alpha=0 and exact fg for alpha=1.
          if (!run_comp_bg_expect(alpha0, bg_cpu)) {
            ++failures;
            ok = false;
          } else if (!run_comp_bg_expect(alpha1, fg_cpu)) {
            ++failures;
            ok = false;
          } else {
            // Restore half-mask alpha and validate mixed output.
            if (!upload_alpha(alpha_cpu)) {
              ++failures;
              std::printf(
                  "[FAIL] OpenCudaVbKernelsSmoke\n  Alpha upload failed: %s\n",
                  err.c_str());
              ok = false;
            } else {
              std::vector<std::uint8_t> want(stride *
                                             static_cast<std::size_t>(h));
              for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                  const std::size_t i = static_cast<std::size_t>(y) * stride +
                                        static_cast<std::size_t>(x) * 3u;
                  const bool fg_side = (x >= (w / 2));
                  const auto *src = fg_side ? &fg_cpu[i] : &bg_cpu[i];
                  want[i + 0] = src[0];
                  want[i + 1] = src[1];
                  want[i + 2] = src[2];
                }
              }
              if (!run_comp_bg_expect(alpha_cpu, want)) {
                ++failures;
                ok = false;
              }
            }
          }

          // Solid background composite (exact cases).
          if (ok) {
            const std::uint8_t sr = 10, sg = 20, sb = 30;
            auto run_comp_solid_expect =
                [&](const std::vector<float> &a,
                    const std::vector<std::uint8_t> &want) -> bool {
              std::string kerr;
              if (!upload_alpha(a)) {
                std::printf("[FAIL] OpenCudaVbKernelsSmoke\n  Alpha upload "
                            "failed: %s\n",
                            err.c_str());
                return false;
              }
              if (!studiocast::cuda::kernels::CompositeAlphaSolidU8x3(
                      fg, alpha_img, sr, sg, sb, out, stream, &kerr)) {
                std::printf("[FAIL] OpenCudaVbKernelsSmoke\n  "
                            "CompositeAlphaSolidU8x3 failed: %s\n",
                            kerr.c_str());
                return false;
              }
              if (!out.DownloadToCpuRgb24(&cuda, out_cpu.data(), stride, stream,
                                          &err) ||
                  !cuda.StreamSynchronize(stream, &err)) {
                std::printf("[FAIL] OpenCudaVbKernelsSmoke\n  Solid composite "
                            "download/sync failed: %s\n",
                            err.c_str());
                return false;
              }
              int mx = 0, my = 0;
              if (first_mismatch_rgb(out_cpu, want, &mx, &my)) {
                const std::size_t i = static_cast<std::size_t>(my) * stride +
                                      static_cast<std::size_t>(mx) * 3u;
                std::printf(
                    "[FAIL] OpenCudaVbKernelsSmoke\n  Solid composite mismatch "
                    "at (%d,%d): got=(%u,%u,%u) want=(%u,%u,%u)\n",
                    mx, my, static_cast<unsigned>(out_cpu[i + 0]),
                    static_cast<unsigned>(out_cpu[i + 1]),
                    static_cast<unsigned>(out_cpu[i + 2]),
                    static_cast<unsigned>(want[i + 0]),
                    static_cast<unsigned>(want[i + 1]),
                    static_cast<unsigned>(want[i + 2]));
                return false;
              }
              return true;
            };

            std::vector<std::uint8_t> solid_cpu(stride *
                                                static_cast<std::size_t>(h));
            for (int y = 0; y < h; ++y) {
              for (int x = 0; x < w; ++x) {
                const std::size_t i = static_cast<std::size_t>(y) * stride +
                                      static_cast<std::size_t>(x) * 3u;
                solid_cpu[i + 0] = sr;
                solid_cpu[i + 1] = sg;
                solid_cpu[i + 2] = sb;
              }
            }

            if (!run_comp_solid_expect(alpha0, solid_cpu)) {
              ++failures;
              ok = false;
            } else if (!run_comp_solid_expect(alpha1, fg_cpu)) {
              ++failures;
              ok = false;
            }
          }
        }

        (void)fg.Free(&cuda, nullptr);
        (void)bg.Free(&cuda, nullptr);
        (void)tmp.Free(&cuda, nullptr);
        (void)blurred.Free(&cuda, nullptr);
        (void)out.Free(&cuda, nullptr);
        (void)alpha_tensor.Free(&cuda, nullptr);
        (void)cuda.DestroyStream(stream, nullptr);
      }
    }
  }

  // Open CUDA matting session smoke test (requires CUDA + ONNX Runtime + CUDA
  // EP).
  {
#if STUDIOCAST_HAVE_ONNXRUNTIME
    studiocast::maxine::CudaDriverApi cuda;
    std::string err;
    if (!cuda.Initialize(&err)) {
      reportCudaSmokeSkip("OpenCudaMattingSessionSmoke", "CUDA unavailable",
                          err);
    } else if (!cuda.EnsureContext(&err)) {
      reportCudaSmokeSkip("OpenCudaMattingSessionSmoke",
                          "no CUDA context/device", err);
    } else {
      studiocast::maxine::CUstream stream = nullptr;
      if (!cuda.CreateStream(&stream, &err)) {
        reportCudaSmokeSkip("OpenCudaMattingSessionSmoke",
                            "CreateStream failed", err);
      } else {
        const auto reg = studiocast::open_video::ModelPackRegistry::Scan(
            std::filesystem::path("tests") / "data" / "models" / "open_cuda");
        const auto packOpt = reg.Find("matting", "mock_model");
        if (!packOpt) {
          ++failures;
          std::printf("[FAIL] OpenCudaMattingSessionSmoke\n  missing "
                      "mock_model fixture\n");
        } else {
          studiocast::open_cuda::OpenCudaMattingSession sess(&cuda, *packOpt);

          studiocast::cuda::CudaImage frame;
          if (!frame.Allocate(&cuda, 320, 240,
                              studiocast::cuda::PixelFormatGpu::rgb_u8, &err)) {
            ++failures;
            std::printf("[FAIL] OpenCudaMattingSessionSmoke\n  frame.Allocate "
                        "failed: %s\n",
                        err.c_str());
          } else {
            // Upload a deterministic pattern.
            const std::size_t stride = static_cast<std::size_t>(frame.w) * 3u;
            std::vector<std::uint8_t> cpu(stride *
                                          static_cast<std::size_t>(frame.h));
            for (int y = 0; y < frame.h; ++y) {
              for (int x = 0; x < frame.w; ++x) {
                const std::size_t i = static_cast<std::size_t>(y) * stride +
                                      static_cast<std::size_t>(x) * 3u;
                cpu[i + 0] = static_cast<std::uint8_t>((x * 3 + y * 7) & 0xFF);
                cpu[i + 1] = static_cast<std::uint8_t>((x * 5 + y * 11) & 0xFF);
                cpu[i + 2] =
                    static_cast<std::uint8_t>((x * 13 + y * 17) & 0xFF);
              }
            }
            if (!frame.UploadFromCpuRgb24(&cuda, cpu.data(), stride, stream,
                                          &err)) {
              ++failures;
              std::printf("[FAIL] OpenCudaMattingSessionSmoke\n  "
                          "UploadFromCpuRgb24 failed: %s\n",
                          err.c_str());
            } else {
              studiocast::cuda::CudaTensor alpha;
              if (!alpha.AllocateNchwF32(&cuda, 1, 1,
                                         packOpt->matting->input.height,
                                         packOpt->matting->input.width, &err)) {
                ++failures;
                std::printf("[FAIL] OpenCudaMattingSessionSmoke\n  "
                            "alpha.AllocateNchwF32 failed: %s\n",
                            err.c_str());
              } else {
                // Run multiple iterations to exercise re-use of
                // buffers/session.
                bool ok = true;
                for (int i = 0; i < 5; ++i) {
                  std::string run_err;
                  if (!sess.Run(stream, frame, &alpha, &run_err)) {
                    // If CUDA EP is missing (CPU-only ORT), treat as SKIP so
                    // self-test stays portable.
                    if (run_err.find("CUDA EP") != std::string::npos ||
                        run_err.find("CUDA") != std::string::npos) {
                      reportCudaSmokeSkip("OpenCudaMattingSessionSmoke",
                                          "ORT CUDA EP unavailable", run_err);
                      ok = false;
                      break;
                    }
                    ++failures;
                    std::printf("[FAIL] OpenCudaMattingSessionSmoke\n  Run "
                                "failed: %s\n",
                                run_err.c_str());
                    ok = false;
                    break;
                  }
                }

                if (ok) {
                  std::vector<float> out;
                  if (!alpha.DownloadToCpuF32(&cuda, &out, stream, &err)) {
                    ++failures;
                    std::printf("[FAIL] OpenCudaMattingSessionSmoke\n  "
                                "DownloadToCpuF32 failed: %s\n",
                                err.c_str());
                  } else {
                    expectIntEq("OpenCudaMattingSessionSmoke.alpha.size",
                                static_cast<int>(out.size()),
                                packOpt->matting->input.width *
                                    packOpt->matting->input.height);

                    bool all_finite = true;
                    for (float v : out) {
                      if (!std::isfinite(v)) {
                        all_finite = false;
                        break;
                      }
                    }
                    expectTrue("OpenCudaMattingSessionSmoke.alpha.all_finite",
                               all_finite);
                  }
                }

                (void)alpha.Free(&cuda, &err);
              }
            }
            (void)frame.Free(&cuda, &err);
          }
        }

        (void)cuda.DestroyStream(stream, &err);
      }
    }
#else
    reportCudaSmokeSkip("OpenCudaMattingSessionSmoke",
                        "built without ONNX Runtime", std::string());
#endif
  }

#if STUDIOCAST_ENABLE_OPEN_VULKAN
  // Open Vulkan resize runtime (optional): skips when no loader/device is
  // available, but validates upload/compute/download against the CPU resize
  // contract when the runtime exists.
  {
    auto runVulkanResizeCheck = [&](int src_w, int src_h, int dst_w,
                                    int dst_h) -> bool {
      const std::size_t src_stride = static_cast<std::size_t>(src_w) * 3u;
      const std::size_t dst_stride = static_cast<std::size_t>(dst_w) * 3u;
      std::vector<std::uint8_t> src_rgb(src_stride *
                                        static_cast<std::size_t>(src_h));
      for (int y = 0; y < src_h; ++y) {
        for (int x = 0; x < src_w; ++x) {
          const std::size_t i = static_cast<std::size_t>(y) * src_stride +
                                static_cast<std::size_t>(x) * 3u;
          src_rgb[i + 0] = static_cast<std::uint8_t>((x * 17 + y * 29) & 0xff);
          src_rgb[i + 1] = static_cast<std::uint8_t>((x * 31 + y * 7) & 0xff);
          src_rgb[i + 2] = static_cast<std::uint8_t>((x * 5 + y * 43) & 0xff);
        }
      }

      std::vector<std::uint8_t> cpu_resized;
      std::string err;
      if (!studiocast::video::ResizeRgb24Bilinear(
              src_rgb.data(), src_w, src_h, src_stride, dst_w, dst_h,
              &cpu_resized, dst_stride, &err)) {
        ++failures;
        std::printf("[FAIL] OpenVulkanResize\n  CPU reference failed: %s\n",
                    err.c_str());
        return false;
      }

      studiocast::vulkan::kernels::ResizeBilinear vk_resize;
      if (!vk_resize.EnsureInitialized(src_w, src_h, dst_w, dst_h, &err)) {
        reportOptionalSkip("OpenVulkanResize", "Vulkan unavailable", err);
        return false;
      }

      std::vector<std::uint8_t> vk_resized(
          dst_stride * static_cast<std::size_t>(dst_h), 0xcd);
      if (!vk_resize.Resize(src_rgb.data(), src_stride, vk_resized.data(),
                            dst_stride, &err)) {
        ++failures;
        std::printf("[FAIL] OpenVulkanResize\n  Resize failed: %s\n",
                    err.c_str());
        return false;
      }

      int max_abs_diff = 0;
      for (std::size_t i = 0; i < cpu_resized.size() && i < vk_resized.size();
           ++i) {
        const int d = static_cast<int>(cpu_resized[i]) -
                      static_cast<int>(vk_resized[i]);
        max_abs_diff = std::max(max_abs_diff, d < 0 ? -d : d);
      }
      if (max_abs_diff > 1) {
        ++failures;
        std::printf("[FAIL] OpenVulkanResize\n  max_abs_diff=%d "
                    "(want <= 1)\n",
                    max_abs_diff);
        return false;
      }
      return true;
    };

    if (runVulkanResizeCheck(4, 4, 8, 8))
      (void)runVulkanResizeCheck(8, 8, 4, 4);
  }
#else
  reportOptionalSkip("OpenVulkanResize", "backend disabled in build",
                     std::string());
#endif

  // CUDA kernels (optional build): bilinear resize + preprocess-to-NCHW.
  {
    studiocast::maxine::CudaDriverApi cuda;
    std::string err;
    if (!cuda.Initialize(&err)) {
      reportCudaSmokeSkip("CudaKernels", "CUDA unavailable", err);
    } else if (!cuda.EnsureContext(&err)) {
      reportCudaSmokeSkip("CudaKernels", "no CUDA context/device", err);
    } else {
      studiocast::maxine::CUstream stream = nullptr;
      if (!cuda.CreateStream(&stream, &err)) {
        ++failures;
        std::printf("[FAIL] CudaKernels\n  CreateStream failed: %s\n",
                    err.c_str());
      } else {
        auto fail = [&](const char *name, const std::string &msg) {
          ++failures;
          std::printf("[FAIL] %s\n  %s\n", name, msg.c_str());
        };

        // ---------------------------
        // ResizeBilinear vs CPU reference
        // ---------------------------
        const int src_w = 13;
        const int src_h = 7;
        const int dst_w = 19;
        const int dst_h = 11;
        const std::size_t src_stride = static_cast<std::size_t>(src_w) * 3u;
        const std::size_t dst_stride = static_cast<std::size_t>(dst_w) * 3u;

        std::vector<std::uint8_t> src_rgb(src_stride *
                                          static_cast<std::size_t>(src_h));
        for (int y = 0; y < src_h; ++y) {
          for (int x = 0; x < src_w; ++x) {
            const std::size_t i = static_cast<std::size_t>(y) * src_stride +
                                  static_cast<std::size_t>(x) * 3u;
            src_rgb[i + 0] = static_cast<std::uint8_t>((x * 3 + y * 7) & 0xFF);
            src_rgb[i + 1] = static_cast<std::uint8_t>((x * 5 + y * 11) & 0xFF);
            src_rgb[i + 2] =
                static_cast<std::uint8_t>((x * 13 + y * 17) & 0xFF);
          }
        }

        std::vector<std::uint8_t> cpu_resized;
        std::string resizeErr;
        if (!studiocast::video::ResizeRgb24Bilinear(
                src_rgb.data(), src_w, src_h, src_stride, dst_w, dst_h,
                &cpu_resized, dst_stride, &resizeErr)) {
          fail("CudaResizeBilinear",
               "CPU reference resize failed: " + resizeErr);
        } else {
          studiocast::cuda::CudaImage gpu_src;
          studiocast::cuda::CudaImage gpu_dst;
          if (!gpu_src.Allocate(&cuda, src_w, src_h,
                                studiocast::cuda::PixelFormatGpu::rgb_u8,
                                &err)) {
            fail("CudaResizeBilinear", "Allocate src failed: " + err);
          } else if (!gpu_dst.Allocate(&cuda, dst_w, dst_h,
                                       studiocast::cuda::PixelFormatGpu::rgb_u8,
                                       &err)) {
            fail("CudaResizeBilinear", "Allocate dst failed: " + err);
          } else if (!gpu_src.UploadFromCpuRgb24(&cuda, src_rgb.data(),
                                                 src_stride, stream, &err)) {
            fail("CudaResizeBilinear", "Upload failed: " + err);
          } else if (!studiocast::cuda::kernels::ResizeBilinear(
                         gpu_src, gpu_dst, stream, &err)) {
            fail("CudaResizeBilinear", "Kernel failed: " + err);
          } else {
            std::vector<std::uint8_t> gpu_resized(
                dst_stride * static_cast<std::size_t>(dst_h), 0xCD);
            if (!gpu_dst.DownloadToCpuRgb24(&cuda, gpu_resized.data(),
                                            dst_stride, stream, &err)) {
              fail("CudaResizeBilinear", "Download failed: " + err);
            } else if (!cuda.StreamSynchronize(stream, &err)) {
              fail("CudaResizeBilinear", "StreamSynchronize failed: " + err);
            } else {
              int max_abs_diff = 0;
              for (std::size_t i = 0;
                   i < cpu_resized.size() && i < gpu_resized.size(); ++i) {
                const int d = static_cast<int>(cpu_resized[i]) -
                              static_cast<int>(gpu_resized[i]);
                const int ad = (d < 0) ? -d : d;
                if (ad > max_abs_diff)
                  max_abs_diff = ad;
              }
              if (max_abs_diff > 1) {
                ++failures;
                std::printf("[FAIL] CudaResizeBilinear\n  max_abs_diff=%d "
                            "(want <= 1)\n",
                            max_abs_diff);
              }
            }
          }

          if (failures == 0) {
            constexpr float crop_x = 2.25f;
            constexpr float crop_y = 1.50f;
            constexpr float crop_w = 7.50f;
            constexpr float crop_h = 4.00f;
            std::vector<std::uint8_t> cpu_crop(
                dst_stride * static_cast<std::size_t>(dst_h), 0);
            for (int y = 0; y < dst_h; ++y) {
              const float src_y = crop_y +
                                  (static_cast<float>(y) + 0.5f) *
                                      (crop_h / static_cast<float>(dst_h)) -
                                  0.5f;
              const float sy =
                  std::clamp(src_y, 0.0f, static_cast<float>(src_h - 1));
              const int y0 = static_cast<int>(sy);
              const int y1 = std::min(y0 + 1, src_h - 1);
              const float ty = sy - static_cast<float>(y0);
              for (int x = 0; x < dst_w; ++x) {
                const float src_x = crop_x +
                                    (static_cast<float>(x) + 0.5f) *
                                        (crop_w / static_cast<float>(dst_w)) -
                                    0.5f;
                const float sx =
                    std::clamp(src_x, 0.0f, static_cast<float>(src_w - 1));
                const int x0 = static_cast<int>(sx);
                const int x1 = std::min(x0 + 1, src_w - 1);
                const float tx = sx - static_cast<float>(x0);
                for (int c = 0; c < 3; ++c) {
                  const auto ch = static_cast<std::size_t>(c);
                  const float p00 =
                      src_rgb[static_cast<std::size_t>(y0) * src_stride +
                              static_cast<std::size_t>(x0) * 3u + ch];
                  const float p10 =
                      src_rgb[static_cast<std::size_t>(y0) * src_stride +
                              static_cast<std::size_t>(x1) * 3u + ch];
                  const float p01 =
                      src_rgb[static_cast<std::size_t>(y1) * src_stride +
                              static_cast<std::size_t>(x0) * 3u + ch];
                  const float p11 =
                      src_rgb[static_cast<std::size_t>(y1) * src_stride +
                              static_cast<std::size_t>(x1) * 3u + ch];
                  const float v0 = p00 + (p10 - p00) * tx;
                  const float v1 = p01 + (p11 - p01) * tx;
                  const int iv =
                      static_cast<int>(std::lround(v0 + (v1 - v0) * ty));
                  cpu_crop[static_cast<std::size_t>(y) * dst_stride +
                           static_cast<std::size_t>(x) * 3u + ch] =
                      static_cast<std::uint8_t>(std::clamp(iv, 0, 255));
                }
              }
            }

            if (!studiocast::cuda::kernels::CropResizeBilinear(
                    gpu_src, gpu_dst, crop_x, crop_y, crop_w, crop_h, stream,
                    &err)) {
              fail("CudaCropResizeBilinear", "Kernel failed: " + err);
            } else {
              std::vector<std::uint8_t> gpu_crop(
                  dst_stride * static_cast<std::size_t>(dst_h), 0xCD);
              if (!gpu_dst.DownloadToCpuRgb24(&cuda, gpu_crop.data(),
                                              dst_stride, stream, &err)) {
                fail("CudaCropResizeBilinear", "Download failed: " + err);
              } else if (!cuda.StreamSynchronize(stream, &err)) {
                fail("CudaCropResizeBilinear",
                     "StreamSynchronize failed: " + err);
              } else {
                int max_abs_diff = 0;
                for (std::size_t i = 0;
                     i < cpu_crop.size() && i < gpu_crop.size(); ++i) {
                  const int d = static_cast<int>(cpu_crop[i]) -
                                static_cast<int>(gpu_crop[i]);
                  max_abs_diff = std::max(max_abs_diff, std::abs(d));
                }
                if (max_abs_diff > 1) {
                  ++failures;
                  std::printf("[FAIL] CudaCropResizeBilinear\n  "
                              "max_abs_diff=%d (want <= 1)\n",
                              max_abs_diff);
                }
              }
            }
          }

          (void)gpu_src.Free(&cuda, nullptr);
          (void)gpu_dst.Free(&cuda, nullptr);
        }

        // ---------------------------
        // PreprocessToTensor vs CPU reference
        // ---------------------------
        {
          studiocast::cuda::kernels::ModelPreprocessSpec spec;
          spec.dst_w = 9;
          spec.dst_h = 5;
          spec.dst_order = studiocast::cuda::kernels::ChannelOrder::rgb;
          spec.mean[0] = 0.5f;
          spec.mean[1] = 0.25f;
          spec.mean[2] = 0.75f;
          spec.std[0] = 0.25f;
          spec.std[1] = 0.5f;
          spec.std[2] = 0.125f;

          studiocast::cuda::CudaImage gpu_src;
          studiocast::cuda::CudaTensor tensor;
          if (!gpu_src.Allocate(&cuda, src_w, src_h,
                                studiocast::cuda::PixelFormatGpu::rgb_u8,
                                &err)) {
            fail("CudaPreprocessToTensor", "Allocate src failed: " + err);
          } else if (!gpu_src.UploadFromCpuRgb24(&cuda, src_rgb.data(),
                                                 src_stride, stream, &err)) {
            fail("CudaPreprocessToTensor", "Upload failed: " + err);
          } else if (!tensor.AllocateNchwF32(&cuda, 1, 3, spec.dst_h,
                                             spec.dst_w, &err)) {
            fail("CudaPreprocessToTensor", "Allocate tensor failed: " + err);
          } else if (!studiocast::cuda::kernels::PreprocessToTensor(
                         gpu_src, tensor, spec, stream, &err)) {
            fail("CudaPreprocessToTensor", "Kernel failed: " + err);
          } else {
            std::vector<float> gpu_out;
            if (!tensor.DownloadToCpuF32(&cuda, &gpu_out, stream, &err)) {
              fail("CudaPreprocessToTensor", "Download failed: " + err);
            } else if (!cuda.StreamSynchronize(stream, &err)) {
              fail("CudaPreprocessToTensor",
                   "StreamSynchronize failed: " + err);
            } else {
              auto clampInt = [](int v, int lo, int hi) {
                if (v < lo)
                  return lo;
                if (v > hi)
                  return hi;
                return v;
              };
              const auto cpu_ref = [&]() {
                std::vector<float> ref(static_cast<std::size_t>(3) *
                                       static_cast<std::size_t>(spec.dst_h) *
                                       static_cast<std::size_t>(spec.dst_w));
                const float scale_x =
                    static_cast<float>(src_w) / static_cast<float>(spec.dst_w);
                const float scale_y =
                    static_cast<float>(src_h) / static_cast<float>(spec.dst_h);
                for (int y = 0; y < spec.dst_h; ++y) {
                  const float sy =
                      (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
                  const int y0 =
                      clampInt(static_cast<int>(std::floor(sy)), 0, src_h - 1);
                  const int y1 = clampInt(y0 + 1, 0, src_h - 1);
                  const float fy = sy - static_cast<float>(y0);
                  const auto *row0 = src_rgb.data() +
                                     static_cast<std::size_t>(y0) * src_stride;
                  const auto *row1 = src_rgb.data() +
                                     static_cast<std::size_t>(y1) * src_stride;
                  for (int x = 0; x < spec.dst_w; ++x) {
                    const float sx =
                        (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
                    const int x0 = clampInt(static_cast<int>(std::floor(sx)), 0,
                                            src_w - 1);
                    const int x1 = clampInt(x0 + 1, 0, src_w - 1);
                    const float fx = sx - static_cast<float>(x0);

                    const auto *p00 = row0 + static_cast<std::size_t>(x0) * 3u;
                    const auto *p10 = row0 + static_cast<std::size_t>(x1) * 3u;
                    const auto *p01 = row1 + static_cast<std::size_t>(x0) * 3u;
                    const auto *p11 = row1 + static_cast<std::size_t>(x1) * 3u;

                    const std::size_t base =
                        static_cast<std::size_t>(y) *
                            static_cast<std::size_t>(spec.dst_w) +
                        static_cast<std::size_t>(x);
                    for (int c = 0; c < 3; ++c) {
                      const float p00f = static_cast<float>(p00[c]);
                      const float p10f = static_cast<float>(p10[c]);
                      const float p01f = static_cast<float>(p01[c]);
                      const float p11f = static_cast<float>(p11[c]);

                      const float v0 = p00f + fx * (p10f - p00f);
                      const float v1 = p01f + fx * (p11f - p01f);
                      const float v = v0 + fy * (v1 - v0);
                      const float vf = v * (1.0f / 255.0f);
                      const float norm = (vf - spec.mean[c]) / spec.std[c];
                      ref[static_cast<std::size_t>(c) *
                              static_cast<std::size_t>(spec.dst_h) *
                              static_cast<std::size_t>(spec.dst_w) +
                          base] = norm;
                    }
                  }
                }
                return ref;
              }();

              float max_abs_diff = 0.0f;
              for (std::size_t i = 0; i < cpu_ref.size() && i < gpu_out.size();
                   ++i) {
                const float d = std::fabs(cpu_ref[i] - gpu_out[i]);
                if (d > max_abs_diff)
                  max_abs_diff = d;
              }
              if (max_abs_diff > 1e-4f) {
                ++failures;
                std::printf("[FAIL] CudaPreprocessToTensor\n  max_abs_diff=%g "
                            "(want <= 1e-4)\n",
                            static_cast<double>(max_abs_diff));
              }
            }
          }

          (void)gpu_src.Free(&cuda, nullptr);
          (void)tensor.Free(&cuda, nullptr);
        }

        (void)cuda.DestroyStream(stream, nullptr);
      }
    }
  }

  // MJPEG preference heuristic (pure logic; used by V4L2 capture negotiation).
  {
    using studiocast::video::ShouldPreferMjpegForResolution;
    expectTrue("ShouldPreferMjpegForResolution(1920x1080)",
               ShouldPreferMjpegForResolution(1920, 1080));
    expectTrue("ShouldPreferMjpegForResolution(1280x720)",
               ShouldPreferMjpegForResolution(1280, 720));
    expectTrue("ShouldPreferMjpegForResolution(invalid) == false",
               !ShouldPreferMjpegForResolution(0, 720));
  }

  // MJPEG decode: encode a simple RGB test pattern to JPEG in-memory, then
  // decode back to RGB24.
  {
    struct JpegErr {
      jpeg_error_mgr pub;
      jmp_buf jmp;
    } jerr;

    auto encodeRgb24ToJpeg = [&](const std::uint8_t *rgb, int w, int h,
                                 int quality,
                                 std::vector<std::uint8_t> *outJpeg) -> bool {
      if (!rgb || w <= 0 || h <= 0 || !outJpeg)
        return false;
      outJpeg->clear();

      jpeg_compress_struct cinfo{};
      cinfo.err = jpeg_std_error(&jerr.pub);
      jerr.pub.error_exit = [](j_common_ptr ci) {
        auto *e = reinterpret_cast<JpegErr *>(ci->err);
        longjmp(e->jmp, 1);
      };

      if (setjmp(jerr.jmp) != 0) {
        jpeg_destroy_compress(&cinfo);
        return false;
      }

      jpeg_create_compress(&cinfo);
      unsigned char *dst = nullptr;
      unsigned long dstLen = 0;
      jpeg_mem_dest(&cinfo, &dst, &dstLen);

      cinfo.image_width = static_cast<JDIMENSION>(w);
      cinfo.image_height = static_cast<JDIMENSION>(h);
      cinfo.input_components = 3;
      cinfo.in_color_space = JCS_RGB;

      jpeg_set_defaults(&cinfo);
      jpeg_set_quality(&cinfo, quality, TRUE);
      jpeg_start_compress(&cinfo, TRUE);

      const std::size_t stride = static_cast<std::size_t>(w) * 3u;
      while (cinfo.next_scanline < cinfo.image_height) {
        JSAMPROW row[1];
        row[0] = const_cast<JSAMPROW>(reinterpret_cast<const JSAMPLE *>(
            rgb + static_cast<std::size_t>(cinfo.next_scanline) * stride));
        if (jpeg_write_scanlines(&cinfo, row, 1) != 1) {
          jpeg_finish_compress(&cinfo);
          jpeg_destroy_compress(&cinfo);
          return false;
        }
      }

      jpeg_finish_compress(&cinfo);
      jpeg_destroy_compress(&cinfo);

      if (!dst || dstLen == 0)
        return false;
      outJpeg->assign(dst, dst + dstLen);
      // `jpeg_mem_dest` uses `malloc`.
      std::free(dst);
      return true;
    };

    const int w = 16;
    const int h = 16;
    std::vector<std::uint8_t> src(static_cast<std::size_t>(w) *
                                  static_cast<std::size_t>(h) * 3u);

    auto setPixel = [&](int x, int y, std::uint8_t r, std::uint8_t g,
                        std::uint8_t b) {
      const std::size_t i =
          (static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
           static_cast<std::size_t>(x)) *
          3u;
      src[i + 0] = r;
      src[i + 1] = g;
      src[i + 2] = b;
    };

    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const bool right = x >= (w / 2);
        const bool bottom = y >= (h / 2);
        if (!right && !bottom) {
          setPixel(x, y, 255, 0, 0); // red
        } else if (right && !bottom) {
          setPixel(x, y, 0, 255, 0); // green
        } else if (!right && bottom) {
          setPixel(x, y, 0, 0, 255); // blue
        } else {
          setPixel(x, y, 255, 255, 255); // white
        }
      }
    }

    std::vector<std::uint8_t> jpeg;
    expectTrue("EncodeRgb24ToJpeg(in-memory)",
               encodeRgb24ToJpeg(src.data(), w, h, 95, &jpeg));

    studiocast::video::Rgb24Frame decoded;
    int dw = 0;
    int dh = 0;
    std::string decErr;
    const bool decOk = studiocast::video::DecodeMjpegToRgb24(
        jpeg.data(), jpeg.size(), decoded, dw, dh, &decErr);
    expectTrue("DecodeMjpegToRgb24", decOk);
    expectTrue("DecodeMjpegToRgb24 error empty on success",
               decOk && decErr.empty());
    expectIntEq("DecodeMjpegToRgb24 width", dw, w);
    expectIntEq("DecodeMjpegToRgb24 height", dh, h);

    // Negative: decode should fail on truncated JPEG and provide an error.
    {
      std::vector<std::uint8_t> truncated = jpeg;
      if (truncated.size() > 8)
        truncated.resize(truncated.size() / 2);
      studiocast::video::Rgb24Frame bad;
      int bw = 0;
      int bh = 0;
      std::string badErr;
      const bool badOk = studiocast::video::DecodeMjpegToRgb24(
          truncated.data(), truncated.size(), bad, bw, bh, &badErr);
      expectTrue("DecodeMjpegToRgb24(truncated) fails", !badOk);
      expectTrue("DecodeMjpegToRgb24(truncated) sets error",
                 !badOk && !badErr.empty());
    }

    auto get = [&](int x, int y) {
      const std::size_t i =
          (static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
           static_cast<std::size_t>(x)) *
          3u;
      struct RGB {
        int r;
        int g;
        int b;
      };
      return RGB{decoded.data()[i + 0], decoded.data()[i + 1],
                 decoded.data()[i + 2]};
    };

    const auto a = get(4, 4);
    const auto b = get(12, 4);
    const auto c = get(4, 12);
    const auto d = get(12, 12);

    expectTrue("DecodeMjpegToRgb24 TL is red-ish",
               a.r >= 180 && a.g <= 80 && a.b <= 80);
    expectTrue("DecodeMjpegToRgb24 TR is green-ish",
               b.g >= 180 && b.r <= 80 && b.b <= 80);
    expectTrue("DecodeMjpegToRgb24 BL is blue-ish",
               c.b >= 180 && c.r <= 80 && c.g <= 80);
    expectTrue("DecodeMjpegToRgb24 BR is white-ish",
               d.r >= 180 && d.g >= 180 && d.b >= 180);
  }

  // `pactl info` parsing helper (deterministic; avoids needing pactl in
  // self-test).
  {
    const std::string info =
        "Server String: /run/user/1000/pulse/native\n"
        "Default Sink:  alsa_output.pci-0000_00_1f.3.analog-stereo  \n"
        "Default Source:\talsa_input.pci-0000_00_1f.3.analog-stereo\n";

    const auto sink = studiocast::audio::pulse::ParseDefaultFromPactlInfo(
        info, "Default Sink:");
    expectEq("ParseDefaultFromPactlInfo(Default Sink)", sink ? *sink : "",
             "alsa_output.pci-0000_00_1f.3.analog-stereo");

    const auto src = studiocast::audio::pulse::ParseDefaultFromPactlInfo(
        info, "Default Source:");
    expectEq("ParseDefaultFromPactlInfo(Default Source)", src ? *src : "",
             "alsa_input.pci-0000_00_1f.3.analog-stereo");

    const auto missing = studiocast::audio::pulse::ParseDefaultFromPactlInfo(
        info, "Does Not Exist:");
    expectTrue("ParseDefaultFromPactlInfo(missing) is nullopt",
               !missing.has_value());
  }

  // Canonical effects model sanity: CPU backend must never be
  // persisted/returned as a real option.
  {
    studiocast::video::CameraEffects legacy;
    legacy.background_backend = studiocast::video::effects::EffectBackend::cpu;

    const auto bfx = studiocast::video::ToBroadcastCameraEffects(legacy);
    expectTrue(
        "ToBroadcastCameraEffects maps legacy cpu backend -> engine auto",
        bfx.engine ==
            studiocast::video::effects::EffectsEnginePreference::auto_select);

    const auto roundtrip = studiocast::video::ToLegacyCameraEffects(bfx);
    expectTrue("ToLegacyCameraEffects never returns cpu backend",
               roundtrip.background_backend !=
                   studiocast::video::effects::EffectBackend::cpu);
  }

  // Canonical Broadcast-style audio effects JSON round-trip + validation.
  {
    using studiocast::audio::effects::BroadcastAudioEffects;
    using studiocast::audio::effects::BroadcastAudioEffectsJsonParseOptions;
    using studiocast::audio::effects::BroadcastAudioEffectsToJson;
    using studiocast::audio::effects::ParseBroadcastAudioEffectsJsonText;
    using studiocast::audio::effects::SuperresMode;

    BroadcastAudioEffects fx;
    fx.microphone.model_id = "mic_model_0";
    fx.microphone.noise_removal_enabled = true;
    fx.microphone.room_echo_removal_enabled = true;
    fx.microphone.strength = 42;
    fx.microphone.studio_voice_enabled = false;
    fx.microphone.aec.enabled = true;
    fx.microphone.aec.reference_source = "monitor_source0";
    fx.microphone.superres.enabled = true;
    fx.microphone.superres.mode = SuperresMode::k8kTo16k;
    fx.speaker.model_id = "spk_model_0";
    fx.speaker.noise_removal_enabled = true;
    fx.speaker.strength = 33;
    fx.speaker.superres.enabled = true;
    fx.speaker.superres.mode = SuperresMode::k16kTo48k;

    const auto json = BroadcastAudioEffectsToJson(fx);
    expectContains("BroadcastAudioEffectsToJson includes mic.aec", json,
                   "\"aec\":{");
    expectContains("BroadcastAudioEffectsToJson includes aec reference_source",
                   json, "\"reference_source\":\"monitor_source0\"");
    expectContains("BroadcastAudioEffectsToJson includes microphone object",
                   json, "\"microphone\":{");
    expectContains("BroadcastAudioEffectsToJson includes mic.superres mode",
                   json, "\"mode\":\"8k_to_16k\"");
    expectContains("BroadcastAudioEffectsToJson includes mic model_id", json,
                   "\"model_id\":\"mic_model_0\"");
    expectContains("BroadcastAudioEffectsToJson includes speaker model_id",
                   json, "\"model_id\":\"spk_model_0\"");
    BroadcastAudioEffects parsed;
    std::vector<std::string> warnings;
    std::string error;
    expectTrue("ParseBroadcastAudioEffectsJsonText: round-trip ok",
               ParseBroadcastAudioEffectsJsonText(
                   json, &parsed, BroadcastAudioEffectsJsonParseOptions{},
                   &warnings, &error));
    expectEq("ParseBroadcastAudioEffectsJsonText: error empty", error, "");
    expectVecEq("ParseBroadcastAudioEffectsJsonText: warnings empty", warnings,
                {});
    expectTrue("BroadcastAudioEffects JSON round-trip equality", parsed == fx);

    // model_id should be optional for forward/backward compatibility.
    {
      const std::string noModel = "{"
                                  "\"schema_version\":2,"
                                  "\"microphone\":{"
                                  "\"noise_removal_enabled\":true"
                                  "},"
                                  "\"speaker\":{"
                                  "\"strength\":50"
                                  "}"
                                  "}";

      BroadcastAudioEffects parsedNoModel;
      warnings.clear();
      error.clear();
      expectTrue("ParseBroadcastAudioEffectsJsonText: missing model_id ok",
                 ParseBroadcastAudioEffectsJsonText(
                     noModel, &parsedNoModel,
                     BroadcastAudioEffectsJsonParseOptions{}, &warnings,
                     &error));
      expectEq(
          "ParseBroadcastAudioEffectsJsonText: missing model_id error empty",
          error, "");
      expectEq("ParseBroadcastAudioEffectsJsonText: missing mic model_id "
               "default empty",
               parsedNoModel.microphone.model_id, "");
      expectEq("ParseBroadcastAudioEffectsJsonText: missing spk model_id "
               "default empty",
               parsedNoModel.speaker.model_id, "");
    }

    // Superres mode validation.
    {
      const std::string badMode = "{"
                                  "\"schema_version\":2,"
                                  "\"microphone\":{"
                                  "\"superres\":{"
                                  "\"enabled\":true,"
                                  "\"mode\":\"12k_to_48k\""
                                  "}"
                                  "}"
                                  "}";

      BroadcastAudioEffects parsedBadMode;
      warnings.clear();
      error.clear();
      expectTrue(
          "ParseBroadcastAudioEffectsJsonText: invalid superres mode rejects",
          !ParseBroadcastAudioEffectsJsonText(
              badMode, &parsedBadMode, BroadcastAudioEffectsJsonParseOptions{},
              &warnings, &error));
      expectContains("ParseBroadcastAudioEffectsJsonText: invalid superres "
                     "mode error text",
                     error, "superres.mode");
    }

    // Studio Voice must be mutually exclusive with mic noise/echo removal.
    const std::string bad = "{"
                            "\"schema_version\":1,"
                            "\"microphone\":{"
                            "\"noise_removal_enabled\":true,"
                            "\"room_echo_removal_enabled\":false,"
                            "\"strength\":50,"
                            "\"studio_voice_enabled\":true"
                            "},"
                            "\"speaker\":{"
                            "\"noise_removal_enabled\":false,"
                            "\"strength\":50"
                            "}"
                            "}";

    BroadcastAudioEffects parsedBad;
    warnings.clear();
    error.clear();
    expectTrue("ParseBroadcastAudioEffectsJsonText: exclusivity rejects",
               !ParseBroadcastAudioEffectsJsonText(
                   bad, &parsedBad, BroadcastAudioEffectsJsonParseOptions{},
                   &warnings, &error));
    expectContains("ParseBroadcastAudioEffectsJsonText: exclusivity error text",
                   error, "mutually exclusive");
  }

  // Effect ordering + compatibility rules (single source of truth).
  {
    using studiocast::video::effects::BroadcastCameraEffects;
    using studiocast::video::effects::BuildBroadcastEffectsPlan;
    using studiocast::video::effects::VirtualBackgroundMode;

    const auto disabledHas =
        [](const studiocast::video::effects::BroadcastEffectsPlan &plan,
           std::string_view id) {
          for (const auto &d : plan.disabled) {
            if (d.id == id)
              return true;
          }
          return false;
        };

    // VB replace requires replace_path.
    {
      BroadcastCameraEffects fx;
      fx.virtual_background.mode = VirtualBackgroundMode::replace;
      fx.virtual_background.replace_path.clear();

      const auto plan = BuildBroadcastEffectsPlan(fx);
      expectVecEq("EffectPlan: vb replace missing path -> no stages",
                  plan.ordered_effect_ids, {});
      expectTrue("EffectPlan: vb replace missing path -> disabled list "
                 "contains vb.replace",
                 disabledHas(plan, studiocast::video::effects::contract::
                                       kEffectIdVirtualBackgroundReplace));
    }

    // Auto Frame and Virtual Background can be enabled simultaneously.
    {
      BroadcastCameraEffects fx;
      fx.auto_frame.enabled = true;
      fx.virtual_background.mode = VirtualBackgroundMode::blur;

      const auto plan = BuildBroadcastEffectsPlan(fx);
      expectVecEq(
          "EffectPlan: vb.blur + auto_frame are both scheduled",
          plan.ordered_effect_ids,
          {std::string(studiocast::video::effects::contract::
                           kEffectIdVirtualBackgroundBlur),
           std::string(
               studiocast::video::effects::contract::kEffectIdAutoFrame)});
      expectTrue("EffectPlan: vb.blur is not disabled when auto_frame enabled",
                 !disabledHas(plan, studiocast::video::effects::contract::
                                        kEffectIdVirtualBackgroundBlur));
    }

    // Ordering + vignette attachment target.
    {
      BroadcastCameraEffects fx;
      fx.eye_contact.enabled = true;
      fx.virtual_background.mode = VirtualBackgroundMode::blur;
      fx.vignette.enabled = true;
      fx.vignette.intensity = 50;
      fx.mirror = true;

      const auto plan = BuildBroadcastEffectsPlan(fx);
      expectVecEq(
          "EffectPlan: ordering eye_contact -> vb.blur -> vignette (mirror "
          "ignored)",
          plan.ordered_effect_ids,
          {std::string(
               studiocast::video::effects::contract::kEffectIdEyeContact),
           std::string(studiocast::video::effects::contract::
                           kEffectIdVirtualBackgroundBlur),
           std::string(
               studiocast::video::effects::contract::kEffectIdVignette)});
      expectTrue(
          "EffectPlan: mirror is reported as disabled",
          disabledHas(plan,
                      studiocast::video::effects::contract::kEffectIdMirror));
      expectEq("EffectPlan: vignette attaches to last GPU stage (vb.blur)",
               plan.vignette_attach_to_effect_id,
               std::string(studiocast::video::effects::contract::
                               kEffectIdVirtualBackgroundBlur));
    }

    // Standalone vignette when no other GPU stage is enabled.
    {
      BroadcastCameraEffects fx;
      fx.vignette.enabled = true;
      fx.vignette.intensity = 25;
      const auto plan = BuildBroadcastEffectsPlan(fx);
      expectVecEq(
          "EffectPlan: vignette-only stage", plan.ordered_effect_ids,
          {std::string(
              studiocast::video::effects::contract::kEffectIdVignette)});
      expectEq("EffectPlan: vignette-only attach target empty",
               plan.vignette_attach_to_effect_id, "");
    }
  }

  {
    using studiocast::video::effects::EffectsEnginePreference;
    EffectsEnginePreference ep{};

    expectTrue("ParseEffectsEnginePreference(maxine)",
               studiocast::video::effects::ParseEffectsEnginePreference(
                   "maxine", &ep) &&
                   ep == EffectsEnginePreference::maxine);

    expectTrue("ParseEffectsEnginePreference(open_cuda)",
               studiocast::video::effects::ParseEffectsEnginePreference(
                   "open_cuda", &ep) &&
                   ep == EffectsEnginePreference::open_cuda);

    expectTrue(
        "ParseEffectsEnginePreference alias onnx",
        studiocast::video::effects::ParseEffectsEnginePreference("onnx", &ep) &&
            ep == EffectsEnginePreference::open_cuda);

    expectTrue(
        "ParseEffectsEnginePreference alias cuda",
        studiocast::video::effects::ParseEffectsEnginePreference("cuda", &ep) &&
            ep == EffectsEnginePreference::open_cuda);

    expectEq("ToString(open_cuda)",
             studiocast::video::effects::ToString(
                 EffectsEnginePreference::open_cuda),
             "open_cuda");

    // Round-trip: ToString() must be parseable.
    expectTrue("EffectsEnginePreference roundtrip auto",
               studiocast::video::effects::ParseEffectsEnginePreference(
                   studiocast::video::effects::ToString(
                       EffectsEnginePreference::auto_select),
                   &ep) &&
                   ep == EffectsEnginePreference::auto_select);
    expectTrue("EffectsEnginePreference roundtrip maxine",
               studiocast::video::effects::ParseEffectsEnginePreference(
                   studiocast::video::effects::ToString(
                       EffectsEnginePreference::maxine),
                   &ep) &&
                   ep == EffectsEnginePreference::maxine);
    expectTrue("EffectsEnginePreference roundtrip open_cuda",
               studiocast::video::effects::ParseEffectsEnginePreference(
                   studiocast::video::effects::ToString(
                       EffectsEnginePreference::open_cuda),
                   &ep) &&
                   ep == EffectsEnginePreference::open_cuda);

    expectTrue(
        "ParseEffectsEnginePreference rejects cpu",
        !studiocast::video::effects::ParseEffectsEnginePreference("cpu", &ep));
  }

  {
    studiocast::video::effects::BroadcastCameraEffects parsed;
    studiocast::video::effects::BroadcastEffectsJsonParseOptions options;
    options.allow_unknown_keys = true;
    options.allow_compat_keys = true;
    std::vector<std::string> warnings;
    std::string perr;

    expectTrue(
        "ParseBroadcastCameraEffectsJsonText engine=maxine",
        studiocast::video::effects::ParseBroadcastCameraEffectsJsonText(
            "{\"engine\":\"maxine\"}", &parsed, options, &warnings, &perr) &&
            parsed.engine ==
                studiocast::video::effects::EffectsEnginePreference::maxine);

    warnings.clear();
    perr.clear();
    expectTrue(
        "ParseBroadcastCameraEffectsJsonText engine=open_cuda",
        studiocast::video::effects::ParseBroadcastCameraEffectsJsonText(
            "{\"engine\":\"open_cuda\"}", &parsed, options, &warnings, &perr) &&
            parsed.engine ==
                studiocast::video::effects::EffectsEnginePreference::open_cuda);

    warnings.clear();
    perr.clear();
    expectTrue("ParseBroadcastCameraEffectsJsonText rejects engine=cpu",
               !studiocast::video::effects::ParseBroadcastCameraEffectsJsonText(
                   "{\"engine\":\"cpu\"}", &parsed, options, &warnings, &perr));
    expectEq("ParseBroadcastCameraEffectsJsonText rejects engine=cpu msg", perr,
             "engine must be one of: auto, maxine, open_cuda");
  }

  {
    studiocast::video::CameraEffects legacy;
    std::string perr;

    expectTrue("ApplyCameraEffectsPatchJsonText rejects engine=cpu",
               !studiocast::video::ApplyCameraEffectsPatchJsonText(
                   "{\"engine\":\"cpu\"}", &legacy, &perr));

    perr.clear();
    expectTrue("ApplyCameraEffectsPatchJsonText rejects background_backend=cpu",
               !studiocast::video::ApplyCameraEffectsPatchJsonText(
                   "{\"background_backend\":\"cpu\"}", &legacy, &perr));

    perr.clear();
    expectTrue(
        "ApplyCameraEffectsPatchJsonText accepts background_backend=maxine",
        studiocast::video::ApplyCameraEffectsPatchJsonText(
            "{\"background_backend\":\"maxine\"}", &legacy, &perr));

    const auto bfx = studiocast::video::ToBroadcastCameraEffects(legacy);
    expectTrue("Legacy patch converts to Broadcast engine=maxine",
               bfx.engine ==
                   studiocast::video::effects::EffectsEnginePreference::maxine);
  }

  {
    studiocast::maxine::NvcvApi api;
    std::string err;
    const bool ok =
        api.Initialize(studiocast::maxine::NvcvApi::Requirement::Minimal, &err);

    // Self-test must be stable without Maxine installed:
    // - If Maxine is present, initialization may succeed.
    // - If Maxine is absent, we expect a clean failure with a non-empty error
    // message.
    if (!ok && err.empty()) {
      ++failures;
      std::printf("[FAIL] NvcvApi.Initialize returned false but provided no "
                  "error message\n");
    }
  }

  // Virtual Background CPU-side math sanity checks.
  {
    auto composite_u8 = [](std::uint8_t fg, std::uint8_t bg,
                           std::uint8_t a) -> std::uint8_t {
      // a is matte alpha for foreground.
      const int af = static_cast<int>(a);
      const int ab = 255 - af;
      const int v = static_cast<int>(fg) * af + static_cast<int>(bg) * ab;
      return static_cast<std::uint8_t>((v + 127) / 255);
    };

    // fg=255, bg=0
    if (composite_u8(255, 0, 0) != 0) {
      ++failures;
      std::printf("[FAIL] composite_u8 alpha=0\n");
    }
    if (composite_u8(255, 0, 255) != 255) {
      ++failures;
      std::printf("[FAIL] composite_u8 alpha=255\n");
    }
    const auto mid = composite_u8(255, 0, 128);
    if (mid < 126 || mid > 129) {
      ++failures;
      std::printf("[FAIL] composite_u8 alpha=128 got=%u\n",
                  static_cast<unsigned>(mid));
    }
  }

  // Auto Frame crop math + smoothing (deterministic; no Maxine runtime needed).
  {
    using studiocast::maxine::effects::ArAutoFrameTracker;
    using studiocast::maxine::effects::AutoFrameKnobs;
    using studiocast::maxine::effects::RectF;

    const int w = 1280;
    const int h = 720;
    const float aspect = 16.0f / 9.0f;

    // Center crop at 1x should be full frame for matching aspect.
    {
      const RectF r = ArAutoFrameTracker::CenterCrop(w, h, aspect, 1.0f);
      if (std::abs(r.x) > 1e-3f || std::abs(r.y) > 1e-3f ||
          std::abs(r.w - 1280.0f) > 1e-3f || std::abs(r.h - 720.0f) > 1e-3f) {
        ++failures;
        std::printf("[FAIL] AutoFrame CenterCrop full-frame mismatch\n");
      }
    }

    // Stronger strength should produce a tighter crop for the same box.
    {
      const RectF face = RectF{540.0f, 180.0f, 200.0f, 200.0f};
      AutoFrameKnobs k0;
      k0.strength = 0;
      k0.smoothing = 0;
      k0.headroom = 0.15f;
      AutoFrameKnobs k1;
      k1.strength = 100;
      k1.smoothing = 0;
      k1.headroom = 0.15f;

      const RectF a = ArAutoFrameTracker::ComputeTargetCropFromBoxPx(
          face, w, h, aspect, k0);
      const RectF b = ArAutoFrameTracker::ComputeTargetCropFromBoxPx(
          face, w, h, aspect, k1);
      if (!(b.w < a.w && b.h < a.h)) {
        ++failures;
        std::printf("[FAIL] AutoFrame strength should tighten crop\n");
      }
    }

    // Smoothing alpha monotonic: smoothing=0 should respond faster than
    // smoothing=100.
    {
      const float a0 = ArAutoFrameTracker::SmoothingAlpha(0);
      const float a1 = ArAutoFrameTracker::SmoothingAlpha(100);
      if (!(a0 > a1 && a0 > 0.5f && a1 < 0.2f)) {
        ++failures;
        std::printf(
            "[FAIL] AutoFrame SmoothingAlpha unexpected mapping a0=%f a1=%f\n",
            a0, a1);
      }
    }
  }

  // PPM loader + resize check (dependency-free replace-image path).
  {
    // 2x2 PPM P6: red, green, blue, white.
    const char *tmpPath = "/tmp/studiocast_selftest_bg.ppm";
    {
      std::ofstream out(tmpPath, std::ios::binary);
      if (!out) {
        ++failures;
        std::printf("[FAIL] failed to open tmp PPM for writing\n");
      } else {
        out << "P6\n2 2\n255\n";
        const std::uint8_t px[] = {
            255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255,
        };
        out.write(reinterpret_cast<const char *>(px),
                  static_cast<std::streamsize>(sizeof(px)));
      }
    }

    int w = 0, h = 0;
    std::vector<std::uint8_t> rgb;
    std::string imgErr;
    if (!studiocast::video::LoadPpmP6Rgb24(tmpPath, &w, &h, &rgb, &imgErr)) {
      ++failures;
      std::printf("[FAIL] LoadPpmP6Rgb24: %s\n", imgErr.c_str());
    } else {
      if (w != 2 || h != 2 || rgb.size() != 12) {
        ++failures;
        std::printf("[FAIL] LoadPpmP6Rgb24 unexpected dims\n");
      }
    }

    std::vector<std::uint8_t> resized;
    std::string resizeErr;
    if (!rgb.empty() &&
        !studiocast::video::ResizeRgb24Bilinear(rgb.data(), 2, 2, 6, 4, 4,
                                                &resized, 12, &resizeErr)) {
      ++failures;
      std::printf("[FAIL] ResizeRgb24Bilinear: %s\n", resizeErr.c_str());
    } else if (!resized.empty()) {
      // Top-left should stay close to red.
      const std::uint8_t r = resized[0];
      const std::uint8_t g = resized[1];
      const std::uint8_t b = resized[2];
      if (r < 200 || g > 80 || b > 80) {
        ++failures;
        std::printf(
            "[FAIL] ResizeRgb24Bilinear unexpected top-left pixel: %u,%u,%u\n",
            static_cast<unsigned>(r), static_cast<unsigned>(g),
            static_cast<unsigned>(b));
      }
    }
  }

  {
    studiocast::maxine::vfx::VfxApi api;
    std::string err;
    const bool ok = api.Initialize(&err);

    // Self-test must be stable without Maxine installed:
    // - If Maxine is present, initialization may succeed.
    // - If Maxine is absent, we expect a clean failure with a non-empty error
    // message.
    if (!ok && err.empty()) {
      ++failures;
      std::printf("[FAIL] VfxApi.Initialize returned false but provided no "
                  "error message\n");
    }

    // If the library is present, CreateEffect should either succeed or return
    // an error code that maps to a useful string.
    if (ok) {
      studiocast::maxine::vfx::NvVFX_Handle h = nullptr;
      const studiocast::maxine::NvCV_Status st = api.f().NvVFX_CreateEffect(
          studiocast::maxine::vfx::NVVFX_FX_GREEN_SCREEN, &h);
      if (st == studiocast::maxine::NVCV_SUCCESS) {
        if (h) {
          api.f().NvVFX_DestroyEffect(h);
        }
      } else {
        const std::string msg = api.StatusToString(st);
        if (msg.empty()) {
          ++failures;
          std::printf("[FAIL] VfxApi.StatusToString returned empty message for "
                      "status=%d\n",
                      st);
        }
      }
    }
  }

  {
    studiocast::maxine::ar::ArApi api;
    std::string err;
    const bool ok = api.Initialize(&err);

    // Self-test must be stable without Maxine installed:
    // - If Maxine is present, initialization may succeed.
    // - If Maxine is absent, we expect a clean failure with a non-empty error
    // message.
    if (!ok && err.empty()) {
      ++failures;
      std::printf("[FAIL] ArApi.Initialize returned false but provided no "
                  "error message\n");
    }

    // If the library is present, Create should either succeed or return an
    // error code that maps to a useful string.
    if (ok) {
      studiocast::maxine::ar::NvAR_FeatureHandle h = nullptr;
      const studiocast::maxine::NvCV_Status st = api.f().NvAR_Create(
          studiocast::maxine::ar::NVAR_FEATURE_GAZE_REDIRECTION, &h);
      if (st == studiocast::maxine::NVCV_SUCCESS) {
        if (h) {
          api.f().NvAR_Destroy(h);
        }
      } else {
        const std::string msg = api.StatusToString(st);
        if (msg.empty()) {
          ++failures;
          std::printf("[FAIL] ArApi.StatusToString returned empty message for "
                      "status=%d\n",
                      st);
        }
      }
    }
  }

  {
    studiocast::maxine::afx::AfxApi api;
    std::string err;
    const bool ok = api.Initialize(&err);

    // Self-test must be stable without Maxine installed:
    // - If Maxine is present, initialization may succeed.
    // - If Maxine is absent, we expect a clean failure with a non-empty error
    // message.
    if (!ok && err.empty()) {
      ++failures;
      std::printf("[FAIL] AfxApi.Initialize returned false but provided no "
                  "error message\n");
    }

    // If the library is present, ensure required symbols resolved.
    if (ok) {
      if (!api.f().NvAFX_CreateEffect || !api.f().NvAFX_DestroyEffect ||
          !api.f().NvAFX_SetU32 || !api.f().NvAFX_SetFloat ||
          !api.f().NvAFX_SetString || !api.f().NvAFX_GetU32 ||
          !api.f().NvAFX_Load || !api.f().NvAFX_Run) {
        ++failures;
        std::printf("[FAIL] AfxApi.Initialize succeeded but required symbols "
                    "are missing\n");
      }
    }
  }

  // Daemon config schema migration + round-trip.
  {
    namespace fs = std::filesystem;

    const char *oldXdg = std::getenv("XDG_CONFIG_HOME");
    const std::string oldXdgStr = oldXdg ? std::string(oldXdg) : std::string();

    char tmpl[] = "/tmp/studiocast_selftest_conf_XXXXXX";
    char *dir = ::mkdtemp(tmpl);
    if (!dir) {
      ++failures;
      std::printf("[FAIL] mkdtemp failed\n");
    } else {
      ::setenv("XDG_CONFIG_HOME", dir, 1);

      std::error_code ec;
      fs::create_directories(fs::path(dir) / "studiocast", ec);
      if (ec) {
        ++failures;
        std::printf("[FAIL] create_directories: %s\n", ec.message().c_str());
      }

      const fs::path confPath = fs::path(dir) / "studiocast" / "daemon.conf";

      {
        const auto dc_default = studiocast::config::LoadDaemonConfig();
        expectTrue("daemon_config default output_format rgb24",
                   dc_default.video_output_format ==
                       studiocast::video::PixelFormat::rgb24);
        const auto vc_default =
            studiocast::config::ToVideoServiceConfig(dc_default);
        expectTrue("ToVideoServiceConfig default output_format rgb24",
                   vc_default.pipeline.output_format ==
                       studiocast::video::PixelFormat::rgb24);
      }

      // Legacy background keys should migrate into canonical
      // `video.effects.json`.
      {
        std::ofstream out(confPath);
        out << "video.mirror = true\n";
        out << "video.scaling.backend = gpu\n";
        out << "video.background = blur\n";
        out << "video.background_backend = maxine\n";
        out << "video.background_strength = 13\n";
        out << "video.background_remove_color = #112233\n";
        out << "video.background_replace_image = /tmp/x.ppm\n";
        out << "video.eye_contact = true\n";
        out << "video.eye_contact_strength = 77\n";
        out << "video.eye_contact_look_away = false\n";
        out << "video.virtual_key_light = true\n";
        out << "video.virtual_key_light_intensity = 42\n";
        out << "video.virtual_key_light_temperature = warm\n";
        out << "video.vignette = true\n";
        out << "video.vignette_intensity = 9\n";
        out << "video.vignette_center_on_face = false\n";
      }

      const auto dc = studiocast::config::LoadDaemonConfig();
      expectEq("daemon_config migrate vb mode",
               studiocast::video::effects::ToString(
                   dc.video_effects.virtual_background.mode),
               "blur");
      expectIntEq("daemon_config migrate vb blur_strength",
                  dc.video_effects.virtual_background.strength, 13);
      expectEq("daemon_config migrate vb remove_color",
               dc.video_effects.virtual_background.remove_color, "#112233");
      expectEq("daemon_config migrate vb replace_path",
               dc.video_effects.virtual_background.replace_path, "/tmp/x.ppm");
      expectTrue("daemon_config migrate mirror", dc.video_effects.mirror);
      expectTrue("daemon_config migrate eye_contact enabled",
                 dc.video_effects.eye_contact.enabled);
      expectIntEq("daemon_config migrate eye_contact strength",
                  dc.video_effects.eye_contact.strength, 77);
      expectTrue("daemon_config migrate key_light enabled",
                 dc.video_effects.virtual_key_light.enabled);
      expectIntEq("daemon_config migrate key_light intensity",
                  dc.video_effects.virtual_key_light.intensity, 42);
      expectTrue("daemon_config default allow CPU resize",
                 dc.video_allow_cpu_resize);
      expectTrue("daemon_config missing output_format defaults rgb24",
                 dc.video_output_format ==
                     studiocast::video::PixelFormat::rgb24);

      const auto vc = studiocast::config::ToVideoServiceConfig(dc);
      expectTrue("ToVideoServiceConfig mirror", vc.pipeline.effects.mirror);
      expectTrue("ToVideoServiceConfig allow CPU resize",
                 vc.pipeline.allow_cpu_resize);
      expectTrue("ToVideoServiceConfig output_format rgb24",
                 vc.pipeline.output_format ==
                     studiocast::video::PixelFormat::rgb24);
      expectTrue("ToVideoServiceConfig scaling backend gpu",
                 vc.pipeline.scaling_backend ==
                     studiocast::video::ScalingBackendPreference::gpu);
      expectTrue("ToVideoServiceConfig vb blur",
                 vc.pipeline.effects.virtual_background.mode ==
                     studiocast::video::effects::VirtualBackgroundMode::blur);
      expectIntEq("ToVideoServiceConfig vb strength",
                  vc.pipeline.effects.virtual_background.strength, 13);
      expectTrue(
          "ToVideoServiceConfig engine maxine",
          vc.pipeline.effects.engine ==
              studiocast::video::effects::EffectsEnginePreference::maxine);
      expectTrue("ToVideoServiceConfig eye_contact enabled",
                 vc.pipeline.effects.eye_contact.enabled);

      // Audio config: ensure new persisted fields are wired.
      {
        auto dc_audio = dc;
        dc_audio.audio_enabled = true;
        dc_audio.audio_create_virtual_mic = true;
        dc_audio.audio_create_virtual_speakers = true;
        dc_audio.audio_speakers_enabled = true;
        dc_audio.audio_speaker_target_sink = "dummy_sink";
        dc_audio.audio_speaker_latency_ms = 15;
        dc_audio.audio_source = "dummy_source";
        dc_audio.audio_effects.microphone.noise_removal_enabled = true;
        dc_audio.audio_effects.microphone.room_echo_removal_enabled = true;
        dc_audio.audio_effects.microphone.strength = 55;
        dc_audio.audio_effects.microphone.studio_voice_enabled = false;

        const auto ac = studiocast::config::ToAudioServiceConfig(dc_audio);
        expectTrue("ToAudioServiceConfig enabled", ac.enabled);
        expectTrue("ToAudioServiceConfig create_virtual_mic",
                   ac.create_virtual_mic);
        expectTrue("ToAudioServiceConfig create_virtual_speakers",
                   ac.create_virtual_speakers);
        expectTrue("ToAudioServiceConfig speakers_enabled",
                   ac.speakers_enabled);
        expectEq("ToAudioServiceConfig speaker_target_sink",
                 ac.speaker_target_sink, "dummy_sink");
        expectIntEq("ToAudioServiceConfig speaker_latency_ms",
                    ac.speaker_latency_ms, 15);
        expectEq("ToAudioServiceConfig source", ac.source_name, "dummy_source");
        expectTrue("ToAudioServiceConfig noise enabled",
                   ac.effects.microphone.noise_removal_enabled);
        expectTrue("ToAudioServiceConfig echo enabled",
                   ac.effects.microphone.room_echo_removal_enabled);
        expectIntEq("ToAudioServiceConfig strength",
                    ac.effects.microphone.strength, 55);
      }

      // New schema should round-trip through Save/Load.
      {
        // Ensure audio fields are persisted too.
        auto dc_save = dc;
        dc_save.audio_enabled = true;
        dc_save.audio_create_virtual_mic = true;
        dc_save.audio_create_virtual_speakers = true;
        dc_save.audio_speakers_enabled = true;
        dc_save.audio_speaker_target_sink = "dummy_sink";
        dc_save.audio_speaker_latency_ms = 15;
        dc_save.audio_source = "dummy_source";
        dc_save.audio_effects.microphone.noise_removal_enabled = true;
        dc_save.audio_effects.microphone.room_echo_removal_enabled = true;
        dc_save.audio_effects.microphone.strength = 55;
        dc_save.video_output_format = studiocast::video::PixelFormat::yuyv;

        std::string err;
        if (!studiocast::config::SaveDaemonConfig(dc_save, &err)) {
          ++failures;
          std::printf("[FAIL] SaveDaemonConfig: %s\n", err.c_str());
        }

        // Saved config should contain only the canonical effects blob (no
        // legacy per-effect keys).
        {
          std::ifstream in(confPath);
          const std::string content((std::istreambuf_iterator<char>(in)),
                                    std::istreambuf_iterator<char>());
          expectTrue("saved config has video.effects.json",
                     content.find("video.effects.json") != std::string::npos);
          expectTrue("saved config has audio.effects.json",
                     content.find("audio.effects.json") != std::string::npos);
          expectTrue("saved config has video.scaling.backend",
                     content.find("video.scaling.backend") !=
                         std::string::npos);
          expectTrue("saved config has video.scaling.allow_cpu_resize true",
                     content.find("video.scaling.allow_cpu_resize = true") !=
                         std::string::npos);
          expectTrue("saved config has video.output_format yuyv",
                     content.find("video.output_format = yuyv") !=
                         std::string::npos);
          expectTrue("saved config removes video.mirror",
                     content.find("video.mirror") == std::string::npos);
          expectTrue("saved config removes video.background",
                     content.find("video.background") == std::string::npos);
          expectTrue("saved config removes video.effects.virtual_background",
                     content.find("video.effects.virtual_background") ==
                         std::string::npos);
        }

        const auto dc2 = studiocast::config::LoadDaemonConfig();
        const auto vc2 = studiocast::config::ToVideoServiceConfig(dc2);
        expectTrue("roundtrip allow CPU resize", vc2.pipeline.allow_cpu_resize);
        expectTrue("roundtrip output_format yuyv",
                   vc2.pipeline.output_format ==
                       studiocast::video::PixelFormat::yuyv);
        expectTrue("roundtrip vb blur",
                   vc2.pipeline.effects.virtual_background.mode ==
                       studiocast::video::effects::VirtualBackgroundMode::blur);
        expectIntEq("roundtrip vb strength",
                    vc2.pipeline.effects.virtual_background.strength, 13);
        expectTrue("roundtrip eye_contact enabled",
                   vc2.pipeline.effects.eye_contact.enabled);
        expectTrue("roundtrip key_light enabled",
                   vc2.pipeline.effects.virtual_key_light.enabled);

        const auto ac2 = studiocast::config::ToAudioServiceConfig(dc2);
        expectTrue("roundtrip audio enabled", ac2.enabled);
        expectTrue("roundtrip audio create_virtual_mic",
                   ac2.create_virtual_mic);
        expectTrue("roundtrip audio create_virtual_speakers",
                   ac2.create_virtual_speakers);
        expectTrue("roundtrip audio speakers_enabled", ac2.speakers_enabled);
        expectEq("roundtrip audio speaker_target_sink", ac2.speaker_target_sink,
                 "dummy_sink");
        expectIntEq("roundtrip audio speaker_latency_ms",
                    ac2.speaker_latency_ms, 15);
        expectEq("roundtrip audio source", ac2.source_name, "dummy_source");
        expectTrue("roundtrip audio noise enabled",
                   ac2.effects.microphone.noise_removal_enabled);
        expectTrue("roundtrip audio echo enabled",
                   ac2.effects.microphone.room_echo_removal_enabled);
        expectIntEq("roundtrip audio strength", ac2.effects.microphone.strength,
                    55);
      }

      // Explicit CPU-resize opt-out should still be respected.
      {
        std::ofstream out(confPath);
        out << "video.scaling.allow_cpu_resize = false\n";
      }
      {
        const auto dc_no_cpu = studiocast::config::LoadDaemonConfig();
        expectTrue("explicit CPU resize opt-out parses false",
                   !dc_no_cpu.video_allow_cpu_resize);
        const auto vc_no_cpu =
            studiocast::config::ToVideoServiceConfig(dc_no_cpu);
        expectTrue("explicit CPU resize opt-out reaches service config",
                   !vc_no_cpu.pipeline.allow_cpu_resize);
      }

      // Invalid persisted output format safely falls back to the default.
      {
        std::ofstream out(confPath);
        out << "video.output_format = not_a_format\n";
      }
      {
        const auto dc_bad_fmt = studiocast::config::LoadDaemonConfig();
        const auto vc_bad_fmt =
            studiocast::config::ToVideoServiceConfig(dc_bad_fmt);
        expectTrue("invalid output_format falls back rgb24",
                   vc_bad_fmt.pipeline.output_format ==
                       studiocast::video::PixelFormat::rgb24);
      }

      // Audio effects JSON parsing should tolerate unknown keys
      // (forward/backward drift).
      {
        std::ofstream out(confPath);
        out << "audio.enabled = true\n";
        out << "audio.source = dummy_source\n";
        out << "audio.effects.json = "
               "{\"schema_version\":1,\"microphone\":{\"noise_removal_"
               "enabled\":true,\"room_echo_removal_enabled\":false,"
               "\"strength\":42,\"studio_voice_enabled\":false,\"future_key\":"
               "123},\"speaker\":{\"enabled\":false}}\n";
      }
      {
        const auto dcu = studiocast::config::LoadDaemonConfig();
        const auto acu = studiocast::config::ToAudioServiceConfig(dcu);
        expectTrue("audio unknown-key parse enabled", acu.enabled);
        expectEq("audio unknown-key parse source", acu.source_name,
                 "dummy_source");
        expectTrue("audio unknown-key parse noise enabled",
                   acu.effects.microphone.noise_removal_enabled);
        expectIntEq("audio unknown-key parse strength",
                    acu.effects.microphone.strength, 42);
      }

      // Legacy auto_frame should migrate into the canonical effects blob.
      {
        std::ofstream out(confPath);
        out << "video.background = auto_frame\n";
        out << "video.auto_frame_strength = 88\n";
        out << "video.auto_frame_smoothing = 12\n";
        out << "video.auto_frame_headroom = 0.33\n";
      }

      const auto dc_af = studiocast::config::LoadDaemonConfig();
      expectTrue("daemon_config migrate auto_frame enabled",
                 dc_af.video_effects.auto_frame.enabled);
      expectIntEq("daemon_config migrate auto_frame zoom",
                  dc_af.video_effects.auto_frame.strength, 88);
      const auto vc_af = studiocast::config::ToVideoServiceConfig(dc_af);
      expectTrue("ToVideoServiceConfig auto_frame enabled",
                 vc_af.pipeline.effects.auto_frame.enabled);

      // Restore env.
      if (oldXdg) {
        ::setenv("XDG_CONFIG_HOME", oldXdgStr.c_str(), 1);
      } else {
        ::unsetenv("XDG_CONFIG_HOME");
      }
    }
  }

  // Effects JSON patch (line-based IPC helper).
  {
    studiocast::video::effects::BroadcastCameraEffects fx;
    std::string jerr;

    // Canonical contract: effect IDs are keys.
    const std::string pretty =
        "{\n"
        "  \"virtual_background.replace\": {\n"
        "    \"enabled\": true,\n"
        "    \"replace_path\": \"/tmp/some path/with spaces/bg.ppm\"\n"
        "  }\n"
        "}\n";

    const std::string minified = studiocast::util::json::Minify(pretty);
    expectTrue("json minify keeps spaces in strings",
               minified.find("/tmp/some path/with spaces/bg.ppm") !=
                   std::string::npos);

    if (!studiocast::video::ApplyBroadcastCameraEffectsPatchJsonText(
            minified, &fx, &jerr)) {
      ++failures;
      std::printf("[FAIL] ApplyBroadcastCameraEffectsPatchJsonText: %s\n",
                  jerr.c_str());
    } else {
      expectTrue(
          "effects patch vb replace",
          fx.virtual_background.mode ==
              studiocast::video::effects::VirtualBackgroundMode::replace);
      expectEq("effects patch replace_path", fx.virtual_background.replace_path,
               "/tmp/some path/with spaces/bg.ppm");
    }

    {
      studiocast::video::effects::BroadcastCameraEffects emptyPathFx;
      const std::string replaceWithoutPath =
          "{\"virtual_background.replace\":{\"enabled\":true,"
          "\"replace_path\":\"\"}}";
      jerr.clear();
      if (!studiocast::video::ApplyBroadcastCameraEffectsPatchJsonText(
              replaceWithoutPath, &emptyPathFx, &jerr)) {
        ++failures;
        std::printf("[FAIL] ApplyBroadcastCameraEffectsPatchJsonText "
                    "replace without path: %s\n",
                    jerr.c_str());
      } else {
        expectTrue("effects patch replace without path keeps replace mode",
                   emptyPathFx.virtual_background.mode ==
                       studiocast::video::effects::VirtualBackgroundMode::
                           replace);
        expectEq("effects patch replace without path keeps empty path",
                 emptyPathFx.virtual_background.replace_path, "");
      }
    }

    const std::string af =
        "{\"auto_frame\":{\"enabled\":true,\"strength\":77}}";
    jerr.clear();
    if (!studiocast::video::ApplyBroadcastCameraEffectsPatchJsonText(af, &fx,
                                                                     &jerr)) {
      ++failures;
      std::printf(
          "[FAIL] ApplyBroadcastCameraEffectsPatchJsonText auto_frame: %s\n",
          jerr.c_str());
    } else {
      expectTrue("effects patch auto_frame enabled", fx.auto_frame.enabled);
      expectIntEq("effects patch auto_frame zoom", fx.auto_frame.strength, 77);
      expectTrue(
          "effects patch auto_frame does not clear vb mode",
          fx.virtual_background.mode ==
              studiocast::video::effects::VirtualBackgroundMode::replace);
    }

    const std::string blur =
        "{\"virtual_background.blur\":{\"enabled\":true,\"strength\":9}}";
    jerr.clear();
    if (!studiocast::video::ApplyBroadcastCameraEffectsPatchJsonText(blur, &fx,
                                                                     &jerr)) {
      ++failures;
      std::printf("[FAIL] ApplyBroadcastCameraEffectsPatchJsonText blur: %s\n",
                  jerr.c_str());
    } else {
      expectTrue("effects patch blur keeps auto_frame enabled",
                 fx.auto_frame.enabled);
      expectTrue("effects patch vb blur",
                 fx.virtual_background.mode ==
                     studiocast::video::effects::VirtualBackgroundMode::blur);
      expectIntEq("effects patch blur_strength", fx.virtual_background.strength,
                  9);
    }

    const std::string mid = "{\"virtual_background.blur\":{\"enabled\":true,"
                            "\"model_id\":\"birefnet_lite\"}}";
    jerr.clear();
    if (!studiocast::video::ApplyBroadcastCameraEffectsPatchJsonText(mid, &fx,
                                                                     &jerr)) {
      ++failures;
      std::printf(
          "[FAIL] ApplyBroadcastCameraEffectsPatchJsonText model_id: %s\n",
          jerr.c_str());
    } else {
      expectEq("effects patch model_id", fx.virtual_background.model_id,
               "birefnet_lite");
    }

    const std::string eng = "{\"engine\":\"open_cuda\"}";
    jerr.clear();
    if (!studiocast::video::ApplyBroadcastCameraEffectsPatchJsonText(eng, &fx,
                                                                     &jerr)) {
      ++failures;
      std::printf(
          "[FAIL] ApplyBroadcastCameraEffectsPatchJsonText engine: %s\n",
          jerr.c_str());
    } else {
      expectTrue(
          "effects patch engine=open_cuda",
          fx.engine ==
              studiocast::video::effects::EffectsEnginePreference::open_cuda);
    }

    const std::string badEng = "{\"engine\":\"wat\"}";
    jerr.clear();
    expectTrue("effects patch engine rejects unknown",
               !studiocast::video::ApplyBroadcastCameraEffectsPatchJsonText(
                   badEng, &fx, &jerr));
    expectEq("effects patch engine unknown msg", jerr,
             "engine must be one of: auto, maxine, open_cuda");

    // Serializer output should be valid JSON and re-applicable.
    const std::string fxJson =
        studiocast::video::BroadcastCameraEffectsContractToJson(fx);
    studiocast::util::json::Value parsed;
    jerr.clear();
    if (!studiocast::util::json::Parse(fxJson, &parsed, &jerr)) {
      ++failures;
      std::printf("[FAIL] BroadcastCameraEffectsContractToJson parseable: %s\n",
                  jerr.c_str());
    }

    studiocast::video::effects::BroadcastCameraEffects fx2;
    const std::string wrapper =
        std::string("{\"video_effects\":") + fxJson + "}";
    jerr.clear();
    if (!studiocast::video::ApplyBroadcastCameraEffectsPatchJsonText(
            wrapper, &fx2, &jerr)) {
      ++failures;
      std::printf(
          "[FAIL] BroadcastCameraEffectsContractToJson roundtrip apply: %s\n",
          jerr.c_str());
    }
    expectTrue("BroadcastCameraEffectsContractToJson roundtrip mode",
               fx2.virtual_background.mode == fx.virtual_background.mode);
    expectEq("BroadcastCameraEffectsContractToJson roundtrip path",
             fx2.virtual_background.replace_path,
             fx.virtual_background.replace_path);
    expectEq("BroadcastCameraEffectsContractToJson roundtrip model_id",
             fx2.virtual_background.model_id, fx.virtual_background.model_id);
  }

  // BroadcastCameraEffects canonical JSON round-trip + strict validation.
  {
    using studiocast::video::effects::BroadcastCameraEffects;
    using studiocast::video::effects::BroadcastCameraEffectsToJson;
    using studiocast::video::effects::BroadcastEffectsJsonParseOptions;
    using studiocast::video::effects::EffectsEnginePreference;
    using studiocast::video::effects::ParseBroadcastCameraEffectsJsonText;
    using studiocast::video::effects::VirtualBackgroundMode;

    BroadcastCameraEffects fx;
    fx.schema_version =
        studiocast::video::effects::kBroadcastEffectsSchemaVersion;
    fx.mirror = true;
    fx.engine = EffectsEnginePreference::open_cuda;
    fx.virtual_background.mode = VirtualBackgroundMode::replace;
    fx.virtual_background.model_id = "birefnet_lite";
    fx.virtual_background.strength = 9;
    fx.virtual_background.replace_path = "/tmp/bg.ppm";
    fx.auto_frame.enabled = false;
    fx.eye_contact.enabled = true;
    fx.eye_contact.strength = 77;
    fx.eye_contact.look_away_enabled = false;
    fx.video_noise_removal.enabled = true;
    fx.video_noise_removal.strength = 22;
    fx.virtual_key_light.enabled = true;
    fx.virtual_key_light.intensity = 42;
    fx.virtual_key_light.temperature = 5000;
    fx.vignette.enabled = true;
    fx.vignette.intensity = 33;

    const std::string json = BroadcastCameraEffectsToJson(fx);
    BroadcastCameraEffects out;
    std::vector<std::string> warnings;
    std::string err;

    BroadcastEffectsJsonParseOptions opt;
    opt.allow_unknown_keys = false;
    opt.allow_compat_keys = true;
    if (!ParseBroadcastCameraEffectsJsonText(json, &out, opt, &warnings,
                                             &err)) {
      ++failures;
      std::printf("[FAIL] ParseBroadcastCameraEffectsJsonText roundtrip: %s\n",
                  err.c_str());
    } else {
      expectTrue("BroadcastCameraEffects roundtrip equals", out == fx);
      expectTrue("BroadcastCameraEffects roundtrip no warnings",
                 warnings.empty());
    }

    // model_id is optional (parser must accept JSON with or without it).
    {
      const std::string withModelId =
          "{\"schema_version\":1,\"virtual_background\":{\"mode\":\"none\","
          "\"model_id\":\"modnet-webnn-256-fp32\"}}";
      BroadcastCameraEffects tmp;
      warnings.clear();
      err.clear();
      opt.allow_unknown_keys = false;
      if (!ParseBroadcastCameraEffectsJsonText(withModelId, &tmp, opt,
                                               &warnings, &err)) {
        ++failures;
        std::printf("[FAIL] BroadcastCameraEffects model_id should parse: %s\n",
                    err.c_str());
      } else {
        expectEq("BroadcastCameraEffects model_id parse",
                 tmp.virtual_background.model_id, "modnet-webnn-256-fp32");
      }

      const std::string withoutModelId =
          "{\"schema_version\":1,\"virtual_background\":{\"mode\":\"none\"}}";
      BroadcastCameraEffects tmp2;
      warnings.clear();
      err.clear();
      if (!ParseBroadcastCameraEffectsJsonText(withoutModelId, &tmp2, opt,
                                               &warnings, &err)) {
        ++failures;
        std::printf(
            "[FAIL] BroadcastCameraEffects missing model_id should parse: %s\n",
            err.c_str());
      } else {
        expectEq("BroadcastCameraEffects missing model_id default",
                 tmp2.virtual_background.model_id, "");
      }
    }

    {
      const std::string replaceWithoutPath =
          "{\"schema_version\":1,\"virtual_background\":{\"mode\":\"replace\","
          "\"replace_path\":\"\"}}";
      BroadcastCameraEffects tmp;
      warnings.clear();
      err.clear();
      opt.allow_unknown_keys = false;
      if (!ParseBroadcastCameraEffectsJsonText(replaceWithoutPath, &tmp, opt,
                                               &warnings, &err)) {
        ++failures;
        std::printf("[FAIL] BroadcastCameraEffects replace without path should "
                    "parse: %s\n",
                    err.c_str());
      } else {
        expectTrue("BroadcastCameraEffects replace without path mode",
                   tmp.virtual_background.mode ==
                       VirtualBackgroundMode::replace);
        expectEq("BroadcastCameraEffects replace without path empty path",
                 tmp.virtual_background.replace_path, "");
        expectTrue(
            "BroadcastCameraEffects replace without path warning",
            std::any_of(warnings.begin(), warnings.end(),
                        [](const std::string &w) {
                          return w.find("replace_path is empty") !=
                                 std::string::npos;
                        }));
      }
    }

    // Unknown key strict vs compat.
    {
      const std::string u = "{\"schema_version\":1,\"unknown\":123}";

      BroadcastCameraEffects tmp;
      warnings.clear();
      err.clear();
      opt.allow_unknown_keys = false;
      if (ParseBroadcastCameraEffectsJsonText(u, &tmp, opt, &warnings, &err)) {
        ++failures;
        std::printf("[FAIL] BroadcastCameraEffects unknown key should fail in "
                    "strict mode\n");
      } else {
        expectContains("BroadcastCameraEffects unknown strict msg", err,
                       "unknown key");
      }

      warnings.clear();
      err.clear();
      opt.allow_unknown_keys = true;
      if (!ParseBroadcastCameraEffectsJsonText(u, &tmp, opt, &warnings, &err)) {
        ++failures;
        std::printf("[FAIL] BroadcastCameraEffects unknown key should pass in "
                    "compat mode: %s\n",
                    err.c_str());
      } else {
        expectTrue("BroadcastCameraEffects unknown compat warns",
                   !warnings.empty());
      }
    }

    // Validation: virtual_background.mode
    {
      const std::string badMode =
          "{\"schema_version\":1,\"virtual_background\":{\"mode\":\"wat\"}}";
      BroadcastCameraEffects tmp;
      warnings.clear();
      err.clear();
      opt.allow_unknown_keys = false;
      if (ParseBroadcastCameraEffectsJsonText(badMode, &tmp, opt, &warnings,
                                              &err)) {
        ++failures;
        std::printf("[FAIL] BroadcastCameraEffects bad mode should fail\n");
      } else {
        expectContains("BroadcastCameraEffects bad mode msg", err,
                       "virtual_background.mode");
      }
    }

    // Validation: auto_frame and virtual_background can run together.
    {
      const std::string paired =
          "{\"schema_version\":1,\"auto_frame\":{\"enabled\":true},\"virtual_"
          "background\":{\"mode\":\"blur\"}}";
      BroadcastCameraEffects tmp;
      warnings.clear();
      err.clear();
      if (!ParseBroadcastCameraEffectsJsonText(paired, &tmp, opt, &warnings,
                                               &err)) {
        ++failures;
        std::printf("[FAIL] BroadcastCameraEffects paired effects should pass: "
                    "%s\n",
                    err.c_str());
      } else {
        expectTrue("BroadcastCameraEffects paired auto_frame preserved",
                   tmp.auto_frame.enabled);
        expectTrue("BroadcastCameraEffects paired virtual_background preserved",
                   tmp.virtual_background.mode ==
                       VirtualBackgroundMode::blur);
      }
    }

    // Compatibility: temperature_preset alias.
    {
      const std::string preset = "{\"schema_version\":1,\"virtual_key_light\":{"
                                 "\"temperature_preset\":\"warm\"}}";
      BroadcastCameraEffects tmp;
      warnings.clear();
      err.clear();
      opt.allow_unknown_keys = false;
      opt.allow_compat_keys = true;
      if (!ParseBroadcastCameraEffectsJsonText(preset, &tmp, opt, &warnings,
                                               &err)) {
        ++failures;
        std::printf(
            "[FAIL] BroadcastCameraEffects temperature_preset compat: %s\n",
            err.c_str());
      } else {
        expectTrue("BroadcastCameraEffects preset sets temp",
                   tmp.virtual_key_light.temperature != 4500);
        expectTrue("BroadcastCameraEffects preset warns", !warnings.empty());
      }
    }
  }

  // Canonical Maxine blocked messaging copy (Task 25).
  {
    const char *oldHome = std::getenv("HOME");
    const std::string oldHomeStr =
        oldHome ? std::string(oldHome) : std::string();

    // Force a deterministic HOME so we can validate "~/.local/share/..."
    // rendering.
    ::setenv("HOME", "/home/studiocast_selftest_home", 1);

    const auto mkDiag = [&] {
      studiocast::maxine::MaxineDiagnostics d;
      d.gpu.ok = true;
      d.driver.ok = true;
      return d;
    };

    // 1) No GPU.
    {
      auto d = mkDiag();
      d.gpu.ok = false;
      const auto c = studiocast::maxine::BuildCanonicalMaxineBlockedCopy(d);
      expectEq("maxine_copy no_gpu summary", c.summary,
               "Maxine unavailable: no supported NVIDIA GPU detected.");
    }

    // 2) Driver too old.
    {
      auto d = mkDiag();
      d.driver.ok = false;
      d.driver.version = "560.0";
      const auto c = studiocast::maxine::BuildCanonicalMaxineBlockedCopy(d);
      expectEq("maxine_copy driver_old summary", c.summary,
               "Maxine unavailable: NVIDIA driver too old (need R570+).");
      const std::string s =
          studiocast::maxine::FormatCanonicalMaxineBlockedCopy(c);
      expectContains("maxine_copy driver_old has probe", s,
                     "Run `studiocast-probe` to verify GPU/driver.");
    }

    // 3) VFX SDK missing: must show exact expected paths.
    {
      auto d = mkDiag();
      d.vfx.root_source = "xdg";
      d.vfx.root_exists = false;
      d.vfx.library_exists = false;
      d.vfx.candidate_roots.push_back("/home/studiocast_selftest_home/.local/"
                                      "share/studiocast/maxine/VideoFX");

      const auto c = studiocast::maxine::BuildCanonicalMaxineBlockedCopy(
          d, studiocast::maxine::MaxineNeed::vfx);
      expectEq(
          "maxine_copy vfx_missing summary", c.summary,
          "Maxine unavailable: VFX SDK not found (expected: "
          "~/.local/share/studiocast/maxine/VideoFX or /usr/local/VideoFX).");
      const std::string s =
          studiocast::maxine::FormatCanonicalMaxineBlockedCopy(c);
      expectContains("maxine_copy vfx_missing has libvideofx hint", s,
                     "Ensure `libVideoFX.so` (or legacy `libnvvfx.so` /");
    }

    // 4) AR SDK missing.
    {
      auto d = mkDiag();
      d.ar.root_source = "xdg";
      d.ar.root_exists = false;
      d.ar.library_exists = false;
      d.ar.candidate_roots.push_back("/home/studiocast_selftest_home/.local/"
                                     "share/studiocast/maxine/ARSDK");

      const auto c = studiocast::maxine::BuildCanonicalMaxineBlockedCopy(
          d, studiocast::maxine::MaxineNeed::ar);
      expectEq("maxine_copy ar_missing summary", c.summary,
               "Maxine unavailable: AR SDK not found (needed for Eye Contact / "
               "Auto Frame).");
      const std::string s =
          studiocast::maxine::FormatCanonicalMaxineBlockedCopy(c);
      expectContains(
          "maxine_copy ar_missing shows expected root", s,
          "Expected AR SDK root: ~/.local/share/studiocast/maxine/ARSDK or "
          "/usr/local/ARSDK.");
    }

    // 4b) AFX SDK missing.
    {
      auto d = mkDiag();
      d.afx.root_source = "xdg";
      d.afx.root_exists = false;
      d.afx.library_exists = false;
      d.afx.candidate_roots.push_back(
          "/home/studiocast_selftest_home/.local/share/studiocast/maxine/"
          "Audio_Effects_SDK");

      const auto c = studiocast::maxine::BuildCanonicalMaxineBlockedCopy(
          d, studiocast::maxine::MaxineNeed::afx);
      expectEq("maxine_copy afx_missing summary", c.summary,
               "Maxine unavailable: Audio Effects SDK not found (needed for "
               "audio effects).");
      const std::string s =
          studiocast::maxine::FormatCanonicalMaxineBlockedCopy(c);
      expectContains("maxine_copy afx_missing shows expected root", s,
                     "Expected AFX SDK root: "
                     "~/.local/share/studiocast/maxine/Audio_Effects_SDK or "
                     "/usr/local/Audio_Effects_SDK.");
    }

    // 5) Features missing.
    {
      auto d = mkDiag();
      d.vfx.ok = true;
      d.vfx.library_loadable = true;
      d.ar.ok = true;
      d.ar.library_loadable = true;
      const auto c = studiocast::maxine::BuildCanonicalMaxineBlockedCopy(d);
      expectEq("maxine_copy features summary", c.summary,
               "Maxine unavailable: feature libraries not installed (run "
               "install_feature.sh).");
    }

    // 5b) AFX features missing.
    {
      auto d = mkDiag();
      d.afx.root_exists = true;
      d.afx.library_exists = true;
      d.afx.ok = true;
      d.afx.library_loadable = true;
      const auto c = studiocast::maxine::BuildCanonicalMaxineBlockedCopy(
          d, studiocast::maxine::MaxineNeed::afx);
      expectEq("maxine_copy afx_features summary", c.summary,
               "Maxine unavailable: Audio Effects features not installed (run "
               "download_features.sh).");
    }

    // Maxine per-effect availability oracle JSON contract.
    // GUI enable/disable decisions rely on `available_effects` +
    // `missing_effects`.
    {
      studiocast::maxine::MaxineDiagnostics d;
      d.ok = true;
      d.supported = true;
      d.available_effects = {"auto_frame"};
      d.available_audio_effects = {"denoiser"};
      d.missing_effects["eye_contact"] = {
          studiocast::maxine::reasons::MissingArFeature("gaze_redirection")};
      d.missing_effects["noise_removal"] = {
          studiocast::maxine::reasons::MissingAfxFeature("denoiser")};

      const std::string js = d.ToJson();
      expectContains("maxine_tojson has available_effects", js,
                     "\"available_effects\":[\"auto_frame\"]");
      expectContains("maxine_tojson has available_audio_effects", js,
                     "\"available_audio_effects\":[\"denoiser\"]");
      expectContains("maxine_tojson has missing_effects", js,
                     "\"missing_effects\":{");
      expectContains(
          "maxine_tojson has eye_contact reasons", js,
          "\"eye_contact\":[\"missing_ar_feature:gaze_redirection\"]");
      expectContains("maxine_tojson has noise_removal reasons", js,
                     "\"noise_removal\":[\"missing_afx_feature:denoiser\"]");
      expectContains("maxine_tojson has components.afx", js,
                     "\"components\":{");
    }

    // Maxine gating decision helper: if a Maxine-backed effect is enabled but
    // not available, the pipeline must be blocked before starting.
    {
      studiocast::video::effects::BroadcastCameraEffects fx;
      fx.virtual_background.mode =
          studiocast::video::effects::VirtualBackgroundMode::blur;

      auto d = mkDiag();
      d.vfx.root_source = "xdg";
      d.vfx.root_exists = false;
      d.vfx.library_exists = false;
      d.vfx.candidate_roots.push_back("/home/studiocast_selftest_home/.local/"
                                      "share/studiocast/maxine/VideoFX");

      const auto gate = studiocast::video::effects::EvaluateMaxineGate(fx, d);
      expectTrue("maxine_gate blur blocked", !gate.ok);
      expectContains("maxine_gate blur blocked message", gate.message,
                     "Maxine unavailable: VFX SDK not found");

      d.vfx.root_exists = true;
      d.vfx.library_exists = true;
      d.vfx.ok = true;
      d.vfx.library_loadable = true;
      d.available_effects = {"virtual_background.blur"};
      const auto gate2 = studiocast::video::effects::EvaluateMaxineGate(fx, d);
      expectTrue("maxine_gate blur allowed when available", gate2.ok);
    }

    // Open CUDA diagnostics + gating decision helper: if an Open CUDA-backed
    // effect is enabled but not available, the service should suppress those
    // effects and keep the pipeline running in pass-through mode.
    {
      using studiocast::video::effects::contract::
          kEffectIdVirtualBackgroundBlur;

      const auto diag = studiocast::open_cuda::DiagnoseOpenCudaDefault();
#if !STUDIOCAST_ENABLE_OPEN_CUDA
      expectTrue("open_cuda_diag blocked when backend disabled in build",
                 !diag.ok);
      const auto it = diag.blocked_effects.find(
          std::string(kEffectIdVirtualBackgroundBlur));
      expectTrue("open_cuda_diag reason disabled_in_build",
                 it != diag.blocked_effects.end() &&
                     it->second == "disabled_in_build");
#elif STUDIOCAST_HAVE_ONNXRUNTIME
      expectTrue("open_cuda_diag has install_hints",
                 !diag.install_hints.empty());
#else
      expectTrue("open_cuda_diag blocked when built without ORT", !diag.ok);
      const auto it = diag.blocked_effects.find(
          std::string(kEffectIdVirtualBackgroundBlur));
      expectTrue("open_cuda_diag reason onnxruntime_not_found",
                 it != diag.blocked_effects.end() &&
                     it->second == "onnxruntime_not_found");
#endif

      studiocast::video::effects::BroadcastCameraEffects fx;
      fx.engine =
          studiocast::video::effects::EffectsEnginePreference::open_cuda;
      fx.virtual_background.mode =
          studiocast::video::effects::VirtualBackgroundMode::blur;

      expectTrue(
          "open_cuda_gate wants_vb",
          studiocast::video::effects::WantsOpenCudaForPlannedEffects(fx));

      studiocast::open_cuda::OpenCudaDiagnostics blocked;
      blocked.ok = false;
      blocked.blocked_effects[std::string(kEffectIdVirtualBackgroundBlur)] =
          "missing_model_packs";
      blocked.install_hints = {"No usable Open CUDA model packs were found."};

      const auto gate =
          studiocast::video::effects::EvaluateOpenCudaGate(fx, blocked);
      expectTrue("open_cuda_gate blur blocked", !gate.ok);
      expectContains(
          "open_cuda_gate blur blocked message", gate.message,
          "virtual_background.blur unavailable (missing_model_packs)");

      auto fx_for_pipeline = fx;
      if (!gate.ok) {
        fx_for_pipeline.virtual_background.mode =
            studiocast::video::effects::VirtualBackgroundMode::none;
      }
      expectTrue("open_cuda_gate suppressed vb mode",
                 fx_for_pipeline.virtual_background.mode ==
                     studiocast::video::effects::VirtualBackgroundMode::none);

      studiocast::open_cuda::OpenCudaDiagnostics available;
      available.ok = true;
      available.available_effects = {"virtual_background.blur"};
      const auto gate2 =
          studiocast::video::effects::EvaluateOpenCudaGate(fx, available);
      expectTrue("open_cuda_gate blur allowed when available", gate2.ok);

      {
        using studiocast::video::effects::contract::kEffectIdVideoNoiseRemoval;

        studiocast::video::effects::BroadcastCameraEffects fx_denoise;
        fx_denoise.engine =
            studiocast::video::effects::EffectsEnginePreference::open_cuda;
        fx_denoise.video_noise_removal.enabled = true;

        studiocast::open_cuda::OpenCudaDiagnostics ort_cuda_missing;
        ort_cuda_missing.ok = true;
        ort_cuda_missing.available_effects = {
            std::string(kEffectIdVideoNoiseRemoval)};
        ort_cuda_missing
            .blocked_effects[std::string(kEffectIdVirtualBackgroundBlur)] =
            "onnxruntime_cuda_provider_unavailable";

        const auto gate_denoise =
            studiocast::video::effects::EvaluateOpenCudaGate(fx_denoise,
                                                             ort_cuda_missing);
        expectTrue("open_cuda_gate denoise allowed without ort cuda ep",
                   gate_denoise.ok);
      }

      // Auto Frame (Open CUDA) is also gated by the Open CUDA diagnostics.
      {
        using studiocast::video::effects::contract::kEffectIdAutoFrame;

        studiocast::video::effects::BroadcastCameraEffects fx2;
        fx2.engine =
            studiocast::video::effects::EffectsEnginePreference::open_cuda;
        fx2.auto_frame.enabled = true;
        fx2.auto_frame.strength = 50;

        expectTrue(
            "open_cuda_gate wants_auto_frame",
            studiocast::video::effects::WantsOpenCudaForPlannedEffects(fx2));

        studiocast::open_cuda::OpenCudaDiagnostics blocked2;
        blocked2.ok = false;
        blocked2.blocked_effects[std::string(kEffectIdAutoFrame)] =
            "missing_model_packs";

        const auto gate_af =
            studiocast::video::effects::EvaluateOpenCudaGate(fx2, blocked2);
        expectTrue("open_cuda_gate auto_frame blocked", !gate_af.ok);
        expectContains("open_cuda_gate auto_frame blocked message",
                       gate_af.message,
                       "auto_frame unavailable (missing_model_packs)");

        auto fx2_for_pipeline = fx2;
        if (!gate_af.ok) {
          fx2_for_pipeline.auto_frame.enabled = false;
        }
        expectTrue("open_cuda_gate suppressed auto_frame",
                   !fx2_for_pipeline.auto_frame.enabled);

        studiocast::open_cuda::OpenCudaDiagnostics available2;
        available2.ok = true;
        available2.available_effects = {"auto_frame"};
        const auto gate_af2 =
            studiocast::video::effects::EvaluateOpenCudaGate(fx2, available2);
        expectTrue("open_cuda_gate auto_frame allowed when available",
                   gate_af2.ok);
      }

      // Virtual Key Light (Open CUDA) is also gated by the Open CUDA
      // diagnostics.
      {
        using studiocast::video::effects::contract::kEffectIdVirtualKeyLight;

        studiocast::video::effects::BroadcastCameraEffects fx3;
        fx3.engine =
            studiocast::video::effects::EffectsEnginePreference::open_cuda;
        fx3.virtual_key_light.enabled = true;
        fx3.virtual_key_light.intensity = 60;

        expectTrue(
            "open_cuda_gate wants_virtual_key_light",
            studiocast::video::effects::WantsOpenCudaForPlannedEffects(fx3));

        studiocast::open_cuda::OpenCudaDiagnostics blocked3;
        blocked3.ok = false;
        blocked3.blocked_effects[std::string(kEffectIdVirtualKeyLight)] =
            "missing_model_packs";

        const auto gate_kl =
            studiocast::video::effects::EvaluateOpenCudaGate(fx3, blocked3);
        expectTrue("open_cuda_gate virtual_key_light blocked", !gate_kl.ok);
        expectContains("open_cuda_gate virtual_key_light blocked message",
                       gate_kl.message,
                       "virtual_key_light unavailable (missing_model_packs)");

        auto fx3_for_pipeline = fx3;
        if (!gate_kl.ok) {
          fx3_for_pipeline.virtual_key_light.enabled = false;
        }
        expectTrue("open_cuda_gate suppressed virtual_key_light",
                   !fx3_for_pipeline.virtual_key_light.enabled);

        studiocast::open_cuda::OpenCudaDiagnostics available3;
        available3.ok = true;
        available3.available_effects = {"virtual_key_light"};
        const auto gate_kl2 =
            studiocast::video::effects::EvaluateOpenCudaGate(fx3, available3);
        expectTrue("open_cuda_gate virtual_key_light allowed when available",
                   gate_kl2.ok);
      }

      auto fx_for_pipeline2 = fx;
      if (!gate2.ok) {
        fx_for_pipeline2.virtual_background.mode =
            studiocast::video::effects::VirtualBackgroundMode::none;
      }
      expectTrue("open_cuda_gate does not suppress when allowed",
                 fx_for_pipeline2.virtual_background.mode ==
                     studiocast::video::effects::VirtualBackgroundMode::blur);
    }

    // Repo model pack templates (metadata-only) must exist and be well-formed.
    //
    // These templates are used by scripts/tools to guide users to install the
    // actual model artifacts. StudioCast intentionally does not commit the
    // model binaries into git.
    {
      namespace fs = std::filesystem;

      auto addFailure = [&](const std::string &msg) {
        ++failures;
        std::printf("[FAIL] %s\n", msg.c_str());
      };

      auto expectPathExists = [&](const char *name, const fs::path &p) -> bool {
        if (fs::exists(p))
          return true;
        ++failures;
        std::printf("[FAIL] %s\n  missing: %s\n", name, p.string().c_str());
        return false;
      };

      auto findRepoModelPacksDir = [&]() -> std::optional<fs::path> {
        const auto tryFrom = [&](fs::path start) -> std::optional<fs::path> {
          for (int i = 0; i < 10; ++i) {
            const auto cand = start / "resources" / "model_packs";
            if (fs::exists(cand) && fs::is_directory(cand))
              return cand;
            if (!start.has_parent_path())
              break;
            start = start.parent_path();
          }
          return std::nullopt;
        };

        // Prefer current working directory (CI runs from repo root).
        {
          std::error_code ec;
          const auto cwd = fs::current_path(ec);
          if (!ec) {
            if (auto r = tryFrom(cwd))
              return r;
          }
        }

        // Fallback: resolve from the executable location.
        try {
          const auto exe = fs::canonical("/proc/self/exe");
          if (exe.has_parent_path()) {
            if (auto r = tryFrom(exe.parent_path()))
              return r;
          }
        } catch (...) {
          // Ignore.
        }

        return std::nullopt;
      };

      auto readTextFile = [&](const fs::path &p) -> std::optional<std::string> {
        std::ifstream f(p);
        if (!f)
          return std::nullopt;
        std::string text((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
        return text;
      };

      auto parseJsonFile = [&](const fs::path &p,
                               studiocast::util::json::Value *out) -> bool {
        const auto text = readTextFile(p);
        if (!text) {
          addFailure(std::string("model pack: failed to read ") + p.string());
          return false;
        }
        std::string err;
        if (!studiocast::util::json::Parse(*text, out, &err)) {
          addFailure(std::string("model pack: invalid JSON in ") + p.string() +
                     ": " + err);
          return false;
        }
        return true;
      };

      const auto packsRootOpt = findRepoModelPacksDir();
      expectTrue("repo_model_packs_dir found", packsRootOpt.has_value());
      if (packsRootOpt) {
        const auto packsRoot = *packsRootOpt;
        (void)expectPathExists("repo_model_packs_dir exists", packsRoot);

        // ---- Open Video templates ----
        const fs::path openVideoRoot = packsRoot / "open_video";
        if (expectPathExists("open_video templates dir", openVideoRoot)) {
          struct Subject {
            const char *dir;
            const char *name;
          };
          const Subject subjects[] = {
              {"matting", "open_video templates: matting"},
              {"eye_contact", "open_video templates: eye_contact"},
              {"face_detection", "open_video templates: face_detection"},
              {"face_landmarks", "open_video templates: face_landmarks"},
              {"video_denoise", "open_video templates: video_denoise"},
          };

          for (const auto &sub : subjects) {
            const auto subjectDir = openVideoRoot / sub.dir;
            if (!expectPathExists(sub.name, subjectDir))
              continue;

            int packCount = 0;
            for (const auto &e : fs::directory_iterator(subjectDir)) {
              if (!e.is_directory())
                continue;
              const auto packDir = e.path();
              const auto manifest = packDir / "model.json";
              const auto license = packDir / "LICENSE.txt";
              if (!fs::exists(manifest) || !fs::exists(license))
                continue;
              ++packCount;

              studiocast::util::json::Value v;
              if (!parseJsonFile(manifest, &v))
                continue;
              const auto *o = v.AsObject();
              if (!o) {
                addFailure(std::string("model pack: expected object in ") +
                           manifest.string());
                continue;
              }
              auto itId = o->find("id");
              auto itName = o->find("display_name");
              auto itTask = o->find("task");
              if (itId == o->end() || !itId->second.AsString() ||
                  itId->second.AsString()->empty()) {
                addFailure(std::string("model pack: missing/empty 'id' in ") +
                           manifest.string());
              }
              if (itName == o->end() || !itName->second.AsString() ||
                  itName->second.AsString()->empty()) {
                addFailure(std::string(
                               "model pack: missing/empty 'display_name' in ") +
                           manifest.string());
              }
              if (itTask == o->end() || !itTask->second.AsString() ||
                  itTask->second.AsString()->empty()) {
                addFailure(std::string("model pack: missing/empty 'task' in ") +
                           manifest.string());
              }
            }

            expectTrue(
                (std::string(sub.name) + " has at least one pack").c_str(),
                packCount > 0);
          }
        }

        // ---- Open Audio templates ----
        const fs::path openAudioRoot = packsRoot / "open_audio";
        if (expectPathExists("open_audio templates dir", openAudioRoot)) {
          std::map<std::string, bool> covered;
          covered[std::string(
              studiocast::audio::effects::contract::kEffectIdNoiseRemoval)] =
              false;
          covered[std::string(
              studiocast::audio::effects::contract::kEffectIdRoomEchoRemoval)] =
              false;
          covered[std::string(
              studiocast::audio::effects::contract::kEffectIdStudioVoice)] =
              false;

          int packCount = 0;
          for (const auto &e : fs::directory_iterator(openAudioRoot)) {
            if (!e.is_directory())
              continue;
            const auto packDir = e.path();
            const auto manifest = packDir / "model.json";
            const auto license = packDir / "LICENSE.txt";
            if (!fs::exists(manifest) || !fs::exists(license))
              continue;
            ++packCount;

            studiocast::util::json::Value v;
            if (!parseJsonFile(manifest, &v))
              continue;
            const auto *o = v.AsObject();
            if (!o) {
              addFailure(std::string("model pack: expected object in ") +
                         manifest.string());
              continue;
            }

            // Basic required fields (templates may omit the actual ONNX
            // binary).
            auto itId = o->find("id");
            auto itName = o->find("display_name");
            auto itOnnx = o->find("onnx_filename");
            auto itEffects = o->find("effects");
            if (itId == o->end() || !itId->second.AsString() ||
                itId->second.AsString()->empty()) {
              addFailure(std::string("model pack: missing/empty 'id' in ") +
                         manifest.string());
            }
            if (itName == o->end() || !itName->second.AsString() ||
                itName->second.AsString()->empty()) {
              addFailure(
                  std::string("model pack: missing/empty 'display_name' in ") +
                  manifest.string());
            }
            if (itOnnx == o->end() || !itOnnx->second.AsString() ||
                itOnnx->second.AsString()->empty()) {
              addFailure(
                  std::string("model pack: missing/empty 'onnx_filename' in ") +
                  manifest.string());
            }
            if (itEffects == o->end() || !itEffects->second.AsArray()) {
              addFailure(
                  std::string("model pack: missing/invalid 'effects' in ") +
                  manifest.string());
            } else {
              for (const auto &ev : *itEffects->second.AsArray()) {
                const auto *s = ev.AsString();
                if (!s)
                  continue;
                auto it = covered.find(*s);
                if (it != covered.end())
                  it->second = true;
              }
            }
          }

          expectTrue("open_audio templates has at least one pack",
                     packCount > 0);
          for (const auto &kv : covered) {
            expectTrue(
                (std::string("open_audio templates cover effect ") + kv.first)
                    .c_str(),
                kv.second);
          }
        }
      }
    }

    // AFX: Broadcast-equivalent microphone planning rules.
    {
      using studiocast::maxine::afx::PlanBroadcastMicrophoneEffect;

      const auto p0 =
          PlanBroadcastMicrophoneEffect(/*studio_voice_enabled=*/false,
                                        /*noise_removal_enabled=*/false,
                                        /*room_echo_removal_enabled=*/false,
                                        /*strength=*/50);
      expectTrue("afx_plan none disabled", !p0.enabled);

      const auto p1 =
          PlanBroadcastMicrophoneEffect(/*studio_voice_enabled=*/true,
                                        /*noise_removal_enabled=*/true,
                                        /*room_echo_removal_enabled=*/true,
                                        /*strength=*/50);
      expectTrue("afx_plan studio_voice enabled", p1.enabled);
      expectEq("afx_plan studio_voice selector", p1.effect_selector,
               "studio_voice_low_latency");
      expectEq("afx_plan studio_voice feature_id", p1.feature_id,
               "studio_voice");

      const auto p2 =
          PlanBroadcastMicrophoneEffect(/*studio_voice_enabled=*/false,
                                        /*noise_removal_enabled=*/true,
                                        /*room_echo_removal_enabled=*/true,
                                        /*strength=*/50);
      expectEq("afx_plan noise+echo selector", p2.effect_selector,
               "dereverb_denoiser");
      expectEq("afx_plan noise+echo feature_id", p2.feature_id,
               "dereverb_denoiser");

      const auto p3 =
          PlanBroadcastMicrophoneEffect(/*studio_voice_enabled=*/false,
                                        /*noise_removal_enabled=*/true,
                                        /*room_echo_removal_enabled=*/false,
                                        /*strength=*/50);
      expectEq("afx_plan noise selector", p3.effect_selector, "denoiser");
      expectEq("afx_plan noise feature_id", p3.feature_id, "denoiser");

      const auto p4 =
          PlanBroadcastMicrophoneEffect(/*studio_voice_enabled=*/false,
                                        /*noise_removal_enabled=*/false,
                                        /*room_echo_removal_enabled=*/true,
                                        /*strength=*/50);
      expectEq("afx_plan echo selector", p4.effect_selector, "dereverb");
      expectEq("afx_plan echo feature_id", p4.feature_id, "dereverb");
    }

    // Canonical audio effects planner: AEC + Superres validation / reason
    // strings.
    {
      using studiocast::audio::effects::BroadcastAudioEffects;
      using studiocast::audio::effects::BroadcastAudioEffectsPlanInputs;
      using studiocast::audio::effects::PlanBroadcastAudioEffects;
      using studiocast::audio::effects::SuperresMode;

      BroadcastAudioEffects fx;
      BroadcastAudioEffectsPlanInputs in;
      in.available_pulse_sources = {"monitor0"};
      in.float32_pcm = true;

      fx.microphone.aec.enabled = true;
      fx.microphone.aec.reference_source.clear();
      const auto p0 = PlanBroadcastAudioEffects(fx, in);
      expectTrue("audio_plan aec missing reference_source disabled",
                 !p0.microphone_aec.enabled);
      expectContains("audio_plan aec missing reference_source reason",
                     p0.microphone_aec.reason, "reference_source is empty");

      fx.microphone.aec.reference_source = "monitor0";
      const auto p1 = PlanBroadcastAudioEffects(fx, in);
      expectTrue("audio_plan aec enabled when reference_source available",
                 p1.microphone_aec.enabled);
      expectEq("audio_plan aec planned reference_source",
               p1.microphone_aec.reference_source, "monitor0");

      fx.microphone.aec.reference_source = "monitor_missing";
      const auto p2 = PlanBroadcastAudioEffects(fx, in);
      expectTrue("audio_plan aec disabled when reference_source unavailable",
                 !p2.microphone_aec.enabled);
      expectContains("audio_plan aec unavailable reason",
                     p2.microphone_aec.reason, "not available");

      fx.microphone.superres.enabled = true;
      fx.microphone.superres.mode = SuperresMode::k16kTo48k;
      in.float32_pcm = false;
      const auto p3 = PlanBroadcastAudioEffects(fx, in);
      expectTrue("audio_plan mic superres disabled when not float32",
                 !p3.microphone_superres.enabled);
      expectContains("audio_plan mic superres not float32 reason",
                     p3.microphone_superres.reason, "not float32");

      in.float32_pcm = true;
      const auto p4 = PlanBroadcastAudioEffects(fx, in);
      expectTrue("audio_plan mic superres enabled when float32",
                 p4.microphone_superres.enabled);
      expectEq("audio_plan mic superres planned mode",
               std::string(studiocast::audio::effects::ToString(
                   p4.microphone_superres.mode)),
               "16k_to_48k");
    }

    // AFX: wrapper error messaging should be clean + actionable even without
    // AFX installed.
    {
      studiocast::maxine::afx::AfxEffect e(nullptr);
      studiocast::maxine::afx::AfxEffectConfig cfg;
      cfg.effect_selector = "denoiser";
      cfg.feature_id = "denoiser";
      cfg.features_dir = "/nonexistent/afx/features";
      cfg.model_path = "/this/does/not/exist.trtpkg";

      std::string err;
      expectTrue("afx_effect missing model fails", !e.Configure(cfg, &err));
      expectContains("afx_effect missing model message", err,
                     "AFX model file not found");
    }
    {
      studiocast::maxine::afx::AfxEffect e(nullptr);
      studiocast::maxine::afx::AfxEffectConfig cfg;
      cfg.effect_selector = "denoiser";
      cfg.feature_id = "denoiser";
      cfg.features_dir = "/nonexistent/afx/features";
      cfg.model_path = "/dev/null";

      std::string err;
      expectTrue("afx_effect missing feature lib fails",
                 !e.Configure(cfg, &err));
      expectContains("afx_effect missing feature lib message", err,
                     "AFX feature library not found");
      expectContains("afx_effect missing feature lib mentions .so", err,
                     "libnv_audiofx_denoiser.so");
    }

    // Optional AFX runtime smoke test: if AFX + GPU are present, ensure we can
    // create+load+run.
    {
      const auto paths = studiocast::maxine::ResolveMaxinePaths();
      if (paths.afx.ok) {
        studiocast::maxine::afx::AfxApi api;
        std::string apiErr;
        if (api.InitializeFromLibraryPath(paths.afx.library, &apiErr)) {
          const auto sel =
              studiocast::maxine::SelectGpu(studiocast::config::GpuSelection{});
          if (sel.selected && sel.selected->compute_capability) {
            studiocast::maxine::afx::AfxEffect fx(&api);
            studiocast::maxine::afx::AfxEffectConfig cfg;
            cfg.effect_selector = "denoiser";
            cfg.feature_id = "denoiser";
            cfg.features_dir = paths.afx.features_dir;
            cfg.compute_capability = sel.selected->compute_capability;
            cfg.sample_rate = 48000;
            cfg.frame_samples = 480;
            cfg.channels = 1;
            cfg.intensity = 0.5f;

            std::string err;
            if (!fx.Configure(cfg, &err)) {
              ++failures;
              std::printf(
                  "[FAIL] afx_effect Configure (runtime)\n  error: %s\n",
                  err.c_str());
            } else if (!fx.Load(&err)) {
              ++failures;
              std::printf("[FAIL] afx_effect Load (runtime)\n  error: %s\n",
                          err.c_str());
            } else {
              std::vector<float> in(cfg.frame_samples * cfg.channels);
              std::vector<float> out(in.size());
              if (!in.empty()) {
                in[0] = 1.0f;
              }
              if (!fx.Run(in.data(), out.data(),
                          static_cast<std::uint32_t>(in.size()), &err)) {
                ++failures;
                std::printf("[FAIL] afx_effect Run (runtime)\n  error: %s\n",
                            err.c_str());
              }
            }
          }
        }
      }
    }

    // TTL cache: deterministic behavior (no sleep).
    {
      studiocast::util::TtlCache<std::string> c;
      int computes = 0;
      constexpr auto ttl = std::chrono::seconds(2);

      const auto t0 = std::chrono::steady_clock::time_point{};

      const auto v1 = c.GetOrCompute(t0 + std::chrono::seconds(1), ttl, [&]() {
        ++computes;
        return std::string("one");
      });
      expectEq("ttl_cache v1", v1, "one");
      expectIntEq("ttl_cache computes after v1", computes, 1);

      // Within TTL: should reuse cached value.
      const auto v2 = c.GetOrCompute(t0 + std::chrono::seconds(2), ttl, [&]() {
        ++computes;
        return std::string("two");
      });
      expectEq("ttl_cache v2 uses cached", v2, "one");
      expectIntEq("ttl_cache computes within ttl", computes, 1);

      // At TTL boundary (>= ttl): should recompute.
      const auto v3 = c.GetOrCompute(t0 + std::chrono::seconds(3), ttl, [&]() {
        ++computes;
        return std::string("three");
      });
      expectEq("ttl_cache v3 recompute", v3, "three");
      expectIntEq("ttl_cache computes after expiry", computes, 2);
    }

    // Restore env.
    if (oldHome) {
      ::setenv("HOME", oldHomeStr.c_str(), 1);
    } else {
      ::unsetenv("HOME");
    }
  }

  if (failures == 0) {
    std::printf("SELFTEST OK\n");
    return 0;
  }
  std::printf("SELFTEST FAILED (%d)\n", failures);
  return 1;
}
} // namespace

int main(int argc, char **argv) {
  if (hasArg(argc, argv, "--self-test")) {
    return RunSelfTest(ParseSelfTestOptions(argc, argv));
  }

  const bool json = hasArg(argc, argv, "--json");
  const bool verbose = hasArg(argc, argv, "--verbose");
  const bool strict = hasArg(argc, argv, "--strict");

  if (hasArg(argc, argv, "--verify-models")) {
    return RunVerifyModels(json, strict);
  }

  const auto report = studiocast::probe::Run(verbose);

  if (json) {
    std::printf("%s\n", report.ToJson().c_str());
  } else {
    std::printf("%s\n", report.ToText().c_str());
  }

  if (!strict)
    return 0;
  return report.AllChecksPassed() ? 0 : 1;
}
