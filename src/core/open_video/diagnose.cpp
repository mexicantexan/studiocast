#include "core/open_video/diagnose.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <set>

#include "core/cuda/cuda_driver_api.h"
#include "core/onnx/ort_session.h"
#include "core/open_video/model_pack_registry.h"
#include "core/util/xdg.h"
#include "core/video/effects/broadcast_effect_contract.h"

namespace studiocast::open_cuda {

namespace {

std::string ToLowerAscii(std::string s) {
  for (char &c : s) {
    const unsigned char uc = static_cast<unsigned char>(c);
    c = static_cast<char>(std::tolower(uc));
  }
  return s;
}

bool OpenCudaTensorRtRequestedFromEnv() {
  const char *v = std::getenv("STUDIOCAST_OPEN_CUDA_TENSORRT");
  if (!v || !*v)
    return false;

  const std::string value = ToLowerAscii(v);
  return value == "1" || value == "true" || value == "yes" || value == "on";
}

bool HasProvider(const std::vector<std::string> &providers,
                 const char *provider) {
  return std::find(providers.begin(), providers.end(), provider) !=
         providers.end();
}

bool LooksLikeDuplicateInstalledModelProblem(const std::string &id,
                                             const std::string &reason,
                                             const std::set<std::string> &ids) {
  if (ids.find(id) == ids.end())
    return false;
  const std::string lower = ToLowerAscii(reason);
  return lower.find("duplicate model id") != std::string::npos;
}

} // namespace

OpenCudaDiagnostics DiagnoseOpenCudaDefault() {
  OpenCudaDiagnostics od;

  const auto ort = studiocast::onnx::OrtSession::QueryRuntimeInfo();
  od.onnxruntime_version = ort.version;
  od.onnxruntime_providers = ort.providers;
  od.onnxruntime_cuda_provider_present = ort.cuda_provider_present;
  od.onnxruntime_tensorrt_provider_present = ort.tensorrt_provider_present;
  od.onnxruntime_cpu_provider_present = ort.cpu_provider_present;
  od.onnxruntime_cuda_ep_v2_build = ort.cuda_ep_v2_build;
  od.onnxruntime_library_path = ort.library_path;
  od.onnxruntime_warnings = ort.warnings;
  od.tensorrt_supported = studiocast::onnx::OrtBuildHasTensorRtEpV2();
  od.tensorrt_available =
      HasProvider(od.onnxruntime_providers, "TensorrtExecutionProvider");
  od.tensorrt_requested = OpenCudaTensorRtRequestedFromEnv();
  od.tensorrt_cache_path =
      studiocast::onnx::DefaultTensorRtCachePath(/*cuda_device_id=*/0)
          .string();
  if (!od.tensorrt_requested) {
    od.tensorrt_status = "not_requested";
  } else if (!od.tensorrt_supported) {
    od.tensorrt_status = "unsupported_in_build";
  } else if (!od.tensorrt_available) {
    od.tensorrt_status = "provider_unavailable";
  } else {
    od.tensorrt_status = "available";
  }

  const auto reg = studiocast::open_video::ModelPackRegistry::ScanDefault();
  [[maybe_unused]] bool has_matting_model = false;
  std::set<std::string> installed_model_ids;
  od.default_model_id = [&reg]() -> std::string {
    if (reg.Find("matting", "modnet-webnn-256-fp32"))
      return "modnet-webnn-256-fp32";
    if (reg.Find("matting", "birefnet_lite"))
      return "birefnet_lite";
    for (const auto &m : reg.ListModels()) {
      if (m.task == "matting")
        return m.id;
    }
    return {};
  }();
  for (const auto &m : reg.ListModels()) {
    if (m.task == "matting")
      has_matting_model = true;
    installed_model_ids.insert(m.id);
    od.installed_models.push_back(m.id);
    OpenCudaDiagnostics::ModelInfo mi;
    mi.id = m.id;
    mi.display_name = m.display_name;
    mi.task = m.task;
    if (m.matting) {
      mi.width = m.matting->input.width;
      mi.height = m.matting->input.height;
    }
    od.models.push_back(std::move(mi));
  }
  for (const auto &[id, reason] : reg.Problems()) {
    if (LooksLikeDuplicateInstalledModelProblem(id, reason,
                                                installed_model_ids)) {
      continue;
    }
    od.missing_models[id] = reason;
  }

  const auto modelsRoot = studiocast::util::StudioCastModelsDir();
  const auto openVideoRoot =
      modelsRoot.empty()
          ? std::string("~/.local/share/studiocast/models/open_video")
          : (modelsRoot / "open_video").string();

  od.install_hints.push_back(std::string("Model packs: ") + openVideoRoot +
                             "/<subject>/<pack_dir>/");
  od.install_hints.push_back("Example: " + openVideoRoot +
                             "/matting/Good Quality/model.json");
  od.install_hints.push_back(
      "Source builds: run ./scripts/install.sh open-video-models to install "
      "curated Open Video packs.");
  od.install_hints.push_back(
      "Each pack must contain: model.json, model.onnx, LICENSE.txt");

  const auto block_open_cuda_effects = [&](const char *reason_code) {
    od.blocked_effects[std::string(
        studiocast::video::effects::contract::kEffectIdVideoNoiseRemoval)] =
        reason_code;
    od.blocked_effects[std::string(
        studiocast::video::effects::contract::kEffectIdVirtualBackgroundBlur)] =
        reason_code;
    od.blocked_effects[std::string(studiocast::video::effects::contract::
                                       kEffectIdVirtualBackgroundRemove)] =
        reason_code;
    od.blocked_effects[std::string(studiocast::video::effects::contract::
                                       kEffectIdVirtualBackgroundReplace)] =
        reason_code;
    od.blocked_effects[std::string(
        studiocast::video::effects::contract::kEffectIdAutoFrame)] =
        reason_code;
    od.blocked_effects[std::string(
        studiocast::video::effects::contract::kEffectIdVirtualKeyLight)] =
        reason_code;
  };

  [[maybe_unused]] const auto block_open_cuda_matting_effects =
      [&](const char *reason_code) {
    od.blocked_effects[std::string(
        studiocast::video::effects::contract::kEffectIdVirtualBackgroundBlur)] =
        reason_code;
    od.blocked_effects[std::string(studiocast::video::effects::contract::
                                       kEffectIdVirtualBackgroundRemove)] =
        reason_code;
    od.blocked_effects[std::string(studiocast::video::effects::contract::
                                       kEffectIdVirtualBackgroundReplace)] =
        reason_code;
    od.blocked_effects[std::string(
        studiocast::video::effects::contract::kEffectIdAutoFrame)] =
        reason_code;
    od.blocked_effects[std::string(
        studiocast::video::effects::contract::kEffectIdVirtualKeyLight)] =
        reason_code;
  };

#if !STUDIOCAST_ENABLE_OPEN_CUDA
  od.ok = false;
  block_open_cuda_effects("disabled_in_build");
  od.install_hints.push_back("Open CUDA backend is disabled in this build.");
  od.install_hints.push_back("Rebuild with -DSTUDIOCAST_ENABLE_OPEN_CUDA=ON "
                             "(requires ONNX Runtime + CUDA EP).");
#elif STUDIOCAST_HAVE_ONNXRUNTIME
  // CUDA driver/device gate. This avoids repeatedly attempting to start the
  // pipeline only to fail deep in Open CUDA initialization when no NVIDIA
  // driver/GPU is available.
  bool cuda_ok = false;
  std::string cuda_err;
  {
    studiocast::cuda::CudaDriverApi cuda;
    std::string e;
    if (!cuda.Initialize(&e)) {
      cuda_err = e.empty() ? std::string("CUDA driver API not available.") : e;
    } else {
      od.cuda_driver_api_available = true;

      if (cuda.f().cuDriverGetVersion) {
        int version = 0;
        const auto st = cuda.f().cuDriverGetVersion(&version);
        if (st == studiocast::cuda::CUDA_SUCCESS) {
          od.cuda_driver_version = version;
        } else if (od.cuda_driver_error.empty()) {
          od.cuda_driver_error =
              "cuDriverGetVersion failed: " + cuda.StatusToString(st);
        }
      }

      if (cuda.f().cuDeviceGetCount) {
        int count = 0;
        const auto st = cuda.f().cuDeviceGetCount(&count);
        if (st == studiocast::cuda::CUDA_SUCCESS) {
          od.cuda_device_count = count;
        } else if (od.cuda_driver_error.empty()) {
          od.cuda_driver_error =
              "cuDeviceGetCount failed: " + cuda.StatusToString(st);
        }
      }

      if (!cuda.EnsureContext(&e)) {
        od.cuda_context_error =
            e.empty() ? std::string("Failed to ensure CUDA context.") : e;
        cuda_err = od.cuda_context_error;
      } else {
        od.cuda_context_available = true;
        cuda_ok = true;
      }
    }
  }

  if (!cuda_ok) {
    od.ok = false;
    block_open_cuda_effects("cuda_unavailable");
    if (od.cuda_driver_error.empty() && !cuda_err.empty() &&
        !od.cuda_driver_api_available) {
      od.cuda_driver_error = cuda_err;
    }
    od.install_hints.push_back(std::string("CUDA not available: ") + cuda_err);
  } else {
    od.ok = true;

    // Open CUDA Video Noise Removal has a CUDA-kernel fallback and must not be
    // blocked only because the ORT CUDA EP is unavailable.
    od.available_effects.push_back(std::string(
        studiocast::video::effects::contract::kEffectIdVideoNoiseRemoval));

    if (!od.onnxruntime_cuda_provider_present) {
      block_open_cuda_matting_effects(
          "onnxruntime_cuda_provider_unavailable");
      od.install_hints.push_back(
          "ONNX Runtime CUDAExecutionProvider is not available; "
          "matting-based Open CUDA effects require the ORT CUDA provider.");
    } else if (!has_matting_model) {
      // Segmentation/matting-based effects require at least one usable model
      // pack.
      block_open_cuda_matting_effects("missing_model_packs");
      od.install_hints.push_back(
          "No usable Open Video matting model packs were found (required for "
          "segmentation-based effects). Video Noise Removal can still run "
          "without packs.");
    } else {
      od.available_effects.push_back(
          std::string(studiocast::video::effects::contract::
                          kEffectIdVirtualBackgroundBlur));
      od.available_effects.push_back(
          std::string(studiocast::video::effects::contract::
                          kEffectIdVirtualBackgroundRemove));
      od.available_effects.push_back(
          std::string(studiocast::video::effects::contract::
                          kEffectIdVirtualBackgroundReplace));
      od.available_effects.push_back(std::string(
          studiocast::video::effects::contract::kEffectIdAutoFrame));
      od.available_effects.push_back(std::string(
          studiocast::video::effects::contract::kEffectIdVirtualKeyLight));
    }
  }
#else
  od.ok = false;
  block_open_cuda_effects("onnxruntime_not_found");
  od.install_hints.push_back(
      "Open CUDA backend is disabled in this build (ONNX Runtime not found). ");
  od.install_hints.push_back("Rebuild with -DSTUDIOCAST_ENABLE_OPEN_CUDA=ON "
                             "and ensure ONNXRUNTIME_ROOT is set.");
#endif

  return od;
}

} // namespace studiocast::open_cuda
