#include <filesystem>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "core/config/settings.h"
#include "core/maxine/effects/vfx_background_blur_effect.h"
#include "core/maxine/effects/vfx_green_screen_effect.h"
#include "core/maxine/nvcv_api.h"
#include "core/maxine/vfx_api.h"
#include "core/probe/probe.h"
#include "core/util/xdg.h"
#include "core/video/camera_pipeline.h"
#include "core/video/convert.h"
#include "core/video/effects/broadcast_effects.h"
#include "core/video/v4l2_capture.h"

namespace fs = std::filesystem;

static void Usage(const char *argv0) {
  std::cout << "StudioCast Maxine Helper\n\n"
            << "Usage:\n"
            << "  " << argv0 << " paths\n"
            << "  " << argv0 << " init\n"
            << "  " << argv0 << " doctor\n"
            << "  " << argv0 << " gpu list\n"
            << "  " << argv0 << " gpu select --auto\n"
            << "  " << argv0 << " gpu select --index <N>\n"
            << "  " << argv0 << " gpu select --uuid <GPU-UUID>\n"
            << "  " << argv0 << " install-hints\n"
            << "  " << argv0
            << " greenscreen-smoke [--device /dev/video0] [--width N] "
               "[--height N] [--fps N]"
            << " [--frames N] [--mode N] [--temporal 0|1] [--model-dir PATH]\n";
}

static void FillSyntheticBgr(std::uint8_t *bgr, int width, int height,
                             std::size_t stride) {
  // Default background: bright green.
  for (int y = 0; y < height; ++y) {
    auto *row = bgr + static_cast<std::size_t>(y) * stride;
    for (int x = 0; x < width; ++x) {
      row[x * 3 + 0] = 0;   // B
      row[x * 3 + 1] = 255; // G
      row[x * 3 + 2] = 0;   // R
    }
  }

  // Foreground rectangle: red.
  const int x0 = width / 4;
  const int x1 = (width * 3) / 4;
  const int y0 = height / 4;
  const int y1 = (height * 3) / 4;
  for (int y = y0; y < y1; ++y) {
    auto *row = bgr + static_cast<std::size_t>(y) * stride;
    for (int x = x0; x < x1; ++x) {
      row[x * 3 + 0] = 0;   // B
      row[x * 3 + 1] = 0;   // G
      row[x * 3 + 2] = 255; // R
    }
  }
}

static void Rgb24ToBgr24(const std::uint8_t *rgb, std::uint8_t *bgr, int width,
                         int height, std::size_t src_stride,
                         std::size_t dst_stride) {
  for (int y = 0; y < height; ++y) {
    const auto *s = rgb + static_cast<std::size_t>(y) * src_stride;
    auto *d = bgr + static_cast<std::size_t>(y) * dst_stride;
    for (int x = 0; x < width; ++x) {
      d[x * 3 + 0] = s[x * 3 + 2];
      d[x * 3 + 1] = s[x * 3 + 1];
      d[x * 3 + 2] = s[x * 3 + 0];
    }
  }
}

static int ParseIntArg(const char *s, int default_value) {
  if (!s)
    return default_value;
  try {
    return std::stoi(s);
  } catch (...) {
    return default_value;
  }
}

static void PrintGpus(const studiocast::probe::Report &rep) {
  if (rep.gpus.empty()) {
    std::cout << "No GPUs detected via nvidia-smi.\n";
    return;
  }

  for (const auto &g : rep.gpus) {
    std::cout << "[" << g.index << "] " << g.name;
    if (!g.uuid.empty())
      std::cout << " (" << g.uuid << ")";
    if (g.compute_cap)
      std::cout << " cc " << *g.compute_cap;
    std::cout << (g.likely_supported ? " [supported]" : " [unsupported]");
    if (g.maxine_gpu_arg)
      std::cout << " (maxine --gpu " << *g.maxine_gpu_arg << ")";
    std::cout << "\n";
  }
}

static std::set<std::string>
UniqueMaxineGpuArgs(const studiocast::probe::Report &rep) {
  std::set<std::string> out;
  for (const auto &g : rep.gpus) {
    if (!g.likely_supported)
      continue;
    if (g.maxine_gpu_arg)
      out.insert(*g.maxine_gpu_arg);
  }
  return out;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    Usage(argv[0]);
    return 1;
  }

  const std::string cmd = argv[1];

  const fs::path base = studiocast::util::StudioCastMaxineDir();
  const fs::path vfx = studiocast::util::DefaultVfxRoot();
  const fs::path ar = studiocast::util::DefaultArRoot();
  const fs::path afx = studiocast::util::DefaultAfxRoot();

  if (cmd == "paths") {
    std::cout << "StudioCast Paths\n";
    std::cout << "  Settings: " << studiocast::config::SettingsPath().string()
              << "\n";
    std::cout << "  Maxine base: " << base.string() << "\n";
    std::cout << "  VFX : " << vfx.string() << "\n";
    std::cout << "  AR  : " << ar.string() << "\n";
    std::cout << "  AFX : " << afx.string() << "\n";
    return 0;
  }

  if (cmd == "init") {
    std::error_code ec;
    fs::create_directories(base, ec);
    if (ec) {
      std::cerr << "Failed to create: " << base.string() << "\n";
      std::cerr << "Error: " << ec.message() << "\n";
      return 2;
    }
    std::cout << "Created (or already existed): " << base.string() << "\n";
    std::cout << "Expected SDK roots:\n";
    std::cout << "  " << vfx.string() << "\n";
    std::cout << "  " << ar.string() << "\n";
    std::cout << "  " << afx.string() << "\n";
    return 0;
  }

  if (cmd == "doctor") {
    const auto rep = studiocast::probe::Run(false);
    std::cout << rep.ToText() << "\n";
    return rep.AllChecksPassed() ? 0 : 3;
  }

  if (cmd == "gpu") {
    if (argc < 3) {
      Usage(argv[0]);
      return 1;
    }

    const std::string sub = argv[2];
    if (sub == "list") {
      const auto rep = studiocast::probe::Run(false);
      PrintGpus(rep);
      std::cout << "\nSelected GPU policy: " << rep.gpu_selection_mode << "\n";
      if (rep.selected_gpu_index) {
        std::cout << "Selected GPU index: " << *rep.selected_gpu_index << "\n";
      }
      if (!rep.selected_gpu_uuid.empty()) {
        std::cout << "Selected GPU uuid: " << rep.selected_gpu_uuid << "\n";
      }
      return 0;
    }

    if (sub == "select") {
      studiocast::config::Settings s;

      // Default: keep existing settings then modify.
      s = studiocast::config::LoadSettings();

      bool changed = false;
      for (int i = 3; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--auto") {
          s.gpu.mode = studiocast::config::GpuSelectMode::Auto;
          s.gpu.index.reset();
          s.gpu.uuid.clear();
          changed = true;
        } else if (arg == "--index" && i + 1 < argc) {
          s.gpu.mode = studiocast::config::GpuSelectMode::Index;
          s.gpu.index = std::atoi(argv[i + 1]);
          s.gpu.uuid.clear();
          changed = true;
          ++i;
        } else if (arg == "--uuid" && i + 1 < argc) {
          s.gpu.mode = studiocast::config::GpuSelectMode::Uuid;
          s.gpu.uuid = argv[i + 1];
          s.gpu.index.reset();
          changed = true;
          ++i;
        }
      }

      if (!changed) {
        std::cerr << "No selection provided.\n";
        return 2;
      }

      std::string err;
      if (!studiocast::config::SaveSettings(s, &err)) {
        std::cerr << "Failed to save settings: " << err << "\n";
        return 3;
      }

      std::cout << "Saved GPU selection to: "
                << studiocast::config::SettingsPath().string() << "\n";
      const auto rep = studiocast::probe::Run(false);
      std::cout << "\n" << rep.ToText() << "\n";
      return rep.AllChecksPassed() ? 0 : 4;
    }

    Usage(argv[0]);
    return 1;
  }

  if (cmd == "install-hints") {
    const auto rep = studiocast::probe::Run(false);
    const auto args = UniqueMaxineGpuArgs(rep);

    std::cout << "StudioCast Maxine Install Hints\n\n";
    std::cout << "Maxine base:\n  " << base.string() << "\n\n";

    std::cout << "GPU policy:\n";
    std::cout << "  settings: " << studiocast::config::SettingsPath().string()
              << "\n";
    std::cout << "  mode: " << rep.gpu_selection_mode << "\n";
    if (rep.selected_gpu_index)
      std::cout << "  selected index: " << *rep.selected_gpu_index << "\n";
    if (!rep.selected_gpu_uuid.empty())
      std::cout << "  selected uuid: " << rep.selected_gpu_uuid << "\n\n";

    std::cout << "Detected GPUs:\n";
    PrintGpus(rep);
    std::cout << "\n";

    std::cout << "VFX core (extract so that '" << vfx.string()
              << "' exists):\n";
    std::cout << "  mkdir -p \"" << base.string() << "\"\n";
    std::cout << "  tar -xvf NVIDIA_VFX_SDK_linux_<version>.tar.gz -C \""
              << base.string() << "\"\n\n";

    std::cout << "AR core (extract so that '" << ar.string() << "' exists):\n";
    std::cout << "  mkdir -p \"" << base.string() << "\"\n";
    std::cout << "  tar -xvf NVIDIA_AR_SDK_linux_<version>.tar.gz -C \""
              << base.string() << "\"\n\n";

    std::cout << "AFX core (create '" << afx.string() << "'):\n";
    std::cout << "  mkdir -p \"" << base.string() << "\"\n";
    std::cout << "  cd \"" << base.string() << "\"\n";
    std::cout << "  tar xvf --one-top-level Audio_Effects_SDK.tar.gz\n\n";

    std::cout << "Library naming note:\n";
    std::cout << "  Current Linux Maxine SDKs typically ship `libVideoFX.so` "
                 "(VFX) and `libnvARPose.so` (AR).\n";
    std::cout << "  StudioCast also accepts legacy aliases such as "
                 "`libnvvfx.so`, `libNvVFX.so`, `libnvar.so`, and "
                 "`libNvAR.so`.\n\n";

    if (args.empty()) {
      std::cout << "VFX/AR feature install:\n";
      std::cout
          << "  No supported GPUs with known --gpu mapping were detected.\n";
      std::cout << "  Run this on a Tensor Core GPU machine (Turing+).\n\n";
    } else {
      std::cout
          << "VFX/AR feature install (run once per unique --gpu value):\n";
      std::cout << "  export NGC_CLI_API_KEY=\"<your_api_key>\"\n";
      for (const auto &a : args) {
        std::cout << "  cd \"" << vfx.string()
                  << "/features\" && ./install_feature.sh --gpu " << a
                  << " --feature all --ngc-org nvidia --ngc-team maxine\n";
        std::cout << "  cd \"" << ar.string()
                  << "/features\" && ./install_feature.sh --gpu " << a
                  << " --feature all --ngc-org nvidia --ngc-team maxine\n";
      }
      std::cout << "\n";
    }

    std::cout << "AFX features (MVP: AEC + Superres):\n";
    std::cout << "  export NGC_API_KEY=\"<your_api_key>\"\n";
    std::cout << "  cd \"" << afx.string()
              << "/features\" && ./download_features.sh --effects "
                 "superres-16k_to_48k,superres-8k_to_16k,aec-16k,aec-48k\n";
    std::cout << "\n";
    std::cout
        << "Optional AFX effects (noise removal / room echo / studio voice):\n";
    std::cout
        << "  cd \"" << afx.string()
        << "/features\" && ./download_features.sh --effects "
           "denoiser-48k,dereverb-48k,dereverb_denoiser-48k,studio_voice-48k\n";

    return 0;
  }

  if (cmd == "greenscreen-smoke") {
    // Defaults.
    std::string device = "/dev/video0";
    int width = 640;
    int height = 480;
    int fps = 30;
    int frames = 1;
    std::uint32_t mode = 0;
    bool temporal = true;
    fs::path model_dir = vfx / "models";

    for (int i = 2; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--device" && i + 1 < argc) {
        device = argv[++i];
      } else if (a == "--width" && i + 1 < argc) {
        width = ParseIntArg(argv[++i], width);
      } else if (a == "--height" && i + 1 < argc) {
        height = ParseIntArg(argv[++i], height);
      } else if (a == "--fps" && i + 1 < argc) {
        fps = ParseIntArg(argv[++i], fps);
      } else if (a == "--frames" && i + 1 < argc) {
        frames = ParseIntArg(argv[++i], frames);
      } else if (a == "--mode" && i + 1 < argc) {
        mode = static_cast<std::uint32_t>(
            ParseIntArg(argv[++i], static_cast<int>(mode)));
      } else if (a == "--temporal" && i + 1 < argc) {
        temporal = ParseIntArg(argv[++i], temporal ? 1 : 0) != 0;
      } else if (a == "--model-dir" && i + 1 < argc) {
        model_dir = argv[++i];
      }
    }

    studiocast::maxine::vfx::VfxApi vfx_api;
    studiocast::maxine::NvcvApi nvcv_api;

    std::string err;
    if (!vfx_api.Initialize(&err)) {
      std::cerr << "Maxine VFX runtime unavailable: " << err << "\n";
      std::cerr
          << "Hint: run '" << argv[0]
          << " install-hints' and ensure VFX SDK + features are installed.\n";
      return 3;
    }
    if (!nvcv_api.Initialize(
            studiocast::maxine::NvcvApi::Requirement::VfxCompat, &err)) {
      std::cerr << "NvCVImage runtime unavailable: " << err << "\n";
      return 4;
    }

    studiocast::maxine::effects::VfxGreenScreenEffect gs(&vfx_api, &nvcv_api,
                                                         model_dir);
    studiocast::video::effects::BroadcastCameraEffects settings;
    settings.virtual_background.greenscreen_mode = mode;
    settings.virtual_background.greenscreen_temporal = temporal;
    if (!gs.Configure(settings, &err)) {
      std::cerr << "Configure failed: " << err << "\n";
      return 5;
    }
    if (!gs.Initialize(&err)) {
      std::cerr << "Initialize failed: " << err << "\n";
      return 6;
    }

    // Attempt to capture from V4L2; fall back to synthetic.
    studiocast::video::V4l2Capture cap;
    bool have_camera = false;
    if (cap.Open(device, width, height, fps,
                 studiocast::video::CapturePixelFormat::yuyv,
                 /*prefer_mjpeg=*/true, &err)) {
      have_camera = true;
      width = cap.Actual().width;
      height = cap.Actual().height;
    } else {
      std::cerr << "Warning: failed to open capture device '" << device
                << "': " << err << "\n";
      std::cerr << "         using a synthetic frame instead.\n";
    }

    const std::size_t rgb_stride = static_cast<std::size_t>(width) * 3u;
    std::vector<std::uint8_t> rgb(static_cast<std::size_t>(height) *
                                  rgb_stride);
    std::vector<std::uint8_t> bgr(static_cast<std::size_t>(height) *
                                  rgb_stride);

    // CPU BGR image view (wraps `bgr` memory).
    studiocast::maxine::NvCVImage cpu_bgr{};
    if (!nvcv_api.f().NvCVImage_Init) {
      std::cerr
          << "NvCVImage_Init missing from NvCVImage runtime (unexpected).\n";
      return 7;
    }
    {
      const auto st = nvcv_api.f().NvCVImage_Init(
          &cpu_bgr, static_cast<unsigned>(width), static_cast<unsigned>(height),
          static_cast<int>(rgb_stride), bgr.data(),
          studiocast::maxine::NVCV_BGR, studiocast::maxine::NVCV_U8,
          studiocast::maxine::NVCV_CHUNKY, studiocast::maxine::NVCV_CPU);
      if (st != studiocast::maxine::NVCV_SUCCESS) {
        std::cerr << "NvCVImage_Init(cpu BGR) failed: " << st << "\n";
        return 8;
      }
    }

    // GPU input image (allocated).
    studiocast::maxine::NvCVImage gpu_bgr{};
    {
      const auto st = nvcv_api.f().NvCVImage_Alloc(
          &gpu_bgr, static_cast<unsigned>(width), static_cast<unsigned>(height),
          studiocast::maxine::NVCV_BGR, studiocast::maxine::NVCV_U8,
          studiocast::maxine::NVCV_CHUNKY, studiocast::maxine::NVCV_GPU, 0);
      if (st != studiocast::maxine::NVCV_SUCCESS) {
        std::cerr << "NvCVImage_Alloc(gpu BGR) failed: " << st << "\n";
        return 9;
      }
    }

    // CPU matte for readback.
    studiocast::maxine::NvCVImage cpu_matte{};
    {
      const auto st = nvcv_api.f().NvCVImage_Alloc(
          &cpu_matte, static_cast<unsigned>(width),
          static_cast<unsigned>(height), studiocast::maxine::NVCV_A,
          studiocast::maxine::NVCV_U8, studiocast::maxine::NVCV_CHUNKY,
          studiocast::maxine::NVCV_CPU, 0);
      if (st != studiocast::maxine::NVCV_SUCCESS) {
        std::cerr << "NvCVImage_Alloc(cpu matte) failed: " << st << "\n";
        (void)nvcv_api.f().NvCVImage_Dealloc(&gpu_bgr);
        return 10;
      }
    }

    for (int fi = 0; fi < frames; ++fi) {
      if (have_camera) {
        studiocast::video::CapturedFrameView fv;
        if (!cap.AcquireFrame(&fv, /*timeout_ms=*/1000, &err)) {
          std::cerr << "Capture AcquireFrame failed: " << err << "\n";
          have_camera = false;
          FillSyntheticBgr(bgr.data(), width, height, rgb_stride);
        } else {
          // Convert YUYV -> RGB24.
          studiocast::video::YuyvToRgb24(fv.data, width, height,
                                         cap.Actual().bytes_per_line,
                                         rgb.data(), rgb_stride);
          (void)cap.ReleaseFrame(fv, &err);
          Rgb24ToBgr24(rgb.data(), bgr.data(), width, height, rgb_stride,
                       rgb_stride);
        }
      } else {
        FillSyntheticBgr(bgr.data(), width, height, rgb_stride);
      }

      // Upload CPU->GPU.
      const auto up = nvcv_api.f().NvCVImage_Transfer(
          &cpu_bgr, &gpu_bgr, 1.0f, gs.cuda_stream(), nullptr);
      if (up != studiocast::maxine::NVCV_SUCCESS) {
        std::cerr << "NvCVImage_Transfer(cpu->gpu) failed: " << up << "\n";
        break;
      }

      studiocast::video::GpuFrame frame;
      frame.width = width;
      frame.height = height;
      frame.nvcv_gpu = &gpu_bgr;
      frame.cuda_stream = gs.cuda_stream();

      err.clear();
      const auto st = gs.Process(frame, &err);
      if (st != studiocast::maxine::NVCV_SUCCESS) {
        std::cerr << "Green Screen Process failed: "
                  << (err.empty() ? std::to_string(st) : err) << "\n";
        break;
      }

      const auto *matte_gpu = gs.MatteGpu();
      if (!matte_gpu) {
        std::cerr << "No matte produced.\n";
        break;
      }

      // Download GPU->CPU.
      const auto down = nvcv_api.f().NvCVImage_Transfer(
          matte_gpu, &cpu_matte, 1.0f, gs.cuda_stream(), nullptr);
      if (down != studiocast::maxine::NVCV_SUCCESS) {
        std::cerr << "NvCVImage_Transfer(gpu->cpu) failed: " << down << "\n";
        break;
      }

      // Compute basic matte statistics.
      const auto *p = static_cast<const std::uint8_t *>(cpu_matte.pixels);
      if (!p) {
        std::cerr << "Matte CPU image has null pixels.\n";
        break;
      }
      std::uint64_t sum = 0;
      std::uint8_t mn = 255;
      std::uint8_t mx = 0;
      for (int y = 0; y < height; ++y) {
        const auto *row = p + static_cast<std::size_t>(y) *
                                  static_cast<std::size_t>(cpu_matte.pitch);
        for (int x = 0; x < width; ++x) {
          const std::uint8_t v = row[x];
          sum += v;
          if (v < mn)
            mn = v;
          if (v > mx)
            mx = v;
        }
      }
      const double mean =
          static_cast<double>(sum) / static_cast<double>(width * height);
      std::cout << "Frame " << fi
                << ": matte stats min=" << static_cast<int>(mn)
                << " max=" << static_cast<int>(mx) << " mean=" << mean << "\n";
    }

    (void)nvcv_api.f().NvCVImage_Dealloc(&cpu_matte);
    (void)nvcv_api.f().NvCVImage_Dealloc(&gpu_bgr);
    cap.Close();
    return 0;
  }

  if (cmd == "background-blur-smoke") {
    // Defaults.
    std::string device = "/dev/video0";
    int width = 640;
    int height = 480;
    int fps = 30;
    int frames = 1;
    std::uint32_t mode = 0;
    bool temporal = true;
    int strength = 32; // UI knob [1..64]
    fs::path model_dir = vfx / "models";

    for (int i = 2; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--device" && i + 1 < argc) {
        device = argv[++i];
      } else if (a == "--width" && i + 1 < argc) {
        width = ParseIntArg(argv[++i], width);
      } else if (a == "--height" && i + 1 < argc) {
        height = ParseIntArg(argv[++i], height);
      } else if (a == "--fps" && i + 1 < argc) {
        fps = ParseIntArg(argv[++i], fps);
      } else if (a == "--frames" && i + 1 < argc) {
        frames = ParseIntArg(argv[++i], frames);
      } else if (a == "--mode" && i + 1 < argc) {
        mode = static_cast<std::uint32_t>(
            ParseIntArg(argv[++i], static_cast<int>(mode)));
      } else if (a == "--temporal" && i + 1 < argc) {
        temporal = ParseIntArg(argv[++i], temporal ? 1 : 0) != 0;
      } else if (a == "--strength" && i + 1 < argc) {
        strength = ParseIntArg(argv[++i], strength);
      } else if (a == "--model-dir" && i + 1 < argc) {
        model_dir = argv[++i];
      }
    }

    studiocast::maxine::vfx::VfxApi vfx_api;
    studiocast::maxine::NvcvApi nvcv_api;

    std::string err;
    if (!vfx_api.Initialize(&err)) {
      std::cerr << "Maxine VFX runtime unavailable: " << err << "\n";
      std::cerr
          << "Hint: run '" << argv[0]
          << " install-hints' and ensure VFX SDK + features are installed.\n";
      return 3;
    }
    if (!nvcv_api.Initialize(
            studiocast::maxine::NvcvApi::Requirement::VfxCompat, &err)) {
      std::cerr << "NvCVImage runtime unavailable: " << err << "\n";
      return 4;
    }

    studiocast::maxine::effects::VfxGreenScreenEffect gs(&vfx_api, &nvcv_api,
                                                         model_dir);
    studiocast::maxine::effects::VfxBackgroundBlurEffect bgblur(
        &vfx_api, &nvcv_api, model_dir);

    studiocast::video::effects::BroadcastCameraEffects settings;
    settings.virtual_background.mode =
        studiocast::video::effects::VirtualBackgroundMode::blur;
    settings.virtual_background.greenscreen_mode = mode;
    settings.virtual_background.greenscreen_temporal = temporal;
    settings.virtual_background.strength = strength;

    if (!gs.Configure(settings, &err) || !bgblur.Configure(settings, &err)) {
      std::cerr << "Configure failed: " << err << "\n";
      return 5;
    }
    if (!gs.Initialize(&err)) {
      std::cerr << "Green Screen Initialize failed: " << err << "\n";
      return 6;
    }
    if (!bgblur.Initialize(&err)) {
      std::cerr << "Background Blur Initialize failed: " << err << "\n";
      return 7;
    }

    // Attempt to capture from V4L2; fall back to synthetic.
    studiocast::video::V4l2Capture cap;
    bool have_camera = false;
    if (cap.Open(device, width, height, fps,
                 studiocast::video::CapturePixelFormat::yuyv,
                 /*prefer_mjpeg=*/true, &err)) {
      have_camera = true;
      width = cap.Actual().width;
      height = cap.Actual().height;
    } else {
      std::cerr << "Warning: failed to open capture device '" << device
                << "': " << err << "\n";
      std::cerr << "         using a synthetic frame instead.\n";
    }

    const std::size_t rgb_stride = static_cast<std::size_t>(width) * 3u;
    std::vector<std::uint8_t> rgb(static_cast<std::size_t>(height) *
                                  rgb_stride);
    std::vector<std::uint8_t> bgr(static_cast<std::size_t>(height) *
                                  rgb_stride);
    std::vector<std::uint8_t> out_bgr(static_cast<std::size_t>(height) *
                                      rgb_stride);

    // CPU BGR image view (wraps `bgr` memory).
    if (!nvcv_api.f().NvCVImage_Init) {
      std::cerr
          << "NvCVImage_Init missing from NvCVImage runtime (unexpected).\n";
      return 8;
    }

    studiocast::maxine::NvCVImage cpu_bgr{};
    {
      const auto st = nvcv_api.f().NvCVImage_Init(
          &cpu_bgr, static_cast<unsigned>(width), static_cast<unsigned>(height),
          static_cast<int>(rgb_stride), bgr.data(),
          studiocast::maxine::NVCV_BGR, studiocast::maxine::NVCV_U8,
          studiocast::maxine::NVCV_CHUNKY, studiocast::maxine::NVCV_CPU);
      if (st != studiocast::maxine::NVCV_SUCCESS) {
        std::cerr << "NvCVImage_Init(cpu BGR) failed: " << st << "\n";
        return 9;
      }
    }

    studiocast::maxine::NvCVImage cpu_out_bgr{};
    {
      const auto st = nvcv_api.f().NvCVImage_Init(
          &cpu_out_bgr, static_cast<unsigned>(width),
          static_cast<unsigned>(height), static_cast<int>(rgb_stride),
          out_bgr.data(), studiocast::maxine::NVCV_BGR,
          studiocast::maxine::NVCV_U8, studiocast::maxine::NVCV_CHUNKY,
          studiocast::maxine::NVCV_CPU);
      if (st != studiocast::maxine::NVCV_SUCCESS) {
        std::cerr << "NvCVImage_Init(cpu out BGR) failed: " << st << "\n";
        return 10;
      }
    }

    // GPU input image (allocated).
    studiocast::maxine::NvCVImage gpu_bgr{};
    {
      const auto st = nvcv_api.f().NvCVImage_Alloc(
          &gpu_bgr, static_cast<unsigned>(width), static_cast<unsigned>(height),
          studiocast::maxine::NVCV_BGR, studiocast::maxine::NVCV_U8,
          studiocast::maxine::NVCV_CHUNKY, studiocast::maxine::NVCV_GPU, 0);
      if (st != studiocast::maxine::NVCV_SUCCESS) {
        std::cerr << "NvCVImage_Alloc(gpu BGR) failed: " << st << "\n";
        return 11;
      }
    }

    for (int fi = 0; fi < frames; ++fi) {
      if (have_camera) {
        studiocast::video::CapturedFrameView fv;
        if (!cap.AcquireFrame(&fv, /*timeout_ms=*/1000, &err)) {
          std::cerr << "Capture AcquireFrame failed: " << err << "\n";
          have_camera = false;
          FillSyntheticBgr(bgr.data(), width, height, rgb_stride);
        } else {
          studiocast::video::YuyvToRgb24(fv.data, width, height,
                                         cap.Actual().bytes_per_line,
                                         rgb.data(), rgb_stride);
          (void)cap.ReleaseFrame(fv, &err);
          Rgb24ToBgr24(rgb.data(), bgr.data(), width, height, rgb_stride,
                       rgb_stride);
        }
      } else {
        FillSyntheticBgr(bgr.data(), width, height, rgb_stride);
      }

      // Upload CPU->GPU.
      const auto up = nvcv_api.f().NvCVImage_Transfer(
          &cpu_bgr, &gpu_bgr, 1.0f, gs.cuda_stream(), nullptr);
      if (up != studiocast::maxine::NVCV_SUCCESS) {
        std::cerr << "NvCVImage_Transfer(cpu->gpu) failed: " << up << "\n";
        break;
      }

      studiocast::video::GpuFrame frame;
      frame.width = width;
      frame.height = height;
      frame.nvcv_gpu = &gpu_bgr;
      frame.cuda_stream = gs.cuda_stream();

      err.clear();
      const auto st = gs.Process(frame, &err);
      if (st != studiocast::maxine::NVCV_SUCCESS) {
        std::cerr << "Green Screen Process failed: "
                  << (err.empty() ? std::to_string(st) : err) << "\n";
        break;
      }

      const auto *matte_gpu = gs.MatteGpu();
      if (!matte_gpu) {
        std::cerr << "No matte produced.\n";
        break;
      }

      frame.matte_gpu = matte_gpu;

      err.clear();
      const auto st2 = bgblur.Process(frame, &err);
      if (st2 != studiocast::maxine::NVCV_SUCCESS) {
        std::cerr << "Background Blur Process failed: "
                  << (err.empty() ? std::to_string(st2) : err) << "\n";
        break;
      }

      const auto *out_gpu = bgblur.OutputGpu();
      if (!out_gpu) {
        std::cerr << "No output produced.\n";
        break;
      }

      const auto down = nvcv_api.f().NvCVImage_Transfer(
          out_gpu, &cpu_out_bgr, 1.0f, bgblur.cuda_stream(), nullptr);
      if (down != studiocast::maxine::NVCV_SUCCESS) {
        std::cerr << "NvCVImage_Transfer(gpu->cpu) failed: " << down << "\n";
        break;
      }

      // Compute a quick checksum-like stat on the output.
      std::uint64_t sum = 0;
      for (const auto v : out_bgr)
        sum += v;
      const double mean =
          static_cast<double>(sum) / static_cast<double>(out_bgr.size());
      std::cout << "Frame " << fi << ": output mean byte=" << mean << "\n";
    }

    (void)nvcv_api.f().NvCVImage_Dealloc(&gpu_bgr);
    cap.Close();
    return 0;
  }

  Usage(argv[0]);
  return 1;
}
