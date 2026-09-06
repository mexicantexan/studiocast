#include "core/open_video/prepared_cuda_dlib_shape_predictor.h"
#include <cmath>
#include <cstdlib>
#include <dlib/image_processing/shape_predictor.h>
#include <dlib/serialize.h>
#include <iostream>
#include <vector>

using namespace studiocast;
int main() {
  const char *root = std::getenv("STUDIOCAST_DLIB_ORACLE_MODELS_ROOT");
  if (!root) {
    std::cout << "SKIP: set STUDIOCAST_DLIB_ORACLE_MODELS_ROOT for opt-in "
                 "native test\n";
    return 0;
  }
  const std::filesystem::path artifact =
      std::filesystem::path(root) / "face_landmarks" / "Dlib 68 (iBUG 300-W)" /
      "shape_predictor_68_face_landmarks.dat";
  dlib::shape_predictor oracle;
  dlib::deserialize(artifact.string()) >> oracle;
  cuda::CudaDriverApi driver;
  std::string error;
  if (!driver.Initialize(&error) || !driver.EnsureContextForDevice(0, &error)) {
    std::cerr << error << '\n';
    return 1;
  }
  cuda::CUcontext context = nullptr;
  const auto current_status = driver.f().cuCtxGetCurrent(&context);
  if (current_status != cuda::CUDA_SUCCESS || !context ||
      !driver.context_identity() || !driver.context_identity()->Valid()) {
    std::cerr << "CUDA context identity unavailable: status=" << current_status
              << " context=" << (context != nullptr)
              << " identity=" << (driver.context_identity() != nullptr) << '\n';
    return 1;
  }
  cuda::CUstream stream = nullptr;
  if (!driver.CreateStream(&stream, &error)) {
    std::cerr << error << '\n';
    return 1;
  }
  open_video::PreparedCudaDlibBinding binding{
      &driver, *driver.context_identity(), context, stream, 1, 1, 1};
  open_video::PreparedCudaDlibShapePredictor gpu;
  open_video::PreparedCudaDlibSetup setup{binding, root, "dlib_68_ibug_300w",
                                          512u << 20};
  if (!gpu.Prepare(setup, &error)) {
    std::cerr << error << '\n';
    return 1;
  }
  constexpr int w = 257, h = 193;
  std::vector<unsigned char> rgb(w * h * 3);
  cuda::CudaImage image;
  if (!image.Allocate(&driver, w, h, cuda::PixelFormatGpu::rgb_u8, &error))
    return 1;
  const std::array<dlib::rectangle, 5> boxes = {
      dlib::rectangle(7, 11, 180, 170), dlib::rectangle(0, 0, 256, 192),
      dlib::rectangle(51, 27, 205, 178), dlib::rectangle(-13, 9, 150, 181),
      dlib::rectangle(100, -19, 270, 150)};
  float worst = 0;
  std::uint64_t sequence = 1;
  for (int pattern = 0; pattern < 4; ++pattern) {
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        auto *p = &rgb[(y * w + x) * 3];
        p[0] =
            static_cast<unsigned char>((x * 17 + y * 3 + pattern * 41) & 255);
        p[1] =
            static_cast<unsigned char>((x * 5 + y * 29 + pattern * 13) & 255);
        p[2] =
            static_cast<unsigned char>((x * 11 + y * 7 + pattern * 67) & 255);
      }
    image.UploadFromCpuRgb24(&driver, rgb.data(), w * 3, stream, &error);
    dlib::array2d<dlib::rgb_pixel> host(h, w);
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const auto *p = &rgb[(y * w + x) * 3];
        host[y][x] = dlib::rgb_pixel(p[0], p[1], p[2]);
      }
    for (const auto &box : boxes) {
      open_video::PreparedCudaFaceBox fb{
          static_cast<float>(box.left()), static_cast<float>(box.top()),
          static_cast<float>(box.right()), static_cast<float>(box.bottom())};
      if (gpu.Execute(binding, image, fb, sequence++, &error).status !=
          open_video::PreparedCudaDlibStatus::success) {
        std::cerr << error << '\n';
        return 1;
      }
      std::vector<float> actual;
      gpu.resident_landmarks().DownloadToCpuF32(&driver, &actual, stream,
                                                &error);
      driver.StreamSynchronize(stream, &error);
      const auto expected = oracle(host, box);
      float case_worst = 0;
      for (unsigned long i = 0; i < 68; ++i) {
        case_worst = std::max(case_worst,
                              std::fabs(actual[2 * i] - expected.part(i).x()));
        case_worst = std::max(
            case_worst, std::fabs(actual[2 * i + 1] - expected.part(i).y()));
      }
      worst = std::max(worst, case_worst);
    }
  }
  if (worst != 0) {
    std::cerr << "GPU/dlib maximum coordinate delta: " << worst << '\n';
    return 1;
  }
  if (!gpu.Shutdown(&error) || !image.Free(&driver, &error) ||
      !driver.DestroyStream(stream, &error)) {
    std::cerr << error << '\n';
    return 1;
  }
  std::cout << "GPU/dlib maximum coordinate delta: 0 across 20 cases\n";
}
