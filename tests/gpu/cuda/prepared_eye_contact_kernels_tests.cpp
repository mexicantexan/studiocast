#include "core/cuda/kernels/prepared_eye_contact_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <string_view>

namespace {
using namespace studiocast::cuda;
using namespace studiocast::cuda::kernels;

struct State {
  CUcontext current = nullptr;
  CUstream launched_stream = nullptr;
  int loads = 0, resolves = 0, launches = 0, unloads = 0;
  CUresult unload_status = CUDA_SUCCESS;
  std::string last_entry;
};
State *g = nullptr;

bool Check(bool value, std::string_view message) {
  if (!value)
    std::cerr << "[FAIL] " << message << '\n';
  return value;
}
CUresult Init(unsigned) { return CUDA_SUCCESS; }
CUresult Load(CUmodule *m, const void *) {
  ++g->loads;
  *m = reinterpret_cast<CUmodule>(0x10);
  return CUDA_SUCCESS;
}
CUresult Resolve(CUfunction *f, CUmodule, const char *name) {
  ++g->resolves;
  g->last_entry = name;
  *f = reinterpret_cast<CUfunction>(g->resolves == 1 ? 0x20 : 0x21);
  return CUDA_SUCCESS;
}
CUresult Unload(CUmodule) {
  ++g->unloads;
  return g->unload_status;
}
CUresult Launch(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned,
                unsigned, unsigned, CUstream stream, void **, void **) {
  ++g->launches;
  g->launched_stream = stream;
  return CUDA_SUCCESS;
}
CUresult Alloc(CUdeviceptr *p, std::size_t *pitch, std::size_t width,
               std::size_t, unsigned) {
  *p = 0x9000;
  *pitch = width;
  return CUDA_SUCCESS;
}
CUresult Free(CUdeviceptr) { return CUDA_SUCCESS; }
CUresult Copy(const CUDA_MEMCPY2D *, CUstream) { return CUDA_SUCCESS; }
CUresult Memset(CUdeviceptr, unsigned char, std::size_t, CUstream) {
  return CUDA_SUCCESS;
}
CUresult StreamCreate(CUstream *s, unsigned) {
  *s = reinterpret_cast<CUstream>(0x30);
  return CUDA_SUCCESS;
}
CUresult StreamDestroy(CUstream) { return CUDA_SUCCESS; }
CUresult StreamSync(CUstream) { return CUDA_SUCCESS; }
CUresult DeviceCount(int *n) {
  *n = 1;
  return CUDA_SUCCESS;
}
CUresult DeviceGet(CUdevice *d, int n) {
  *d = n;
  return CUDA_SUCCESS;
}
CUresult DeviceUuid(CUuuid *u, CUdevice) {
  std::memset(u, 0, sizeof(*u));
  return CUDA_SUCCESS;
}
CUresult DevicePci(char *p, int n, CUdevice) {
  if (n < 2)
    return 1;
  std::strcpy(p, "0");
  return CUDA_SUCCESS;
}
CUresult PrimaryRetain(CUcontext *c, CUdevice) {
  *c = reinterpret_cast<CUcontext>(0x40);
  return CUDA_SUCCESS;
}
CUresult PrimaryRelease(CUdevice) { return CUDA_SUCCESS; }
CUresult CtxSet(CUcontext c) {
  g->current = c;
  return CUDA_SUCCESS;
}
CUresult CtxGet(CUcontext *c) {
  *c = g->current;
  return CUDA_SUCCESS;
}
CUresult CtxDevice(CUdevice *d) {
  *d = 0;
  return CUDA_SUCCESS;
}

CudaDriverApi Driver(State *state) {
  g = state;
  CudaDriverApi::Functions f;
  f.cuInit = &Init;
  f.cuModuleLoadData = &Load;
  f.cuModuleGetFunction = &Resolve;
  f.cuModuleUnload = &Unload;
  f.cuLaunchKernel = &Launch;
  f.cuMemAllocPitch = &Alloc;
  f.cuMemFree = &Free;
  f.cuMemcpy2DAsync = &Copy;
  f.cuMemsetD8Async = &Memset;
  f.cuStreamCreate = &StreamCreate;
  f.cuStreamDestroy = &StreamDestroy;
  f.cuStreamSynchronize = &StreamSync;
  f.cuDeviceGetCount = &DeviceCount;
  f.cuDeviceGet = &DeviceGet;
  f.cuDeviceGetUuid = &DeviceUuid;
  f.cuDeviceGetPCIBusId = &DevicePci;
  f.cuDevicePrimaryCtxRetain = &PrimaryRetain;
  f.cuDevicePrimaryCtxRelease = &PrimaryRelease;
  f.cuCtxSetCurrent = &CtxSet;
  f.cuCtxGetCurrent = &CtxGet;
  f.cuCtxGetDevice = &CtxDevice;
  CudaDriverApi driver;
  std::string error;
  (void)driver.InitializeForTesting(f, &error);
  return driver;
}

CudaTensor TensorAt(CUdeviceptr p, int c, int h, int w) {
  CudaTensor t;
  t.ptr = p;
  t.pitch = static_cast<std::size_t>(c * h * w) * 4;
  t.bytes = t.pitch;
  t.n = 1;
  t.c = c;
  t.h = h;
  t.w = w;
  return t;
}
CudaImage ImageAt(CUdeviceptr p) {
  CudaImage i;
  i.ptr = p;
  i.pitch = 303;
  i.w = 101;
  i.h = 79;
  i.format = PixelFormatGpu::rgb_u8;
  return i;
}

struct CpuCrop {
  int left, top, width, height;
  bool valid;
};
CpuCrop Crop(const std::vector<float> &marks, int eye, int fw, int fh) {
  const int base = eye == 0 ? 42 : 36;
  const float x0 = marks[2 * base], y0 = marks[2 * base + 1];
  const float x3 = marks[2 * (base + 3)], y3 = marks[2 * (base + 3) + 1];
  const float cx = (x0 + x3) * .5f, cy = (y0 + y3) * .5f;
  const float half = std::abs(x3 - x0) * .75f, box_h = 1.5f * half;
  const int left =
      std::clamp(static_cast<int>(std::floor(cx - half)), 0, fw - 2);
  const int right =
      std::clamp(static_cast<int>(std::ceil(cx + half)), 1, fw - 1);
  const int top = std::clamp(
      static_cast<int>(std::floor(cy - box_h * (7.f / 12.f))), 0, fh - 2);
  const int bottom = std::clamp(
      static_cast<int>(std::ceil(cy + box_h * (5.f / 12.f))), 1, fh - 1);
  return {left, top, std::max(1, right - left), std::max(1, bottom - top),
          right - left >= 2 && bottom - top >= 2};
}
std::uint8_t Byte(float value) {
  return static_cast<std::uint8_t>(std::clamp(
      static_cast<int>(std::floor(std::clamp(value, 0.f, 255.f) + .5f)), 0,
      255));
}
float SampleModel(const std::vector<float> &model, int channel, float x,
                  float y) {
  x = std::clamp(x, 0.f, 63.f);
  y = std::clamp(y, 0.f, 47.f);
  const int x0 = static_cast<int>(std::floor(x));
  const int y0 = static_cast<int>(std::floor(y));
  const int x1 = std::min(63, x0 + 1), y1 = std::min(47, y0 + 1);
  const float fx = x - x0, fy = y - y0;
  const auto at = [&](int px, int py) {
    return static_cast<float>(Byte(
        std::clamp(model[channel * 48 * 64 + py * 64 + px], 0.f, 1.f) * 255.f));
  };
  return (at(x0, y0) * (1 - fx) + at(x1, y0) * fx) * (1 - fy) +
         (at(x0, y1) * (1 - fx) + at(x1, y1) * fx) * fy;
}
std::vector<std::uint8_t> Expected(const std::vector<std::uint8_t> &input,
                                   const std::vector<float> &marks, int width,
                                   int height,
                                   const std::vector<float> &left_model,
                                   const std::vector<float> &right_model,
                                   float strength, bool left_ok,
                                   bool right_ok) {
  auto out = input;
  const CpuCrop crops[2] = {Crop(marks, 0, width, height),
                            Crop(marks, 1, width, height)};
  const std::vector<float> *models[2] = {&left_model, &right_model};
  for (int eye = 0; eye < 2; ++eye) {
    const auto &crop = crops[eye];
    const bool ok = eye == 0 ? left_ok : right_ok;
    if (!ok || !crop.valid || strength <= 0)
      continue;
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < width; ++x) {
        const auto &c = crops[eye];
        if (x >= c.left && y >= c.top && x < c.left + c.width &&
            y < c.top + c.height) {
          const float mx = (x - c.left + .5f) * 64.f / c.width - .5f;
          const float my = (y - c.top + .5f) * 48.f / c.height - .5f;
          for (int channel = 0; channel < 3; ++channel) {
            const auto i =
                static_cast<std::size_t>((y * width + x) * 3 + channel);
            const float corrected =
                Byte(SampleModel(*models[eye], channel, mx, my));
            out[i] = Byte(out[i] + (corrected - out[i]) * strength);
          }
        }
      }
  }
  return out;
}

struct CpuInputs {
  std::vector<float> geometry, left_rgb, right_rgb, left_anchors, right_anchors,
      left_angles, right_angles;
};
float SampleFrame(const std::vector<std::uint8_t> &frame, int fw, int fh,
                  int channel, float x, float y) {
  x = std::clamp(x, 0.f, static_cast<float>(fw - 1));
  y = std::clamp(y, 0.f, static_cast<float>(fh - 1));
  const int x0 = static_cast<int>(std::floor(x)),
            y0 = static_cast<int>(std::floor(y));
  const int x1 = std::min(fw - 1, x0 + 1), y1 = std::min(fh - 1, y0 + 1);
  const float fx = x - x0, fy = y - y0;
  const auto at = [&](int px, int py) {
    return static_cast<float>(frame[(py * fw + px) * 3 + channel]);
  };
  return (at(x0, y0) * (1 - fx) + at(x1, y0) * fx) * (1 - fy) +
         (at(x0, y1) * (1 - fx) + at(x1, y1) * fx) * fy;
}
CpuInputs ExpectedInputs(const std::vector<std::uint8_t> &frame,
                         const std::vector<float> &marks, int fw, int fh,
                         float strength) {
  CpuInputs result;
  result.geometry.resize(10);
  result.left_rgb.resize(3 * 48 * 64);
  result.right_rgb.resize(3 * 48 * 64);
  result.left_anchors.resize(12 * 48 * 64);
  result.right_anchors.resize(12 * 48 * 64);
  result.left_angles = result.right_angles = {0.f, -.10f * strength};
  for (int eye = 0; eye < 2; ++eye) {
    const auto crop = Crop(marks, eye, fw, fh);
    const float geometry[5] = {
        static_cast<float>(crop.left), static_cast<float>(crop.top),
        static_cast<float>(crop.width), static_cast<float>(crop.height),
        crop.valid ? 1.f : 0.f};
    std::copy(std::begin(geometry), std::end(geometry),
              result.geometry.begin() + eye * 5);
    auto &rgb = eye == 0 ? result.left_rgb : result.right_rgb;
    auto &anchors = eye == 0 ? result.left_anchors : result.right_anchors;
    const int base = eye == 0 ? 42 : 36;
    const int left_order[6] = {3, 2, 1, 0, 5, 4};
    for (int c = 0; c < 3; ++c)
      for (int y = 0; y < 48; ++y)
        for (int x = 0; x < 64; ++x) {
          const float sx =
              std::clamp(crop.left + (x + .5f) * crop.width / 64.f - .5f,
                         static_cast<float>(crop.left),
                         static_cast<float>(crop.left + crop.width - 1));
          const float sy =
              std::clamp(crop.top + (y + .5f) * crop.height / 48.f - .5f,
                         static_cast<float>(crop.top),
                         static_cast<float>(crop.top + crop.height - 1));
          rgb[c * 48 * 64 + y * 64 + x] =
              Byte(SampleFrame(frame, fw, fh, c, sx, sy)) / 255.f;
        }
    for (int c = 0; c < 12; ++c) {
      const int point = base + (eye == 0 ? left_order[c / 2] : c / 2);
      const int rx = static_cast<int>(
          std::floor((marks[2 * point] - crop.left) * 64.f / crop.width + .5f));
      const int ry = static_cast<int>(std::floor(
          (marks[2 * point + 1] - crop.top) * 48.f / crop.height + .5f));
      for (int y = 0; y < 48; ++y)
        for (int x = 0; x < 64; ++x)
          anchors[c * 48 * 64 + y * 64 + x] =
              static_cast<float>((c & 1) ? y - ry : x - rx);
    }
  }
  return result;
}

bool Near(const std::vector<float> &actual, const std::vector<float> &expected,
          float tolerance = 1e-6f) {
  if (actual.size() != expected.size())
    return false;
  for (std::size_t i = 0; i < actual.size(); ++i)
    if (std::abs(actual[i] - expected[i]) > tolerance)
      return false;
  return true;
}
float MaxDelta(const std::vector<float> &actual,
               const std::vector<float> &expected) {
  float delta = 0;
  for (std::size_t i = 0; i < std::min(actual.size(), expected.size()); ++i)
    delta = std::max(delta, std::abs(actual[i] - expected[i]));
  return actual.size() == expected.size() ? delta : 1e9f;
}
bool BytesNear(const std::vector<std::uint8_t> &actual,
               const std::vector<std::uint8_t> &expected, int tolerance) {
  if (actual.size() != expected.size())
    return false;
  for (std::size_t i = 0; i < actual.size(); ++i)
    if (std::abs(static_cast<int>(actual[i]) - static_cast<int>(expected[i])) >
        tolerance)
      return false;
  return true;
}

bool FakeContract() {
  State state;
  auto driver = Driver(&state);
  auto context = reinterpret_cast<CUcontext>(0x40);
  auto stream = reinterpret_cast<CUstream>(0x30);
  state.current = context;
  PreparedEyeContactKernels k;
  std::string error;
  bool ok = Check(PrepareEyeContactKernels(driver, context, stream, &k, &error),
                  "prepare") &&
            Check(state.loads == 1 && state.resolves == 2,
                  "prepare must freeze one module/two entries");
  auto image = ImageAt(0x100000);
  auto lm = TensorAt(0x200000, 1, 1, 136), geo = TensorAt(0x210000, 2, 1, 5);
  auto le = TensorAt(0x220000, 3, 48, 64), re = TensorAt(0x230000, 3, 48, 64);
  auto la = TensorAt(0x240000, 12, 48, 64), ra = TensorAt(0x270000, 12, 48, 64);
  auto lang = TensorAt(0x2a0000, 2, 1, 1), rang = TensorAt(0x2b0000, 2, 1, 1);
  PreparedEyeContactInputs inputs{&lm, &geo, &le, &re, &la, &ra, &lang, &rang};
  ok &= Check(LaunchEyeContactInputsPrepared(k, image, inputs, .5f, &error) &&
                  state.launches == 1 && state.launched_stream == stream,
              "input launch/stream");
  auto one_pixel = ImageAt(0x500000);
  one_pixel.w = 1;
  ok &= Check(
      !LaunchEyeContactInputsPrepared(k, one_pixel, inputs, .5f, &error) &&
          state.launches == 1,
      "one-pixel input geometry rejected before launch");
  auto left_out = TensorAt(0x300000, 3, 48, 64),
       right_out = TensorAt(0x310000, 3, 48, 64);
  auto output = ImageAt(0x400000);
  ok &= Check(LaunchEyeContactCompositePrepared(k, image, geo, left_out,
                                                right_out, true, false, .5f,
                                                &output, &error) &&
                  state.launches == 2,
              "composite launch");
  auto wide_span_source = ImageAt(0x100000000ull);
  wide_span_source.w = 2;
  wide_span_source.h = 3;
  wide_span_source.pitch = 0x7fffffffu;
  auto wide_span_output = ImageAt(0x300000000ull);
  wide_span_output.w = 2;
  wide_span_output.h = 3;
  wide_span_output.pitch = 0x7fffffffu;
  ok &= Check(LaunchEyeContactCompositePrepared(
                  k, wide_span_source, geo, left_out, right_out, true, true,
                  .5f, &wide_span_output, &error) &&
                  state.launches == 3,
              "64-bit image row offsets accepted without 32-bit overflow");
  output.ptr = image.ptr + 1;
  ok &= Check(!LaunchEyeContactCompositePrepared(k, image, geo, left_out,
                                                 right_out, true, true, .5f,
                                                 &output, &error) &&
                  state.launches == 3,
              "partial alias rejected");
  output.ptr = 0x400000;
  geo.ptr = left_out.ptr;
  ok &= Check(!LaunchEyeContactCompositePrepared(k, image, geo, left_out,
                                                 right_out, true, true, .5f,
                                                 &output, &error),
              "input tensor alias rejected");
  state.unload_status = 1;
  ok &= Check(!DestroyEyeContactKernelsPrepared(&k, &error) && k.live,
              "failed destroy remains retryable");
  state.unload_status = CUDA_SUCCESS;
  ok &= Check(DestroyEyeContactKernelsPrepared(&k, &error) && !k.live &&
                  state.unloads == 2,
              "destroy retry");
  return ok;
}

int NativeOracle() {
  // This gate precedes CUDA initialization/device discovery by contract.
  if (!std::getenv("STUDIOCAST_REQUIRE_CUDA_EYE_CONTACT_TEST")) {
    std::cout << "[SKIP] native CUDA eye-contact oracle not requested\n";
    return 77;
  }
  CudaDriverApi driver;
  std::string error;
  if (!driver.Initialize(&error) || !driver.EnsureContextForDevice(0, &error)) {
    std::cerr << "[FAIL] required CUDA unavailable: " << error << '\n';
    return 1;
  }
  CUcontext context = nullptr;
  CUstream stream = nullptr;
  if (!driver.f().cuCtxGetCurrent ||
      driver.f().cuCtxGetCurrent(&context) != CUDA_SUCCESS || !context ||
      !driver.CreateStream(&stream, &error)) {
    std::cerr << "[FAIL] CUDA setup: " << error << '\n';
    return 1;
  }
  // Odd geometry and boundary-near eyes exercise clamping. Pixel verification
  // is deliberately performed after one final synchronization/readback.
  constexpr int w = 101, h = 79;
  std::vector<std::uint8_t> host(w * h * 3), out(host.size());
  for (std::size_t i = 0; i < host.size(); ++i)
    host[i] = static_cast<std::uint8_t>((i * 29 + 7) & 255);
  std::vector<float> marks(136, 0.f);
  for (int i = 0; i < 68; ++i) {
    marks[2 * i] = 50.f;
    marks[2 * i + 1] = 35.f;
  }
  const float lx[6] = {78, 80, 82, 86, 82, 80}, ly[6] = {5, 3, 4, 5, 7, 7};
  const float rx[6] = {2, 4, 6, 10, 6, 4}, ry[6] = {68, 66, 67, 68, 70, 70};
  for (int i = 0; i < 6; ++i) {
    marks[2 * (42 + i)] = lx[i];
    marks[2 * (42 + i) + 1] = ly[i];
    marks[2 * (36 + i)] = rx[i];
    marks[2 * (36 + i) + 1] = ry[i];
  }
  std::vector<float> left_model(3 * 48 * 64), right_model(3 * 48 * 64);
  for (int c = 0; c < 3; ++c)
    for (int y = 0; y < 48; ++y)
      for (int x = 0; x < 64; ++x) {
        const auto i = c * 48 * 64 + y * 64 + x;
        left_model[i] = ((17 * x + 11 * y + 31 * c) % 254 + .75f) / 255.f;
        right_model[i] = ((7 * x + 19 * y + 47 * c + 23) % 254 + .75f) / 255.f;
      }
  CudaImage source, output;
  CudaTensor lm, geo, le, re, la, ra, langle, rangle, lout, rout;
  PreparedEyeContactKernels kernels;
  bool ok = source.Allocate(&driver, w, h, PixelFormatGpu::rgb_u8, &error) &&
            output.Allocate(&driver, w, h, PixelFormatGpu::rgb_u8, &error) &&
            lm.AllocateNchwF32(&driver, 1, 1, 1, 136, &error) &&
            geo.AllocateNchwF32(&driver, 1, 2, 1, 5, &error) &&
            le.AllocateNchwF32(&driver, 1, 3, 48, 64, &error) &&
            re.AllocateNchwF32(&driver, 1, 3, 48, 64, &error) &&
            la.AllocateNchwF32(&driver, 1, 12, 48, 64, &error) &&
            ra.AllocateNchwF32(&driver, 1, 12, 48, 64, &error) &&
            langle.AllocateNchwF32(&driver, 1, 2, 1, 1, &error) &&
            rangle.AllocateNchwF32(&driver, 1, 2, 1, 1, &error) &&
            lout.AllocateNchwF32(&driver, 1, 3, 48, 64, &error) &&
            rout.AllocateNchwF32(&driver, 1, 3, 48, 64, &error) &&
            lout.UploadFromCpuF32(&driver, left_model.data(), left_model.size(),
                                  stream, &error) &&
            rout.UploadFromCpuF32(&driver, right_model.data(),
                                  right_model.size(), stream, &error) &&
            PrepareEyeContactKernels(driver, context, stream, &kernels, &error);
  PreparedEyeContactInputs inputs{&lm, &geo, &le,     &re,
                                  &la, &ra,  &langle, &rangle};
  const auto run = [&](const std::vector<float> &landmarks, float strength,
                       bool left_ok, bool right_ok, std::string_view label) {
    std::fill(out.begin(), out.end(), 0);
    std::vector<float> actual_geometry, actual_left_rgb, actual_right_rgb,
        actual_left_anchors, actual_right_anchors, actual_left_angles,
        actual_right_angles;
    bool pass =
        source.UploadFromCpuRgb24(&driver, host.data(), w * 3, stream,
                                  &error) &&
        lm.UploadFromCpuF32(&driver, landmarks.data(), landmarks.size(), stream,
                            &error) &&
        LaunchEyeContactInputsPrepared(kernels, source, inputs, strength,
                                       &error) &&
        geo.DownloadToCpuF32(&driver, &actual_geometry, stream, &error) &&
        le.DownloadToCpuF32(&driver, &actual_left_rgb, stream, &error) &&
        re.DownloadToCpuF32(&driver, &actual_right_rgb, stream, &error) &&
        la.DownloadToCpuF32(&driver, &actual_left_anchors, stream, &error) &&
        ra.DownloadToCpuF32(&driver, &actual_right_anchors, stream, &error) &&
        langle.DownloadToCpuF32(&driver, &actual_left_angles, stream, &error) &&
        rangle.DownloadToCpuF32(&driver, &actual_right_angles, stream,
                                &error) &&
        LaunchEyeContactCompositePrepared(kernels, source, geo, lout, rout,
                                          left_ok, right_ok, strength, &output,
                                          &error) &&
        output.DownloadToCpuRgb24(&driver, out.data(), w * 3, stream, &error) &&
        driver.StreamSynchronize(stream, &error);
    const auto expected = Expected(host, landmarks, w, h, left_model,
                                   right_model, strength, left_ok, right_ok);
    const auto expected_inputs =
        ExpectedInputs(host, landmarks, w, h, strength);
    const bool frame_ok = BytesNear(out, expected, 0);
    const bool geometry_ok = Near(actual_geometry, expected_inputs.geometry);
    const bool left_rgb_ok = Near(actual_left_rgb, expected_inputs.left_rgb);
    const bool right_rgb_ok = Near(actual_right_rgb, expected_inputs.right_rgb);
    const bool left_anchors_ok =
        Near(actual_left_anchors, expected_inputs.left_anchors);
    const bool right_anchors_ok =
        Near(actual_right_anchors, expected_inputs.right_anchors);
    const bool left_angles_ok =
        Near(actual_left_angles, expected_inputs.left_angles);
    const bool right_angles_ok =
        Near(actual_right_angles, expected_inputs.right_angles);
    pass = pass && frame_ok && geometry_ok && left_rgb_ok && right_rgb_ok &&
           left_anchors_ok && right_anchors_ok && left_angles_ok &&
           right_angles_ok;
    if (!pass)
      std::cerr << "[FAIL] native " << label << ": " << error
                << " frame=" << frame_ok << " geometry=" << geometry_ok
                << " left_rgb=" << left_rgb_ok << "("
                << MaxDelta(actual_left_rgb, expected_inputs.left_rgb) << ")"
                << " right_rgb=" << right_rgb_ok << "("
                << MaxDelta(actual_right_rgb, expected_inputs.right_rgb) << ")"
                << " left_anchors=" << left_anchors_ok
                << " right_anchors=" << right_anchors_ok
                << " left_angles=" << left_angles_ok
                << " right_angles=" << right_angles_ok << '\n';
    return pass;
  };
  ok = ok && run(marks, 0.f, true, true, "strength zero exact copy");
  ok = ok && run(marks, .01f, true, true, "strength one percent");
  ok = ok && run(marks, .5f, true, true, "both eyes intermediate strength");
  ok = ok && run(marks, 1.f, false, true, "left model failure preservation");
  ok = ok && run(marks, 1.f, true, false, "right model failure preservation");
  auto overlapping = marks;
  for (int i = 0; i < 6; ++i) {
    overlapping[2 * (42 + i)] = 47.f + i;
    overlapping[2 * (42 + i) + 1] = 38.f + (i & 1);
    overlapping[2 * (36 + i)] = 49.f + i;
    overlapping[2 * (36 + i) + 1] = 38.f + (i & 1);
  }
  ok = ok && run(overlapping, .5f, true, true,
                 "overlapping eyes composite left then right");
  auto one_invalid = marks;
  for (int point = 0; point < 6; ++point) {
    one_invalid[2 * (42 + point)] = 50.f;
    one_invalid[2 * (42 + point) + 1] = 35.f;
  }
  ok = ok && run(one_invalid, 1.f, true, true, "one resident crop invalid");
  auto invalid = marks;
  for (int eye_base : {36, 42})
    for (int point = 0; point < 6; ++point) {
      invalid[2 * (eye_base + point)] = 50.f;
      invalid[2 * (eye_base + point) + 1] = 35.f;
    }
  ok = ok && run(invalid, 1.f, true, true, "both resident crops invalid");
  if (kernels.live) {
    (void)driver.StreamSynchronize(stream, nullptr);
    (void)DestroyEyeContactKernelsPrepared(&kernels, nullptr);
  }
  for (CudaTensor *t :
       {&rout, &lout, &rangle, &langle, &ra, &la, &re, &le, &geo, &lm})
    (void)t->Free(&driver, nullptr);
  (void)output.Free(&driver, nullptr);
  (void)source.Free(&driver, nullptr);
  if (stream)
    (void)driver.DestroyStream(stream, nullptr);
  if (ok)
    std::cout << "[PASS] native CUDA eye-contact CPU oracle: odd=101x79 "
                 "strengths=0,0.01,0.5,1 both/left-failed/right-failed/"
                 "one-invalid/both-invalid; geometry/eye-rgb/anchors/angles/"
                 "composite verified\n";
  return ok ? 0 : 1;
}
} // namespace

int main(int argc, char **argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--native")
    return NativeOracle();
  if (!FakeContract())
    return 1;
  std::cout << "[PASS] prepared CUDA eye-contact fake contract: ownership, "
               "geometry, 64-bit spans, aliasing, launch, destroy retry\n";
  return 0;
}
