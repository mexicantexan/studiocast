#include "core/open_video/dlib_shape_predictor_data.h"
#include "core/open_video/ncnn_vulkan_context.h"
#include "core/open_video/prepared_cuda_auto_frame.h"
#include "core/open_video/yunet_face_detector.h"
#include "core/vulkan/kernels/tracking_kernels.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <dlib/image_processing/shape_predictor.h>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace vk = studiocast::vulkan;
namespace ov = studiocast::open_vulkan;
namespace video = studiocast::open_video;
using vk::kernels::TrackingBufferBinding;
using vk::kernels::TrackingDispatch;
namespace {
void Need(bool good, const std::string &message) {
  if (!good)
    throw std::runtime_error(message);
}
std::uint32_t Bits(float v) { return std::bit_cast<std::uint32_t>(v); }
TrackingBufferBinding Bind(const vk::VulkanBuffer &b) {
  return {b.buffer(), b.size(), b.context_identity()};
}
struct Runner {
  vk::VulkanDevice device;
  vk::kernels::TrackingKernels kernels;
  vk::VulkanBuffer dummy;
  // Recorded preprocessing bindings remain borrowed until kernel shutdown.
  vk::VulkanBuffer preprocess_image, preprocess_output;
  std::string error;
  unsigned slot = 0;
  double maximum_execute_ms = 0;
  Runner() {
    vk::VulkanDeviceConfig config;
    config.fence_timeout_ns = 100000000;
    Need(ov::InitializeProductionVulkanMattingDevice(&device, {}, config,
                                                     &error, true),
         error);
    Need(device.shader_float64_enabled(), "enabled FP64 proof missing");
    Allocate(dummy, 16, false);
    Need(kernels.Prepare(&device, true, 32, 64, &error), error);
  }
  void Allocate(vk::VulkanBuffer &b, std::size_t bytes, bool mapped) {
    Need(b.Allocate(&device, bytes,
                    vk::VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                        vk::VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                        vk::VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    mapped ? vk::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                           : vk::VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    mapped ? vk::VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                           : vk::VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    mapped, &error),
         error);
  }
  void Copy(const vk::VulkanBuffer &src, const vk::VulkanBuffer &dst,
            std::size_t bytes) {
    vk::VkCommandBufferAllocateInfo a{};
    a.commandPool = device.command_pool();
    a.commandBufferCount = 1;
    vk::VkCommandBuffer cmd = nullptr;
    Need(device.CheckDriverResult(
             device.f().vkAllocateCommandBuffers(device.device(), &a, &cmd),
             "test allocate", false, &error),
         error);
    vk::VkCommandBufferBeginInfo begin{};
    Need(device.CheckDriverResult(device.f().vkBeginCommandBuffer(cmd, &begin),
                                  "test begin", false, &error),
         error);
    vk::VkBufferMemoryBarrier barrier{};
    barrier.buffer = src.buffer();
    barrier.size = bytes;
    barrier.srcAccessMask = vk::VK_ACCESS_HOST_WRITE_BIT |
                            vk::VK_ACCESS_TRANSFER_WRITE_BIT |
                            vk::VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = vk::VK_ACCESS_TRANSFER_READ_BIT;
    device.f().vkCmdPipelineBarrier(cmd,
                                    vk::VK_PIPELINE_STAGE_ALL_COMMANDS_BIT |
                                        vk::VK_PIPELINE_STAGE_HOST_BIT,
                                    vk::VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                                    nullptr, 1, &barrier, 0, nullptr);
    const vk::VkBufferCopy copy{0, 0, bytes};
    device.f().vkCmdCopyBuffer(cmd, src.buffer(), dst.buffer(), 1, &copy);
    barrier.buffer = dst.buffer();
    barrier.srcAccessMask = vk::VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask =
        vk::VK_ACCESS_SHADER_READ_BIT | vk::VK_ACCESS_HOST_READ_BIT;
    device.f().vkCmdPipelineBarrier(cmd, vk::VK_PIPELINE_STAGE_TRANSFER_BIT,
                                    vk::VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                        vk::VK_PIPELINE_STAGE_HOST_BIT,
                                    0, 0, nullptr, 1, &barrier, 0, nullptr);
    Need(device.CheckDriverResult(device.f().vkEndCommandBuffer(cmd),
                                  "test end", false, &error),
         error);
    const auto start = std::chrono::steady_clock::now();
    const bool completed = device.SubmitAndWait(cmd, &error);
    Need(completed,
         error + " transfer_bytes=" + std::to_string(bytes) + " duration_ms=" +
             std::to_string(std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - start)
                                .count()));
    device.f().vkFreeCommandBuffers(device.device(), device.command_pool(), 1,
                                    &cmd);
  }
  void Upload(vk::VulkanBuffer &dst, const void *data, std::size_t bytes) {
    vk::VulkanBuffer host;
    Allocate(host, bytes, true);
    std::memcpy(host.mapped(), data, bytes);
    Need(host.Flush(&error), error);
    Copy(host, dst, bytes);
  }
  template <class T>
  std::vector<T> Read(const vk::VulkanBuffer &src, std::size_t count) {
    vk::VulkanBuffer host;
    Allocate(host, count * sizeof(T), true);
    Copy(src, host, count * sizeof(T));
    Need(host.Invalidate(&error), error);
    std::vector<T> out(count);
    std::memcpy(out.data(), host.mapped(), out.size() * sizeof(T));
    return out;
  }
  TrackingDispatch Dispatch(unsigned op) {
    TrackingDispatch d;
    d.buffers.fill(Bind(dummy));
    d.parameters[0] = op;
    return d;
  }
  vk::VkCommandBuffer Record(std::span<const TrackingDispatch> ds) {
    vk::VkCommandBuffer cmd = nullptr;
    Need(kernels.Record(slot++, ds, &cmd, &error), error);
    return cmd;
  }
  void Execute(vk::VkCommandBuffer command) {
    const auto begin = std::chrono::steady_clock::now();
    const bool completed = device.SubmitAndWait(command, &error);
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - begin)
                          .count();
    Need(completed, error + " execute_duration_ms=" + std::to_string(ms));
    maximum_execute_ms = std::max(maximum_execute_ms, ms);
  }
};
void DlibOracle(Runner &r, const char *root) {
  const auto registry = video::ModelPackRegistry::Scan(root);
  const auto pack = registry.Find("face_landmarks", "dlib_68_ibug_300w");
  Need(pack.has_value(), "dlib oracle pack absent");
  const auto identity =
      video::ValidateModelArtifactIdentity(*pack, pack->files[0]);
  Need(identity.ok(), identity.detail);
  video::DlibPredictorData data;
  Need(video::LoadDlibPredictorData(*pack, identity.identity, 128u << 20, &data,
                                    &r.error),
       r.error);
  std::vector<std::uint32_t> words;
  auto d = r.Dispatch(0);
  d.shader = vk::kernels::TrackingShader::dlib_predictor;
  const auto add = [&](const auto &a, unsigned offset) {
    d.parameters[offset] = words.size();
    auto old = words.size();
    words.resize(old + (a.size() * sizeof(a[0]) + 3) / 4);
    std::memcpy(words.data() + old, a.data(), a.size() * sizeof(a[0]));
  };
  d.parameters[4] = data.cascade_trees.size() - 1;
  add(data.initial_shape, 5);
  add(data.cascade_trees, 6);
  add(data.tree_splits, 7);
  add(data.tree_leaves, 8);
  add(data.splits, 9);
  add(data.leaves, 10);
  add(data.cascade_features, 11);
  add(data.feature_anchors, 12);
  add(data.feature_deltas, 13);
  if (words.size() & 1)
    words.push_back(0);
  add(data.rectangle_basis_pseudoinverse, 14);
  d.parameters[15] = data.maximum_features;
  vk::VulkanBuffer image, forest, scratch, points, state;
  constexpr unsigned w = 257, h = 193;
  r.Allocate(image, w * h * 4, false);
  r.Allocate(forest, words.size() * 4, false);
  r.Allocate(scratch, (136 + data.maximum_features + data.maximum_trees) * 4,
             false);
  r.Allocate(points, 136 * 4, false);
  r.Allocate(state, 5512 * 4, false);
  r.Upload(forest, words.data(), words.size() * 4);
  d.parameters[1] = w;
  d.parameters[2] = h;
  d.parameters[3] = w;
  d.buffers[0] = Bind(image);
  d.buffers[1] = Bind(forest);
  d.buffers[2] = Bind(scratch);
  d.buffers[3] = Bind(points);
  d.buffers[5] = Bind(state);
  const auto command = r.Record(std::span(&d, 1));
  dlib::shape_predictor oracle;
  dlib::deserialize(pack->files[0].path.string()) >> oracle;
  const std::array<dlib::rectangle, 5> boxes{
      dlib::rectangle(7, 11, 180, 170), dlib::rectangle(0, 0, 256, 192),
      dlib::rectangle(51, 27, 205, 178), dlib::rectangle(-13, 9, 150, 181),
      dlib::rectangle(100, -19, 270, 150)};
  dlib::array2d<dlib::rgb_pixel> host(h, w);
  std::vector<std::uint32_t> rgba(w * h), metadata(5512);
  for (unsigned pattern = 0; pattern < 4; pattern++) {
    for (unsigned y = 0; y < h; y++)
      for (unsigned x = 0; x < w; x++) {
        unsigned red = (x * 17 + y * 3 + pattern * 41) & 255,
                 green = (x * 5 + y * 29 + pattern * 13) & 255,
                 blue = (x * 11 + y * 7 + pattern * 67) & 255;
        rgba[y * w + x] = red | (green << 8) | (blue << 16) | 0xff000000;
        host[y][x] = dlib::rgb_pixel(red, green, blue);
      }
    r.Upload(image, rgba.data(), rgba.size() * 4);
    for (const auto &box : boxes) {
      metadata[5] = metadata[7] = 1;
      metadata[16] = Bits(box.left());
      metadata[17] = Bits(box.top());
      metadata[18] = Bits(box.right() - box.left());
      metadata[19] = Bits(box.bottom() - box.top());
      r.Upload(state, metadata.data(), metadata.size() * 4);
      r.Execute(command);
      const auto actual = r.Read<float>(points, 136);
      const auto expected = oracle(host, box);
      for (unsigned i = 0; i < 68; i++)
        if (actual[2 * i] != expected.part(i).x() ||
            actual[2 * i + 1] != expected.part(i).y())
          throw std::runtime_error(
              "GPU dlib coordinate mismatch pattern=" +
              std::to_string(pattern) + " point=" + std::to_string(i) +
              " actual=" + std::to_string(actual[2 * i]) + "," +
              std::to_string(actual[2 * i + 1]) +
              " expected=" + std::to_string(expected.part(i).x()) + "," +
              std::to_string(expected.part(i).y()));
    }
  }
  std::cout << "PASS native GPU dlib: exact136coordinates across20cases "
               "max_execute_ms="
            << r.maximum_execute_ms << std::endl;
}
// Independent literal oracle for YunetFaceDetector::ComputeLetterbox and
// FillInputTensorBgr: uniform inverse scale, zero padding, clamped border
// weights, RGB byte input and planar BGR float output in [0,255].
std::vector<float> CanonicalYunetInput(const std::vector<std::uint8_t> &rgb,
                                       int w, int h, std::size_t stride) {
  constexpr int extent = 640;
  std::vector<float> out(3 * extent * extent, 0.F);
  if (w <= 1 || h <= 1)
    return out;
  const float scale =
      std::min(float(extent) / float(w), float(extent) / float(h));
  const float inverse = 1.F / scale;
  const int cw = std::max(1, int(std::lround(float(w) * scale))),
            ch = std::max(1, int(std::lround(float(h) * scale))),
            px = (extent - cw) / 2, py = (extent - ch) / 2;
  for (int y = py; y < py + ch; ++y) {
    const float sy = (float(y - py) + .5F) * inverse - .5F;
    int y0 = int(std::floor(sy));
    float fy = sy - float(y0);
    if (y0 < 0) {
      y0 = 0;
      fy = 0;
    }
    int y1 = y0 + 1;
    if (y1 >= h) {
      y1 = h - 1;
      fy = 0;
    }
    for (int x = px; x < px + cw; ++x) {
      const float sx = (float(x - px) + .5F) * inverse - .5F;
      int x0 = int(std::floor(sx));
      float fx = sx - float(x0);
      if (x0 < 0) {
        x0 = 0;
        fx = 0;
      }
      int x1 = x0 + 1;
      if (x1 >= w) {
        x1 = w - 1;
        fx = 0;
      }
      for (unsigned c = 0; c < 3; ++c) {
        const auto value = [&](int xx, int yy) {
          return float(
              rgb[std::size_t(yy) * stride + std::size_t(xx) * 3 + 2 - c]);
        };
        const float a = value(x0, y0) + (value(x1, y0) - value(x0, y0)) * fx,
                    b = value(x0, y1) + (value(x1, y1) - value(x0, y1)) * fx;
        out[c * extent * extent + unsigned(y * extent + x)] = a + (b - a) * fy;
      }
    }
  }
  return out;
}
void YunetPreprocessOracle(Runner &r) {
  auto &image = r.preprocess_image;
  auto &output = r.preprocess_output;
  r.Allocate(image, 1283u * 1280u * 4u, false);
  r.Allocate(output, 3u * 640u * 640u * 4u, false);
  const std::array<std::array<unsigned, 2>, 8> shapes{{{64, 64},
                                                       {67, 51},
                                                       {1280, 719},
                                                       {719, 1280},
                                                       {641, 479},
                                                       {320, 640},
                                                       {640, 320},
                                                       {1, 23}}};
  for (const auto &shape : shapes) {
    const unsigned w = shape[0], h = shape[1], pitch = w + 3;
    const std::size_t stride = std::size_t(w) * 3 + 5;
    std::vector<std::uint8_t> rgb(stride * h, 0xa5);
    std::vector<std::uint32_t> packed(std::size_t(pitch) * h, 0xdeadbeef);
    for (unsigned y = 0; y < h; ++y)
      for (unsigned x = 0; x < w; ++x) {
        const std::array<unsigned, 3> channels{
            (x * 17 + y * 31 + 13) & 255, (x * 43 + y * 7 + 89) & 255,
            w == 64 ? (x == 0 ? 64u : 192u) : (x * 11 + y * 53 + 157) & 255};
        packed[std::size_t(y) * pitch + x] = 0xff000000;
        for (unsigned c = 0; c < 3; ++c) {
          rgb[std::size_t(y) * stride + std::size_t(x) * 3 + c] =
              static_cast<std::uint8_t>(channels[c]);
          packed[std::size_t(y) * pitch + x] |= channels[c] << (8 * c);
        }
      }
    const auto expected = CanonicalYunetInput(rgb, int(w), int(h), stride);
    Need(w != 64 || expected[0] == 64.F,
         "canonical first blue sample must clamp to64, not extrapolate6.4");
    auto d = r.Dispatch(1);
    d.buffers[0] = Bind(image);
    d.buffers[1] = Bind(output);
    d.parameters[1] = w;
    d.parameters[2] = h;
    d.parameters[3] = pitch;
    d.parameters[25] = Bits(1.F / 255.F);
    d.parameters[26] = Bits(1.F / (1.F / 255.F));
    d.invocations = 3 * 640 * 640;
    r.Upload(image, packed.data(), packed.size() * 4);
    r.Execute(r.Record(std::span(&d, 1)));
    const auto actual = r.Read<float>(output, expected.size());
    float maximum = 0;
    std::size_t worst = 0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
      Need(std::isfinite(actual[i]), "nonfinite YuNet input tensor");
      const float error = std::abs(actual[i] - expected[i]);
      if (error > maximum) {
        maximum = error;
        worst = i;
      }
    }
    std::cout << "YuNet preprocess " << w << 'x' << h
              << " max_error=" << maximum << " index=" << worst
              << " expected=" << expected[worst] << " actual=" << actual[worst]
              << std::endl;
    Need(maximum <= .0001F,
         "GPU YuNet preprocessing differs from canonical BGR/letterbox/border");
  }
  std::cout << "PASS native GPU YuNet preprocess:8geometries clampededges/"
               "uniformscale/oddaspect/letterbox/BGR/pitchedinput\n";
}
void YunetOracle(Runner &r) {
  vk::VulkanBuffer cls, obj, boxes, candidates, masks, state, frame;
  r.Allocate(cls, 6400 * 4, false);
  r.Allocate(obj, 6400 * 4, false);
  r.Allocate(boxes, 6400 * 16, false);
  r.Allocate(candidates, 16384 * 32, false);
  r.Allocate(masks, 5000 * 157 * 4, false);
  r.Allocate(state, 5512 * 4, false);
  r.Allocate(frame, 16, false);
  std::vector<float> scores(6400), box(6400 * 4);
  video::YunetDetectionScratch expected;
  Need(expected.Configure(5000), "CPU YuNet scratch");
  for (unsigned i = 0; i < 6400; i++) {
    scores[i] = i < 6000 ? .95F : .91F;
    box[i * 4] = .5F;
    box[i * 4 + 1] = .5F;
    box[i * 4 + 2] = std::log(float(1 + i % 3));
    box[i * 4 + 3] = std::log(float(1 + (i / 3) % 3));
    const float score = std::sqrt(scores[i] * scores[i]),
                w = std::exp(box[i * 4 + 2]) * 8,
                h = std::exp(box[i * 4 + 3]) * 8;
    expected.Consider({(float(i % 80) + .5F) * 8 - w * .5F,
                       (float(i / 80) + .5F) * 8 - h * .5F, w, h, score});
  }
  const auto &gold = expected.Finalize(1, 0, 0, 640, 640, .3F);
  r.Upload(cls, scores.data(), scores.size() * 4);
  r.Upload(obj, scores.data(), scores.size() * 4);
  r.Upload(boxes, box.data(), box.size() * 4);
  std::array<std::uint32_t, 5512> zero{};
  std::array<std::uint32_t, 4> sequence{1, 0, 0, 0};
  r.Upload(state, zero.data(), sizeof(zero));
  r.Upload(frame, sequence.data(), sizeof(sequence));
  auto base = r.Dispatch(0);
  base.parameters[1] = base.parameters[2] = 640;
  base.buffers[4] = Bind(candidates);
  base.buffers[5] = Bind(state);
  base.buffers[7] = Bind(frame);
  std::vector<TrackingDispatch> ds;
  ds.push_back(base);
  auto clear = base;
  clear.parameters[0] = 2;
  clear.invocations = 16384;
  ds.push_back(clear);
  auto decode = base;
  decode.parameters[0] = 3;
  decode.parameters[5] = 8;
  decode.buffers[1] = Bind(cls);
  decode.buffers[2] = Bind(obj);
  decode.buffers[3] = Bind(boxes);
  decode.invocations = 6400;
  ds.push_back(decode);
  for (unsigned k = 2; k <= 16384; k *= 2)
    for (unsigned j = k / 2; j; j /= 2) {
      auto d = base;
      d.parameters[0] = 4;
      d.parameters[11] = k;
      d.parameters[12] = j;
      d.invocations = 16384;
      ds.push_back(d);
    }
  auto mask = base;
  mask.parameters[0] = 5;
  mask.buffers[6] = Bind(masks);
  mask.invocations = 5000 * 157;
  ds.push_back(mask);
  auto choose = base;
  choose.parameters[0] = 6;
  choose.buffers[6] = Bind(masks);
  ds.push_back(choose);
  const auto cmd = r.Record(ds);
  r.Execute(cmd);
  const auto actual_state = r.Read<std::uint32_t>(state, 5512);
  const auto actual_candidates = r.Read<float>(candidates, 16384 * 8);
  Need(actual_state[3] == 1 && actual_state[4] == gold.size(),
       "GPU YuNet kept count differs: actual=" +
           std::to_string(actual_state[4]) +
           " expected=" + std::to_string(gold.size()));
  for (unsigned i = 0; i < gold.size(); i++) {
    const auto index = actual_state[512 + i];
    const auto *a = actual_candidates.data() + index * 8;
    const auto &e = gold[i];
    Need(std::abs(a[0] - e.x) < .0001F && std::abs(a[1] - e.y) < .0001F &&
             std::abs(a[2] - e.w) < .0001F && std::abs(a[3] - e.h) < .0001F,
         "GPU YuNet stable topK/NMS ordering differs");
  }
  const video::FaceDetection *eye = nullptr, *automatic = nullptr;
  for (const auto &f : gold) {
    if (!eye || f.score > eye->score + 1e-5F ||
        (std::abs(f.score - eye->score) < 1e-5F && f.w * f.h > eye->w * eye->h))
      eye = &f;
    if (!automatic || f.w * f.h > automatic->w * automatic->h)
      automatic = &f;
  }
  const auto sf = [&](unsigned i) {
    return std::bit_cast<float>(actual_state[i]);
  };
  Need(eye && automatic && sf(16) == eye->x && sf(17) == eye->y &&
           std::abs(sf(24) - (automatic->x - automatic->w * .4F)) < .0001F,
       "GPU Eye/Auto Frame face choices differ");
  std::cout << "PASS native GPU YuNet:6400tiedcandidates/topK5000/NMS kept="
            << gold.size() << " distinctfacechoices\n";
}
std::vector<float> Floats(const std::filesystem::path &path,
                          std::size_t count) {
  std::ifstream in(path, std::ios::binary);
  Need(bool(in), "missing numerical fixture " + path.string());
  std::vector<float> out(count);
  in.read(reinterpret_cast<char *>(out.data()), count * 4);
  Need(std::size_t(in.gcount()) == count * 4 &&
           in.peek() == std::char_traits<char>::eof(),
       "numerical fixture size " + path.string());
  return out;
}
void FlxTailOracle(Runner &r, const char *fixtures) {
  vk::VulkanBuffer rgb, flow, lcm, output, state;
  r.Allocate(rgb, 9216 * 4, false);
  r.Allocate(flow, 6144 * 4, false);
  r.Allocate(lcm, 6144 * 4, false);
  r.Allocate(output, 9216 * 4, false);
  r.Allocate(state, 5512 * 4, false);
  std::array<std::uint32_t, 5512> meta{};
  meta[5] = meta[8] = 1;
  r.Upload(state, meta.data(), sizeof(meta));
  auto d = r.Dispatch(1);
  d.shader = vk::kernels::TrackingShader::eye_contact;
  d.buffers[0] = Bind(rgb);
  d.buffers[1] = Bind(flow);
  d.buffers[2] = Bind(lcm);
  d.buffers[3] = Bind(output);
  d.buffers[5] = Bind(state);
  d.invocations = 9216;
  const auto cmd = r.Record(std::span(&d, 1));
  float worst = 0;
  for (const auto eye : {"left", "right"})
    for (const auto pattern :
         {"black", "white", "gradient", "checker", "seeded_noise"})
      for (const auto angle : {"0", "-0.05", "-0.1"}) {
        const auto base = std::filesystem::path(fixtures) / "work" / eye /
                          "numerical" /
                          (std::string(pattern) + "-angle" + angle);
        const auto input = Floats(base.string() + ".input0.f32", 9216),
                   f = Floats(base.string() + ".expected0.f32", 6144),
                   l = Floats(base.string() + ".expected1.f32", 6144),
                   gold = Floats(base.string() + ".rgb.expected.f32", 9216);
        r.Upload(rgb, input.data(), input.size() * 4);
        r.Upload(flow, f.data(), f.size() * 4);
        r.Upload(lcm, l.data(), l.size() * 4);
        r.Execute(cmd);
        const auto actual = r.Read<float>(output, 9216);
        float max = 0;
        for (unsigned i = 0; i < 9216; i++)
          max = std::max(max, std::abs(actual[i] - gold[i]));
        worst = std::max(worst, max);
        Need(max <= .000002F,
             "GPU FLX native tail/original ONNX error=" + std::to_string(max) +
                 " fixture=" + base.string());
      }
  std::cout << "PASS native GPU FLX sampler/lighting:30originalONNXcases max="
            << worst << "\n";
}
void EyeInputsOracle(Runner &r) {
  constexpr unsigned w = 101, h = 79;
  vk::VulkanBuffer image, points, state;
  std::array<vk::VulkanBuffer, 2> rgb, anchors, angles;
  r.Allocate(image, w * h * 4, false);
  r.Allocate(points, 136 * 4, false);
  r.Allocate(state, 5512 * 4, false);
  std::vector<std::uint32_t> pixels(w * h), metadata(5512);
  std::array<float, 136> marks{};
  for (unsigned y = 0; y < h; y++)
    for (unsigned x = 0; x < w; x++)
      pixels[y * w + x] = x | (y << 8) | ((x + y) << 16) | 0xff000000;
  const std::array<std::array<float, 2>, 6> offsets{
      {{0, 0}, {5, -2}, {15, -2}, {20, 0}, {15, 2}, {5, 2}}};
  for (unsigned e = 0; e < 2; e++)
    for (unsigned i = 0; i < 6; i++) {
      unsigned base = e == 0 ? 42 : 36;
      marks[(base + i) * 2] = (e == 0 ? 30 : 60) + offsets[i][0];
      marks[(base + i) * 2 + 1] = 30 + offsets[i][1];
    }
  r.Upload(image, pixels.data(), pixels.size() * 4);
  r.Upload(points, marks.data(), sizeof(marks));
  metadata[5] = metadata[8] = 1;
  r.Upload(state, metadata.data(), metadata.size() * 4);
  for (unsigned e = 0; e < 2; e++) {
    r.Allocate(rgb[e], 9216 * 4, false);
    r.Allocate(anchors[e], 36864 * 4, false);
    r.Allocate(angles[e], 2 * 4, false);
  }
  for (float strength : {0.F, .01F, .5F, 1.F}) {
    std::array<TrackingDispatch, 2> ds;
    for (unsigned e = 0; e < 2; e++) {
      auto &d = ds[e];
      d = r.Dispatch(0);
      d.shader = vk::kernels::TrackingShader::eye_contact;
      d.parameters[1] = w;
      d.parameters[2] = h;
      d.parameters[3] = w;
      d.parameters[5] = e;
      d.parameters[6] = Bits(strength);
      d.buffers[0] = Bind(image);
      d.buffers[1] = Bind(points);
      d.buffers[2] = Bind(rgb[e]);
      d.buffers[3] = Bind(anchors[e]);
      d.buffers[4] = Bind(angles[e]);
      d.buffers[5] = Bind(state);
      d.invocations = 36864;
    }
    r.Execute(r.Record(ds));
    const auto geometry = r.Read<std::uint32_t>(state, 5512);
    for (unsigned e = 0; e < 2; e++) {
      const float left = e == 0 ? 25 : 55, top = 16, cw = 30, ch = 24;
      const auto actual = r.Read<float>(rgb[e], 9216),
                 guidance = r.Read<float>(anchors[e], 36864),
                 target = r.Read<float>(angles[e], 2);
      Need(std::bit_cast<float>(geometry[60 + e * 5]) == left &&
               std::bit_cast<float>(geometry[61 + e * 5]) == top &&
               std::bit_cast<float>(geometry[62 + e * 5]) == cw &&
               std::bit_cast<float>(geometry[63 + e * 5]) == ch &&
               geometry[64 + e * 5] == 1,
           "GPU eye geometry");
      Need(target[0] == 0 && target[1] == -.10F * strength,
           "GPU target angles");
      for (unsigned y = 0; y < 48; y++)
        for (unsigned x = 0; x < 64; x++) {
          const float sx = std::clamp(left + (x + .5F) * cw / 64 - .5F, left,
                                      left + cw - 1),
                      sy = std::clamp(top + (y + .5F) * ch / 48 - .5F, top,
                                      top + ch - 1);
          const unsigned q = y * 64 + x;
          for (unsigned c = 0; c < 3; c++) {
            float value = c == 0 ? sx : c == 1 ? sy : sx + sy;
            float expected = std::floor(value + .5F) / 255.F;
            Need(std::abs(actual[c * 3072 + q] - expected) < .000001F,
                 "GPU eye RGB ramp sampling");
          }
          const std::array<unsigned, 6> order{3, 2, 1, 0, 5, 4};
          for (unsigned c = 0; c < 12; c++) {
            unsigned base = e == 0 ? 42 : 36,
                     point = base + (e == 0 ? order[c / 2] : c / 2);
            int rx = std::floor((marks[point * 2] - left) * 64 / cw + .5F),
                ry = std::floor((marks[point * 2 + 1] - top) * 48 / ch + .5F);
            float expected = (c & 1) ? int(y) - ry : int(x) - rx;
            Need(guidance[c * 3072 + q] == expected,
                 "GPU anchor orientation/offset");
          }
        }
    }
  }
  std::cout << "PASS native GPU eye inputs: odd101x79 strength0/.01/.5/1 "
               "exactanchors/angles/geometry\n";
}
void CropAndVignetteOracle(Runner &r) {
  constexpr unsigned w = 64, h = 48;
  vk::VulkanBuffer image, cropped, vignetted, state, frame;
  r.Allocate(image, w * h * 4, false);
  r.Allocate(cropped, w * h * 4, false);
  r.Allocate(vignetted, w * h * 4, false);
  r.Allocate(state, 5512 * 4, false);
  r.Allocate(frame, 16, false);
  std::vector<std::uint32_t> pixels(w * h), zero(5512);
  for (unsigned y = 0; y < h; y++)
    for (unsigned x = 0; x < w; x++)
      pixels[y * w + x] = ((x * 3 + y) & 255) | (((y * 4 + x) & 255) << 8) |
                          (((x * 2 + y * 2) & 255) << 16) | 0xff000000;
  r.Upload(image, pixels.data(), pixels.size() * 4);
  r.Upload(state, zero.data(), zero.size() * 4);
  auto base = r.Dispatch(0);
  base.parameters[1] = w;
  base.parameters[2] = h;
  base.buffers[5] = Bind(state);
  base.buffers[7] = Bind(frame);
  const auto begin = r.Record(std::span(&base, 1));
  auto plan = base;
  plan.parameters[0] = 8;
  plan.parameters[13] = 50;
  plan.parameters[14] = 50;
  plan.parameters[15] = Bits(.15F);
  auto crop = base;
  crop.parameters[0] = 9;
  crop.parameters[3] = crop.parameters[4] = w;
  crop.buffers[0] = Bind(image);
  crop.buffers[1] = Bind(cropped);
  crop.invocations = w * h;
  const std::array<TrackingDispatch, 2> phases{plan, crop};
  const auto plan_crop = r.Record(phases);
  const auto crop_only = r.Record(std::span(&crop, 1));
  auto vig = base;
  vig.parameters[0] = 10;
  vig.parameters[3] = vig.parameters[4] = w;
  vig.parameters[16] = Bits(.7F);
  vig.parameters[17] = 1;
  vig.buffers[0] = Bind(cropped);
  vig.buffers[1] = Bind(vignetted);
  vig.invocations = w * h;
  const auto vignette = r.Record(std::span(&vig, 1));
  using A = video::PreparedCudaAutoFrame;
  video::PreparedCudaAutoFrameRect previous{};
  bool have = false;
  std::uint64_t previous_seq = 0;
  unsigned previous_source = 0;
  for (unsigned n = 0; n < 6; n++) {
    const std::uint64_t seq = std::array<std::uint64_t, 6>{
        1, 2, 11, 12, 4294967295ULL, 4294967296ULL}[n];
    const std::array<std::uint32_t, 4> parameters{unsigned(seq),
                                                  unsigned(seq >> 32), 1, 0};
    r.Upload(frame, parameters.data(), sizeof(parameters));
    r.Execute(begin);
    auto meta = r.Read<std::uint32_t>(state, 5512);
    const bool found = n != 3;
    const unsigned source = found ? 1 : 0;
    const video::PreparedCudaAutoFrameRect box{float(n == 1 || n == 5 ? 32 : 8),
                                               6, 16, 20};
    meta[6] = found;
    for (unsigned i = 0; i < 4; i++)
      meta[24 + i] = Bits(std::array<float, 4>{box.x, box.y, box.w, box.h}[i]);
    r.Upload(state, meta.data(), meta.size() * 4);
    video::PreparedCudaAutoFrameTrackingBox signal{
        video::PreparedCudaAutoFrameSource::face, box, .99F};
    auto target =
        found ? A::TargetCrop(signal, w, h, float(w) / h, {50, 50, .15F})
              : A::CenterCrop(w, h, float(w) / h, 1.25F);
    if (seq != previous_seq + 1 || source != previous_source)
      have = false;
    if (have) {
      float a = A::SmoothingAlpha(50);
      target = {previous.x + (target.x - previous.x) * a,
                previous.y + (target.y - previous.y) * a,
                previous.w + (target.w - previous.w) * a,
                previous.h + (target.h - previous.h) * a};
    }
    previous = target;
    have = true;
    previous_seq = seq;
    previous_source = source;
    r.Execute(plan_crop);
    meta = r.Read<std::uint32_t>(state, 5512);
    const auto sf = [&](unsigned i) { return std::bit_cast<float>(meta[i]); };
    Need(std::abs(sf(40) - target.x) < .0001F &&
             std::abs(sf(41) - target.y) < .0001F &&
             std::abs(sf(42) - target.w) < .0001F &&
             std::abs(sf(43) - target.h) < .0001F,
         "GPU crop geometry/smoothing/reset differs from canonical CPU "
         "reference");
    const float cx = found ? (box.x + box.w * .5F - target.x) / target.w : .5F,
                cy = found ? (box.y + box.h * .5F - target.y) / target.h : .5F;
    Need(std::abs(sf(50) - cx) < .0001F && std::abs(sf(51) - cy) < .0001F,
         "GPU postcrop tracked center");
    const auto actual = r.Read<std::uint32_t>(cropped, w * h);
    for (unsigned y = 0; y < h; y++)
      for (unsigned x = 0; x < w; x++) {
        const float sx = target.x + (x + .5F) * (target.w / w) - .5F,
                    sy = target.y + (y + .5F) * (target.h / h) - .5F;
        const int x0 = std::clamp(int(sx), 0, int(w) - 1),
                  y0 = std::clamp(int(sy), 0, int(h) - 1),
                  x1 = std::min(int(w) - 1, x0 + 1),
                  y1 = std::min(int(h) - 1, y0 + 1);
        const float fx = sx - x0, fy = sy - y0;
        for (unsigned c = 0; c < 3; c++) {
          const auto value = [&](unsigned xx, unsigned yy) {
            return float((pixels[yy * w + xx] >> (c * 8)) & 255);
          };
          const float a = value(x0, y0) + fx * (value(x1, y0) - value(x0, y0)),
                      b = value(x0, y1) + fx * (value(x1, y1) - value(x0, y1));
          const int gold = std::clamp(int(a + fy * (b - a) + .5F), 0, 255);
          Need(int((actual[y * w + x] >> (c * 8)) & 255) == gold,
               "GPU crop RGB byte oracle n=" + std::to_string(n) +
                   " xy=" + std::to_string(x) + "," + std::to_string(y) +
                   " c=" + std::to_string(c) +
                   " expected=" + std::to_string(gold) + " actual=" +
                   std::to_string((actual[y * w + x] >> (c * 8)) & 255) +
                   " cropx_cpu=" + std::to_string(target.x) +
                   " cropx_gpu=" + std::to_string(sf(40)));
        }
      }
    r.Execute(vignette);
    const auto result = r.Read<std::uint32_t>(vignetted, w * h);
    for (unsigned y = 0; y < h; y++)
      for (unsigned x = 0; x < w; x++) {
        float nx = (x + .5F - w * cx) * (2.F / w),
              ny = (y + .5F - h * cy) * (2.F / h),
              radius = std::clamp(
                  std::sqrt(nx * nx + ny * ny) * .7071067811865476F, 0.F, 1.F),
              factor = std::max(0.F, 1.F - .7F * (radius * radius));
        for (unsigned c = 0; c < 3; c++) {
          const int gold = std::nearbyint(
              float((actual[y * w + x] >> (c * 8)) & 255) * factor);
          Need(std::abs(int((result[y * w + x] >> (c * 8)) & 255) - gold) <= 1,
               "GPU tracked vignette byte oracle");
        }
      }
  }
  auto metadata = r.Read<std::uint32_t>(state, 5512);
  metadata[40] = Bits(.1F);
  metadata[41] = Bits(.2F);
  metadata[42] = Bits(w - .2F);
  metadata[43] = Bits(h - .2F);
  metadata[52] = 1;
  r.Upload(state, metadata.data(), metadata.size() * 4);
  r.Execute(crop_only);
  Need(r.Read<std::uint32_t>(cropped, w * h) == pixels,
       "near-identity canonical crop must preserve exact input");
  std::cout
      << "PASS native GPU crop/sequence/smoothing/tracked vignette: canonical "
         "geometry/centers/exactcropbytes, nearidentitypreservation\n";
}
void MatteCadenceOracle(Runner &r) {
  vk::VulkanBuffer alpha, partials, state, frame;
  r.Allocate(alpha, 32 * 24 * 4, false);
  r.Allocate(partials, 3 * 20, false);
  r.Allocate(state, 5512 * 4, false);
  r.Allocate(frame, 16, false);
  std::vector<std::uint32_t> zero(5512);
  r.Upload(state, zero.data(), zero.size() * 4);
  auto base = r.Dispatch(0);
  base.parameters[1] = 64;
  base.parameters[2] = 48;
  base.buffers[5] = Bind(state);
  base.buffers[7] = Bind(frame);
  const auto begin = r.Record(std::span(&base, 1));
  auto reduction = base;
  reduction.parameters[0] = 7;
  reduction.parameters[20] = 32;
  reduction.parameters[21] = 24;
  reduction.parameters[22] = 32;
  reduction.buffers[0] = Bind(alpha);
  reduction.buffers[6] = Bind(partials);
  reduction.invocations = 3;
  auto plan = base;
  plan.parameters[0] = 8;
  plan.parameters[13] = plan.parameters[14] = 50;
  plan.parameters[15] = Bits(.15F);
  plan.parameters[20] = 32;
  plan.parameters[21] = 24;
  plan.parameters[23] = 1;
  plan.buffers[6] = Bind(partials);
  std::array<TrackingDispatch, 2> phases{reduction, plan};
  const auto fallback = r.Record(phases);
  phases[1].parameters[24] = 1;
  const auto shared = r.Record(phases);
  std::array<float, 32 * 24> matte{};
  for (unsigned seq = 1; seq <= 14; seq++) {
    std::array<std::uint32_t, 4> parameters{seq, 0, seq >= 13 ? 0u : 1u, 0};
    r.Upload(frame, parameters.data(), sizeof(parameters));
    r.Execute(begin);
    matte.fill(0);
    const unsigned left = seq == 1 ? 4 : seq <= 11 ? 12 : 20;
    for (unsigned y = 6; y < 14; y++)
      for (unsigned x = left; x < left + 8; x++)
        matte[y * 32 + x] = .8F;
    r.Upload(alpha, matte.data(), sizeof(matte));
    if (seq == 14) {
      auto meta = r.Read<std::uint32_t>(state, 5512);
      meta[6] = 1;
      meta[24] = Bits(8);
      meta[25] = Bits(6);
      meta[26] = Bits(16);
      meta[27] = Bits(20);
      r.Upload(state, meta.data(), meta.size() * 4);
    }
    r.Execute(seq >= 12 ? shared : fallback);
    const auto meta = r.Read<std::uint32_t>(state, 5512);
    const float x = std::bit_cast<float>(meta[32]);
    if (seq <= 10)
      Need(x == 8 && meta[48] == 1,
           "GPU matte bounds fallback cadence changed before10captureinterval");
    if (seq == 11)
      Need(x == 24 && meta[48] == 11,
           "GPU matte bounds fallback cadence failed torefresh");
    if (seq == 12)
      Need(x == 40 && meta[48] == 12,
           "GPU shared same-frame matte was not consumed immediately");
    if (seq == 13)
      Need(meta[52] == 0 && meta[48] == 12,
           "unavailablematte must preserveinput without stalealpha");
    if (seq == 14)
      Need(meta[52] == 1 && meta[45] == 1 && meta[48] == 12,
           "unavailablematte suppressed unrelated validfacecrop");
  }
  std::cout
      << "PASS native GPU matte: "
         "tiledbounds/10capturecadence/sharedcurrentframe/failureisolation\n";
}
void EyeCompositeOracle(Runner &r) {
  constexpr unsigned w = 101, h = 79;
  vk::VulkanBuffer image, left, right, output, state;
  r.Allocate(image, w * h * 4, false);
  r.Allocate(left, 9216 * 4, false);
  r.Allocate(right, 9216 * 4, false);
  r.Allocate(output, w * h * 4, false);
  r.Allocate(state, 5512 * 4, false);
  std::vector<std::uint32_t> pixels(w * h), metadata(5512);
  std::array<std::vector<float>, 2> model{std::vector<float>(9216),
                                          std::vector<float>(9216)};
  for (unsigned y = 0; y < h; y++)
    for (unsigned x = 0; x < w; x++)
      pixels[y * w + x] = ((x * 3 + y) & 255) | (((y * 4 + x) & 255) << 8) |
                          (((x * 2 + y * 2) & 255) << 16) | 0xff000000;
  for (unsigned e = 0; e < 2; e++)
    for (unsigned i = 0; i < 9216; i++)
      model[e][i] = (int((i * 17 + e * 29) % 93) - 5) / 70.F;
  r.Upload(image, pixels.data(), pixels.size() * 4);
  r.Upload(left, model[0].data(), 9216 * 4);
  r.Upload(right, model[1].data(), 9216 * 4);
  const auto byte = [](float x) {
    return unsigned(
        std::clamp(int(std::floor(std::clamp(x, 0.F, 255.F) + .5F)), 0, 255));
  };
  const auto sample = [&](unsigned eye, unsigned c, float x, float y) {
    x = std::clamp(x, 0.F, 63.F);
    y = std::clamp(y, 0.F, 47.F);
    unsigned x0 = std::floor(x), y0 = std::floor(y), x1 = std::min(63u, x0 + 1),
             y1 = std::min(47u, y0 + 1);
    float fx = x - x0, fy = y - y0;
    const auto at = [&](unsigned xx, unsigned yy) {
      return float(byte(
          std::clamp(model[eye][c * 3072 + yy * 64 + xx], 0.F, 1.F) * 255.F));
    };
    float a = at(x0, y0) * (1 - fx) + at(x1, y0) * fx,
          b = at(x0, y1) * (1 - fx) + at(x1, y1) * fx;
    return a * (1 - fy) + b * fy;
  };
  unsigned case_number = 0;
  for (float strength : {0.F, .01F, .5F, 1.F}) {
    auto d = r.Dispatch(2);
    d.shader = vk::kernels::TrackingShader::eye_contact;
    d.parameters[1] = w;
    d.parameters[2] = h;
    d.parameters[3] = d.parameters[4] = w;
    d.parameters[6] = Bits(strength);
    d.buffers[0] = Bind(image);
    d.buffers[1] = Bind(left);
    d.buffers[2] = Bind(right);
    d.buffers[3] = Bind(output);
    d.buffers[5] = Bind(state);
    d.invocations = w * h;
    const auto cmd = r.Record(std::span(&d, 1));
    for (unsigned flags = 0; flags < (strength == 1.F ? 4u : 1u); flags++) {
      const std::array<std::array<float, 4>, 2> geometry{
          {{25, 16, 30, 24}, {35, 20, 30, 24}}};
      for (unsigned e = 0; e < 2; e++) {
        for (unsigned i = 0; i < 4; i++)
          metadata[60 + e * 5 + i] = Bits(geometry[e][i]);
        metadata[64 + e * 5] = ((flags >> e) & 1) == 0;
      }
      r.Upload(state, metadata.data(), metadata.size() * 4);
      r.Execute(cmd);
      const auto actual = r.Read<std::uint32_t>(output, w * h);
      for (unsigned y = 0; y < h; y++)
        for (unsigned x = 0; x < w; x++)
          for (unsigned c = 0; c < 3; c++) {
            unsigned expected = (pixels[y * w + x] >> (c * 8)) & 255;
            for (unsigned e = 0; e < 2; e++) {
              const auto g = geometry[e];
              if (!metadata[64 + e * 5] || x < g[0] || y < g[1] ||
                  x >= g[0] + g[2] || y >= g[1] + g[3] || strength <= 0)
                continue;
              const float mx = (x - g[0] + .5F) * 64 / g[2] - .5F,
                          my = (y - g[1] + .5F) * 48 / g[3] - .5F;
              const float corrected = byte(sample(e, c, mx, my));
              expected = byte(expected + (corrected - expected) * strength);
            }
            Need(
                ((actual[y * w + x] >> (c * 8)) & 255) == expected,
                "GPU eye byteblend/left-before-right overlap/clamping differs");
          }
      ++case_number;
    }
  }
  std::cout << "PASS native GPU eye composite:" << case_number
            << " exactbytecases "
               "includingoverlap/invalidcrops/zero/intermediatestrength\n";
}
} // namespace
int main() {
  const auto *required = std::getenv("STUDIOCAST_REQUIRE_VULKAN_TRACKING");
  if (!required || std::strcmp(required, "1") != 0) {
    std::cout << "SKIP: opt-in resident tracking kernels\n";
    return 77;
  }
  try {
    const auto *root = std::getenv("STUDIOCAST_DLIB_ORACLE_MODELS_ROOT");
    Need(root, "set STUDIOCAST_DLIB_ORACLE_MODELS_ROOT");
    Runner runner;
    const auto *preprocess_only =
        std::getenv("STUDIOCAST_VULKAN_TRACKING_ONLY_PREPROCESS");
    if (preprocess_only && std::strcmp(preprocess_only, "1") == 0) {
      YunetPreprocessOracle(runner);
      Need(runner.kernels.Shutdown(&runner.error), runner.error);
      return 0;
    }
    const auto *diagnostic_subset =
        std::getenv("STUDIOCAST_VULKAN_TRACKING_ONLY_CROP");
    if (diagnostic_subset && std::strcmp(diagnostic_subset, "1") == 0) {
      CropAndVignetteOracle(runner);
      MatteCadenceOracle(runner);
      EyeCompositeOracle(runner);
      Need(runner.kernels.Shutdown(&runner.error), runner.error);
      return 0;
    }
    YunetPreprocessOracle(runner);
    DlibOracle(runner, root);
    YunetOracle(runner);
    EyeInputsOracle(runner);
    CropAndVignetteOracle(runner);
    MatteCadenceOracle(runner);
    EyeCompositeOracle(runner);
    const auto *fixtures =
        std::getenv("STUDIOCAST_VULKAN_MODEL_CANDIDATE_ROOT");
    Need(fixtures, "required full oracle needs "
                   "STUDIOCAST_VULKAN_MODEL_CANDIDATE_ROOT numerical fixtures");
    FlxTailOracle(runner, fixtures);
    Need(runner.kernels.Shutdown(&runner.error), runner.error);
    std::cout << "PASS native tracking kernel oracle;maximum_execute_ms="
              << runner.maximum_execute_ms << "\n";
    return 0;
  } catch (const std::exception &x) {
    std::cerr << x.what() << '\n';
    return 1;
  }
}
