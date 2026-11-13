#include "core/open_video/dlib_shape_predictor_data.h"
#include "core/open_video/ncnn_vulkan_context.h"
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
} // namespace
int main() {
  if (!std::getenv("STUDIOCAST_REQUIRE_VULKAN_TRACKING")) {
    std::cout << "SKIP: opt-in resident tracking kernels\n";
    return 77;
  }
  try {
    const auto *root = std::getenv("STUDIOCAST_DLIB_ORACLE_MODELS_ROOT");
    Need(root, "set STUDIOCAST_DLIB_ORACLE_MODELS_ROOT");
    Runner runner;
    DlibOracle(runner, root);
    YunetOracle(runner);
    EyeInputsOracle(runner);
    if (const auto *fixtures =
            std::getenv("STUDIOCAST_VULKAN_MODEL_CANDIDATE_ROOT"))
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
