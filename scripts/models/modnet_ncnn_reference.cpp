// Offline artifact reference only: CPU buffers here never attest GPU residency.
// Build against ncnn at modnet-ncnn.lock.json's commit:
// c++ -O2 -std=c++17 -I<source>/src -I<build>/src this.cpp \
//   <build>/src/libncnn.a -pthread -o modnet-ncnn-reference
#include <net.h>

#include <cstdio>
#include <cstring>

int main(int argc, char **argv) {
  if (argc == 2 && std::strcmp(argv[1], "--describe") == 0) {
    // Versioned helper protocol. Caller records the actual executable checksum.
    std::printf("{\"protocol\":\"studiocast-modnet-ncnn-reference-v1\","
                "\"ncnn_version\":\"%s\",\"avx2\":%d,\"fma\":%d,"
                "\"avxvnni\":%d,\"runtime_cpu\":%d}\n",
                NCNN_VERSION_STRING, NCNN_AVX2, NCNN_FMA, NCNN_AVXVNNI,
                NCNN_RUNTIME_CPU);
    return 0;
  }
  if (argc != 7)
    return 2;
  ncnn::Net net;
  net.opt.num_threads = 1;
  net.opt.use_packing_layout = false;
  net.opt.use_fp16_packed = false;
  net.opt.use_fp16_storage = false;
  net.opt.use_fp16_arithmetic = false;
  net.opt.use_bf16_storage = false;
  net.opt.use_vulkan_compute = false;
  // Prefer direct FP32 convolution for the offline numerical reference. Fast
  // convolution transforms amplify round-off before near-constant InstanceNorm.
  net.opt.use_winograd_convolution = false;
  net.opt.use_sgemm_convolution = false;
  if (net.load_param(argv[1]) || net.load_model(argv[2]))
    return 3;
  ncnn::Mat input(256, 256, 3);
  FILE *file = std::fopen(argv[3], "rb");
  if (!file)
    return 4;
  bool valid = true;
  for (int channel = 0; channel < 3; ++channel)
    valid &= std::fread(input.channel(channel), sizeof(float), 256 * 256,
                        file) == 256 * 256;
  valid &= std::fgetc(file) == EOF;
  std::fclose(file);
  if (!valid)
    return 5;
  auto extractor = net.create_extractor();
  ncnn::Mat alpha;
  if (extractor.input(argv[5], input) || extractor.extract(argv[6], alpha))
    return 6;
  if (alpha.dims != 3 || alpha.w != 256 || alpha.h != 256 || alpha.c != 1 ||
      alpha.elempack != 1 || alpha.elemsize != sizeof(float))
    return 7;
  file = std::fopen(argv[4], "wb");
  if (!file)
    return 8;
  valid = std::fwrite(alpha.channel(0), sizeof(float), 256 * 256, file) ==
          256 * 256;
  valid &= std::fclose(file) == 0;
  return valid ? 0 : 9;
}
