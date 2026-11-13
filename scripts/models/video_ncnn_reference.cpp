// Offline artifact oracle only. Never linked into the production daemon.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <net.h>
#include <string>
int main(int argc, char **argv) {
  if (argc == 2 && !std::strcmp(argv[1], "--describe")) {
    std::printf("{\"protocol\":\"studiocast-video-ncnn-reference-v1\",\"ncnn_"
                "version\":\"%s\",\"avx2\":%d,\"fma\":%d,\"runtime_cpu\":%d}\n",
                NCNN_VERSION_STRING, NCNN_AVX2, NCNN_FMA, NCNN_RUNTIME_CPU);
    return 0;
  }
  if (argc < 6)
    return 2;
  ncnn::Net net;
  net.opt.use_vulkan_compute = false;
  net.opt.use_winograd_convolution = false;
  net.opt.use_sgemm_convolution = false;
  net.opt.num_threads = 1;
  net.opt.use_packing_layout = false;
  net.opt.use_fp16_packed = false;
  net.opt.use_fp16_storage = false;
  net.opt.use_fp16_arithmetic = false;
  if (net.load_param(argv[1]) || net.load_model(argv[2]))
    return 3;
  auto ex = net.create_extractor();
  int k = 3;
  int ni = atoi(argv[k++]);
  if (ni < 1 || ni > 3 || argc < 5 + ni * 5)
    return 2;
  for (int i = 0; i < ni; i++) {
    const char *name = argv[k++];
    int w = atoi(argv[k++]), h = atoi(argv[k++]), c = atoi(argv[k++]);
    if (w <= 0 || h <= 0 || c <= 0 ||
        std::uint64_t(w) * std::uint64_t(h) * std::uint64_t(c) > (1u << 28))
      return 2;
    FILE *f = fopen(argv[k++], "rb");
    if (!f)
      return 4;
    ncnn::Mat a = c == 1 && h == 1 ? ncnn::Mat(w) : ncnn::Mat(w, h, c);
    for (int z = 0; z < c; z++)
      if (fread(a.channel(z), 4, w * h, f) != (size_t)(w * h))
        return 5;
    if (fgetc(f) != EOF)
      return 5;
    fclose(f);
    if (ex.input(name, a))
      return 6;
  }
  int no = atoi(argv[k++]);
  if (no < 1 || no > 12 || argc != k + no * 2)
    return 2;
  for (int i = 0; i < no; i++) {
    const char *name = argv[k++];
    const char *path = argv[k++];
    ncnn::Mat a;
    if (ex.extract(name, a))
      return 7;
    printf("%s %d %d %d %d %zu\n", name, a.dims, a.w, a.h, a.c, a.elemsize);
    FILE *f = fopen(path, "wb");
    if (!f)
      return 8;
    for (int z = 0; z < a.c; z++)
      fwrite(a.channel(z), 4, a.w * a.h, f);
    fclose(f);
  }
  return 0;
}
