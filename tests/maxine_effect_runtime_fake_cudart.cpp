#include <cstdlib>
#include <cstring>

extern "C" int cudaMalloc(void **pointer, std::size_t size) {
  *pointer = std::malloc(size);
  return *pointer ? 0 : 1;
}
extern "C" int cudaFree(void *pointer) {
  std::free(pointer);
  return 0;
}
extern "C" int cudaMemset(void *pointer, int value, std::size_t size) {
  std::memset(pointer, value, size);
  return 0;
}
extern "C" int cudaMemsetAsync(void *pointer, int value, std::size_t size,
                                void *) {
  std::memset(pointer, value, size);
  return 0;
}
extern "C" const char *cudaGetErrorString(int) { return "fake cuda error"; }
