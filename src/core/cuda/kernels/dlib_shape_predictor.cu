#include <cstdint>
#include <cuda_runtime.h>

struct Split {
  unsigned int a, b;
  float threshold;
};

// PTX_GENERATE: nvcc -ptx -O3 --fmad=false -arch=compute_52
// dlib_shape_predictor.cu

// One thread is deliberate: dlib accumulates trees in file order, and changing
// that order changes float results and sometimes later cascade decisions.
extern "C" __global__ void dlib_shape_predictor_68(
    const unsigned char *rgb, unsigned int pitch, unsigned int width,
    unsigned int height, float left, float top, float right, float bottom,
    const float *initial, const unsigned int *cascade_trees,
    const unsigned int *tree_splits, const unsigned int *tree_leaves,
    const Split *splits, const float *leaves,
    const unsigned int *cascade_features, const unsigned int *anchors,
    const float2 *deltas, unsigned int cascades, float *shape, float *features,
    float *output) {
  if (blockIdx.x || threadIdx.x)
    return;
  for (int i = 0; i < 136; ++i)
    shape[i] = initial[i];
  for (unsigned int c = 0; c < cascades; ++c) {
    double mx0 = 0, mx1 = 0, my0 = 0, my1 = 0;
    for (int i = 0; i < 68; ++i) {
      mx0 += initial[2 * i];
      my0 += initial[2 * i + 1];
      mx1 += shape[2 * i];
      my1 += shape[2 * i + 1];
    }
    mx0 /= 68;
    my0 /= 68;
    mx1 /= 68;
    my1 /= 68;
    double aa = 0, bb = 0, den = 0;
    for (int i = 0; i < 68; ++i) {
      double x = initial[2 * i] - mx0, y = initial[2 * i + 1] - my0,
             u = shape[2 * i] - mx1, v = shape[2 * i + 1] - my1;
      aa += x * u + y * v;
      bb += x * v - y * u;
      den += x * x + y * y;
    }
    const float a = static_cast<float>(aa / den);
    const float b = static_cast<float>(bb / den);
    unsigned int fb = cascade_features[c], fe = cascade_features[c + 1];
    for (unsigned int f = fb; f < fe; ++f) {
      unsigned int k = anchors[f];
      float dx = deltas[f].x, dy = deltas[f].y;
      float nx = shape[2 * k] + a * dx - b * dy,
            ny = shape[2 * k + 1] + b * dx + a * dy;
      const double mapped_x =
          static_cast<double>(left) + static_cast<double>(nx) * (right - left);
      const double mapped_y =
          static_cast<double>(top) + static_cast<double>(ny) * (bottom - top);
      long x = static_cast<long>(floor(mapped_x + 0.5));
      long y = static_cast<long>(floor(mapped_y + 0.5));
      float value = 0;
      if (x >= 0 && y >= 0 && x < (long)width && y < (long)height) {
        const unsigned char *p = rgb + y * pitch + 3 * x;
        value = (p[0] + p[1] + p[2]) / 3;
      }
      features[f - fb] = value;
    }
    for (unsigned int t = cascade_trees[c]; t < cascade_trees[c + 1]; ++t) {
      unsigned int sb = tree_splits[t], n = 0;
      while (n < tree_splits[t + 1] - sb) {
        Split s = splits[sb + n];
        n = (features[s.a] - features[s.b] > s.threshold) ? 2 * n + 1
                                                          : 2 * n + 2;
      }
      unsigned int leaf = tree_leaves[t] + (n - (tree_splits[t + 1] - sb));
      const float *v = leaves + leaf * 136;
      for (int i = 0; i < 136; ++i)
        shape[i] += v[i];
    }
  }
  for (int i = 0; i < 68; ++i) {
    output[2 * i] =
        floor(static_cast<double>(left) +
              static_cast<double>(shape[2 * i]) * (right - left) + 0.5);
    output[2 * i + 1] =
        floor(static_cast<double>(top) +
              static_cast<double>(shape[2 * i + 1]) * (bottom - top) + 0.5);
  }
}
