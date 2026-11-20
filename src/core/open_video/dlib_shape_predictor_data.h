#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "core/open_video/model_pack_registry.h"

namespace studiocast::open_video {

struct DlibPredictorSplit {
  std::uint32_t feature_a = 0;
  std::uint32_t feature_b = 0;
  float threshold = 0;
};
static_assert(sizeof(DlibPredictorSplit) == 12);

// Setup-only wire representation shared by native accelerator predictors. No
// backend or dlib SDK types escape this header. Every offset is an element
// index, not a byte offset; leaves contain 136 floats per selected leaf.
struct DlibPredictorData {
  std::array<float, 136> initial_shape{};
  std::vector<std::uint32_t> cascade_trees;
  std::vector<std::uint32_t> tree_splits;
  std::vector<std::uint32_t> tree_leaves;
  std::vector<DlibPredictorSplit> splits;
  std::vector<float> leaves;
  std::vector<std::uint32_t> cascade_features;
  std::vector<std::uint32_t> feature_anchors;
  std::vector<std::array<float, 2>> feature_deltas;
  // pinv(P), where P has columns (0,0,1), (1,0,1), (1,1,1).
  // Compute with the configured dlib algebra backend once at setup. The GPU
  // forms Q*pinv(P), retaining the native predictor's half-pixel decisions.
  std::array<double, 9> rectangle_basis_pseudoinverse{};
  std::uint32_t maximum_features = 0;
  std::uint32_t maximum_trees = 0;
  std::size_t resident_bytes = 0;
};

bool LoadDlibPredictorData(const ModelPack &pack,
                           const ValidatedModelArtifactIdentity &artifact,
                           std::size_t maximum_bytes, DlibPredictorData *out,
                           std::string *error);

} // namespace studiocast::open_video
