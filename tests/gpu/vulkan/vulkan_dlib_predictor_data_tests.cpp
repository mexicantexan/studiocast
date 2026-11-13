#include "core/open_video/dlib_shape_predictor_data.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <dlib/image_processing/shape_predictor.h>
#include <dlib/serialize.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>

using namespace studiocast::open_video;
namespace {
void Require(bool yes, const std::string &message) {
  if (!yes)
    throw std::runtime_error(message);
}
struct Fixture {
  std::filesystem::path root;
  Fixture() {
    std::array<char, 128> path{};
    const std::string pattern =
        (std::filesystem::temp_directory_path() / "studiocast-dlib-data-XXXXXX")
            .string();
    std::copy(pattern.begin(), pattern.end(), path.begin());
    Require(::mkdtemp(path.data()), "mkdtemp");
    root = path.data();
  }
  ~Fixture() { std::filesystem::remove_all(root); }
};
ModelPack Pack(const std::filesystem::path &root) {
  const auto dir = root / "face_landmarks" / "tiny";
  std::string hash, error;
  Require(studiocast::model_integrity_internal::ComputeSha256File(
              dir / "model.dat", &hash, &error),
          error);
  std::ofstream(dir / "LICENSE.txt")
      << "Synthetic test data created by this test.\n";
  std::ofstream(dir / "model.json")
      << R"({"schema_version":2,"id":"tiny-dlib68","display_name":"tiny","task":"face_landmarks","files":[{"name":"model.dat","kind":"dlib_shape_predictor","role":"main","sha256":")"
      << hash
      << R"("}],"dlib":{"predictor_kind":"shape_predictor_68_face_landmarks","num_landmarks":68}})";
  auto registry = ModelPackRegistry::Scan(root);
  const auto found = registry.Find("face_landmarks", "tiny-dlib68");
  Require(found.has_value(), "synthetic pack discovery");
  return *found;
}
dlib::shape_predictor Predictor() {
  dlib::matrix<float, 0, 1> initial(136);
  for (int i = 0; i < 68; ++i) {
    initial(2 * i) = .2F + .006F * i;
    initial(2 * i + 1) = .25F + .003F * (i % 11);
  }
  dlib::impl::regression_tree tree;
  tree.splits.push_back({0, 1, 2.F});
  tree.leaf_values.emplace_back(136);
  tree.leaf_values.emplace_back(136);
  tree.leaf_values[0] = .0003F;
  tree.leaf_values[1] = -.0002F;
  std::vector<std::vector<dlib::impl::regression_tree>> forests{{tree, tree},
                                                                {tree}};
  std::vector<std::vector<dlib::vector<float, 2>>> pixels{
      {{.3F, .4F}, {.6F, .7F}}, {{.2F, .6F}, {.7F, .1F}}};
  return dlib::shape_predictor(initial, forests, pixels);
}
std::array<float, 136> Oracle(const DlibPredictorData &d,
                              const dlib::array2d<dlib::rgb_pixel> &image,
                              const dlib::rectangle &box) {
  std::array<double, 6> m{};
  const std::array<double, 3> x{double(float(box.left())),
                                double(float(box.right())),
                                double(float(box.right()))};
  const std::array<double, 3> y{double(float(box.top())),
                                double(float(box.top())),
                                double(float(box.bottom()))};
  for (unsigned j = 0; j < 3; ++j) {
    m[j] = x[0] * d.rectangle_basis_pseudoinverse[j];
    m[j] += x[1] * d.rectangle_basis_pseudoinverse[3 + j];
    m[j] += x[2] * d.rectangle_basis_pseudoinverse[6 + j];
    m[3 + j] = y[0] * d.rectangle_basis_pseudoinverse[j];
    m[3 + j] += y[1] * d.rectangle_basis_pseudoinverse[3 + j];
    m[3 + j] += y[2] * d.rectangle_basis_pseudoinverse[6 + j];
  }
  auto shape = d.initial_shape;
  std::vector<float> features(d.maximum_features);
  for (unsigned c = 0; c + 1 < d.cascade_trees.size(); ++c) {
    double mx0 = 0, mx1 = 0, my0 = 0, my1 = 0;
    for (int i = 0; i < 68; i++) {
      mx0 += d.initial_shape[2 * i];
      my0 += d.initial_shape[2 * i + 1];
      mx1 += shape[2 * i];
      my1 += shape[2 * i + 1];
    }
    mx0 /= 68;
    my0 /= 68;
    mx1 /= 68;
    my1 /= 68;
    double aa = 0, bb = 0, den = 0;
    for (int i = 0; i < 68; i++) {
      double x0 = d.initial_shape[2 * i] - mx0,
             y0 = d.initial_shape[2 * i + 1] - my0;
      double u = shape[2 * i] - mx1, v = shape[2 * i + 1] - my1;
      aa += x0 * u + y0 * v;
      bb += x0 * v - y0 * u;
      den += x0 * x0 + y0 * y0;
    }
    const float a = aa / den, b = bb / den;
    for (unsigned f = d.cascade_features[c]; f < d.cascade_features[c + 1];
         f++) {
      const auto k = d.feature_anchors[f];
      const auto delta = d.feature_deltas[f];
      const float nx = shape[2 * k] + a * delta[0] - b * delta[1];
      const float ny = shape[2 * k + 1] + b * delta[0] + a * delta[1];
      const int x0 = std::floor(m[0] * nx + m[1] * ny + m[2] + .5);
      const int y0 = std::floor(m[3] * nx + m[4] * ny + m[5] + .5);
      float value = 0;
      if (x0 >= 0 && y0 >= 0 && x0 < image.nc() && y0 < image.nr()) {
        const auto px = image[y0][x0];
        value = (int(px.red) + px.green + px.blue) / 3;
      }
      features[f - d.cascade_features[c]] = value;
    }
    for (unsigned t = d.cascade_trees[c]; t < d.cascade_trees[c + 1]; t++) {
      const unsigned sb = d.tree_splits[t], count = d.tree_splits[t + 1] - sb;
      unsigned n = 0;
      while (n < count) {
        const auto s = d.splits[sb + n];
        n = features[s.feature_a] - features[s.feature_b] > s.threshold
                ? 2 * n + 1
                : 2 * n + 2;
      }
      const unsigned leaf = d.tree_leaves[t] + n - count;
      for (unsigned i = 0; i < 136; i++)
        shape[i] += d.leaves[leaf * 136 + i];
    }
  }
  for (unsigned i = 0; i < 68; i++) {
    const float x = shape[2 * i], y = shape[2 * i + 1];
    shape[2 * i] = std::floor(m[0] * x + m[1] * y + m[2] + .5);
    shape[2 * i + 1] = std::floor(m[3] * x + m[4] * y + m[5] + .5);
  }
  return shape;
}
void Compare(const DlibPredictorData &data,
             const dlib::shape_predictor &predictor) {
  const std::array<dlib::rectangle, 5> boxes{
      dlib::rectangle(7, 11, 180, 170), dlib::rectangle(0, 0, 256, 192),
      dlib::rectangle(51, 27, 205, 178), dlib::rectangle(-13, 9, 150, 181),
      dlib::rectangle(100, -19, 270, 150)};
  dlib::array2d<dlib::rgb_pixel> image(193, 257);
  for (int pattern = 0; pattern < 4; pattern++) {
    for (int y = 0; y < 193; y++)
      for (int x = 0; x < 257; x++)
        image[y][x] = dlib::rgb_pixel((x * 17 + y * 3 + pattern * 41) & 255,
                                      (x * 5 + y * 29 + pattern * 13) & 255,
                                      (x * 11 + y * 7 + pattern * 67) & 255);
    for (const auto &box : boxes) {
      const auto actual = Oracle(data, image, box);
      const auto expected = predictor(image, box);
      for (unsigned i = 0; i < 68; i++)
        Require(actual[2 * i] == expected.part(i).x() &&
                    actual[2 * i + 1] == expected.part(i).y(),
                "flattened predictor differs from dlib oracle");
    }
  }
}
} // namespace
int main() {
  try {
    Fixture fixture;
    const auto dir = fixture.root / "face_landmarks" / "tiny";
    std::filesystem::create_directories(dir);
    const auto predictor = Predictor();
    dlib::serialize((dir / "model.dat").string()) << predictor;
    auto pack = Pack(fixture.root);
    auto validation = ValidateModelArtifactIdentity(pack, pack.files[0]);
    Require(validation.ok(), validation.detail);
    DlibPredictorData data;
    std::string error;
    Require(LoadDlibPredictorData(pack, validation.identity, 1u << 20, &data,
                                  &error),
            error);
    Require(data.cascade_trees.size() == 3 && data.splits.size() == 3 &&
                data.maximum_features == 2,
            "flattened dimensions");
    Compare(data, predictor);
    Require(
        !LoadDlibPredictorData(pack, validation.identity, 16, &data, &error),
        "low budget accepted");
    auto wrong = validation.identity;
    wrong.artifact_sha256[0] ^= 1;
    Require(!LoadDlibPredictorData(pack, wrong, 1u << 20, &data, &error),
            "wrong identity accepted");
    std::ofstream(dir / "model.dat", std::ios::app) << "trailing";
    Require(!LoadDlibPredictorData(pack, validation.identity, 1u << 20, &data,
                                   &error),
            "changed artifact accepted");
    pack = Pack(fixture.root);
    validation = ValidateModelArtifactIdentity(pack, pack.files[0]);
    Require(!LoadDlibPredictorData(pack, validation.identity, 1u << 20, &data,
                                   &error),
            "trailing wire data accepted");
    if (const char *root = std::getenv("STUDIOCAST_DLIB_ORACLE_MODELS_ROOT")) {
      const auto registry = ModelPackRegistry::Scan(root);
      const auto real = registry.Find("face_landmarks", "dlib_68_ibug_300w");
      Require(real.has_value(), "real predictor pack missing");
      const auto id = ValidateModelArtifactIdentity(*real, real->files[0]);
      Require(id.ok(), id.detail);
      Require(
          LoadDlibPredictorData(*real, id.identity, 256u << 20, &data, &error),
          error);
      dlib::shape_predictor reference;
      dlib::deserialize(real->files[0].path.string()) >> reference;
      Compare(data, reference);
      std::cout << "Real bounded predictor/dlib oracle: exact136coordinates "
                   "across20cases; resident="
                << data.resident_bytes << "\n";
    }
    std::cout << "PASS: sealed bounded forest loader, identities, budgets, "
                 "wire validation, affine and20case CPU oracle\n";
    return 0;
  } catch (const std::exception &x) {
    std::cerr << x.what() << '\n';
    return 1;
  }
}
