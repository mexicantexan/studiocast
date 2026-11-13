#include "core/open_video/dlib_shape_predictor_data.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <streambuf>

#if STUDIOCAST_HAVE_DLIB
#include <dlib/matrix.h>
#include <dlib/serialize.h>
#endif

namespace studiocast::open_video {
namespace {
#if STUDIOCAST_HAVE_DLIB
class SnapshotStream final : public std::streambuf {
public:
  explicit SnapshotStream(std::FILE *file) : file_(file) {}

private:
  int_type underflow() override {
    if (gptr() && gptr() < egptr())
      return traits_type::to_int_type(*gptr());
    const auto n = std::fread(buffer_.data(), 1, buffer_.size(), file_);
    if (!n)
      return traits_type::eof();
    setg(buffer_.data(), buffer_.data(), buffer_.data() + n);
    return traits_type::to_int_type(*gptr());
  }
  std::FILE *file_;
  std::array<char, 8192> buffer_{};
};

struct Reader {
  std::istream &in;
  std::size_t maximum, used = 0;
  void Charge(std::size_t count, std::size_t size) {
    if (used > maximum || count > (maximum - used) / size)
      throw std::runtime_error("dlib predictor resource budget exhausted");
    used += count * size;
  }
  unsigned long Count() {
    unsigned long n = 0;
    dlib::deserialize(n, in);
    if (n > std::numeric_limits<std::uint32_t>::max())
      throw std::runtime_error("dlib predictor count exceeds uint32");
    return n;
  }
  float Scalar() {
    float x = 0;
    dlib::deserialize(x, in);
    if (!std::isfinite(x))
      throw std::runtime_error("non-finite dlib predictor value");
    return x;
  }
  void Shape() {
    long rows = 0, cols = 0;
    dlib::deserialize(rows, in);
    dlib::deserialize(cols, in);
    if (rows != -136 || cols != -1)
      throw std::runtime_error("invalid dlib 68-point shape dimensions");
  }
};
#endif
} // namespace

bool LoadDlibPredictorData(const ModelPack &pack,
                           const ValidatedModelArtifactIdentity &artifact,
                           std::size_t maximum_bytes, DlibPredictorData *out,
                           std::string *error) {
  if (error)
    error->clear();
  if (!out || !maximum_bytes || pack.task != "face_landmarks" ||
      !artifact.Valid()) {
    if (error)
      *error = "invalid dlib predictor setup";
    return false;
  }
#if !STUDIOCAST_HAVE_DLIB
  if (error)
    *error = "dlib predictor support is unavailable in this build";
  return false;
#else
  try {
    const ModelFile *file = nullptr;
    for (const auto &candidate : pack.files) {
      if (candidate.kind == "dlib_shape_predictor" &&
          candidate.role == "main") {
        if (file)
          throw std::runtime_error("duplicate dlib predictor artifacts");
        file = &candidate;
      }
    }
    if (!file)
      throw std::runtime_error("missing dlib predictor artifact");
    const auto validation = ValidateModelArtifactIdentity(pack, *file);
    if (!validation.ok() || !(validation.identity == artifact))
      throw std::runtime_error("selected dlib predictor identity mismatch");
    // Missing BYO manifest checksums remain policy-compatible. Selection still
    // computes an exact artifact identity; seal the consumed bytes against that
    // digest, including when the manifest intentionally reports "unchecked".
    constexpr char hex[] = "0123456789abcdef";
    std::string selected_digest(64, '0');
    for (std::size_t i = 0; i < validation.identity.artifact_sha256.size();
         ++i) {
      const auto byte = validation.identity.artifact_sha256[i];
      selected_digest[i * 2] = hex[byte >> 4];
      selected_digest[i * 2 + 1] = hex[byte & 15];
    }
    auto snapshot = OpenVerifiedModelFileSnapshot(file->path, selected_digest,
                                                  maximum_bytes, error);
    if (!snapshot)
      return false;
    SnapshotStream stream(snapshot.get());
    std::istream in(&stream);
    Reader reader{in, maximum_bytes};
    DlibPredictorData data;
    int version = 0;
    dlib::deserialize(version, in);
    if (version != 1)
      throw std::runtime_error("unsupported dlib predictor wire version");
    reader.Shape();
    reader.Charge(136, sizeof(float));
    for (auto &x : data.initial_shape)
      x = reader.Scalar();
    const auto cascades = reader.Count();
    if (!cascades || cascades > 1024)
      throw std::runtime_error("invalid dlib cascade count");
    reader.Charge(cascades + 1, sizeof(std::uint32_t) * 2);
    data.cascade_trees.push_back(0);
    data.tree_splits.push_back(0);
    data.tree_leaves.push_back(0);
    reader.Charge(2, sizeof(std::uint32_t));
    for (unsigned long c = 0; c < cascades; ++c) {
      const auto trees = reader.Count();
      if (!trees || trees > 100000)
        throw std::runtime_error("invalid dlib tree count");
      data.maximum_trees =
          std::max(data.maximum_trees, static_cast<std::uint32_t>(trees));
      reader.Charge(trees, 2 * sizeof(std::uint32_t));
      for (unsigned long t = 0; t < trees; ++t) {
        const auto count = reader.Count();
        reader.Charge(count, sizeof(DlibPredictorSplit));
        for (unsigned long s = 0; s < count; ++s) {
          const auto a = reader.Count(), b = reader.Count();
          data.splits.push_back({static_cast<std::uint32_t>(a),
                                 static_cast<std::uint32_t>(b),
                                 reader.Scalar()});
        }
        const auto leaves = reader.Count();
        if (!leaves || leaves != count + 1 || (leaves & (leaves - 1)))
          throw std::runtime_error("incomplete dlib regression tree");
        reader.Charge(leaves, 136 * sizeof(float));
        for (unsigned long leaf = 0; leaf < leaves; ++leaf) {
          reader.Shape();
          for (int p = 0; p < 136; ++p)
            data.leaves.push_back(reader.Scalar());
        }
        if (data.splits.size() > std::numeric_limits<std::uint32_t>::max() ||
            data.leaves.size() / 136 >
                std::numeric_limits<std::uint32_t>::max())
          throw std::runtime_error("dlib flattened table offset overflow");
        data.tree_splits.push_back(
            static_cast<std::uint32_t>(data.splits.size()));
        data.tree_leaves.push_back(
            static_cast<std::uint32_t>(data.leaves.size() / 136));
      }
      data.cascade_trees.push_back(
          static_cast<std::uint32_t>(data.tree_splits.size() - 1));
    }
    if (reader.Count() != cascades)
      throw std::runtime_error("dlib feature cascade count mismatch");
    data.cascade_features.push_back(0);
    for (unsigned long c = 0; c < cascades; ++c) {
      const auto count = reader.Count();
      if (!count)
        throw std::runtime_error("empty dlib cascade features");
      reader.Charge(count, sizeof(std::uint32_t));
      for (unsigned long f = 0; f < count; ++f) {
        const auto anchor = reader.Count();
        if (anchor >= 68)
          throw std::runtime_error("invalid dlib feature anchor");
        data.feature_anchors.push_back(static_cast<std::uint32_t>(anchor));
      }
      if (data.feature_anchors.size() >
          std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("dlib feature table offset overflow");
      data.cascade_features.push_back(
          static_cast<std::uint32_t>(data.feature_anchors.size()));
      data.maximum_features =
          std::max(data.maximum_features, static_cast<std::uint32_t>(count));
      for (auto t = data.cascade_trees[c]; t < data.cascade_trees[c + 1]; ++t)
        for (auto s = data.tree_splits[t]; s < data.tree_splits[t + 1]; ++s)
          if (data.splits[s].feature_a >= count ||
              data.splits[s].feature_b >= count)
            throw std::runtime_error("invalid dlib split feature index");
    }
    if (reader.Count() != cascades)
      throw std::runtime_error("dlib delta cascade count mismatch");
    for (unsigned long c = 0; c < cascades; ++c) {
      const auto count = reader.Count();
      if (count != data.cascade_features[c + 1] - data.cascade_features[c])
        throw std::runtime_error("dlib feature/delta count mismatch");
      reader.Charge(count, 2 * sizeof(float));
      for (unsigned long f = 0; f < count; ++f) {
        const float x = reader.Scalar(), y = reader.Scalar();
        data.feature_deltas.push_back({x, y});
      }
    }
    if (in.peek() != std::char_traits<char>::eof() ||
        std::ferror(snapshot.get()))
      throw std::runtime_error("dlib predictor trailing bytes/read failure");
    dlib::matrix<double, 3, 0> basis(3, 3);
    basis = 0, 1, 1, 0, 0, 1, 1, 1, 1;
    const dlib::matrix<double, 0, 3> inverse = dlib::pinv(basis);
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c)
        data.rectangle_basis_pseudoinverse[r * 3 + c] = inverse(r, c);
    reader.Charge(136 * 2, sizeof(float));
    reader.Charge(data.maximum_features, sizeof(float));
    reader.Charge(data.maximum_trees, sizeof(std::uint32_t));
    reader.Charge(9, sizeof(double));
    data.resident_bytes = reader.used;
    *out = std::move(data);
    return true;
  } catch (const std::exception &x) {
    if (error)
      *error = std::string("invalid dlib predictor: ") + x.what();
    return false;
  }
#endif
}
} // namespace studiocast::open_video
