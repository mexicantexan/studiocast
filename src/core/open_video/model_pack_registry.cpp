#include "core/open_video/model_pack_registry.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <map>
#include <sstream>

#include "core/util/fs.h"
#include "core/util/json.h"
#include "core/util/xdg.h"

namespace fs = std::filesystem;

namespace studiocast::model_integrity_internal {
namespace {

constexpr std::array<std::uint32_t, 64> kSha256K{
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
    0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
    0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
    0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
    0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
    0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
    0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
    0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
    0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

std::uint32_t RotateRight(std::uint32_t v, int n) {
  return (v >> n) | (v << (32 - n));
}

class Sha256 {
public:
  void Update(const std::uint8_t *data, std::size_t len) {
    for (std::size_t i = 0; i < len; ++i) {
      data_[datalen_++] = data[i];
      if (datalen_ == data_.size()) {
        Transform(data_.data());
        bitlen_ += 512;
        datalen_ = 0;
      }
    }
  }

  std::array<std::uint8_t, 32> Final() {
    std::size_t i = datalen_;

    if (datalen_ < 56) {
      data_[i++] = 0x80u;
      while (i < 56)
        data_[i++] = 0x00u;
    } else {
      data_[i++] = 0x80u;
      while (i < 64)
        data_[i++] = 0x00u;
      Transform(data_.data());
      data_.fill(0);
    }

    bitlen_ += static_cast<std::uint64_t>(datalen_) * 8u;
    for (int j = 0; j < 8; ++j) {
      const std::size_t idx = 63u - static_cast<std::size_t>(j);
      data_[idx] = static_cast<std::uint8_t>((bitlen_ >> (j * 8)) & 0xffu);
    }
    Transform(data_.data());

    std::array<std::uint8_t, 32> hash{};
    for (std::size_t j = 0; j < state_.size(); ++j) {
      hash[j * 4 + 0] = static_cast<std::uint8_t>((state_[j] >> 24) & 0xffu);
      hash[j * 4 + 1] = static_cast<std::uint8_t>((state_[j] >> 16) & 0xffu);
      hash[j * 4 + 2] = static_cast<std::uint8_t>((state_[j] >> 8) & 0xffu);
      hash[j * 4 + 3] = static_cast<std::uint8_t>(state_[j] & 0xffu);
    }
    return hash;
  }

private:
  void Transform(const std::uint8_t *data) {
    std::array<std::uint32_t, 64> m{};
    for (std::size_t i = 0, j = 0; i < 16; ++i, j += 4) {
      m[i] = (static_cast<std::uint32_t>(data[j]) << 24) |
             (static_cast<std::uint32_t>(data[j + 1]) << 16) |
             (static_cast<std::uint32_t>(data[j + 2]) << 8) |
             static_cast<std::uint32_t>(data[j + 3]);
    }
    for (std::size_t i = 16; i < 64; ++i) {
      const std::uint32_t s0 = RotateRight(m[i - 15], 7) ^
                               RotateRight(m[i - 15], 18) ^ (m[i - 15] >> 3);
      const std::uint32_t s1 = RotateRight(m[i - 2], 17) ^
                               RotateRight(m[i - 2], 19) ^ (m[i - 2] >> 10);
      m[i] = m[i - 16] + s0 + m[i - 7] + s1;
    }

    std::uint32_t a = state_[0];
    std::uint32_t b = state_[1];
    std::uint32_t c = state_[2];
    std::uint32_t d = state_[3];
    std::uint32_t e = state_[4];
    std::uint32_t f = state_[5];
    std::uint32_t g = state_[6];
    std::uint32_t h = state_[7];

    for (std::size_t i = 0; i < 64; ++i) {
      const std::uint32_t s1 =
          RotateRight(e, 6) ^ RotateRight(e, 11) ^ RotateRight(e, 25);
      const std::uint32_t ch = (e & f) ^ (~e & g);
      const std::uint32_t temp1 = h + s1 + ch + kSha256K[i] + m[i];
      const std::uint32_t s0 =
          RotateRight(a, 2) ^ RotateRight(a, 13) ^ RotateRight(a, 22);
      const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temp2 = s0 + maj;

      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }

    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<std::uint8_t, 64> data_{};
  std::size_t datalen_ = 0;
  std::uint64_t bitlen_ = 0;
  std::array<std::uint32_t, 8> state_{0x6a09e667u, 0xbb67ae85u,
                                      0x3c6ef372u, 0xa54ff53au,
                                      0x510e527fu, 0x9b05688cu,
                                      0x1f83d9abu, 0x5be0cd19u};
};

std::string HashToHex(const std::array<std::uint8_t, 32> &hash) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string out;
  out.reserve(64);
  for (const auto b : hash) {
    out.push_back(kHex[(b >> 4) & 0x0f]);
    out.push_back(kHex[b & 0x0f]);
  }
  return out;
}

} // namespace

std::string NormalizeSha256Hex(std::string s) {
  for (char &c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return s;
}

bool IsSha256Hex(const std::string &s) {
  if (s.size() != 64)
    return false;
  for (const char c : s) {
    if (!std::isxdigit(static_cast<unsigned char>(c)))
      return false;
  }
  return true;
}

bool ComputeSha256File(const fs::path &path, std::string *out,
                       std::string *error) {
  if (out)
    out->clear();
  if (error)
    error->clear();

  std::ifstream f(path, std::ios::binary);
  if (!f) {
    if (error)
      *error = std::string("failed to open file: ") + path.string();
    return false;
  }

  Sha256 sha;
  std::array<char, 64 * 1024> buf{};
  while (f) {
    f.read(buf.data(), static_cast<std::streamsize>(buf.size()));
    const std::streamsize got = f.gcount();
    if (got > 0) {
      sha.Update(reinterpret_cast<const std::uint8_t *>(buf.data()),
                 static_cast<std::size_t>(got));
    }
  }
  if (f.bad()) {
    if (error)
      *error = std::string("failed to read file: ") + path.string();
    return false;
  }

  if (out)
    *out = HashToHex(sha.Final());
  return true;
}

} // namespace studiocast::model_integrity_internal

namespace studiocast::open_video {
namespace {

std::string PathForError(const fs::path &p) {
  // Avoid platform-specific quoting. We only need human-readable paths.
  return p.string();
}

bool IsSafeRelativePath(const fs::path &p) {
  if (p.empty())
    return false;
  if (p.is_absolute())
    return false;
  for (const auto &part : p) {
    if (part == "." || part == "..")
      return false;
  }
  return true;
}

bool Fail(std::string *error, std::string msg) {
  if (error)
    *error = std::move(msg);
  return false;
}

const util::json::Value::Object *
AsObject(const util::json::Value &v, std::string *error, const char *what) {
  const auto *o = v.AsObject();
  if (o)
    return o;
  if (error)
    *error = std::string("model.json: expected object for ") + what;
  return nullptr;
}

const util::json::Value *Get(const util::json::Value::Object &o,
                             const char *key) {
  auto it = o.find(key);
  if (it == o.end())
    return nullptr;
  return &it->second;
}

bool GetStringRequired(const util::json::Value::Object &o, const char *key,
                       std::string *out, std::string *error) {
  const auto *v = Get(o, key);
  if (!v)
    return Fail(error, std::string("model.json: missing required field '") +
                           key + "'");
  const auto *s = v->AsString();
  if (!s)
    return Fail(error, std::string("model.json: field '") + key +
                           "' must be a string");
  if (s->empty())
    return Fail(error, std::string("model.json: field '") + key +
                           "' must be non-empty");
  *out = *s;
  return true;
}

bool GetStringOptional(const util::json::Value::Object &o, const char *key,
                       std::string *out, std::string *error) {
  const auto *v = Get(o, key);
  if (!v)
    return true;
  const auto *s = v->AsString();
  if (!s)
    return Fail(error, std::string("model.json: field '") + key +
                           "' must be a string");
  *out = *s;
  return true;
}

bool GetIntOptional(const util::json::Value::Object &o, const char *key,
                    int *out, std::string *error) {
  const auto *v = Get(o, key);
  if (!v)
    return true;
  const auto *n = v->AsNumber();
  if (!n)
    return Fail(error, std::string("model.json: field '") + key +
                           "' must be a number");
  const int i = static_cast<int>(*n);
  if (static_cast<double>(i) != *n) {
    return Fail(error, std::string("model.json: field '") + key +
                           "' must be an integer");
  }
  *out = i;
  return true;
}

bool GetIntRequired(const util::json::Value::Object &o, const char *key,
                    int *out, std::string *error) {
  const auto *v = Get(o, key);
  if (!v)
    return Fail(error, std::string("model.json: missing required field '") +
                           key + "'");
  const auto *n = v->AsNumber();
  if (!n)
    return Fail(error, std::string("model.json: field '") + key +
                           "' must be a number");
  const int i = static_cast<int>(*n);
  if (static_cast<double>(i) != *n) {
    return Fail(error, std::string("model.json: field '") + key +
                           "' must be an integer");
  }
  *out = i;
  return true;
}

bool GetNumberArray3Required(const util::json::Value::Object &o,
                             const char *key, std::array<double, 3> *out,
                             std::string *error) {
  const auto *v = Get(o, key);
  if (!v)
    return Fail(error, std::string("model.json: missing required field '") +
                           key + "'");
  const auto *a = v->AsArray();
  if (!a)
    return Fail(error, std::string("model.json: field '") + key +
                           "' must be an array");
  if (a->size() != 3)
    return Fail(error, std::string("model.json: field '") + key +
                           "' must have length 3");
  for (std::size_t i = 0; i < 3; ++i) {
    const auto *n = (*a)[i].AsNumber();
    if (!n)
      return Fail(error, std::string("model.json: field '") + key +
                             "' must contain only numbers");
    (*out)[i] = *n;
  }
  return true;
}

bool IsOneOf(const std::string &v,
             std::initializer_list<const char *> allowed) {
  for (const char *a : allowed) {
    if (v == a)
      return true;
  }
  return false;
}

bool IsPlaceholderModelId(const std::string &id) {
  return id.find("placeholder") != std::string::npos;
}

bool IsDownloadOnlyChecksum(const ModelPack &pack, const ModelFile &file) {
  return pack.task == "matting" &&
         (pack.id == "birefnet_lite" || pack.id == "birefnet_portrait") &&
         file.name.ends_with(".onnx");
}

std::string ClassifyLoadError(const std::string &err) {
  if (err.find("missing model file") != std::string::npos ||
      err.find("missing ONNX file") != std::string::npos ||
      err.find("missing model.json") != std::string::npos ||
      err.find("No such") != std::string::npos) {
    return "missing";
  }
  return "invalid_manifest";
}

std::string ClassifiedMessage(const std::string &status,
                              const std::string &message) {
  if (message.empty())
    return status;
  if (message.starts_with(status + ":"))
    return message;
  return status + ": " + message;
}

bool GetStringArrayOptional(const util::json::Value::Object &o, const char *key,
                            std::vector<std::string> *out, std::string *error) {
  const auto *v = Get(o, key);
  if (!v)
    return true;
  const auto *a = v->AsArray();
  if (!a)
    return Fail(error, std::string("model.json: field '") + key +
                           "' must be an array");
  out->clear();
  out->reserve(a->size());
  for (const auto &el : *a) {
    const auto *s = el.AsString();
    if (!s)
      return Fail(error, std::string("model.json: field '") + key +
                             "' must contain only strings");
    if (s->empty())
      continue;
    out->push_back(*s);
  }
  return true;
}

bool ParseMattingSpecFromV1Fields(const util::json::Value::Object &root,
                                  MattingSpec *out, std::string *error) {
  const auto *inputV = Get(root, "input");
  if (!inputV)
    return Fail(error, "model.json: missing required field 'input'");
  const auto *inputObj = AsObject(*inputV, error, "'input'");
  if (!inputObj)
    return false;

  if (!GetStringRequired(*inputObj, "name", &out->input.name, error))
    return false;
  if (!GetStringRequired(*inputObj, "layout", &out->input.layout, error))
    return false;
  if (!GetStringRequired(*inputObj, "dtype", &out->input.dtype, error))
    return false;
  if (!GetIntRequired(*inputObj, "width", &out->input.width, error))
    return false;
  if (!GetIntRequired(*inputObj, "height", &out->input.height, error))
    return false;
  if (!GetIntRequired(*inputObj, "channels", &out->input.channels, error))
    return false;

  if (!IsOneOf(out->input.layout, {"nchw", "nhwc"})) {
    return Fail(error, "model.json: input.layout must be one of: nchw, nhwc");
  }
  if (!IsOneOf(out->input.dtype, {"float32", "float16"})) {
    return Fail(error,
                "model.json: input.dtype must be one of: float32, float16");
  }
  if (out->input.width <= 0 || out->input.height <= 0 ||
      out->input.channels <= 0) {
    return Fail(error,
                "model.json: input width/height/channels must be positive");
  }

  const auto *outputV = Get(root, "output");
  if (!outputV)
    return Fail(error, "model.json: missing required field 'output'");
  const auto *outputObj = AsObject(*outputV, error, "'output'");
  if (!outputObj)
    return false;

  if (!GetStringRequired(*outputObj, "name", &out->output.name, error))
    return false;
  if (!GetStringRequired(*outputObj, "kind", &out->output.kind, error))
    return false;
  if (!GetStringRequired(*outputObj, "dtype", &out->output.dtype, error))
    return false;

  if (!IsOneOf(out->output.kind, {"alpha"})) {
    return Fail(error, "model.json: output.kind must be 'alpha'");
  }
  if (!IsOneOf(out->output.dtype, {"float32", "float16"})) {
    return Fail(error,
                "model.json: output.dtype must be one of: float32, float16");
  }

  const auto *ppV = Get(root, "preprocess");
  if (!ppV)
    return Fail(error, "model.json: missing required field 'preprocess'");
  const auto *ppObj = AsObject(*ppV, error, "'preprocess'");
  if (!ppObj)
    return false;

  if (!GetNumberArray3Required(*ppObj, "mean", &out->preprocess.mean, error))
    return false;
  if (!GetNumberArray3Required(*ppObj, "std", &out->preprocess.std, error))
    return false;
  if (!GetStringRequired(*ppObj, "color", &out->preprocess.color, error))
    return false;
  if (!GetStringRequired(*ppObj, "range", &out->preprocess.range, error))
    return false;

  if (!IsOneOf(out->preprocess.color, {"rgb"})) {
    return Fail(error, "model.json: preprocess.color must be 'rgb'");
  }
  if (!IsOneOf(out->preprocess.range, {"0..1"})) {
    return Fail(error, "model.json: preprocess.range must be '0..1'");
  }

  return true;
}

const ModelFile *FindUniqueModelFile(const ModelPack &pack,
                                     const std::string &name,
                                     const std::string &kind,
                                     std::string *error) {
  const ModelFile *found = nullptr;
  for (const auto &file : pack.files) {
    if (file.name != name)
      continue;
    if (found) {
      Fail(error, "model.json: ncnn_vulkan artifact '" + name +
                      "' is declared more than once in files[]");
      return nullptr;
    }
    found = &file;
  }
  if (!found) {
    Fail(error, "model.json: ncnn_vulkan artifact '" + name +
                    "' must be declared in files[]");
    return nullptr;
  }
  if (found->kind != kind) {
    Fail(error, "model.json: ncnn_vulkan artifact '" + name +
                    "' must use files[].kind='" + kind + "'");
    return nullptr;
  }
  if (!studiocast::model_integrity_internal::IsSha256Hex(found->sha256)) {
    Fail(error, "model.json: ncnn_vulkan artifact '" + name +
                    "' requires a 64-character files[].sha256 digest");
    return nullptr;
  }
  return found;
}

bool ParseNcnnVulkanMattingSpec(const util::json::Value::Object &root,
                                ModelPack *pack, std::string *error) {
  pack->ncnn_vulkan.reset();
  const auto *spec_value = Get(root, "ncnn_vulkan");
  if (!spec_value)
    return true;
  if (pack->schema_version != 2 || pack->task != "matting") {
    return Fail(error,
                "model.json: ncnn_vulkan is supported only for schema-v2 "
                "matting packs");
  }

  const auto *spec_object = AsObject(*spec_value, error, "'ncnn_vulkan'");
  if (!spec_object)
    return false;

  std::string param_file;
  std::string bin_file;
  NcnnVulkanMattingSpec spec;
  if (!GetStringRequired(*spec_object, "param_file", &param_file, error) ||
      !GetStringRequired(*spec_object, "bin_file", &bin_file, error) ||
      !GetStringRequired(*spec_object, "input_blob", &spec.input_blob,
                         error) ||
      !GetStringRequired(*spec_object, "output_blob", &spec.output_blob,
                         error) ||
      !GetStringRequired(*spec_object, "precision", &spec.precision, error)) {
    return false;
  }
  if (!IsSafeRelativePath(fs::path(param_file)) ||
      !IsSafeRelativePath(fs::path(bin_file))) {
    return Fail(error,
                "model.json: ncnn_vulkan param_file/bin_file must be safe "
                "relative paths");
  }
  if (param_file == bin_file) {
    return Fail(error,
                "model.json: ncnn_vulkan param_file and bin_file must differ");
  }
  if (!IsOneOf(spec.precision, {"fp32", "fp16"})) {
    return Fail(error,
                "model.json: ncnn_vulkan.precision must be one of: fp32, fp16");
  }

  const auto *converter_value = Get(*spec_object, "converter");
  if (!converter_value) {
    return Fail(error,
                "model.json: ncnn_vulkan missing required field 'converter'");
  }
  const auto *converter = AsObject(*converter_value, error,
                                   "'ncnn_vulkan.converter'");
  if (!converter ||
      !GetStringRequired(*converter, "name", &spec.converter_name, error) ||
      !GetStringRequired(*converter, "version", &spec.converter_version,
                         error)) {
    return false;
  }

  const ModelFile *param =
      FindUniqueModelFile(*pack, param_file, "ncnn_param", error);
  if (!param)
    return false;
  const ModelFile *bin = FindUniqueModelFile(*pack, bin_file, "ncnn_bin", error);
  if (!bin)
    return false;

  std::error_code ec;
  spec.param_path = fs::absolute(param->path, ec).lexically_normal();
  if (ec)
    return Fail(error, "model.json: failed to resolve ncnn_vulkan param_file");
  spec.bin_path = fs::absolute(bin->path, ec).lexically_normal();
  if (ec)
    return Fail(error, "model.json: failed to resolve ncnn_vulkan bin_file");
  spec.param_sha256 =
      studiocast::model_integrity_internal::NormalizeSha256Hex(param->sha256);
  spec.bin_sha256 =
      studiocast::model_integrity_internal::NormalizeSha256Hex(bin->sha256);
  pack->ncnn_vulkan = std::move(spec);
  return true;
}

bool ParseModelJsonV1(const fs::path &pack_dir, ModelPack *out,
                      std::string *error) {
  const fs::path manifestPath = pack_dir / "model.json";
  out->root_dir = pack_dir;
  out->manifest_path = manifestPath;
  out->schema_version = 1;

  const auto textOpt = util::ReadTextFile(manifestPath.string());
  if (!textOpt) {
    return Fail(error, std::string("missing model.json at ") +
                           PathForError(manifestPath));
  }

  util::json::Value root;
  std::string parseErr;
  if (!util::json::Parse(*textOpt, &root, &parseErr)) {
    return Fail(error, std::string("model.json: ") + parseErr);
  }

  const auto *obj = root.AsObject();
  if (!obj)
    return Fail(error, "model.json: root must be an object");

  if (!GetStringRequired(*obj, "id", &out->id, error))
    return false;
  if (!GetStringOptional(*obj, "display_name", &out->display_name, error))
    return false;
  if (!GetStringRequired(*obj, "task", &out->task, error))
    return false;

  std::string onnx_filename;
  if (!GetStringRequired(*obj, "onnx_filename", &onnx_filename, error))
    return false;

  const fs::path rel = onnx_filename;
  if (!IsSafeRelativePath(rel)) {
    return Fail(error,
                "model.json: onnx_filename must be a safe relative path");
  }

  ModelFile f;
  f.name = onnx_filename;
  f.kind = "onnx";
  f.role = "main";
  f.sha256 = "";
  f.path = pack_dir / rel;

  std::error_code ec;
  if (!fs::exists(f.path, ec) || ec) {
    return Fail(error,
                std::string("missing model file: ") + PathForError(f.path));
  }
  if (!fs::is_regular_file(f.path, ec) || ec) {
    return Fail(error, std::string("model file is not a regular file: ") +
                           PathForError(f.path));
  }

  out->files.clear();
  out->files.push_back(std::move(f));

  out->matting.reset();
  if (out->task == "matting") {
    MattingSpec spec;
    std::string perr;
    if (!ParseMattingSpecFromV1Fields(*obj, &spec, &perr)) {
      return Fail(error, perr);
    }
    out->matting = std::move(spec);
  }

  return true;
}

bool ParseModelJsonV2(const fs::path &pack_dir, ModelPack *out,
                      std::string *error) {
  const fs::path manifestPath = pack_dir / "model.json";
  out->root_dir = pack_dir;
  out->manifest_path = manifestPath;
  out->schema_version = 2;

  const auto textOpt = util::ReadTextFile(manifestPath.string());
  if (!textOpt) {
    return Fail(error, std::string("missing model.json at ") +
                           PathForError(manifestPath));
  }

  util::json::Value root;
  std::string parseErr;
  if (!util::json::Parse(*textOpt, &root, &parseErr)) {
    return Fail(error, std::string("model.json: ") + parseErr);
  }

  const auto *obj = root.AsObject();
  if (!obj)
    return Fail(error, "model.json: root must be an object");

  if (!GetStringRequired(*obj, "id", &out->id, error))
    return false;
  if (!GetStringOptional(*obj, "display_name", &out->display_name, error))
    return false;
  if (!GetStringRequired(*obj, "task", &out->task, error))
    return false;

  if (!GetStringArrayOptional(*obj, "depends_on", &out->depends_on, error))
    return false;

  const auto *filesVal = Get(*obj, "files");
  if (!filesVal)
    return Fail(error, "model.json: missing required field 'files'");
  const auto *filesArr = filesVal->AsArray();
  if (!filesArr)
    return Fail(error, "model.json: field 'files' must be an array");
  if (filesArr->empty())
    return Fail(error, "model.json: field 'files' must be non-empty");

  out->files.clear();
  out->files.reserve(filesArr->size());
  for (const auto &el : *filesArr) {
    const auto *fo = el.AsObject();
    if (!fo)
      return Fail(error, "model.json: files[] entries must be objects");

    ModelFile f;
    if (!GetStringRequired(*fo, "name", &f.name, error))
      return false;
    if (!GetStringRequired(*fo, "kind", &f.kind, error))
      return false;
    if (!GetStringOptional(*fo, "role", &f.role, error))
      return false;
    if (!GetStringOptional(*fo, "sha256", &f.sha256, error))
      return false;

    const fs::path rel = f.name;
    if (!IsSafeRelativePath(rel)) {
      return Fail(error,
                  "model.json: files[].name must be a safe relative path");
    }
    f.path = pack_dir / rel;

    std::error_code ec;
    if (!fs::exists(f.path, ec) || ec) {
      std::ostringstream oss;
      oss << "missing model file: " << PathForError(f.path);
      if (!out->task.empty())
        oss << " (task=" << out->task << ")";
      return Fail(error, oss.str());
    }
    if (!fs::is_regular_file(f.path, ec) || ec) {
      return Fail(error, std::string("model file is not a regular file: ") +
                             PathForError(f.path));
    }

    out->files.push_back(std::move(f));
  }

  out->matting.reset();
  if (out->task == "matting") {
    MattingSpec spec;
    std::string perr;
    if (!ParseMattingSpecFromV1Fields(*obj, &spec, &perr)) {
      return Fail(error, perr);
    }
    out->matting = std::move(spec);
  }

  if (!ParseNcnnVulkanMattingSpec(*obj, out, error))
    return false;

  return true;
}

std::string PackDirKey(const fs::path &root, const fs::path &pack_dir) {
  // Use a stable, human-readable key for error reporting.
  // Prefer the relative path from the scan root, prefixed with the root
  // directory name. Example (when scanning
  // ~/.local/share/studiocast/models/open_video):
  //   open_video/matting/Better Quality
  std::error_code ec;
  fs::path rel = fs::relative(pack_dir, root, ec);
  std::string rels;
  if (!ec && !rel.empty() && rel != ".") {
    rels = rel.generic_string();
  } else {
    rels = pack_dir.filename().string();
  }

  const std::string rootName = root.filename().string();
  if (rootName.empty())
    return rels;
  if (rels.empty())
    return rootName;
  return rootName + "/" + rels;
}

int BestEffortReadSchemaVersion(const fs::path &manifest) {
  const auto textOpt = util::ReadTextFile(manifest.string());
  if (!textOpt)
    return 1;

  util::json::Value root;
  std::string parseErr;
  if (!util::json::Parse(*textOpt, &root, &parseErr))
    return 1;

  const auto *obj = root.AsObject();
  if (!obj)
    return 1;

  int schema = 1;
  std::string ignore;
  if (!GetIntOptional(*obj, "schema_version", &schema, &ignore))
    return 1;
  return schema;
}

ModelFileVerification VerifyFile(const ModelPack &pack, const ModelFile &f) {
  ModelFileVerification out;
  out.name = f.name;
  out.kind = f.kind;
  out.role = f.role;
  out.path = f.path;
  out.expected_sha256 =
      studiocast::model_integrity_internal::NormalizeSha256Hex(f.sha256);

  std::error_code ec;
  if (!fs::exists(f.path, ec) || ec) {
    out.status = "missing";
    out.message = std::string("missing model file: ") + PathForError(f.path);
    out.ok = false;
    return out;
  }
  if (!fs::is_regular_file(f.path, ec) || ec) {
    out.status = "missing";
    out.message =
        std::string("model file is not a regular file: ") + PathForError(f.path);
    out.ok = false;
    return out;
  }

  if (out.expected_sha256.empty()) {
    out.status = "unchecked";
    out.message = "no sha256 in model.json";
    out.ok = true;
    return out;
  }
  if (!studiocast::model_integrity_internal::IsSha256Hex(
          out.expected_sha256)) {
    out.status = "invalid_manifest";
    out.message = "model.json: files[].sha256 must be a 64-character hex "
                  "SHA-256 digest";
    out.ok = false;
    return out;
  }
  if (IsDownloadOnlyChecksum(pack, f)) {
    out.checksum_kind = "download_sha256";
    out.status = "unchecked";
    out.message = "files[].sha256 is a download checksum for this patched "
                  "BiRefNet install; installed-file SHA is not declared";
    out.ok = true;
    return out;
  }
  out.checksum_kind = "installed_sha256";

  std::string err;
  if (!studiocast::model_integrity_internal::ComputeSha256File(
          f.path, &out.actual_sha256, &err)) {
    out.status = "read_error";
    out.message = err.empty() ? std::string("failed to hash model file") : err;
    out.ok = false;
    return out;
  }

  if (out.actual_sha256 != out.expected_sha256) {
    out.status = "checksum_mismatch";
    out.message = "checksum_mismatch: expected " + out.expected_sha256 +
                  ", got " + out.actual_sha256;
    out.ok = false;
    return out;
  }

  out.status = "ok";
  out.message = "sha256 OK";
  out.ok = true;
  return out;
}

ModelPackVerification VerifyParsedPack(const ModelPack &pack) {
  ModelPackVerification out;
  out.id = pack.id;
  out.display_name = pack.display_name;
  out.task = pack.task;
  out.root_dir = pack.root_dir;
  out.manifest_path = pack.manifest_path;

  if (IsPlaceholderModelId(pack.id)) {
    out.status = "placeholder";
    out.message =
        "placeholder: model id contains 'placeholder' and is skipped by default";
    out.ok = false;
    return out;
  }

  out.status = "ok";
  out.message = "ok";
  out.ok = true;
  out.files.reserve(pack.files.size());
  for (const auto &f : pack.files) {
    auto vf = VerifyFile(pack, f);
    if (!vf.ok && out.ok) {
      out.ok = false;
      out.status = vf.status;
      out.message = vf.message;
    }
    out.files.push_back(std::move(vf));
  }
  return out;
}

ModelPackVerification FailedVerification(const fs::path &scan_root,
                                         const fs::path &pack_dir,
                                         const fs::path &manifest,
                                         const ModelPack &pack,
                                         const std::string &err) {
  ModelPackVerification out;
  out.id = pack.id.empty() ? PackDirKey(scan_root, pack_dir) : pack.id;
  out.display_name = pack.display_name;
  out.task = pack.task;
  out.root_dir = pack_dir;
  out.manifest_path = manifest;

  out.status =
      IsPlaceholderModelId(pack.id) ? "placeholder" : ClassifyLoadError(err);
  const std::string msg =
      IsPlaceholderModelId(pack.id)
          ? "model id contains 'placeholder' and is skipped by default"
          : (err.empty() ? "failed to load model pack" : err);
  out.message = ClassifiedMessage(out.status, msg);
  out.ok = false;
  return out;
}

} // namespace

bool ValidateProductionNcnnVulkanMattingPack(const ModelPack &pack,
                                             std::string *error) {
  if (error)
    error->clear();
  if (pack.task != "matting")
    return Fail(error, "production ncnn Vulkan requires task=matting");
  if (!pack.ncnn_vulkan) {
    return Fail(error,
                "model.json: missing required production ncnn_vulkan metadata");
  }

  const auto &spec = *pack.ncnn_vulkan;
  std::error_code ec;
  const fs::path canonical_root = fs::weakly_canonical(pack.root_dir, ec);
  if (ec)
    return Fail(error, "failed to resolve model pack root");

  const auto validate_artifact = [&](const fs::path &path,
                                     const std::string &expected_sha256,
                                     const char *label) {
    if (!studiocast::model_integrity_internal::IsSha256Hex(expected_sha256)) {
      return Fail(error, std::string("model.json: ncnn_vulkan ") + label +
                             " requires a valid SHA-256 digest");
    }
    const fs::path canonical_path = fs::weakly_canonical(path, ec);
    if (ec)
      return Fail(error, std::string("failed to resolve ncnn_vulkan ") + label);
    const fs::path relative = canonical_path.lexically_relative(canonical_root);
    if (relative.empty() || relative.is_absolute() ||
        *relative.begin() == "..") {
      return Fail(error, std::string("ncnn_vulkan ") + label +
                             " escapes the model pack root");
    }
    if (!fs::is_regular_file(canonical_path, ec) || ec) {
      return Fail(error, std::string("ncnn_vulkan ") + label +
                             " is missing or is not a regular file");
    }
    std::string actual_sha256;
    std::string hash_error;
    if (!studiocast::model_integrity_internal::ComputeSha256File(
            canonical_path, &actual_sha256, &hash_error)) {
      return Fail(error, hash_error.empty()
                             ? std::string("failed to hash ncnn_vulkan ") + label
                             : hash_error);
    }
    if (actual_sha256 != expected_sha256) {
      return Fail(error, std::string("ncnn_vulkan ") + label +
                             " checksum_mismatch: expected " + expected_sha256 +
                             ", got " + actual_sha256);
    }
    return true;
  };

  return validate_artifact(spec.param_path, spec.param_sha256, "param_file") &&
         validate_artifact(spec.bin_path, spec.bin_sha256, "bin_file");
}

ModelPackRegistry
ModelPackRegistry::Scan(const std::filesystem::path &open_video_models_dir) {
  ModelPackRegistry reg;
  reg.root_ = open_video_models_dir;

  if (open_video_models_dir.empty()) {
    reg.problems_["(open_video)"] = "open_video models directory is empty";
    return reg;
  }

  if (!fs::exists(open_video_models_dir)) {
    // Not an error; just no models installed.
    return reg;
  }

  std::error_code ec;
  fs::recursive_directory_iterator it(
      open_video_models_dir, fs::directory_options::skip_permission_denied, ec);
  if (ec) {
    reg.problems_[open_video_models_dir.string()] =
        std::string("failed to scan directory: ") + ec.message();
    return reg;
  }

  for (const auto &entry : it) {
    if (!entry.is_regular_file())
      continue;
    if (entry.path().filename() != "model.json")
      continue;

    const fs::path manifest = entry.path();
    const fs::path pack_dir = manifest.parent_path();

    ModelPack pack;
    std::string err;

    const int schema = BestEffortReadSchemaVersion(manifest);

    bool ok = false;
    if (schema == 2) {
      ok = ParseModelJsonV2(pack_dir, &pack, &err);
    } else {
      ok = ParseModelJsonV1(pack_dir, &pack, &err);
    }

    if (!ok) {
      const std::string dirKey = PackDirKey(open_video_models_dir, pack_dir);
      const std::string key = pack.id.empty() ? dirKey : pack.id;
      const std::string status =
          IsPlaceholderModelId(pack.id) ? "placeholder" : ClassifyLoadError(err);
      const std::string msg =
          IsPlaceholderModelId(pack.id)
              ? "model id contains 'placeholder' and is skipped by default"
              : (err.empty() ? "failed to load model pack" : err);
      reg.problems_[key] = ClassifiedMessage(status, msg);
      continue;
    }

    if (IsPlaceholderModelId(pack.id)) {
      reg.problems_[pack.id] =
          "placeholder: model id contains 'placeholder' and is skipped by "
          "default";
      continue;
    }

    const fs::path lic = pack_dir / "LICENSE.txt";
    if (fs::exists(lic)) {
      pack.license_path = lic;
    }

    // Task-specific validation.
    // Matting packs must include at least one ONNX file, since the runtime
    // expects an ONNX model.
    if (pack.task == "matting") {
      bool has_onnx = false;
      for (const auto &f : pack.files) {
        if (f.kind == "onnx") {
          has_onnx = true;
          break;
        }
      }
      if (!has_onnx) {
        const std::string dirKey = PackDirKey(open_video_models_dir, pack_dir);
        const std::string key = pack.id.empty() ? dirKey : pack.id;
        reg.problems_[key] = "missing ONNX file (kind=onnx)";
        continue;
      }
    }

    reg.models_.push_back(std::move(pack));
  }

  // Sort deterministically.
  std::sort(reg.models_.begin(), reg.models_.end(),
            [](const ModelPack &a, const ModelPack &b) {
              if (a.task != b.task)
                return a.task < b.task;
              return a.id < b.id;
            });

  // Deduplicate by id (keep first, record problem for duplicates).
  std::map<std::string, fs::path> seen;
  std::vector<ModelPack> unique;
  unique.reserve(reg.models_.size());
  for (auto &m : reg.models_) {
    if (auto it2 = seen.find(m.id); it2 != seen.end()) {
      reg.problems_[m.id] = std::string("duplicate model id '") + m.id +
                            "' in " + PackDirKey(reg.root_, m.root_dir) +
                            " (already provided by " +
                            PackDirKey(reg.root_, it2->second) + ")";
      continue;
    }
    seen[m.id] = m.root_dir;
    unique.push_back(std::move(m));
  }
  reg.models_.swap(unique);

  // Build task -> model id index (deterministic ordering matches models_).
  reg.tasks_.clear();
  for (const auto &m : reg.models_) {
    reg.tasks_[m.task].push_back(m.id);
  }

  return reg;
}

std::vector<ModelPackVerification>
ModelPackRegistry::Verify(const std::filesystem::path &open_video_models_dir) {
  std::vector<ModelPackVerification> out;

  if (open_video_models_dir.empty()) {
    ModelPackVerification r;
    r.id = "(open_video)";
    r.status = "missing";
    r.message = "missing: open_video models directory is empty";
    r.ok = false;
    out.push_back(std::move(r));
    return out;
  }

  std::error_code ec;
  if (!fs::exists(open_video_models_dir, ec) || ec) {
    return out;
  }
  if (!fs::is_directory(open_video_models_dir, ec) || ec) {
    ModelPackVerification r;
    r.id = open_video_models_dir.filename().string();
    r.root_dir = open_video_models_dir;
    r.status = "invalid_manifest";
    r.message = "invalid_manifest: open_video models path is not a directory: " +
                PathForError(open_video_models_dir);
    r.ok = false;
    out.push_back(std::move(r));
    return out;
  }

  fs::recursive_directory_iterator it(
      open_video_models_dir, fs::directory_options::skip_permission_denied, ec);
  if (ec) {
    ModelPackVerification r;
    r.id = open_video_models_dir.string();
    r.root_dir = open_video_models_dir;
    r.status = "read_error";
    r.message = std::string("failed to scan directory: ") + ec.message();
    r.ok = false;
    out.push_back(std::move(r));
    return out;
  }

  for (const auto &entry : it) {
    if (!entry.is_regular_file())
      continue;
    if (entry.path().filename() != "model.json")
      continue;

    const fs::path manifest = entry.path();
    const fs::path pack_dir = manifest.parent_path();

    ModelPack pack;
    std::string err;
    const int schema = BestEffortReadSchemaVersion(manifest);

    const bool ok = schema == 2 ? ParseModelJsonV2(pack_dir, &pack, &err)
                                : ParseModelJsonV1(pack_dir, &pack, &err);
    if (!ok) {
      out.push_back(
          FailedVerification(open_video_models_dir, pack_dir, manifest, pack, err));
      continue;
    }

    out.push_back(VerifyParsedPack(pack));
  }

  std::sort(out.begin(), out.end(),
            [](const ModelPackVerification &a,
               const ModelPackVerification &b) {
              if (a.task != b.task)
                return a.task < b.task;
              return a.id < b.id;
            });
  return out;
}

ModelPackRegistry ModelPackRegistry::ScanDefault() {
  const auto modelsRoot = util::StudioCastModelsDir();
  if (modelsRoot.empty())
    return {};
  return Scan(modelsRoot / "open_video");
}

std::vector<ModelPackVerification> ModelPackRegistry::VerifyDefault() {
  const auto modelsRoot = util::StudioCastModelsDir();
  if (modelsRoot.empty())
    return {};
  return Verify(modelsRoot / "open_video");
}

std::optional<ModelPack>
ModelPackRegistry::ResolveModel(const std::string &id) const {
  // models_ is sorted by (task, id) for human-friendly grouping in tools.
  // Model counts are small, so a linear search is fine and avoids maintaining a
  // separate index.
  for (const auto &m : models_) {
    if (m.id == id)
      return m;
  }
  return std::nullopt;
}

std::optional<ModelPack> ModelPackRegistry::Find(const std::string &task,
                                                 const std::string &id) const {
  if (id.empty())
    return std::nullopt;
  const auto m = ResolveModel(id);
  if (!m.has_value())
    return std::nullopt;
  if (!task.empty() && m->task != task)
    return std::nullopt;
  return m;
}

std::string
ModelPackRegistry::DefaultModelIdForTask(const std::string &task) const {
  if (task == "matting") {
    // Prefer the lightest matting model by default (good enough for
    // tracking/segmentation and keeps latency low on mid-range GPUs).
    if (Find("matting", "modnet-webnn-256-fp32"))
      return "modnet-webnn-256-fp32";

    // Fall back to a higher-quality option when MODNet isn't installed.
    if (Find("matting", "birefnet_lite"))
      return "birefnet_lite";
  }

  if (!task.empty()) {
    for (const auto &m : models_) {
      if (m.task == task)
        return m.id;
    }
  }
  if (models_.empty())
    return {};
  return models_.front().id;
}

} // namespace studiocast::open_video
