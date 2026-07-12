#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>

namespace studiocast::video::detail {

struct PreparedReplaceBackgroundSource {
  std::filesystem::path path;
  std::filesystem::file_time_type mtime{};
  bool valid = false;
  std::string error;
};

using ReplaceBackgroundFileStat =
    std::function<bool(const std::filesystem::path &,
                       std::filesystem::file_time_type *, std::string *)>;

bool PrepareReplaceBackgroundSourceForConfig(
    const std::filesystem::path &path, const ReplaceBackgroundFileStat &stat,
    PreparedReplaceBackgroundSource *out, std::string *error);

bool PrepareReplaceBackgroundSourceForConfig(
    const std::filesystem::path &path, PreparedReplaceBackgroundSource *out,
    std::string *error);

struct ReplaceBackgroundSourceCacheSnapshot {
  std::filesystem::path path;
  std::filesystem::file_time_type mtime{};
  bool valid = false;
  std::uint64_t generation = 0;
};

struct ReplaceBackgroundResizeCacheSnapshot {
  int width = 0;
  int height = 0;
  bool valid = false;
  std::uint64_t source_generation = 0;
};

struct ReplaceBackgroundFrameCacheDecision {
  bool refresh_source = false;
  bool refresh_resized_destination = false;
  std::uint64_t source_generation_after_refresh = 0;
};

ReplaceBackgroundFrameCacheDecision DecideReplaceBackgroundFrameCache(
    const PreparedReplaceBackgroundSource &prepared,
    const ReplaceBackgroundSourceCacheSnapshot &source_cache,
    const ReplaceBackgroundResizeCacheSnapshot &resize_cache, int frame_width,
    int frame_height);

} // namespace studiocast::video::detail
