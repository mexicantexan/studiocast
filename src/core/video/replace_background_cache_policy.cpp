#include "core/video/replace_background_cache_policy.h"

#include <system_error>
#include <utility>

namespace studiocast::video::detail {
namespace {

bool StatReplaceBackgroundFile(const std::filesystem::path &path,
                               std::filesystem::file_time_type *mtime,
                               std::string *error) {
  if (error)
    error->clear();
  std::error_code ec;
  const auto value = std::filesystem::last_write_time(path, ec);
  if (ec) {
    if (error)
      *error = "failed to stat replace image: " + ec.message();
    return false;
  }
  if (mtime)
    *mtime = value;
  return true;
}

} // namespace

bool PrepareReplaceBackgroundSourceForConfig(
    const std::filesystem::path &path, const ReplaceBackgroundFileStat &stat,
    PreparedReplaceBackgroundSource *out, std::string *error) {
  if (error)
    error->clear();

  PreparedReplaceBackgroundSource prepared;
  prepared.path = path;

  const auto fail = [&](std::string message) {
    prepared.valid = false;
    prepared.error = std::move(message);
    if (out)
      *out = prepared;
    if (error)
      *error = prepared.error;
    return false;
  };

  if (path.empty())
    return fail("virtual_background.replace_path not set.");
  if (!stat)
    return fail("replace background stat function not set.");

  std::filesystem::file_time_type mtime{};
  std::string stat_error;
  if (!stat(path, &mtime, &stat_error)) {
    if (stat_error.empty())
      stat_error = "failed to stat replace image.";
    return fail(std::move(stat_error));
  }

  prepared.mtime = mtime;
  prepared.valid = true;
  prepared.error.clear();
  if (out)
    *out = prepared;
  return true;
}

bool PrepareReplaceBackgroundSourceForConfig(
    const std::filesystem::path &path, PreparedReplaceBackgroundSource *out,
    std::string *error) {
  return PrepareReplaceBackgroundSourceForConfig(
      path, ReplaceBackgroundFileStat(StatReplaceBackgroundFile), out, error);
}

ReplaceBackgroundFrameCacheDecision DecideReplaceBackgroundFrameCache(
    const PreparedReplaceBackgroundSource &prepared,
    const ReplaceBackgroundSourceCacheSnapshot &source_cache,
    const ReplaceBackgroundResizeCacheSnapshot &resize_cache, int frame_width,
    int frame_height) {
  ReplaceBackgroundFrameCacheDecision decision;
  if (!prepared.valid || frame_width <= 0 || frame_height <= 0)
    return decision;

  const bool source_hit = source_cache.valid &&
                          source_cache.path == prepared.path &&
                          source_cache.mtime == prepared.mtime;
  decision.refresh_source = !source_hit;
  decision.source_generation_after_refresh = decision.refresh_source
                                                 ? source_cache.generation + 1
                                                 : source_cache.generation;

  decision.refresh_resized_destination =
      decision.refresh_source || !resize_cache.valid ||
      resize_cache.width != frame_width ||
      resize_cache.height != frame_height ||
      resize_cache.source_generation !=
          decision.source_generation_after_refresh;
  return decision;
}

} // namespace studiocast::video::detail
