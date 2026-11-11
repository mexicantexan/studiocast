#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>

#include "core/video/replace_background_cache_policy.h"

namespace {

using studiocast::video::detail::DecideReplaceBackgroundFrameCache;
using studiocast::video::detail::PreparedReplaceBackgroundSource;
using studiocast::video::detail::PrepareReplaceBackgroundSourceForConfig;
using studiocast::video::detail::ReplaceBackgroundResizeCacheSnapshot;
using studiocast::video::detail::ReplaceBackgroundSourceCacheSnapshot;

bool Expect(bool condition, const std::string &message) {
  if (!condition) {
    std::cerr << message << "\n";
    return false;
  }
  return true;
}

std::filesystem::file_time_type Mtime(int value) {
  return std::filesystem::file_time_type{} + std::chrono::seconds(value);
}

void ApplyDecision(const PreparedReplaceBackgroundSource &prepared, int width,
                   int height, ReplaceBackgroundSourceCacheSnapshot *source,
                   ReplaceBackgroundResizeCacheSnapshot *resize,
                   int *source_refreshes, int *resize_refreshes) {
  const auto decision = DecideReplaceBackgroundFrameCache(
      prepared, *source, *resize, width, height);
  if (decision.refresh_source) {
    ++(*source_refreshes);
    source->path = prepared.path;
    source->mtime = prepared.mtime;
    source->valid = true;
    source->generation = decision.source_generation_after_refresh;
  }
  if (decision.refresh_resized_destination) {
    ++(*resize_refreshes);
    resize->width = width;
    resize->height = height;
    resize->valid = true;
    resize->source_generation = decision.source_generation_after_refresh;
  }
}

bool TestRepeatedFramePathDoesNotStatReplaceImage() {
  int stat_calls = 0;
  const auto stat = [&](const std::filesystem::path &,
                        std::filesystem::file_time_type *mtime,
                        std::string *error) {
    ++stat_calls;
    if (error)
      error->clear();
    *mtime = Mtime(10);
    return true;
  };

  PreparedReplaceBackgroundSource prepared;
  std::string error;
  if (!PrepareReplaceBackgroundSourceForConfig("/tmp/background.png", stat,
                                               &prepared, &error)) {
    std::cerr << error << "\n";
    return false;
  }

  ReplaceBackgroundSourceCacheSnapshot source;
  ReplaceBackgroundResizeCacheSnapshot resize;
  int source_refreshes = 0;
  int resize_refreshes = 0;

  for (int i = 0; i < 8; ++i) {
    ApplyDecision(prepared, 1280, 720, &source, &resize, &source_refreshes,
                  &resize_refreshes);
  }

  return Expect(stat_calls == 1,
                "replace image should be statted once at config time") &&
         Expect(source_refreshes == 1,
                "repeated same-size frames should reuse source upload") &&
         Expect(resize_refreshes == 1,
                "repeated same-size frames should reuse resized background");
}

bool TestSamePathExplicitRefreshUsesPreparedMtime() {
  int stat_calls = 0;
  auto next_mtime = Mtime(20);
  const auto stat = [&](const std::filesystem::path &,
                        std::filesystem::file_time_type *mtime,
                        std::string *error) {
    ++stat_calls;
    if (error)
      error->clear();
    *mtime = next_mtime;
    return true;
  };

  PreparedReplaceBackgroundSource prepared;
  std::string error;
  if (!PrepareReplaceBackgroundSourceForConfig("/tmp/background.png", stat,
                                               &prepared, &error)) {
    std::cerr << error << "\n";
    return false;
  }

  ReplaceBackgroundSourceCacheSnapshot source;
  ReplaceBackgroundResizeCacheSnapshot resize;
  int source_refreshes = 0;
  int resize_refreshes = 0;
  ApplyDecision(prepared, 1280, 720, &source, &resize, &source_refreshes,
                &resize_refreshes);

  next_mtime = Mtime(21);
  if (!PrepareReplaceBackgroundSourceForConfig("/tmp/background.png", stat,
                                               &prepared, &error)) {
    std::cerr << error << "\n";
    return false;
  }
  for (int i = 0; i < 4; ++i) {
    ApplyDecision(prepared, 1280, 720, &source, &resize, &source_refreshes,
                  &resize_refreshes);
  }

  return Expect(stat_calls == 2,
                "explicit same-path refresh should stat at config time only") &&
         Expect(source_refreshes == 2,
                "changed prepared mtime should refresh source once") &&
         Expect(resize_refreshes == 2,
                "changed source should refresh resized background once");
}

bool TestFrameSizeChangeOnlyRefreshesResizedBackground() {
  int stat_calls = 0;
  const auto stat = [&](const std::filesystem::path &,
                        std::filesystem::file_time_type *mtime,
                        std::string *error) {
    ++stat_calls;
    if (error)
      error->clear();
    *mtime = Mtime(30);
    return true;
  };

  PreparedReplaceBackgroundSource prepared;
  std::string error;
  if (!PrepareReplaceBackgroundSourceForConfig("/tmp/background.png", stat,
                                               &prepared, &error)) {
    std::cerr << error << "\n";
    return false;
  }

  ReplaceBackgroundSourceCacheSnapshot source;
  ReplaceBackgroundResizeCacheSnapshot resize;
  int source_refreshes = 0;
  int resize_refreshes = 0;
  ApplyDecision(prepared, 1280, 720, &source, &resize, &source_refreshes,
                &resize_refreshes);
  ApplyDecision(prepared, 1920, 1080, &source, &resize, &source_refreshes,
                &resize_refreshes);
  ApplyDecision(prepared, 1920, 1080, &source, &resize, &source_refreshes,
                &resize_refreshes);

  return Expect(stat_calls == 1,
                "frame size changes should not stat the replace image") &&
         Expect(source_refreshes == 1,
                "frame size changes should not reload the source image") &&
         Expect(resize_refreshes == 2,
                "frame size changes should refresh only resized output");
}

} // namespace

namespace studiocast::tests {

bool TestReplaceBackgroundRepeatedFramePathDoesNotStatImage() {
  return TestRepeatedFramePathDoesNotStatReplaceImage();
}

bool TestReplaceBackgroundSamePathExplicitRefreshUsesPreparedMtime() {
  return TestSamePathExplicitRefreshUsesPreparedMtime();
}

bool TestReplaceBackgroundFrameSizeChangeOnlyRefreshesResizedBackground() {
  return TestFrameSizeChangeOnlyRefreshesResizedBackground();
}

} // namespace studiocast::tests
