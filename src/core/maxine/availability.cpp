#include "availability.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "core/maxine/maxine_manager.h"

namespace studiocast::maxine {

namespace {

namespace fs = std::filesystem;

std::string ToLowerCopy(std::string s) {
  for (char &c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return s;
}

std::string PrettyPathForCopy(const fs::path &p) {
  const std::string s = p.string();
  const char *home = std::getenv("HOME");
  if (home && *home) {
    const std::string h(home);
    if (s.rfind(h, 0) == 0) {
      return std::string("~").append(s, h.size(), std::string::npos);
    }
  }
  return s;
}

std::string JoinOr(const std::vector<std::string> &items) {
  if (items.empty())
    return {};
  std::ostringstream oss;
  for (size_t i = 0; i < items.size(); ++i) {
    if (i)
      oss << " or ";
    oss << items[i];
  }
  return oss.str();
}

fs::path FindCandidateRoot(const std::vector<fs::path> &cands, const char *leaf,
                           const char *must_contain_lower) {
  for (const auto &p : cands) {
    if (p.filename() != leaf)
      continue;
    if (must_contain_lower) {
      const std::string ps = ToLowerCopy(p.string());
      if (ps.find(must_contain_lower) == std::string::npos)
        continue;
    }
    return p;
  }
  return {};
}

std::string ExpectedRootsFor(const ComponentDiagnostics &c, const char *leaf,
                             const fs::path &system_default) {
  std::vector<std::string> items;
  items.reserve(3);

  // If the user explicitly set the env root, show it first.
  if (c.root_source == "env" && !c.root.empty()) {
    items.push_back(PrettyPathForCopy(c.root));
  }

  // Prefer our known XDG default candidate.
  const auto xdg =
      FindCandidateRoot(c.candidate_roots, leaf, "/studiocast/maxine/");
  if (!xdg.empty()) {
    items.push_back(PrettyPathForCopy(xdg));
  }

  // Always show the canonical system fallback.
  items.push_back(system_default.string());

  // Dedup while preserving order.
  std::vector<std::string> uniq;
  std::set<std::string> seen;
  for (const auto &it : items) {
    if (seen.insert(it).second) {
      uniq.push_back(it);
    }
  }

  // Keep the headline short.
  if (uniq.size() > 3) {
    uniq.resize(3);
  }
  return JoinOr(uniq);
}

bool NeedsMaxineVfx(const MaxineDiagnostics &d) {
  return d.gpu.ok && d.driver.ok && d.vfx.ok && d.vfx.library_loadable;
}

bool NeedsMaxineAr(const MaxineDiagnostics &d) {
  return d.gpu.ok && d.driver.ok && d.ar.ok && d.ar.library_loadable;
}

bool NeedsMaxineAfx(const MaxineDiagnostics &d) {
  return d.gpu.ok && d.driver.ok && d.afx.ok && d.afx.library_loadable;
}

} // namespace

CanonicalMaxineBlockedCopy
BuildCanonicalMaxineBlockedCopy(const MaxineDiagnostics &d, MaxineNeed need) {
  CanonicalMaxineBlockedCopy out;

  auto add_step = [&](const std::string &s) {
    if (s.empty())
      return;
    for (const auto &e : out.steps) {
      if (e == s)
        return;
    }
    if (out.steps.size() < 3) {
      out.steps.push_back(s);
    }
  };

  const bool gpu_ok = d.gpu.ok;
  const bool driver_ok = d.driver.ok;
  const bool vfx_ready = NeedsMaxineVfx(d);
  const bool ar_ready = NeedsMaxineAr(d);
  const bool afx_ready = NeedsMaxineAfx(d);

  const bool require_vfx = (need == MaxineNeed::vfx);
  const bool require_ar = (need == MaxineNeed::ar);
  const bool require_afx = (need == MaxineNeed::afx);

  // Headline selection priority (matches Task 25 examples).
  if (!gpu_ok) {
    out.summary = "Maxine unavailable: no supported NVIDIA GPU detected.";
    add_step("Run `studiocast-probe` to verify GPU/driver.");
    return out;
  }

  if (!driver_ok) {
    if (d.driver.version.empty()) {
      out.summary = "Maxine unavailable: NVIDIA driver not detected.";
    } else {
      out.summary = "Maxine unavailable: NVIDIA driver too old (need R570+).";
    }
    add_step("Run `studiocast-probe` to verify GPU/driver.");
    add_step("Update the NVIDIA driver to R570+ (then reboot).");
    return out;
  }

  // Component-specific blocking.
  if (require_vfx ||
      (need == MaxineNeed::any && !vfx_ready && !ar_ready && !afx_ready)) {
    if (!d.vfx.root_exists || !d.vfx.library_exists) {
      const auto expected =
          ExpectedRootsFor(d.vfx, "VideoFX", fs::path("/usr/local/VideoFX"));
      out.summary =
          "Maxine unavailable: VFX SDK not found (expected: " + expected + ").";
      add_step("Run `studiocast-probe` to verify GPU/driver.");
      add_step("Run `studiocast-maxine init` then follow `studiocast-maxine "
               "install-hints`.");
      add_step("Ensure `libVideoFX.so` (or legacy `libnvvfx.so` / "
               "`libNvVFX.so`) is under `<VFX_ROOT>/lib/` and feature "
               "installs exist under `<VFX_ROOT>/features/`.");
      return out;
    }
    if (!d.vfx.library_loadable) {
      out.summary =
          "Maxine unavailable: VFX SDK not found (expected: " +
          ExpectedRootsFor(d.vfx, "VideoFX", fs::path("/usr/local/VideoFX")) +
          ").";
      add_step("Run `studiocast-maxine init` then follow `studiocast-maxine "
               "install-hints`.");
      add_step("Ensure `libVideoFX.so` (or legacy `libnvvfx.so` / "
               "`libNvVFX.so`) is under `<VFX_ROOT>/lib/` and feature "
               "installs exist under `<VFX_ROOT>/features/`.");
      return out;
    }
  }

  if (require_ar ||
      (need == MaxineNeed::any && !ar_ready && !vfx_ready && !afx_ready)) {
    if (!d.ar.root_exists || !d.ar.library_exists || !d.ar.library_loadable) {
      const auto expected =
          ExpectedRootsFor(d.ar, "ARSDK", fs::path("/usr/local/ARSDK"));
      out.summary = "Maxine unavailable: AR SDK not found (needed for Eye "
                    "Contact / Auto Frame).";
      add_step("Run `studiocast-probe` to verify GPU/driver.");
      add_step("Run `studiocast-maxine init` then follow `studiocast-maxine "
               "install-hints`.");
      add_step("Expected AR SDK root: " + expected + ".");
      add_step("Ensure `libnvARPose.so` (or legacy `libnvar.so` / "
               "`libNvAR.so`) is under `<AR_ROOT>/lib/` and feature "
               "installs exist under `<AR_ROOT>/features/`.");
      return out;
    }
  }

  if (require_afx ||
      (need == MaxineNeed::any && !afx_ready && !vfx_ready && !ar_ready)) {
    if (!d.afx.root_exists || !d.afx.library_exists) {
      const auto expected = ExpectedRootsFor(
          d.afx, "Audio_Effects_SDK", fs::path("/usr/local/Audio_Effects_SDK"));
      out.summary = "Maxine unavailable: Audio Effects SDK not found (needed "
                    "for audio effects).";
      add_step("Run `studiocast-probe` to verify GPU/driver.");
      add_step("Run `studiocast-maxine init` then follow `studiocast-maxine "
               "install-hints`.");
      add_step("Expected AFX SDK root: " + expected + ".");
      add_step("Ensure `libnv_audiofx.so` is under `<AFX_ROOT>/nvafx/lib/` and "
               "feature installs exist under `<AFX_ROOT>/features/`.");
      return out;
    }
    if (!d.afx.library_loadable) {
      out.summary =
          "Maxine unavailable: Audio Effects SDK library could not be loaded.";
      add_step("Run `studiocast-probe` to verify GPU/driver.");
      add_step("Ensure `libnv_audiofx.so` is under `<AFX_ROOT>/nvafx/lib/`.");
      return out;
    }
  }

  // Default: SDK present but effect libraries/features are missing.
  if (require_afx) {
    out.summary = "Maxine unavailable: Audio Effects features not installed "
                  "(run download_features.sh).";
    add_step("Run `studiocast-probe` to verify GPU/driver.");
    add_step("Run `studiocast-maxine init` then follow `studiocast-maxine "
             "install-hints`.");
    add_step("Export `NGC_API_KEY` (do not commit it).");
    add_step(
        "Run: `cd <AFX_ROOT>/features && ./download_features.sh --effects "
        "denoiser-48k,dereverb-48k,dereverb_denoiser-48k,studio_voice-48k`.");
    add_step("Ensure `libnv_audiofx.so` is under `<AFX_ROOT>/nvafx/lib/` and "
             "feature installs exist under `<AFX_ROOT>/features/`.");
    return out;
  }

  out.summary = "Maxine unavailable: feature libraries not installed (run "
                "install_feature.sh).";
  add_step("Run `studiocast-probe` to verify GPU/driver.");
  add_step("Run `studiocast-maxine init` then follow `studiocast-maxine "
           "install-hints`.");
  add_step("Ensure `libVideoFX.so` (or legacy `libnvvfx.so` / "
           "`libNvVFX.so`) is under `<VFX_ROOT>/lib/` and feature "
           "installs exist under `<VFX_ROOT>/features/`.");
  return out;
}

std::string
FormatCanonicalMaxineBlockedCopy(const CanonicalMaxineBlockedCopy &c) {
  if (c.summary.empty() && c.steps.empty()) {
    return {};
  }
  std::ostringstream oss;
  if (!c.summary.empty()) {
    oss << c.summary;
  }
  for (const auto &s : c.steps) {
    if (s.empty())
      continue;
    if (!oss.str().empty())
      oss << "\n";
    oss << "- " << s;
  }
  return oss.str();
}

bool BackendBuilt() {
#ifdef STUDIOCAST_WITH_MAXINE
  return true;
#else
  return false;
#endif
}

bool RuntimeAvailable(std::string *reason) {
#ifndef STUDIOCAST_WITH_MAXINE
  if (reason) {
    *reason = "Maxine unavailable: Maxine support not enabled in this build.";
  }
  return false;
#else
  MaxineManager mgr;
  const auto d = mgr.Diagnose(false);
  if (reason) {
    if (d.ok) {
      *reason = {};
    } else {
      *reason = FormatCanonicalMaxineBlockedCopy(
          BuildCanonicalMaxineBlockedCopy(d, MaxineNeed::any));
    }
  }
  return d.ok;
#endif
}

} // namespace studiocast::maxine
