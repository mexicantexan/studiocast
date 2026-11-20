#include "core/audio/audio_device_safety.h"

#include <algorithm>
#include <sstream>

#include "core/audio/pulse/pactl.h"
#include "core/util/strings.h"

namespace studiocast::audio {
namespace {

constexpr const char *kVirtualMicSinkName = "studiocast_sink";
constexpr const char *kVirtualMicSourceName = "studiocast_mic";
constexpr const char *kVirtualSpeakersSinkName = "studiocast_speakers";

std::string Trimmed(std::string s) { return studiocast::util::TrimCopy(s); }

std::string JoinWarnings(const std::vector<std::string> &warnings) {
  std::ostringstream oss;
  for (std::size_t i = 0; i < warnings.size(); ++i) {
    if (i)
      oss << " ";
    oss << warnings[i];
  }
  return oss.str();
}

} // namespace

bool IsStudioCastVirtualSourceName(const std::string &name) {
  const std::string n = Trimmed(name);
  return n == kVirtualMicSourceName ||
         n == std::string(kVirtualMicSinkName) + ".monitor" ||
         n == std::string(kVirtualSpeakersSinkName) + ".monitor";
}

bool IsPulseMonitorSourceName(const std::string &name) {
  const std::string n = Trimmed(name);
  return n.ends_with(".monitor") || n.find(".monitor.") != std::string::npos ||
         n.find(".monitor") != std::string::npos;
}

bool IsUnsafeInputSourceName(const std::string &name, std::string *reason) {
  const std::string n = Trimmed(name);
  if (reason)
    reason->clear();
  if (n.empty())
    return false;

  if (IsStudioCastVirtualSourceName(n)) {
    if (reason) {
      *reason = "Pulse source '" + n +
                "' is a StudioCast virtual source and would feed "
                "StudioCast audio back into itself.";
    }
    return true;
  }

  if (IsPulseMonitorSourceName(n)) {
    if (reason) {
      *reason = "Pulse source '" + n +
                "' is a monitor source. Choose a microphone/input source, "
                "not a sink monitor.";
    }
    return true;
  }

  return false;
}

bool IsStudioCastVirtualSinkName(const std::string &name) {
  const std::string n = Trimmed(name);
  return n == kVirtualMicSinkName || n == kVirtualSpeakersSinkName;
}

bool IsUnsafeSpeakerTargetSinkName(const std::string &name,
                                   std::string *reason) {
  const std::string n = Trimmed(name);
  if (reason)
    reason->clear();
  if (n.empty())
    return false;

  if (IsStudioCastVirtualSinkName(n)) {
    if (reason) {
      *reason = "Pulse sink '" + n +
                "' is a StudioCast virtual sink. Choose a physical output "
                "sink to avoid a feedback loop.";
    }
    return true;
  }

  if (IsPulseMonitorSourceName(n)) {
    if (reason) {
      *reason = "Pulse endpoint '" + n +
                "' is a monitor source, not an output sink. Choose a "
                "physical sink.";
    }
    return true;
  }

  return false;
}

AudioSourceResolution
ResolveSafeInputSourceName(const std::string &configured_source) {
  return ResolveSafeInputSourceName(configured_source, nullptr);
}

AudioSourceResolution ResolveSafeInputSourceName(
    const std::string &configured_source,
    const std::atomic_bool *stop_requested) {
  AudioSourceResolution out;

  std::string chosen = Trimmed(configured_source);
  if (chosen == "auto")
    chosen.clear();

  std::string reason;
  if (!chosen.empty()) {
    if (IsUnsafeInputSourceName(chosen, &reason)) {
      out.error = reason;
      return out;
    }
    out.ok = true;
    out.source_name = chosen;
    return out;
  }

  std::string defaultErr;
  auto def = pulse::GetDefaultSourceName(&defaultErr, stop_requested);
  if (def) {
    chosen = Trimmed(*def);
    if (!IsUnsafeInputSourceName(chosen, &reason)) {
      out.ok = true;
      out.source_name = chosen;
      return out;
    }
    out.warnings.push_back(
        "Pulse default source is unsafe for StudioCast capture: " + reason);
  } else {
    out.warnings.push_back("Pulse default source could not be determined" +
                           (defaultErr.empty()
                                ? std::string(".")
                                : std::string(": ") + defaultErr));
  }

  std::string listErr;
  const auto sources = pulse::ListSources(&listErr, stop_requested);
  for (const auto &source : sources) {
    std::string candidateReason;
    if (!source.name.empty() &&
        !IsUnsafeInputSourceName(source.name, &candidateReason)) {
      out.ok = true;
      out.source_name = source.name;
      if (!chosen.empty()) {
        out.warnings.push_back("Using safe source '" + source.name +
                               "' instead of unsafe Pulse default source '" +
                               chosen + "'.");
      } else {
        out.warnings.push_back("Using first safe Pulse source '" + source.name +
                               "'.");
      }
      return out;
    }
  }

  out.error = "No safe Pulse microphone source was found. ";
  if (!chosen.empty()) {
    out.error += "The Pulse default source '" + chosen +
                 "' is unsafe for StudioCast capture. ";
  }
  if (!listErr.empty()) {
    out.error += "Pulse source list failed: " + listErr + ". ";
  }
  if (!out.warnings.empty())
    out.error += JoinWarnings(out.warnings) + " ";
  out.error +=
      "Select a physical microphone/input source before enabling audio.";
  return out;
}

std::optional<std::string>
ChooseSafeSpeakerTargetSinkName(const std::string &configured_target,
                                std::string *error) {
  return ChooseSafeSpeakerTargetSinkName(configured_target, error, nullptr);
}

std::optional<std::string> ChooseSafeSpeakerTargetSinkName(
    const std::string &configured_target, std::string *error,
    const std::atomic_bool *stop_requested) {
  if (error)
    error->clear();

  std::string chosen = Trimmed(configured_target);
  if (chosen == "auto")
    chosen.clear();

  std::string reason;
  if (!chosen.empty()) {
    if (IsUnsafeSpeakerTargetSinkName(chosen, &reason)) {
      if (error)
        *error = reason;
      return std::nullopt;
    }
    return chosen;
  }

  std::string defaultErr;
  auto def = pulse::GetDefaultSinkName(&defaultErr, stop_requested);
  if (def) {
    chosen = Trimmed(*def);
    if (!IsUnsafeSpeakerTargetSinkName(chosen, &reason))
      return chosen;
  }

  std::string listErr;
  const auto sinks = pulse::ListSinks(&listErr, stop_requested);
  for (const auto &sink : sinks) {
    std::string candidateReason;
    if (!sink.name.empty() &&
        !IsUnsafeSpeakerTargetSinkName(sink.name, &candidateReason)) {
      return sink.name;
    }
  }

  if (error) {
    *error = "Failed to choose a physical speaker target sink. ";
    if (!chosen.empty()) {
      *error += "The Pulse default sink '" + chosen +
                "' is unsafe for StudioCast speaker routing. ";
    }
    if (!defaultErr.empty())
      *error += "Default sink note: " + defaultErr + ". ";
    if (!listErr.empty())
      *error += "Sink list note: " + listErr + ". ";
    *error += "Choose a physical output sink.";
  }
  return std::nullopt;
}

} // namespace studiocast::audio
