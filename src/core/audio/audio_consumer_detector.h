#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "core/audio/virtual_audio_service.h"

namespace studiocast::audio {

class AudioConsumerDetector {
public:
  virtual ~AudioConsumerDetector() = default;

  virtual AudioConsumerSnapshot
  DetectSourceConsumersByName(const std::string &source_name) = 0;
  virtual AudioConsumerSnapshot
  DetectSinkConsumersByName(const std::string &sink_name) = 0;
};

struct PactlSubscriptionPollResult {
  bool event_received = false;
  bool disconnected = false;
  std::string error;
};

// A long-lived `pactl subscribe` event source. Poll must be non-blocking and
// event delivery is intentionally coalesced: the detector keeps one dirty bit,
// not an event queue.
class PactlSubscriptionMonitor {
public:
  virtual ~PactlSubscriptionMonitor() = default;
  virtual PactlSubscriptionPollResult Poll() = 0;
  virtual void Stop() = 0;
};

using PactlSubscriptionMonitorFactory =
    std::function<std::unique_ptr<PactlSubscriptionMonitor>(std::string *)>;

// Production detector. It prefers a libpulse subscription-backed cache and
// falls back to one long-lived pactl subscription plus event-invalidated
// snapshots if the libpulse subscription monitor is unavailable.
std::unique_ptr<AudioConsumerDetector> CreateDefaultAudioConsumerDetector(
    const std::atomic_bool *stop_requested = nullptr);

// Deterministic fallback seam used by tests. Production callers use the
// default factory above.
std::unique_ptr<AudioConsumerDetector>
CreatePactlSubscriptionAudioConsumerDetectorForTesting(
    PactlSubscriptionMonitorFactory monitor_factory,
    const std::atomic_bool *stop_requested = nullptr,
    std::chrono::milliseconds reconnect_delay = std::chrono::seconds(5));

} // namespace studiocast::audio
