#pragma once

#include <atomic>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "core/util/exec.h"

namespace studiocast::audio::pulse {

struct PactlModule {
  int id = -1;
  std::string name;
  std::string args;
};

struct PactlSource {
  int id = -1;
  std::string name;
};

struct PactlSink {
  int id = -1;
  std::string name;
};

struct PactlSourceOutput {
  int id = -1;
  std::string source;
};

struct PactlSinkInput {
  int id = -1;
  std::string sink;
};

struct PactlPort {
  std::string name;        // e.g. "analog-input-internal-mic"
  std::string description; // e.g. "Internal Microphone"
  bool available = true;
};

struct PactlSourceInfo {
  int id = -1;
  std::string name;        // Pulse source name
  std::string description; // Human-friendly description (may be empty)
  std::string active_port; // Port name (may be empty)
  std::vector<PactlPort> ports;
};

std::vector<PactlSourceInfo> ListSourcesDetailed(std::string *error);
bool SetSourcePort(const std::string &source_name, const std::string &port_name,
                   std::string *error);

bool PactlAvailable(std::string *details);

using PactlExecCaptureHook =
    std::function<studiocast::util::ExecResult(const std::string &)>;
using PactlExecCaptureStopAwareHook =
    std::function<studiocast::util::ExecResult(const std::string &,
                                               const std::atomic_bool *)>;

void SetPactlExecCaptureHookForTesting(PactlExecCaptureHook hook);
void SetPactlExecCaptureStopAwareHookForTesting(
    PactlExecCaptureStopAwareHook hook);

std::optional<int> LoadModule(const std::string &module,
                              const std::string &args, std::string *error);
std::optional<int> LoadModule(const std::string &module,
                              const std::vector<std::string> &args,
                              std::string *error);
bool UnloadModule(int id, std::string *error);

std::vector<PactlModule> ListModules(std::string *error);
std::vector<PactlSource>
ListSources(std::string *error,
            const std::atomic_bool *stop_requested = nullptr);
std::vector<PactlSink>
ListSinks(std::string *error,
          const std::atomic_bool *stop_requested = nullptr);
std::vector<PactlSourceOutput>
ListSourceOutputs(std::string *error,
                  const std::atomic_bool *stop_requested = nullptr);
std::vector<PactlSinkInput>
ListSinkInputs(std::string *error,
               const std::atomic_bool *stop_requested = nullptr);

std::optional<std::string>
GetDefaultSourceName(std::string *error,
                     const std::atomic_bool *stop_requested = nullptr);
std::optional<std::string>
GetDefaultSinkName(std::string *error,
                   const std::atomic_bool *stop_requested = nullptr);

// Deterministic parsing helper (used for `pactl info` fallbacks and self-test).
// Example keys: "Default Source:", "Default Sink:".
std::optional<std::string>
ParseDefaultFromPactlInfo(const std::string &pactl_info_text,
                          const std::string &key);

bool UpdateSinkProplist(const std::string &sink_name_or_index,
                        const std::vector<std::string> &kv_pairs,
                        std::string *error);

bool UpdateSourceProplist(const std::string &source_name_or_index,
                          const std::vector<std::string> &kv_pairs,
                          std::string *error);

} // namespace studiocast::audio::pulse
