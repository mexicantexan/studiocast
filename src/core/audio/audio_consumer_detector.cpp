#include "core/audio/audio_consumer_detector.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include "core/audio/pulse/pactl.h"

#if STUDIOCAST_HAVE_PULSE_SIMPLE
#include <pulse/context.h>
#include <pulse/error.h>
#include <pulse/introspect.h>
#include <pulse/operation.h>
#include <pulse/subscribe.h>
#include <pulse/thread-mainloop.h>
#endif

namespace studiocast::audio {
namespace {

using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

bool SetNonBlocking(int fd) {
  const int flags = ::fcntl(fd, F_GETFL, 0);
  return flags >= 0 && ::fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0;
}

bool WaitForChild(pid_t pid, std::chrono::milliseconds timeout, int *status) {
  const auto deadline = Clock::now() + timeout;
  for (;;) {
    const pid_t result = ::waitpid(pid, status, WNOHANG);
    if (result == pid)
      return true;
    if (result < 0 && errno != EINTR)
      return true;
    if (Clock::now() >= deadline)
      return false;
    std::this_thread::sleep_for(2ms);
  }
}

void SignalChildProcessGroup(pid_t pid, int signal) {
  if (pid <= 0)
    return;
  if (::kill(-pid, signal) != 0 && errno == ESRCH)
    (void)::kill(pid, signal);
}

class PactlSubscribeProcess final : public PactlSubscriptionMonitor {
public:
  static std::unique_ptr<PactlSubscribeProcess> Create(std::string *error) {
    if (error)
      error->clear();

    int pipeFds[2] = {-1, -1};
    if (::pipe2(pipeFds, O_CLOEXEC) != 0) {
      if (error)
        *error = "pactl subscription pipe creation failed";
      return nullptr;
    }

    const pid_t pid = ::fork();
    if (pid < 0) {
      ::close(pipeFds[0]);
      ::close(pipeFds[1]);
      if (error)
        *error = "pactl subscription process creation failed";
      return nullptr;
    }

    if (pid == 0) {
      (void)::setpgid(0, 0);
      ::close(pipeFds[0]);
      (void)::dup2(pipeFds[1], STDOUT_FILENO);
      (void)::dup2(pipeFds[1], STDERR_FILENO);
      ::close(pipeFds[1]);
      ::execlp("pactl", "pactl", "subscribe", static_cast<char *>(nullptr));
      _exit(127);
    }

    (void)::setpgid(pid, pid);
    ::close(pipeFds[1]);
    if (!SetNonBlocking(pipeFds[0])) {
      ::close(pipeFds[0]);
      int status = 0;
      SignalChildProcessGroup(pid, SIGKILL);
      (void)WaitForChild(pid, 250ms, &status);
      if (error)
        *error = "pactl subscription pipe setup failed";
      return nullptr;
    }

    return std::unique_ptr<PactlSubscribeProcess>(
        new PactlSubscribeProcess(pid, pipeFds[0]));
  }

  ~PactlSubscribeProcess() override { Stop(); }

  PactlSubscriptionPollResult Poll() override {
    PactlSubscriptionPollResult result;
    if (pid_ <= 0 || read_fd_ < 0) {
      result.disconnected = true;
      result.error = "pactl subscription is not running";
      return result;
    }

    // Drain at most a fixed amount. The application stores only a dirty bit;
    // the kernel pipe is bounded and remaining events are handled next poll.
    std::array<char, 4096> buffer{};
    std::size_t drained = 0;
    bool pipeDisconnected = false;
    constexpr std::size_t kMaxDrainBytes = 64 * 1024;
    while (drained < kMaxDrainBytes) {
      const ssize_t n = ::read(read_fd_, buffer.data(), buffer.size());
      if (n > 0) {
        drained += static_cast<std::size_t>(n);
        result.event_received = true;
        continue;
      }
      if (n < 0 && errno == EINTR)
        continue;
      if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
        break;
      pipeDisconnected = true;
      break;
    }

    int status = 0;
    const pid_t waited = ::waitpid(pid_, &status, WNOHANG);
    if (waited == pid_ || (waited < 0 && errno != EINTR)) {
      result.disconnected = true;
      if (waited == pid_ && WIFEXITED(status)) {
        result.error = "pactl subscription exited with status " +
                       std::to_string(WEXITSTATUS(status));
      } else {
        result.error = "pactl subscription disconnected";
      }
      ::close(read_fd_);
      read_fd_ = -1;
      pid_ = -1;
    } else if (pipeDisconnected) {
      result.disconnected = true;
      result.error = "pactl subscription output disconnected";
    }
    return result;
  }

  void Stop() override {
    if (read_fd_ >= 0) {
      ::close(read_fd_);
      read_fd_ = -1;
    }
    if (pid_ <= 0)
      return;

    int status = 0;
    SignalChildProcessGroup(pid_, SIGTERM);
    if (!WaitForChild(pid_, 50ms, &status)) {
      SignalChildProcessGroup(pid_, SIGKILL);
      (void)WaitForChild(pid_, 250ms, &status);
    }
    pid_ = -1;
  }

private:
  PactlSubscribeProcess(pid_t pid, int read_fd)
      : pid_(pid), read_fd_(read_fd) {}

  pid_t pid_ = -1;
  int read_fd_ = -1;
};

std::unique_ptr<PactlSubscriptionMonitor>
CreatePactlSubscribeProcess(std::string *error) {
  return PactlSubscribeProcess::Create(error);
}

struct PactlConsumerCache {
  std::vector<pulse::PactlSource> sources;
  std::vector<pulse::PactlSink> sinks;
  std::vector<pulse::PactlSourceOutput> source_outputs;
  std::vector<pulse::PactlSinkInput> sink_inputs;
};

class PactlSubscriptionAudioConsumerDetector final
    : public AudioConsumerDetector {
public:
  PactlSubscriptionAudioConsumerDetector(
      PactlSubscriptionMonitorFactory monitor_factory,
      const std::atomic_bool *stop_requested,
      std::chrono::milliseconds reconnect_delay)
      : monitor_factory_(std::move(monitor_factory)),
        stop_requested_(stop_requested), reconnect_delay_(reconnect_delay) {}

  ~PactlSubscriptionAudioConsumerDetector() override { Suspend(); }

  AudioConsumerSnapshot
  DetectSourceConsumersByName(const std::string &source_name) override {
    if (!PrepareSnapshot())
      return UnavailableSnapshot();

    AudioConsumerSnapshot out;
    bool found = false;
    for (const auto &source : cache_.sources) {
      if (source.name == source_name)
        found = true;
    }
    if (!found) {
      out.error = "Pulse source '" + source_name + "' is not present.";
      return out;
    }
    for (const auto &output : cache_.source_outputs) {
      if (output.source == source_name) {
        ++out.count;
        continue;
      }
      for (const auto &source : cache_.sources) {
        if (source.name == source_name &&
            output.source == std::to_string(source.id)) {
          ++out.count;
          break;
        }
      }
    }
    out.present = out.count > 0;
    return out;
  }

  AudioConsumerSnapshot
  DetectSinkConsumersByName(const std::string &sink_name) override {
    if (!PrepareSnapshot())
      return UnavailableSnapshot();

    AudioConsumerSnapshot out;
    bool found = false;
    for (const auto &sink : cache_.sinks) {
      if (sink.name == sink_name)
        found = true;
    }
    if (!found) {
      out.error = "Pulse sink '" + sink_name + "' is not present.";
      return out;
    }
    for (const auto &input : cache_.sink_inputs) {
      if (input.sink == sink_name) {
        ++out.count;
        continue;
      }
      for (const auto &sink : cache_.sinks) {
        if (sink.name == sink_name && input.sink == std::to_string(sink.id)) {
          ++out.count;
          break;
        }
      }
    }
    out.present = out.count > 0;
    return out;
  }

  void Suspend() {
    if (monitor_) {
      monitor_->Stop();
      monitor_.reset();
    }
    snapshot_valid_ = false;
    dirty_ = true;
  }

private:
  bool EnsureMonitor() {
    if (monitor_)
      return true;
    if (stop_requested_ && stop_requested_->load(std::memory_order_acquire)) {
      last_error_ = "pactl consumer subscription stopped";
      return false;
    }

    const auto now = Clock::now();
    if (now < next_monitor_attempt_)
      return false;

    std::string error;
    monitor_ = monitor_factory_ ? monitor_factory_(&error) : nullptr;
    if (!monitor_) {
      last_error_ = error.empty() ? "pactl subscription unavailable" : error;
      next_monitor_attempt_ = now + reconnect_delay_;
      return false;
    }

    last_error_.clear();
    next_monitor_attempt_ = {};
    dirty_ = true;
    return true;
  }

  bool PrepareSnapshot() {
    if (!EnsureMonitor())
      return false;

    const auto event = monitor_->Poll();
    if (event.disconnected) {
      last_error_ =
          event.error.empty() ? "pactl subscription disconnected" : event.error;
      monitor_->Stop();
      monitor_.reset();
      snapshot_valid_ = false;
      dirty_ = true;
      next_monitor_attempt_ = Clock::now() + reconnect_delay_;
      return false;
    }
    if (event.event_received)
      dirty_ = true;

    if (!dirty_)
      return snapshot_valid_;

    std::string error;
    if (!RefreshSnapshot(&error)) {
      last_error_ = error.empty() ? "pactl consumer snapshot refresh failed"
                                  : std::move(error);
      monitor_->Stop();
      monitor_.reset();
      snapshot_valid_ = false;
      dirty_ = true;
      next_monitor_attempt_ = Clock::now() + reconnect_delay_;
      return false;
    }

    last_error_.clear();
    dirty_ = false;
    snapshot_valid_ = true;
    return true;
  }

  bool RefreshSnapshot(std::string *error) {
    PactlConsumerCache next;
    std::string detail;
    next.sources = pulse::ListSources(&detail, stop_requested_);
    if (!detail.empty()) {
      if (error)
        *error = "Failed to list Pulse sources: " + detail;
      return false;
    }

    next.sinks = pulse::ListSinks(&detail, stop_requested_);
    if (!detail.empty()) {
      if (error)
        *error = "Failed to list Pulse sinks: " + detail;
      return false;
    }

    next.source_outputs = pulse::ListSourceOutputs(&detail, stop_requested_);
    if (!detail.empty()) {
      if (error)
        *error = "Failed to list Pulse source outputs: " + detail;
      return false;
    }

    next.sink_inputs = pulse::ListSinkInputs(&detail, stop_requested_);
    if (!detail.empty()) {
      if (error)
        *error = "Failed to list Pulse sink inputs: " + detail;
      return false;
    }

    cache_ = std::move(next);
    return true;
  }

  AudioConsumerSnapshot UnavailableSnapshot() const {
    AudioConsumerSnapshot out;
    out.error = "Pulse consumer subscription unavailable: " + last_error_;
    return out;
  }

  PactlSubscriptionMonitorFactory monitor_factory_;
  const std::atomic_bool *stop_requested_ = nullptr;
  std::chrono::milliseconds reconnect_delay_{};
  std::unique_ptr<PactlSubscriptionMonitor> monitor_;
  Clock::time_point next_monitor_attempt_{};
  PactlConsumerCache cache_;
  bool dirty_ = true;
  bool snapshot_valid_ = false;
  std::string last_error_;
};

struct ConsumerDetectionResult {
  AudioConsumerSnapshot snapshot;
  bool detector_unavailable = false;
};

#if STUDIOCAST_HAVE_PULSE_SIMPLE

std::string PulseErrorString(int pa_error) {
  const char *s = ::pa_strerror(pa_error);
  if (!s || *s == '\0') {
    return "PulseAudio error " + std::to_string(pa_error);
  }
  return std::string(s);
}

struct EndpointRef {
  int id = -1;
  std::string name;
};

struct SourceOutputRef {
  int id = -1;
  int source_id = -1;
};

struct SinkInputRef {
  int id = -1;
  int sink_id = -1;
};

class PulseSubscriptionConsumerDetector final {
public:
  static std::unique_ptr<PulseSubscriptionConsumerDetector>
  Create(std::string *error) {
    auto detector = std::unique_ptr<PulseSubscriptionConsumerDetector>(
        new PulseSubscriptionConsumerDetector());
    if (!detector->Initialize(error))
      return nullptr;
    return detector;
  }

  ~PulseSubscriptionConsumerDetector() { Shutdown(); }

  PulseSubscriptionConsumerDetector(
      const PulseSubscriptionConsumerDetector &) = delete;
  PulseSubscriptionConsumerDetector &
  operator=(const PulseSubscriptionConsumerDetector &) = delete;

  ConsumerDetectionResult
  DetectSourceConsumersByName(const std::string &source_name) {
    ConsumerDetectionResult result;

    ::pa_threaded_mainloop_lock(mainloop_);
    if (!ContextReadyLocked()) {
      result.snapshot.error = "Pulse consumer monitor is disconnected.";
      result.detector_unavailable = true;
      ::pa_threaded_mainloop_unlock(mainloop_);
      return result;
    }

    std::string err;
    if (dirty_ && !RefreshLocked(&err)) {
      result.snapshot.error = err;
      result.detector_unavailable = true;
      ::pa_threaded_mainloop_unlock(mainloop_);
      return result;
    }

    std::vector<int> sourceIds;
    for (const auto &source : sources_) {
      if (source.name == source_name)
        sourceIds.push_back(source.id);
    }
    if (sourceIds.empty()) {
      result.snapshot.error =
          "Pulse source '" + source_name + "' is not present.";
      ::pa_threaded_mainloop_unlock(mainloop_);
      return result;
    }

    for (const auto &output : source_outputs_) {
      if (std::find(sourceIds.begin(), sourceIds.end(), output.source_id) !=
          sourceIds.end()) {
        ++result.snapshot.count;
      }
    }
    result.snapshot.present = result.snapshot.count > 0;
    ::pa_threaded_mainloop_unlock(mainloop_);
    return result;
  }

  ConsumerDetectionResult
  DetectSinkConsumersByName(const std::string &sink_name) {
    ConsumerDetectionResult result;

    ::pa_threaded_mainloop_lock(mainloop_);
    if (!ContextReadyLocked()) {
      result.snapshot.error = "Pulse consumer monitor is disconnected.";
      result.detector_unavailable = true;
      ::pa_threaded_mainloop_unlock(mainloop_);
      return result;
    }

    std::string err;
    if (dirty_ && !RefreshLocked(&err)) {
      result.snapshot.error = err;
      result.detector_unavailable = true;
      ::pa_threaded_mainloop_unlock(mainloop_);
      return result;
    }

    std::vector<int> sinkIds;
    for (const auto &sink : sinks_) {
      if (sink.name == sink_name)
        sinkIds.push_back(sink.id);
    }
    if (sinkIds.empty()) {
      result.snapshot.error = "Pulse sink '" + sink_name + "' is not present.";
      ::pa_threaded_mainloop_unlock(mainloop_);
      return result;
    }

    for (const auto &input : sink_inputs_) {
      if (std::find(sinkIds.begin(), sinkIds.end(), input.sink_id) !=
          sinkIds.end()) {
        ++result.snapshot.count;
      }
    }
    result.snapshot.present = result.snapshot.count > 0;
    ::pa_threaded_mainloop_unlock(mainloop_);
    return result;
  }

private:
  PulseSubscriptionConsumerDetector() = default;

  struct OperationWaitState {
    PulseSubscriptionConsumerDetector *self = nullptr;
    bool done = false;
    std::string error;
  };

  struct SuccessOperationState : OperationWaitState {
    bool success = false;
  };

  struct SourceListState : OperationWaitState {
    std::vector<EndpointRef> items;
  };

  struct SinkListState : OperationWaitState {
    std::vector<EndpointRef> items;
  };

  struct SourceOutputListState : OperationWaitState {
    std::vector<SourceOutputRef> items;
  };

  struct SinkInputListState : OperationWaitState {
    std::vector<SinkInputRef> items;
  };

  bool Initialize(std::string *error) {
    if (error)
      error->clear();

    mainloop_ = ::pa_threaded_mainloop_new();
    if (!mainloop_) {
      if (error)
        *error = "PulseAudio threaded mainloop allocation failed.";
      return false;
    }
    if (::pa_threaded_mainloop_start(mainloop_) < 0) {
      if (error)
        *error = "PulseAudio threaded mainloop start failed.";
      ::pa_threaded_mainloop_free(mainloop_);
      mainloop_ = nullptr;
      return false;
    }

    ::pa_threaded_mainloop_lock(mainloop_);
    context_ = ::pa_context_new(::pa_threaded_mainloop_get_api(mainloop_),
                                "studiocast-consumer-monitor");
    if (!context_) {
      if (error)
        *error = "PulseAudio context allocation failed.";
      ::pa_threaded_mainloop_unlock(mainloop_);
      Shutdown();
      return false;
    }

    ::pa_context_set_state_callback(
        context_, &PulseSubscriptionConsumerDetector::ContextStateCb, this);
    if (::pa_context_connect(context_, nullptr, PA_CONTEXT_NOFLAGS, nullptr) <
        0) {
      if (error) {
        *error =
            "PulseAudio context connect failed: " + ContextErrorLocked();
      }
      ::pa_threaded_mainloop_unlock(mainloop_);
      Shutdown();
      return false;
    }
    if (!WaitForContextReadyLocked(error)) {
      ::pa_threaded_mainloop_unlock(mainloop_);
      Shutdown();
      return false;
    }

    ::pa_context_set_subscribe_callback(
        context_, &PulseSubscriptionConsumerDetector::SubscribeEventCb, this);
    SuccessOperationState subState;
    subState.self = this;
    const auto mask = static_cast<pa_subscription_mask_t>(
        PA_SUBSCRIPTION_MASK_SOURCE | PA_SUBSCRIPTION_MASK_SINK |
        PA_SUBSCRIPTION_MASK_SOURCE_OUTPUT |
        PA_SUBSCRIPTION_MASK_SINK_INPUT | PA_SUBSCRIPTION_MASK_SERVER);
    pa_operation *op = ::pa_context_subscribe(
        context_, mask, &PulseSubscriptionConsumerDetector::SuccessCb,
        &subState);
    if (!WaitForSuccessOperationLocked(op, &subState,
                                       "Pulse subscription setup", error)) {
      ::pa_threaded_mainloop_unlock(mainloop_);
      Shutdown();
      return false;
    }

    std::string refreshErr;
    if (!RefreshLocked(&refreshErr)) {
      if (error)
        *error = refreshErr;
      ::pa_threaded_mainloop_unlock(mainloop_);
      Shutdown();
      return false;
    }

    ::pa_threaded_mainloop_unlock(mainloop_);
    return true;
  }

  bool WaitForContextReadyLocked(std::string *error) {
    while (true) {
      const pa_context_state_t state = ::pa_context_get_state(context_);
      if (state == PA_CONTEXT_READY)
        return true;
      if (!PA_CONTEXT_IS_GOOD(state)) {
        if (error) {
          *error = "PulseAudio context is not ready: " + ContextErrorLocked();
        }
        return false;
      }
      ::pa_threaded_mainloop_wait(mainloop_);
    }
  }

  bool ContextReadyLocked() const {
    return context_ && ::pa_context_get_state(context_) == PA_CONTEXT_READY;
  }

  bool WaitForOperationLocked(pa_operation *op, OperationWaitState *state,
                              const char *label, std::string *error) {
    if (!op) {
      if (error)
        *error = std::string(label) + " request failed: " +
                 ContextErrorLocked();
      return false;
    }

    while (!state->done &&
           ::pa_operation_get_state(op) == PA_OPERATION_RUNNING) {
      if (!ContextReadyLocked()) {
        state->error =
            std::string(label) + " failed: " + ContextErrorLocked();
        break;
      }
      ::pa_threaded_mainloop_wait(mainloop_);
    }
    ::pa_operation_unref(op);

    if (!state->error.empty()) {
      if (error)
        *error = state->error;
      return false;
    }
    if (!state->done) {
      if (error)
        *error = std::string(label) + " did not complete.";
      return false;
    }
    return true;
  }

  bool WaitForSuccessOperationLocked(pa_operation *op,
                                     SuccessOperationState *state,
                                     const char *label, std::string *error) {
    if (!WaitForOperationLocked(op, state, label, error))
      return false;
    if (!state->success) {
      if (error)
        *error = std::string(label) + " was rejected: " +
                 ContextErrorLocked();
      return false;
    }
    return true;
  }

  bool RefreshLocked(std::string *error) {
    SourceListState sources;
    sources.self = this;
    pa_operation *sourceOp = ::pa_context_get_source_info_list(
        context_, &PulseSubscriptionConsumerDetector::SourceInfoCb, &sources);
    if (!WaitForOperationLocked(sourceOp, &sources, "Pulse source list",
                                error)) {
      dirty_ = true;
      return false;
    }

    SinkListState sinks;
    sinks.self = this;
    pa_operation *sinkOp = ::pa_context_get_sink_info_list(
        context_, &PulseSubscriptionConsumerDetector::SinkInfoCb, &sinks);
    if (!WaitForOperationLocked(sinkOp, &sinks, "Pulse sink list", error)) {
      dirty_ = true;
      return false;
    }

    SourceOutputListState sourceOutputs;
    sourceOutputs.self = this;
    pa_operation *sourceOutputOp = ::pa_context_get_source_output_info_list(
        context_, &PulseSubscriptionConsumerDetector::SourceOutputInfoCb,
        &sourceOutputs);
    if (!WaitForOperationLocked(sourceOutputOp, &sourceOutputs,
                                "Pulse source-output list", error)) {
      dirty_ = true;
      return false;
    }

    SinkInputListState sinkInputs;
    sinkInputs.self = this;
    pa_operation *sinkInputOp = ::pa_context_get_sink_input_info_list(
        context_, &PulseSubscriptionConsumerDetector::SinkInputInfoCb,
        &sinkInputs);
    if (!WaitForOperationLocked(sinkInputOp, &sinkInputs,
                                "Pulse sink-input list", error)) {
      dirty_ = true;
      return false;
    }

    sources_ = std::move(sources.items);
    sinks_ = std::move(sinks.items);
    source_outputs_ = std::move(sourceOutputs.items);
    sink_inputs_ = std::move(sinkInputs.items);
    dirty_ = false;
    return true;
  }

  std::string ContextErrorLocked() const {
    if (!context_)
      return "connection state is unavailable";
    return PulseErrorString(::pa_context_errno(context_));
  }

  void SignalLocked() {
    if (mainloop_)
      ::pa_threaded_mainloop_signal(mainloop_, 0);
  }

  void Shutdown() {
    if (!mainloop_)
      return;

    ::pa_threaded_mainloop_lock(mainloop_);
    if (context_) {
      ::pa_context_set_subscribe_callback(context_, nullptr, nullptr);
      ::pa_context_set_state_callback(context_, nullptr, nullptr);
      ::pa_context_disconnect(context_);
      ::pa_context_unref(context_);
      context_ = nullptr;
    }
    ::pa_threaded_mainloop_signal(mainloop_, 0);
    ::pa_threaded_mainloop_unlock(mainloop_);
    ::pa_threaded_mainloop_stop(mainloop_);
    ::pa_threaded_mainloop_free(mainloop_);
    mainloop_ = nullptr;
  }

  static void ContextStateCb(pa_context *, void *userdata) {
    auto *self =
        static_cast<PulseSubscriptionConsumerDetector *>(userdata);
    self->SignalLocked();
  }

  static void SubscribeEventCb(pa_context *, pa_subscription_event_type_t type,
                               uint32_t, void *userdata) {
    auto *self =
        static_cast<PulseSubscriptionConsumerDetector *>(userdata);
    const auto facility =
        static_cast<pa_subscription_event_type_t>(
            type & PA_SUBSCRIPTION_EVENT_FACILITY_MASK);
    switch (facility) {
    case PA_SUBSCRIPTION_EVENT_SOURCE:
    case PA_SUBSCRIPTION_EVENT_SINK:
    case PA_SUBSCRIPTION_EVENT_SOURCE_OUTPUT:
    case PA_SUBSCRIPTION_EVENT_SINK_INPUT:
    case PA_SUBSCRIPTION_EVENT_SERVER:
      self->dirty_ = true;
      self->SignalLocked();
      break;
    default:
      break;
    }
  }

  static void SuccessCb(pa_context *c, int success, void *userdata) {
    auto *state = static_cast<SuccessOperationState *>(userdata);
    state->success = (success != 0);
    if (!state->success) {
      state->error =
          "Pulse operation failed: " + PulseErrorString(::pa_context_errno(c));
    }
    state->done = true;
    state->self->SignalLocked();
  }

  static void SourceInfoCb(pa_context *c, const pa_source_info *info, int eol,
                           void *userdata) {
    auto *state = static_cast<SourceListState *>(userdata);
    if (eol < 0) {
      state->error =
          "Pulse source list failed: " + PulseErrorString(::pa_context_errno(c));
      state->done = true;
      state->self->SignalLocked();
      return;
    }
    if (eol > 0) {
      state->done = true;
      state->self->SignalLocked();
      return;
    }
    if (info && info->name) {
      state->items.push_back(
          EndpointRef{static_cast<int>(info->index), info->name});
    }
  }

  static void SinkInfoCb(pa_context *c, const pa_sink_info *info, int eol,
                         void *userdata) {
    auto *state = static_cast<SinkListState *>(userdata);
    if (eol < 0) {
      state->error =
          "Pulse sink list failed: " + PulseErrorString(::pa_context_errno(c));
      state->done = true;
      state->self->SignalLocked();
      return;
    }
    if (eol > 0) {
      state->done = true;
      state->self->SignalLocked();
      return;
    }
    if (info && info->name) {
      state->items.push_back(
          EndpointRef{static_cast<int>(info->index), info->name});
    }
  }

  static void SourceOutputInfoCb(pa_context *c,
                                 const pa_source_output_info *info, int eol,
                                 void *userdata) {
    auto *state = static_cast<SourceOutputListState *>(userdata);
    if (eol < 0) {
      state->error = "Pulse source-output list failed: " +
                     PulseErrorString(::pa_context_errno(c));
      state->done = true;
      state->self->SignalLocked();
      return;
    }
    if (eol > 0) {
      state->done = true;
      state->self->SignalLocked();
      return;
    }
    if (info) {
      state->items.push_back(SourceOutputRef{
          static_cast<int>(info->index), static_cast<int>(info->source)});
    }
  }

  static void SinkInputInfoCb(pa_context *c, const pa_sink_input_info *info,
                              int eol, void *userdata) {
    auto *state = static_cast<SinkInputListState *>(userdata);
    if (eol < 0) {
      state->error = "Pulse sink-input list failed: " +
                     PulseErrorString(::pa_context_errno(c));
      state->done = true;
      state->self->SignalLocked();
      return;
    }
    if (eol > 0) {
      state->done = true;
      state->self->SignalLocked();
      return;
    }
    if (info) {
      state->items.push_back(SinkInputRef{
          static_cast<int>(info->index), static_cast<int>(info->sink)});
    }
  }

  pa_threaded_mainloop *mainloop_ = nullptr;
  pa_context *context_ = nullptr;
  bool dirty_ = true;

  std::vector<EndpointRef> sources_;
  std::vector<EndpointRef> sinks_;
  std::vector<SourceOutputRef> source_outputs_;
  std::vector<SinkInputRef> sink_inputs_;
};

#endif // STUDIOCAST_HAVE_PULSE_SIMPLE

class DefaultAudioConsumerDetector final : public AudioConsumerDetector {
public:
  explicit DefaultAudioConsumerDetector(const std::atomic_bool *stop_requested)
      : pactl_(CreatePactlSubscribeProcess, stop_requested,
               kPactlReconnectDelay) {}

  AudioConsumerSnapshot
  DetectSourceConsumersByName(const std::string &source_name) override {
#if STUDIOCAST_HAVE_PULSE_SIMPLE
    if (auto result = DetectWithPulseSource(source_name);
        result.has_value && !result.detection.detector_unavailable) {
      pactl_.Suspend();
      return result.detection.snapshot;
    }
#endif
    return FinishPactlFallback(
        pactl_.DetectSourceConsumersByName(source_name));
  }

  AudioConsumerSnapshot
  DetectSinkConsumersByName(const std::string &sink_name) override {
#if STUDIOCAST_HAVE_PULSE_SIMPLE
    if (auto result = DetectWithPulseSink(sink_name);
        result.has_value && !result.detection.detector_unavailable) {
      pactl_.Suspend();
      return result.detection.snapshot;
    }
#endif
    return FinishPactlFallback(pactl_.DetectSinkConsumersByName(sink_name));
  }

private:
  static constexpr auto kPactlReconnectDelay = std::chrono::seconds(5);

  PactlSubscriptionAudioConsumerDetector pactl_;

#if STUDIOCAST_HAVE_PULSE_SIMPLE
  struct MaybePulseResult {
    bool has_value = false;
    ConsumerDetectionResult detection;
  };

  MaybePulseResult DetectWithPulseSource(const std::string &source_name) {
    EnsurePulseDetector();
    if (!pulse_)
      return {};

    auto result = pulse_->DetectSourceConsumersByName(source_name);
    if (result.detector_unavailable) {
      last_pulse_error_ = result.snapshot.error;
      pulse_.reset();
      next_pulse_attempt_ =
          std::chrono::steady_clock::now() + kPulseReconnectDelay;
    }
    return MaybePulseResult{true, std::move(result)};
  }

  MaybePulseResult DetectWithPulseSink(const std::string &sink_name) {
    EnsurePulseDetector();
    if (!pulse_)
      return {};

    auto result = pulse_->DetectSinkConsumersByName(sink_name);
    if (result.detector_unavailable) {
      last_pulse_error_ = result.snapshot.error;
      pulse_.reset();
      next_pulse_attempt_ =
          std::chrono::steady_clock::now() + kPulseReconnectDelay;
    }
    return MaybePulseResult{true, std::move(result)};
  }

  void EnsurePulseDetector() {
    if (pulse_)
      return;

    const auto now = std::chrono::steady_clock::now();
    if (now < next_pulse_attempt_)
      return;

    std::string err;
    pulse_ = PulseSubscriptionConsumerDetector::Create(&err);
    if (pulse_) {
      last_pulse_error_.clear();
      next_pulse_attempt_ = {};
      return;
    }

    last_pulse_error_ = err.empty() ? "libpulse subscription unavailable" : err;
    next_pulse_attempt_ = now + kPulseReconnectDelay;
  }

  AudioConsumerSnapshot FinishPactlFallback(AudioConsumerSnapshot fallback) {
    if (!last_pulse_error_.empty() && !fallback.error.empty()) {
      fallback.error = "Pulse consumer monitor unavailable: " +
                       last_pulse_error_ + "; pactl fallback failed: " +
                       fallback.error;
    }
    return fallback;
  }

  static constexpr auto kPulseReconnectDelay = std::chrono::seconds(5);

  std::unique_ptr<PulseSubscriptionConsumerDetector> pulse_;
  std::chrono::steady_clock::time_point next_pulse_attempt_{};
  std::string last_pulse_error_;
#else
  AudioConsumerSnapshot FinishPactlFallback(AudioConsumerSnapshot fallback) {
    return fallback;
  }
#endif
};

} // namespace

std::unique_ptr<AudioConsumerDetector>
CreateDefaultAudioConsumerDetector(const std::atomic_bool *stop_requested) {
  return std::make_unique<DefaultAudioConsumerDetector>(stop_requested);
}

std::unique_ptr<AudioConsumerDetector>
CreatePactlSubscriptionAudioConsumerDetectorForTesting(
    PactlSubscriptionMonitorFactory monitor_factory,
    const std::atomic_bool *stop_requested,
    std::chrono::milliseconds reconnect_delay) {
  return std::make_unique<PactlSubscriptionAudioConsumerDetector>(
      std::move(monitor_factory), stop_requested, reconnect_delay);
}

} // namespace studiocast::audio
