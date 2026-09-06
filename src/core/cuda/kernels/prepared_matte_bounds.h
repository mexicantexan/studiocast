#pragma once

#include "core/cuda/cuda_driver_api.h"
#include "core/cuda/cuda_image.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace studiocast::cuda::kernels {

extern const char kPreparedMatteBoundsPtx[];

struct DeviceMatteBoundsRecord {
  std::int32_t x0 = 0;
  std::int32_t y0 = 0;
  std::int32_t x1 = -1;
  std::int32_t y1 = -1;
  std::uint32_t foreground_count = 0;
};

struct PreparedMatteBoundsBudget {
  std::size_t device_bytes = 0;
  std::uint32_t resource_count = 0;
};

struct PreparedMatteBounds {
  CudaDriverApi *driver = nullptr;
  CUcontext owner_context = nullptr;
  CUstream owner_stream = nullptr;
  CUmodule module = nullptr;
  CUfunction initialize = nullptr;
  CUfunction reduce = nullptr;
  CUdeviceptr device_record = 0;
  std::size_t device_record_bytes = 0;
  bool live = false;

  PreparedMatteBounds() = default;
  PreparedMatteBounds(const PreparedMatteBounds &) = delete;
  PreparedMatteBounds &operator=(const PreparedMatteBounds &) = delete;
  PreparedMatteBounds(PreparedMatteBounds &&other) noexcept;
  PreparedMatteBounds &operator=(PreparedMatteBounds &&) = delete;
};

// Logical lower bound used for preflight only. Pitched allocation may be
// larger; PreparedMatteBoundsReservedBudget is the exact post-allocation
// authority.
constexpr PreparedMatteBoundsBudget MinimumMatteBoundsBudget() noexcept {
  return {sizeof(DeviceMatteBoundsRecord), 1}; // one owned device allocation
}

bool PrepareMatteBounds(CudaDriverApi &driver, CUcontext owner_context,
                        CUstream owner_stream, std::size_t device_budget_bytes,
                        PreparedMatteBounds *out, std::string *error_out);

// Enqueues initialization and reduction only. The caller owns completion,
// bounded metadata readback, and synchronization. Alpha remains device-local.
// A false result may follow the initialization enqueue; the caller must treat
// every launch attempt as potentially pending and quiesce before destruction.
bool LaunchMatteBoundsPrepared(const PreparedMatteBounds &prepared,
                               const CudaImage &alpha, std::string *error_out);

CUdeviceptr MatteBoundsDeviceRecord(const PreparedMatteBounds &) noexcept;
std::size_t MatteBoundsDeviceRecordBytes(const PreparedMatteBounds &) noexcept;
PreparedMatteBoundsBudget
PreparedMatteBoundsReservedBudget(const PreparedMatteBounds &) noexcept;
bool MatteBoundsForegroundFound(const DeviceMatteBoundsRecord &, int width,
                                int height) noexcept;

// Caller must quiesce owner_stream and make owner_context current. Failures
// retain every resource not successfully released so destruction can retry.
bool DestroyMatteBoundsPrepared(PreparedMatteBounds *, std::string *error_out);

} // namespace studiocast::cuda::kernels
