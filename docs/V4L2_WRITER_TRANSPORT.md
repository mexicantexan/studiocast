# V4L2 Writer Transport Contract

This contract governs the latency-sensitive V4L2 output boundary used by the
camera pipeline, test-pattern feed, and video tool. It supplements
`LATENCY_POLICY.md` and `PIPELINE_POLICY.md`; those policies remain
authoritative.

## Prepared transport

- The writer opens the device once with `O_NONBLOCK | O_CLOEXEC`, preferring
  `O_WRONLY` and falling back to `O_RDWR` only during setup.
- Format and capability negotiation occur during open, explicit output setup,
  or the failure-driven recovery described below. There is no periodic
  per-frame format probe.
- There is no userspace output queue (`max queued frames = 0`). The frame
  presented to the writer is always the current frame.

## Write and maximum-wait semantics

- A normal frame performs one direct nonblocking `write()` and no preceding
  `poll()`, `ioctl()`, wait, or synchronization point.
- A full-frame result commits the frame. No error string is constructed on this
  successful path.
- `EINTR` is retried at most four times after the initial call, for a maximum of
  five nonblocking write attempts. Stop is checked before the first attempt and
  between retries. There is no retry sleep or wall-clock wait.
- The transport therefore adds no userspace wait interval: its declared bound
  is five nonblocking syscall attempts. The hermetic acceptance bound for a
  full backpressured kernel transport is 25 ms on the validation host. OS
  scheduling and a defective driver that violates `O_NONBLOCK` are outside a
  process-level hard real-time guarantee and must be reported as such.
- The camera's existing capture wait is independently bounded at 1000 ms; this
  output contract removes the previously unbounded additional write wait.

## Stop, drop, and frame integrity

- A set stop token returns `stopped` without entering `write()`. Stop becoming
  set during repeated `EINTR` is observed before the next attempt.
- `EAGAIN`/`EWOULDBLOCK` returns `would_block_dropped`. The current frame is
  dropped immediately and a later frame replaces it. It does not trigger a
  probe, error formatting, retry loop, or queue insertion.
- A positive short write is fatal. The remainder is never written because doing
  so could cross a V4L2 frame boundary. A zero-byte write is also fatal. The
  concrete writer closes immediately after either result and must be reopened
  before another frame can be written.
- `ENODEV`, `ENXIO`, `EPIPE`, and `EBADF` are visible disconnect failures.
  Other undeclared I/O errors are fatal. Missing output is never silently
  replaced.

## Recovery and renegotiation

- `EINVAL` and `EMSGSIZE` are the only write failures classified as potential
  consumer format mismatch. The camera may perform one failure-driven
  `RefreshActual()` and drop that frame if the kernel format actually changed.
- An unchanged or failed refresh does not hide the write failure. Disconnect,
  partial-frame, retry-limit, and generic I/O failures do not trigger an
  opportunistic format probe.
- A successful refresh rebuilds output layout, conversion scratch, pacing, and
  the raw-YUYV passthrough plan before the next frame. No stale layout is reused.

## Measurement scope

Release validation records the baseline and candidate successful-write
average, p95, p99, minimum, maximum, and syscall count with preallocated 720p
and 1080p frames after warmup. Hermetic tests cover full-frame, partial, zero,
`EINTR`, `EAGAIN`, disconnect, format mismatch, stopped, and full-pipe behavior.
The full-pipe test also demonstrates that the former blocking mode ignores an
application stop request until the kernel transport is drained.

Real v4l2loopback measurements are optional only when a pre-existing, safe,
writable device is available. Installing packages, loading modules, changing
services, or renegotiating a user's active device is outside automated
validation and requires explicit authorization.
