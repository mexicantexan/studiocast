# V4L2 Writer Latency Measurements

## Comparison

- Audit comparison baseline: `d2a6f112811297243043f633404fede820f4073b`
- Candidate: the Issue 6 topic commit containing this document
- Build: CMake Release, Ninja, Open CUDA off, Open Audio off, dlib off,
  benchmarks enabled
- Host: Intel Core i7-8750H (6 cores/12 threads, 0.8-4.1 GHz), x86-64
- Kernel: Ubuntu `6.8.0-100-generic`, PREEMPT_DYNAMIC
- Frame data: preallocated YUYV-sized buffers; no effects; output geometry
  1280x720 and 1920x1080
- Warmup/measured: 1,000/20,000 writes per implementation and geometry
- Timing clock: `std::chrono::steady_clock`, one sample per complete write call
- Successful transport: pre-opened `/dev/null` with `O_NONBLOCK`; this isolates
  application call-path/syscall overhead without touching a video device

The benchmark reproduces the old blocking remainder loop directly for the
baseline row. The candidate row invokes the same direct-syscall typed contract
used by `V4l2Writer`; the scripted function-pointer seam is not measured as the
production path.

| Path | Geometry | avg us | p95 us | p99 us | min us | max us | writes/frame |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline blocking loop | 1280x720 | 0.672481 | 0.662000 | 0.856000 | 0.653000 | 19.196000 | 1.000000 |
| Candidate typed nonblocking | 1280x720 | 0.696858 | 0.702000 | 1.228000 | 0.658000 | 7.013000 | 1.000000 |
| Baseline blocking loop | 1920x1080 | 0.674159 | 0.685000 | 0.872000 | 0.652000 | 6.735000 | 1.000000 |
| Candidate typed nonblocking | 1920x1080 | 0.695105 | 0.694000 | 1.183000 | 0.657000 | 11.986000 | 1.000000 |

The candidate adds no successful-path syscall. The measured average delta is
0.024377 us at 720p and 0.020946 us at 1080p for typed results and live
counters. The values are dominated by syscall/clock overhead because
`/dev/null` does not copy a real video frame; they demonstrate call-path cost,
not physical v4l2loopback throughput.

## Backpressure and stop evidence

Hermetic tests fill a real kernel pipe, then compare its blocking and
nonblocking modes:

- The blocking baseline remains inside `write()` after an application stop is
  set and completes only after another thread drains the pipe.
- The candidate full-pipe call returns `would_block_dropped` in one syscall and
  below the declared 25 ms host acceptance bound (0 us at microsecond clock
  resolution in the recorded run).
- A worker repeatedly presented with the full nonblocking pipe joins below 25
  ms after stop, with no queue and no drain required (47 us in the recorded
  run).
- Scripted transport coverage proves at most five calls under perpetual
  `EINTR`, a stop check between attempts, no remainder call after partial
  writes, and typed handling for zero, `EAGAIN`, disconnect, and format
  mismatch.

## Static before/after accounting

| Behavior | Baseline | Candidate |
|---|---|---|
| Successful frame | one or more blocking writes | exactly one nonblocking write |
| Pre-write wait/probe | none | none |
| `EINTR` | unbounded retry | initial call plus at most four retries |
| `EAGAIN` | fatal generic error | current frame dropped; zero queue |
| Partial write | writes remainder as another call | fatal; no remainder; close/reopen |
| Periodic output refresh | every 30 frames; at least `G_FMT` + `G_PARM` ioctls | none |
| Renegotiation refresh | any generic write failure | only typed `EINVAL`/`EMSGSIZE` mismatch |

The removed periodic refresh eliminates at least two ioctls per 30 frames
(0.0667 ioctl/frame) in steady output, with additional format queries possible
in fallback cases. Setup and explicit output-open negotiation remain unchanged.

## External prerequisite

Real v4l2loopback avg/p95/p99, driver-specific partial-write behavior, and
consumer-driven renegotiation require an explicitly authorized inactive output
device. `/dev/video*` nodes were present, but automated validation did not open,
write, or renegotiate them because ownership/activity was not established. No
package, service, module, or device state was changed.
