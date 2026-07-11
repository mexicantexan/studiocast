# Vulkan Matting ncnn Spike

Milestone 4 evaluates ncnn Vulkan as a candidate inference runtime for a future
Open Vulkan virtual-background backend. This is a spike only; it does not change
the production model-pack format or runtime selection.

## Build

The spike tool is opt-in and must not be required by default CI:

```bash
cmake -S . -B build-vulkan-ncnn -G Ninja \
  -DSTUDIOCAST_ENABLE_NCNN_SPIKE=ON \
  -DSTUDIOCAST_ENABLE_OPEN_VULKAN=ON
cmake --build build-vulkan-ncnn --target studiocast-vulkan-matting-spike
```

If ncnn is installed, point CMake at its package config:

```bash
cmake -S . -B build-vulkan-ncnn -G Ninja \
  -DSTUDIOCAST_ENABLE_NCNN_SPIKE=ON \
  -DSTUDIOCAST_ENABLE_OPEN_VULKAN=ON \
  -Dncnn_DIR=/path/to/ncnn/lib/cmake/ncnn
```

Use `-DSTUDIOCAST_REQUIRE_NCNN=ON` only on machines where ncnn with Vulkan is
expected. The default is diagnostics-only when ncnn is missing.

## Model Conversion

Conversion is developer tooling and must not run in the frame loop. The current
default matting pack is `modnet-webnn-256-fp32`, with manifest-fixed
`1x3x256x256` input and `1x1x256x256` alpha output.

Preferred pnnx path:

```bash
cd ~/.local/share/studiocast/models/open_video/matting/Good\ Quality
pnnx model.onnx inputshape=[1,3,256,256] \
  ncnnparam=model.ncnn.param ncnnbin=model.ncnn.bin fp16=0 optlevel=2
```

Legacy fallback:

```bash
python3 -m onnxsim model.onnx model.sim.onnx
onnx2ncnn model.sim.onnx model.ncnn.param model.ncnn.bin
ncnnoptimize model.ncnn.param model.ncnn.bin \
  model.ncnn.opt.param model.ncnn.opt.bin 65536
```

MODNet conversion needs explicit review because the installed ONNX graph is known
to contain dynamic-shape helpers and `InstanceNormalization`. Those can cause
conversion failures or ncnn Vulkan CPU fallback. BiRefNet conversion must happen
after the installer's alpha/Sigmoid output patch.

## Run

```bash
./build-vulkan-ncnn/studiocast-vulkan-matting-spike \
  --model-id modnet-webnn-256-fp32 \
  --fixture synthetic \
  --warmup 20 \
  --iterations 120
```

For a direct pack root or test fixture root:

```bash
./build-vulkan-ncnn/studiocast-vulkan-matting-spike \
  --models-root tests/data/models/open_cuda \
  --model-id mock_model \
  --param /path/to/model.ncnn.param \
  --bin /path/to/model.ncnn.bin \
  --require-reference
```

The tool reports:

- ncnn runtime and ncnn Vulkan compile availability
- StudioCast Open Vulkan runtime availability
- converted model presence
- model/operator compatibility notes
- CPU/ORT reference latency when available
- ncnn candidate startup and steady-state latency when runnable
- alpha MAE, max absolute error, and thresholded mask IoU
- transfer/residency status

## Current Feasibility Boundary

ncnn can run Vulkan inference with its own `VkMat` tensors, but StudioCast's
current Vulkan buffers are owned by a separate minimal `VulkanDevice` wrapper.
Direct `VulkanTensor` or `VulkanImage` zero-copy interop with ncnn is therefore
not proven in this milestone.

The contained spike uses CPU `ncnn::Mat` input/output, so it measures ncnn
Vulkan runtime behavior and output quality, but it explicitly reports device
residency as `no`. A production Open Vulkan backend would need either ncnn-owned
preprocess/composite kernels or a shared Vulkan device/allocator design.
