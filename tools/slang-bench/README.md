# Slang full-pipe benchmark

Times a typical scene-referred edit on `mire1.cr2` (from the integration test
images) three ways: CPU, OpenCL, and the Slang GPU path (Vulkan on Linux and
Windows, Metal on macOS). The edit is denoise (profiled), lens correction,
exposure, tone equalizer, color calibration, sharpen, color equalizer and AgX,
exported at 2048 px.

This is a proof of concept from darktable-org/darktable#22732, not a
darktable feature.

## build

You need `slangc` (tested with 2026.20, from
https://github.com/shader-slang/slang/releases) on the `PATH`, and:

- Linux: the Vulkan loader and headers (`libvulkan-dev` on Debian/Ubuntu) and
  a Vulkan driver for your GPU
- macOS: the Command Line Tools are enough; darktable then compiles the
  Metal source of each kernel the first time it runs it. With Xcode and its
  `metal` compiler the kernels are compiled at build time instead. Recent
  Xcode versions ship it as a separate download:
  `xcodebuild -downloadComponent MetalToolchain`
- Windows: untested

```bash
git clone --recurse-submodules -b slang-benchmark https://github.com/nicknezis/darktable
cd darktable
cmake -B build -DCMAKE_BUILD_TYPE=Release -DUSE_SLANG=ON
cmake --build build -j
```

`--recurse-submodules` is needed for the test image, which lives in
`src/tests/integration`.

## run

```bash
mkdir /tmp/slang-bench && cd /tmp/slang-bench
bash <darktable>/tools/slang-bench/bench.sh <darktable> 5
```

It does 5 exports per path in one `darktable-cli` process each and writes
`fullpipe-results/summary.md`: devices, pipe time per export, which modules
ran on the GPU, per-module times, and the dE between the three outputs. The
first export includes warm-up, such as compiling GPU kernels the first time
after an install, so the summary reports the median of the later ones as
the steady state and the first one as its overhead over that.

A third argument sets the export's longest side in pixels (`0`, the default,
is full resolution). `RESOURCES` sets darktable's `resourcelevel`: the default,
`default`, is what users get, and the summary lists any module that had to be
tiled because of it. Extra `darktable-cli` arguments go in `BENCH_ARGS`, for
example to pick an OpenCL platform.

### heavy edit, other raws

`XMP=heavy` runs `heavy.xmp` instead: the same edit plus diffuse, with
drawn masks on three modules (circle, ellipse, path, brush and gradient,
nested mask groups, an inverted group), feathering, mask blur, parametric
ranges and details refinement. `XMP` also takes the path of your own xmp.
`IMG` exports another raw, for example a 30 or 60 MP one:

```bash
IMG=/path/to/large.raw XMP=heavy bash <darktable>/tools/slang-bench/bench.sh <darktable> 3
```

`heavy.xmp` leaves the raw's black levels and crop to the camera and sets
white balance to camera reference, so it suits any raw. The default
`scene-referred.xmp` carries mire1's black levels and white balance, so
other raws come out with a strong color cast under it. For each path the
summary adds the image size, the memory darktable allowed the GPU modules,
and the time spent rendering the drawn masks on the host.

### devices

With several Vulkan devices, Slang uses a discrete GPU first, then an
integrated one. To choose another, pass its index or part of its name, for
example `BENCH_ARGS="--conf slang_gpu_device=nvidia"`; `slang.log` lists
every device it found.

## report

Post `summary.md` on darktable-org/darktable#22732, with your
driver version, and say whether a laptop was on battery. If a run fails, attach `fullpipe-results/slang.log`.
