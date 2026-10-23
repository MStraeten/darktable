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
- macOS: Xcode, for the `metal` compiler (Command Line Tools alone fall back
  to compiling Metal source at run time). Recent Xcode versions ship it as a
  separate download: `xcodebuild -downloadComponent MetalToolchain`
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
first export includes warm-up; report the later ones as steady state.

A third argument sets the export's longest side in pixels (`0`, the default,
is full resolution). `RESOURCES` sets darktable's `resourcelevel`: the default,
`default`, is what users get, and the summary lists any module that had to be
tiled because of it. Extra `darktable-cli` arguments go in `BENCH_ARGS`, for
example to pick an OpenCL platform. With several Vulkan GPUs the first integrated or discrete
one is used.

## report

Post `summary.md` on darktable-org/darktable#22732, with your
driver version, and say whether a laptop was on battery. If a run fails, attach `fullpipe-results/slang.log`.
