#!/bin/bash
# full-pipe benchmark of a typical scene-referred edit (denoise profiled, lens,
# exposure, tone equalizer, color calibration, sharpen, color equalizer, AgX)
# on mire1.cr2: CPU vs OpenCL vs Slang, N exports in one darktable-cli process each
#
# usage: bench.sh <darktable checkout with build/> [N=5] [SIZE=0]
# SIZE is the export's longest side in pixels; 0 exports at full resolution
# writes results to ./fullpipe-results/ (logs, pngs, summary.md)
# RESOURCES sets darktable's resourcelevel (default: "default", what users get;
# "reference" uses fixed, machine-independent memory limits and tiles more)
# extra darktable-cli arguments for every run can be passed in BENCH_ARGS
set -u
SRC=$(cd "$1" && pwd); N=${2:-5}; SIZE=${3:-0}
RESOURCES=${RESOURCES:-default}
HERE=$(cd "$(dirname "$0")" && pwd)
CLI=$SRC/build/bin/darktable-cli
[ -x "$CLI" ] || CLI=$(ls -d $SRC/build/bin/darktable-cli* 2>/dev/null | head -1)
IMG=$SRC/src/tests/integration/images/mire1.cr2
DE=$SRC/src/tests/integration/deltae
OUT=$PWD/fullpipe-results
rm -rf "$OUT"; mkdir -p "$OUT/in"
for i in $(seq 1 $N); do ln -s "$IMG" "$OUT/in/img$i.cr2"; done

run() { # <name> [extra args]
  local name=$1; shift
  mkdir -p "$OUT/$name"
  "$CLI" "$OUT/in" "$HERE/scene-referred.xmp" "$OUT/$name/\$(FILE_NAME).png" \
    --width $SIZE --height $SIZE --hq true --apply-custom-presets false \
    --core --configdir "$OUT/config-$name" --conf host_memory_limit=8192 --conf resourcelevel=$RESOURCES \
    --conf plugins/lighttable/export/force_lcms2=FALSE \
    --conf plugins/lighttable/export/pixel_interpolator=lanczos3 \
    --conf plugins/lighttable/export/pixel_interpolator_warp=bicubic \
    --conf plugins/lighttable/export/iccintent=0 -d perf -d pipe -d opencl ${BENCH_ARGS:-} "$@" > "$OUT/$name.log" 2>&1
  echo "$name exit $?"
}
run cpu    --disable-opencl --conf slang_gpu=FALSE
run opencl --conf opencl=TRUE --conf slang_gpu=FALSE
run slang  --disable-opencl --conf slang_gpu=TRUE

{
  cpu=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2-)
  echo "# full-pipe benchmark, $(date '+%Y-%m-%d %H:%M')"
  echo
  echo "- os: $(uname -srm)"
  echo "- cpu: ${cpu# }"
  echo "- commit: $(git -C "$SRC" rev-parse --short HEAD 2>/dev/null)"
  echo "- slang device: $(grep -m1 -oE 'using (Vulkan|Metal) device .*' "$OUT/slang.log" | cut -d' ' -f2-)"
  echo "- opencl devices: $(grep -E "^ +DEVICE: +[0-9]+: '" "$OUT/opencl.log" | sed -E "s/.*DEVICE: +[0-9]+: ('[^']*').*/\1/" | tr '\n' ' ')"
  echo "- export size: $([ "$SIZE" = 0 ] && echo full resolution || echo "${SIZE} px"), resourcelevel: $RESOURCES"
  for p in cpu opencl slang; do
    tl=$(grep -E '\*tiled\*|process (fast )?tiled' "$OUT/$p.log" | sed -E 's/.*\[export\] +([a-z0-9_]+).*/\1/' | sort | uniq -c | awk '{printf "%s(%s) ", $2, $1}')
    echo "- tiled on $p: ${tl:-none}"
  done
  echo; echo "pipe totals in ms, one process per path, $N exports each"; echo
  echo "| path | per export |"; echo "|---|---|"
  for p in cpu opencl slang; do
    t=$(grep 'pixel pipeline processing took' "$OUT/$p.log" | sed -E 's/.*took ([0-9.]+) secs.*/\1/' | awk '{printf "%s%.0f", (NR>1?" / ":""), $1*1000}')
    echo "| $p | $t |"
  done
  echo; echo "module runs over all $N exports:"; echo
  for p in opencl slang; do
    echo "- $p: $(grep -c "processed .* on GPU" "$OUT/$p.log") on GPU; on CPU: $(grep "processed .* on CPU" "$OUT/$p.log" | sed -E "s/.*processed .([a-z0-9_]+). on CPU.*/\1/" | sort | uniq -c | awk '{printf "%s(%s) ", $2, $1}')"
  done
  echo "- slang fallbacks/failures: $(grep -ciE 'slang (fallback|failure)' "$OUT/slang.log")"
  echo; echo "slang per-module time, last export (ms; GPU work is async, so trust the pipe total):"; echo
  awk '/pixel pipeline processing took/{last=buf; buf=""; next} /processed .* on /{buf=buf $0 "\n"} END{printf "%s", last}' "$OUT/slang.log" | sed -E 's/.*took ([0-9.]+) secs.*processed .([a-z0-9_]+). on ([A-Z]+).*/- \2 (\3): \1/' | awk '{printf "- %s %s %.0f\n", $2, $3, $4*1000}'
  echo; echo "max / avg dE on the first export:"; echo
  for pair in "cpu opencl" "cpu slang" "opencl slang"; do
    set -- $pair
    r=$("$DE" "$OUT/$1/img1.png" "$OUT/$2/img1.png" 2>&1 | grep -E 'Max dE|Avg dE' | awk '{print $NF}' | tr '\n' ' ')
    echo "- $1 vs $2: $r"
  done
} > "$OUT/summary.md"
cat "$OUT/summary.md"
