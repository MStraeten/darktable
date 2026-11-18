#!/bin/bash
# darkroom timings of the bench edit: opens the raw in the darkroom, then
# zooms to 100%, pans and zooms back to fit (darkroom.lua), CPU vs OpenCL
# vs Slang, one darktable process per path. Needs darktable built with Lua.
#
# usage: darkroom.sh <darktable checkout with build/> [PANS=8]
# writes results to ./darkroom-results/ (logs, summary.md)
# IMG and XMP as for bench.sh (XMP=heavy for heavy.xmp); WAIT_MS is how
# long each step waits for the pipes (default 4000); WINDOW is the
# window size in points (default 1600x1000); extra darktable arguments go
# in BENCH_ARGS; PATHS picks the paths (default "cpu opencl slang"). On
# Linux without a display it runs under xvfb-run.
set -u
export LC_ALL=C
SRC=$(cd "$1" && pwd); PANS=${2:-8}
HERE=$(cd "$(dirname "$0")" && pwd)
DT=$SRC/build/bin/darktable
[ -x "$DT" ] || { echo "no darktable binary in $SRC/build/bin" >&2; exit 1; }
DEFAULT_IMG=$SRC/src/tests/integration/images/mire1.cr2
IMG=${IMG:-$DEFAULT_IMG}
[ -f "$IMG" ] || { echo "no such image: $IMG" >&2; exit 1; }
XMP=${XMP:-scene-referred}
case $XMP in */*|*.xmp) ;; *) XMP=$HERE/$XMP.xmp ;; esac
[ -f "$XMP" ] || { echo "no such edit: $XMP" >&2; exit 1; }
WAIT_MS=${WAIT_MS:-4000}
WINDOW=${WINDOW:-1600x1000}
EXT=${IMG##*.}
OUT=$PWD/darkroom-results
rm -rf "$OUT"; mkdir -p "$OUT"
# the longest a run may take: every step waits WAIT_MS, plus startup
LIMIT=$(( (6 + PANS) * WAIT_MS / 1000 + 120 ))

XRUN=()
if [ "$(uname)" = Linux ] && [ -z "${DISPLAY:-}" ]; then
  XRUN=(xvfb-run -a -s "-screen 0 2560x1600x24")
fi

run() { # <name> [extra args]
  local name=$1; shift
  # a private copy, so that the edit is the image's sidecar and the
  # darktable database of the run starts empty
  local work=$OUT/work-$name
  mkdir -p "$work"
  cp "$IMG" "$work/img.$EXT"
  cp "$XMP" "$work/img.$EXT.xmp"
  BENCH_IMG=$work/img.$EXT BENCH_PANS=$PANS BENCH_WAIT_MS=$WAIT_MS \
    ${XRUN[@]+"${XRUN[@]}"} "$DT" --configdir "$OUT/config-$name" --library :memory: \
    --conf ui_last/window_w=${WINDOW%x*} --conf ui_last/window_h=${WINDOW#*x} \
    --conf ui_last/maximized=FALSE --conf ui_last/fullscreen=FALSE \
    --conf host_memory_limit=8192 --conf resourcelevel=default \
    --conf ui/show_welcome_screen=FALSE \
    --luacmd "dofile('$HERE/darkroom.lua')" \
    -d lua -d pipe -d perf -d opencl -d slang -d tiling ${BENCH_ARGS:-} "$@" > "$OUT/$name.log" 2>&1 &
  local pid=$!
  # darktable can't quit itself from the Lua thread that drives it, so the
  # run ends when the script says so
  local t=0
  while kill -0 $pid 2>/dev/null && [ $t -lt $LIMIT ] \
        && ! grep -qE 'bench (done|error)' "$OUT/$name.log"; do
    sleep 1; t=$((t + 1))
  done
  sleep 1
  pkill -TERM -P $pid 2>/dev/null; kill -TERM $pid 2>/dev/null
  sleep 2
  pkill -KILL -P $pid 2>/dev/null; kill -KILL $pid 2>/dev/null
  wait $pid 2>/dev/null
  rm -rf "$work"
  if grep -q 'bench done' "$OUT/$name.log"; then echo "$name done"
  else echo "$name did not finish, see $OUT/$name.log"; fi
}
PATHS=${PATHS:-cpu opencl slang}
for p in $PATHS; do
  case $p in
    cpu)    run cpu    --disable-opencl --conf slang_gpu=FALSE ;;
    opencl) run opencl --conf opencl=TRUE --conf slang_gpu=FALSE ;;
    slang)  run slang  --disable-opencl --conf slang_gpu=TRUE ;;
  esac
done

# per step: the time from the step to the end of the first full and
# preview pipe after it, if they finished before the next step
steps() { # <log>
  awk '
    /LUA bench step / { step = $NF; t0 = $1; full = prev = 0; next }
    /LUA bench (done|error)/ { step = ""; next }
    step != "" && /pipe finished/ && /\[full\]/ && !full {
      full = 1; ms = ($1 - t0) * 1000
      print step, "full", ms, $(NF - 4) }
    step != "" && /pipe finished/ && /\[preview\]/ && !prev {
      prev = 1; ms = ($1 - t0) * 1000
      print step, "preview", ms, $(NF - 4) }' "$1"
}
median() { sort -n | awk '{ v[NR] = $1 } END { if(NR) printf "%d", (NR % 2) ? v[(NR + 1) / 2] : (v[NR / 2] + v[NR / 2 + 1]) / 2; else printf "-" }'; }

{
  cpu=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2-)
  echo "# darkroom benchmark, $(date '+%Y-%m-%d %H:%M')"
  echo
  echo "- os: $(uname -srm)"
  echo "- cpu: ${cpu# }"
  echo "- commit: $(git -C "$SRC" rev-parse --short HEAD 2>/dev/null)"
  echo "- image: $([ "$IMG" = "$DEFAULT_IMG" ] && echo mire1.cr2 || echo "a .$EXT raw")"
  echo "- edit: $(basename "$XMP")"
  echo "- window: $WINDOW points; open, zoom 100%, $PANS pans, fit; ${WAIT_MS} ms per step"
  [ -f "$OUT/slang.log" ] && echo "- slang device: $(grep -m1 -oE 'using (Vulkan|Metal) device .*' "$OUT/slang.log" | cut -d' ' -f2-)"
  [ -f "$OUT/opencl.log" ] && echo "- opencl devices: $(grep -E "^ +DEVICE: +[0-9]+: '" "$OUT/opencl.log" | sed -E "s/.*DEVICE: +[0-9]+: ('[^']*').*/\1/" | tr '\n' ' ')"
  echo
  echo "ms from the step to the end of the full pipe (pans: median, and the"
  echo "number of pans whose pipe ended in time); the 100% view's size is the"
  echo "full pipe's output"
  echo
  echo "| path | open | zoom 100% | pan | fit | preview after open | 100% view |"
  echo "|---|---|---|---|---|---|---|"
  for p in $PATHS; do
    s=$(steps "$OUT/$p.log")
    pick() { # <step> <pipe>: the median
      echo "$s" | awk -v s="$1" -v p="$2" '$1 == s && $2 == p { print $3 }' | median
    }
    pans=$(echo "$s" | awk '$1 == "pan" && $2 == "full"' | wc -l | tr -d ' ')
    size=$(echo "$s" | awk '$1 == "zoom100" && $2 == "full" { print $4; exit }')
    echo "| $p | $(pick open full) | $(pick zoom100 full) | $(pick pan full) ($pans/$PANS) | $(pick fit full) | $(pick open preview) | ${size:--} |"
  done
  echo
  echo "a '-' or missing pans mean no full pipe ended before the next step:"
  echo "it was cached (fit usually is) or still running (raise WAIT_MS)"
  echo
  for p in $PATHS; do
    echo "- tiled on $p: $(grep -oE "(fast tiling|\*tiled\*)[^[]*\[T?(full|preview)\] +[a-z0-9_]+" "$OUT/$p.log" | awk '{print $NF}' | sort | uniq -c | awk '{printf "%s(%s) ", $2, $1}')"
  done
  [ -f "$OUT/slang.log" ] && echo "- slang fallbacks/failures: $(grep -ciE 'slang (fallback|failure)' "$OUT/slang.log")"
} > "$OUT/summary.md"
cat "$OUT/summary.md"
