-- drives the darkroom for darkroom.sh: opens one image, zooms to 100%,
-- pans BENCH_PANS times and zooms back to fit. Every step is logged as
-- "bench step <name>" so that darkroom.sh can pair it with the pipe runs
-- that follow it in the -d pipe log. Lua has no event for the end of the
-- full pipe, so each step waits BENCH_WAIT_MS; darkroom.sh flags a step
-- whose pipe was still running when the next step began.
local dt = require "darktable"

local file = os.getenv("BENCH_IMG")
local pans = tonumber(os.getenv("BENCH_PANS") or "8")
local wait = tonumber(os.getenv("BENCH_WAIT_MS") or "4000")

local function step(name, action, effect)
  dt.print_log("bench step " .. name)
  if action then dt.gui.action(action, "", effect or "", 1.0) end
  dt.control.sleep(wait)
end

-- the edit comes with the image, as its sidecar
local img = dt.database.import(file)
if not img then
  dt.print_log("bench error: the image did not import")
  return
end

dt.gui.selection({ img })
dt.print_log("bench step open")
dt.gui.current_view(dt.gui.views.darkroom)
dt.control.sleep(2 * wait)

-- after the first 100% view, every pan shows a new part of the image, as
-- the darkroom keeps what it has shown and going back costs nothing
step("zoom100", "views/darkroom/zoom close-up")
local dir = "up"
for k = 1, pans do
  if k % 4 == 0 then
    step("pan", "views/darkroom/move/vertical", "down")
    dir = dir == "up" and "down" or "up"
  else
    step("pan", "views/darkroom/move/horizontal", dir)
  end
end
step("fit", "views/darkroom/zoom close-up")

-- darkroom.sh ends darktable: quitting from this thread would wait for it
dt.print_log("bench done")
