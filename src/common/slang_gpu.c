/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    darktable is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with darktable.  If not, see <http://www.gnu.org/licenses/>.
*/

// device error accounting and settings shared by the Vulkan and Metal
// backends of slang_gpu.h

#include "common/slang_gpu.h"
#include "common/darktable.h"
#include "common/opencl.h"
#include "control/conf.h"
#include "control/control.h"
#include "common/utility.h"

static gint _errors = 0;
static gint _failed_runs = 0;
static gint _stopped = FALSE;

static void _stop(void)
{
  if(!g_atomic_int_compare_and_exchange(&_stopped, FALSE, TRUE)) return;
  dt_print(DT_DEBUG_ALWAYS, "[slang_gpu] disabling the Slang kernels for this session");
  dt_control_log(_("GPU errors in the Slang kernels; running on the CPU for the rest of this session."
                   " run darktable with `-d opencl' to see them"));
}

void dt_slang_gpu_report_error(const char *what, const gboolean fatal)
{
  const int count = g_atomic_int_add(&_errors, 1) + 1;
  dt_print(DT_DEBUG_SLANG, "[slang_gpu] device error %d: %s%s", count, what,
           fatal ? ", the device can't be used any more" : "");
  if(fatal) _stop();
}

int dt_slang_gpu_errors(void)
{
  return g_atomic_int_get(&_errors);
}

void dt_slang_gpu_run_failed(void)
{
  if(g_atomic_int_add(&_failed_runs, 1) + 1 >= DT_SLANG_GPU_MAX_ERRORS) _stop();
}

gboolean dt_slang_gpu_stopped(void)
{
  return g_atomic_int_get(&_stopped);
}

float dt_slang_gpu_unified_fraction(const float automatic)
{
  const float fraction = dt_conf_get_float("slang_gpu_unified_fraction");
  return fraction > 0.0f ? CLAMP(fraction, 0.02f, 0.9f) : automatic;
}

size_t dt_slang_gpu_memory_limit(const size_t device_memory, const gboolean unified)
{
  // the same numbers as OpenCL (opencl.c, dt_opencl_update_settings()), so
  // that both paths keep the same margin and tile the same modules. Unlike
  // OpenCL, tuned headroom doesn't need a second device: Slang has only one
  const dt_sys_resources_t *res = &darktable.dtresources;
  const int level = res->level;
  const size_t lowmem = 256ul * DT_MEGA;
  const size_t headroom = DT_OPENCL_DEFAULT_HEADROOM * DT_MEGA;
  size_t limit;
  if(level < 0)
    limit = (size_t)res->refresource[4 * (-level - 1) + 3] * DT_MEGA;
  else if(!dt_gimpmode() && dt_conf_get_bool("opencl_tune_headroom"))
    limit = device_memory > headroom ? device_memory - headroom : lowmem;
  else
  {
    const size_t disposable = device_memory > headroom ? device_memory - headroom : 0;
    const int fraction = MIN(1024, res->fractions[4 * level + 3]);
    limit = MAX(lowmem, disposable / 1024ul * fraction);
  }
  if(unified)
    limit = MIN(limit, (size_t)(res->total_memory * dt_slang_gpu_unified_fraction(0.25f)));
  return limit;
}

gboolean dt_slang_gpu_avoid_module(const char *op)
{
  const char *avoid = dt_conf_get_string_const("slang_gpu_avoid");
  return avoid && *avoid && dt_str_commasubstring(avoid, op);
}

static void _device_free(gpointer data)
{
  dt_slang_gpu_device_t *device = data;
  g_free(device->name);
  g_free(device);
}

void dt_slang_gpu_devices_free(GList *devices)
{
  g_list_free_full(devices, _device_free);
}
