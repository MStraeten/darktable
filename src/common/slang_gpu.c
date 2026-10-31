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

// device error accounting shared by the Vulkan and Metal backends of slang_gpu.h

#include "common/slang_gpu.h"
#include "common/darktable.h"
#include "control/control.h"

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
