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

// Slang port of dt_box_mean() in src/common/box_filters.cc: one pass of the
// box mean along x when dir is (1, 0) and along y when it is (0, 1). As on
// the CPU, the window is cut at the image border and averages only the
// pixels inside it. The CPU code keeps a running sum; each work item here
// sums its own window.

import common;

[shader("compute")]
[numthreads(16, 16, 1)]
void box_mean_4c(uint3 gid : SV_DispatchThreadID,
                 uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
                 uniform int radius, uniform int2 dir)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const int pos = dot(int2(x, y), dir);
  const int size = dot(int2(width, height), dir);
  const int lo = max(0, pos - radius);
  const int hi = min(size - 1, pos + radius);

  float4 sum = float4(0.0f);
  for(int i = lo; i <= hi; i++) sum += in.readA(int2(x, y) + (i - pos) * dir);
  out.write(int2(x, y), sum / float(hi - lo + 1));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void box_mean_1c(uint3 gid : SV_DispatchThreadID,
                 uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
                 uniform int radius, uniform int2 dir)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const int pos = dot(int2(x, y), dir);
  const int size = dot(int2(width, height), dir);
  const int lo = max(0, pos - radius);
  const int hi = min(size - 1, pos + radius);

  float sum = 0.0f;
  for(int i = lo; i <= hi; i++) sum += in.readA(int2(x, y) + (i - pos) * dir).x;
  out.write(int2(x, y), float4(sum / float(hi - lo + 1)));
}
