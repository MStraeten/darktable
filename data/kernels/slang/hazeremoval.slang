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

// Slang port of ../hazeremoval.cl. The box filters walk whole rows or
// columns, one work item each, as dt_slang_gpu_run_1d_args() lays them out.

import common;

// the running minimum (sign 1) or maximum (sign -1) over a 2w+1 window along
// a line of n values, which in and out read and write at position i
float _box_init(ImageIn in, int line, int n, int w, bool along_x, float sign)
{
  float m = asfloat(0x7f800000u);
  for(int i = 0, i_end = min(w + 1, n); i < i_end; ++i)
    m = min(sign * in.readsingle(along_x ? i : line, along_x ? line : i), m);
  return m;
}

void _box(ImageIn in, ImageOut out, int line, int n, int w, bool along_x, float sign)
{
  float m = _box_init(in, line, n, w, along_x, sign);
  for(int i = 0; i < n; i++)
  {
    out.write(along_x ? int2(i, line) : int2(line, i), float4(sign * m));
    if(i - w >= 0 && sign * in.readsingle(along_x ? i - w : line, along_x ? line : i - w) == m)
    {
      m = asfloat(0x7f800000u);
      for(int j = max(i - w + 1, 0), j_end = min(i + w + 2, n); j < j_end; ++j)
        m = min(sign * in.readsingle(along_x ? j : line, along_x ? line : j), m);
    }
    if(i + w + 1 < n) m = min(sign * in.readsingle(along_x ? i + w + 1 : line, along_x ? line : i + w + 1), m);
  }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void hazeremoval_box_min_x(uint3 gid : SV_DispatchThreadID,
                           uniform int width, uniform int height, uniform ImageIn in, uniform ImageOut out,
                           uniform int w)
{
  const int y = gid.y * 16 + gid.x;
  if(y >= height) return;
  _box(in, out, y, width, w, true, 1.0f);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void hazeremoval_box_min_y(uint3 gid : SV_DispatchThreadID,
                           uniform int width, uniform int height, uniform ImageIn in, uniform ImageOut out,
                           uniform int w)
{
  const int x = gid.y * 16 + gid.x;
  if(x >= width) return;
  _box(in, out, x, height, w, false, 1.0f);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void hazeremoval_box_max_x(uint3 gid : SV_DispatchThreadID,
                           uniform int width, uniform int height, uniform ImageIn in, uniform ImageOut out,
                           uniform int w)
{
  const int y = gid.y * 16 + gid.x;
  if(y >= height) return;
  _box(in, out, y, width, w, true, -1.0f);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void hazeremoval_box_max_y(uint3 gid : SV_DispatchThreadID,
                           uniform int width, uniform int height, uniform ImageIn in, uniform ImageOut out,
                           uniform int w)
{
  const int x = gid.y * 16 + gid.x;
  if(x >= width) return;
  _box(in, out, x, height, w, false, -1.0f);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void hazeremoval_transision_map(uint3 gid : SV_DispatchThreadID,
                                uniform int width, uniform int height, uniform ImageIn in, uniform ImageOut out,
                                uniform float strength, uniform float A0_r, uniform float A0_g,
                                uniform float A0_b)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float4 pixel = in.read(int2(x, y));
  float m = pixel.x / A0_r;
  m = min(pixel.y / A0_g, m);
  m = min(pixel.z / A0_b, m);
  out.write(int2(x, y), float4(1.f - m * strength));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void hazeremoval_dehaze(uint3 gid : SV_DispatchThreadID,
                        uniform int width, uniform int height, uniform ImageIn in, uniform ImageIn trans_map,
                        uniform ImageOut out, uniform float t_min, uniform float4 A0)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float4 pixel = in.read(int2(x, y));
  const float t = max(trans_map.readsingle(x, y), t_min);

  out.write(int2(x, y), float4((pixel.x - A0.x) / t + A0.x, (pixel.y - A0.y) / t + A0.y,
                               (pixel.z - A0.z) / t + A0.z, pixel.w));
}
