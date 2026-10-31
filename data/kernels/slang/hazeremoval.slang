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

// Slang port of ../hazeremoval.cl. The OpenCL box filters walk whole rows or
// columns, one work item each, which leaves most of a GPU idle; here each
// work item takes one pixel, as the windows are small (w1 is at most 6).

import common;

// the minimum (sign 1) or maximum (sign -1) over the 2w+1 window around x, y
// along x or y, clipped to the image, as the running filter of OpenCL finds it
float _box(ImageIn in, int x, int y, int width, int height, int w, bool along_x, float sign)
{
  const int i = along_x ? x : y;
  const int n = along_x ? width : height;
  float m = asfloat(0x7f800000u);
  for(int j = max(i - w, 0), j_end = min(i + w + 1, n); j < j_end; j++)
    m = min(sign * in.readsingle(along_x ? j : x, along_x ? y : j), m);
  return sign * m;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void hazeremoval_box_min_x(uint3 gid : SV_DispatchThreadID,
                           uniform int width, uniform int height, uniform ImageIn in, uniform ImageOut out,
                           uniform int w)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;
  out.write(int2(x, y), float4(_box(in, x, y, width, height, w, true, 1.0f)));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void hazeremoval_box_min_y(uint3 gid : SV_DispatchThreadID,
                           uniform int width, uniform int height, uniform ImageIn in, uniform ImageOut out,
                           uniform int w)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;
  out.write(int2(x, y), float4(_box(in, x, y, width, height, w, false, 1.0f)));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void hazeremoval_box_max_x(uint3 gid : SV_DispatchThreadID,
                           uniform int width, uniform int height, uniform ImageIn in, uniform ImageOut out,
                           uniform int w)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;
  out.write(int2(x, y), float4(_box(in, x, y, width, height, w, true, -1.0f)));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void hazeremoval_box_max_y(uint3 gid : SV_DispatchThreadID,
                           uniform int width, uniform int height, uniform ImageIn in, uniform ImageOut out,
                           uniform int w)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;
  out.write(int2(x, y), float4(_box(in, x, y, width, height, w, false, -1.0f)));
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
