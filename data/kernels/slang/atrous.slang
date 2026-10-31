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

// Slang port of ../atrous.cl

import common;

float4 _weight(const float4 c1, const float4 c2, const float sharpen)
{
  const float wc = dtcl_exp(-((c1.y - c2.y) * (c1.y - c2.y) + (c1.z - c2.z) * (c1.z - c2.z)) * sharpen);
  const float wl = dtcl_exp(-(c1.x - c2.x) * (c1.x - c2.x) * sharpen);
  return float4(wl, wc, wc, 1.0f);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void eaw_decompose(uint3 gid : SV_DispatchThreadID,
                   uniform ImageIn in, uniform ImageOut coarse, uniform ImageOut detail,
                   uniform int width, uniform int height, uniform int scale, uniform float sharpen,
                   uniform StructuredBuffer<float> filter)
{
  const int x = gid.x;
  const int y = gid.y;

  if(x >= width || y >= height) return;

  const int mult = 1 << scale;

  const float4 pixel = in.read(int2(x, y));
  float4 sum = float4(0.0f);
  float4 wgt = float4(0.0f);
  for(int j = 0; j < 5; j++)
    for(int i = 0; i < 5; i++)
    {
      const float4 px = in.read(int2(mult * (i - 2) + x, mult * (j - 2) + y));
      const float4 w = filter[j * 5 + i] * _weight(pixel, px, sharpen);

      sum += w * px;
      wgt += w;
    }
  sum /= wgt;
  sum.w = pixel.w;

  detail.write(int2(x, y), pixel - sum);
  coarse.write(int2(x, y), sum);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void eaw_synthesize(uint3 gid : SV_DispatchThreadID,
                    uniform ImageOut out, uniform ImageIn coarse, uniform ImageIn detail,
                    uniform int width, uniform int height, uniform float4 threshold, uniform float4 boost)
{
  const int x = gid.x;
  const int y = gid.y;

  if(x >= width || y >= height) return;

  const float4 c = coarse.read(int2(x, y));
  const float4 d = detail.read(int2(x, y));
  const float4 amount = copysign(max(float4(0.0f), abs(d) - threshold), d);
  float4 sum = c + boost * amount;
  sum.w = c.w;
  out.write(int2(x, y), sum);
}
