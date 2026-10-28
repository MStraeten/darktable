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

// Slang port of ../soften.cl. The blur is the box mean of box_filters.slang,
// as in process(), where OpenCL approximates it with a gaussian.

import common;
import colorspace;

[shader("compute")]
[numthreads(16, 16, 1)]
void soften_overexposed(uint3 gid : SV_DispatchThreadID,
                        uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
                        uniform float saturation, uniform float brightness)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  float4 hsl = RGB_2_HSL(in.Areadpixel(x, y));
  hsl.y = clipf(hsl.y * saturation);
  hsl.z = clipf(hsl.z * brightness);
  out.write(int2(x, y), HSL_2_RGB(hsl));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void soften_mix(uint3 gid : SV_DispatchThreadID,
                uniform ImageIn in_a, uniform ImageIn in_b, uniform ImageOut out,
                uniform int width, uniform int height, uniform float amount)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float4 original = in_a.Areadpixel(x, y);
  const float4 processed = in_b.Areadpixel(x, y);
  float4 pixel = original * (1.0f - amount) + processed * amount;
  pixel.w = original.w;
  out.write(int2(x, y), pixel);
}
