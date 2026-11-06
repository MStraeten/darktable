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

// Slang port of ../highpass.cl, for Lab: only L is inverted, blurred and
// mixed, and a and b are desaturated. The blur is the box mean of
// box_filters.slang, as in process(), where OpenCL approximates it with a
// gaussian.

import common;

[shader("compute")]
[numthreads(16, 16, 1)]
void highpass_invert(uint3 gid : SV_DispatchThreadID,
                     uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  float4 pixel = in.Areadpixel(x, y);
  pixel.x = clamp(100.0f - pixel.x, 0.0f, 100.0f);
  out.write(int2(x, y), pixel);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void highpass_mix(uint3 gid : SV_DispatchThreadID,
                  uniform ImageIn in_a, uniform ImageIn in_b, uniform ImageOut out,
                  uniform int width, uniform int height, uniform float contrast_scale)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  // as in process(): contrast_scale includes the 1/2 of the average
  const float L = (in_b.Areadsingle(x, y) + in_a.Areadsingle(x, y)) - 100.0f;
  out.write(int2(x, y), float4(clamp(L * contrast_scale + 50.0f, 0.0f, 100.0f), 0.0f, 0.0f, 0.0f));
}
