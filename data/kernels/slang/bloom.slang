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

// Slang port of ../bloom.cl, on single channel images of the thresholded L.
// The blur is the box mean of box_filters.slang, which cuts its window at
// the image border as process() does, where OpenCL repeats the border
// pixels.

import common;

[shader("compute")]
[numthreads(16, 16, 1)]
void bloom_threshold(uint3 gid : SV_DispatchThreadID,
                     uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
                     uniform float scale, uniform float threshold)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float L = in.Areadpixel(x, y).x * scale;
  out.write(int2(x, y), float4(L > threshold ? L : 0.0f));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void bloom_mix(uint3 gid : SV_DispatchThreadID,
               uniform ImageIn in_a, uniform ImageIn in_b, uniform ImageOut out,
               uniform int width, uniform int height)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  float4 pixel = in_a.Areadpixel(x, y);
  const float processed = in_b.Areadsingle(x, y);
  pixel.x = 100.0f - (((100.0f - pixel.x) * (100.0f - processed)) / 100.0f); // screen blend
  out.write(int2(x, y), pixel);
}
