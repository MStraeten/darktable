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

// Slang port of process() in src/iop/hotpixels.c, which has no OpenCL code.
// Each work item writes its own sensel, either copied or replaced by the
// largest of its four like-colored neighbors.

import common;

// mode 0 is Bayer, with the neighbors 2 sensels away; 1 is a monochrome
// sensor, with the direct neighbors; 2 is X-Trans, with the neighbors from
// offsets[(row % 6) * 6 + col % 6], as (x, y) pairs
[shader("compute")]
[numthreads(16, 16, 1)]
void hotpixels(uint3 gid : SV_DispatchThreadID,
               uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
               uniform float threshold, uniform float multiplier, uniform int min_neighbours,
               uniform int mode, uniform StructuredBuffer<int4> offsets,
               uniform RWStructuredBuffer<uint> fixed)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float pixel = in.Areadsingle(x, y);
  float o = pixel;
  const int margin = mode == 1 ? 1 : 2;
  if(pixel > threshold && x >= margin && y >= margin && x < width - margin && y < height - margin)
  {
    int2 n[4];
    if(mode == 2)
    {
      const int k = (y % 6) * 6 + x % 6;
      const int4 a = offsets[2 * k];
      const int4 b = offsets[2 * k + 1];
      n[0] = a.xy;
      n[1] = a.zw;
      n[2] = b.xy;
      n[3] = b.zw;
    }
    else
    {
      // the order of the CPU code, where it matters only for the count
      const int s = mode == 1 ? 1 : 2;
      n[0] = int2(-s, 0);
      n[1] = int2(0, -s);
      n[2] = int2(s, 0);
      n[3] = int2(0, s);
    }

    const float mid = pixel * multiplier;
    int count = 0;
    float maxin = 0.0f;
    for(int i = 0; i < 4; i++)
    {
      const float other = in.Areadsingle(x + n[i].x, y + n[i].y);
      if(mid > other)
      {
        count++;
        maxin = max(maxin, other);
      }
    }
    if(count >= min_neighbours)
    {
      o = maxin;
      InterlockedAdd(fixed[0], 1u);
    }
  }
  out.write(int2(x, y), float4(o));
}
