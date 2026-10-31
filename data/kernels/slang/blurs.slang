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


// Slang port of ../blurs.cl

import common;

// the sparse convolution of lens and motion blurs, with the neighbors
// clamped to the image. Offsets are pixel deltas, values the kernel weights.
[shader("compute")]
[numthreads(16, 16, 1)]
void blurs_convolve_sparse(uint3 gid : SV_DispatchThreadID,
                           uniform ImageIn in, uniform StructuredBuffer<int> offsets_x,
                           uniform StructuredBuffer<int> offsets_y, uniform StructuredBuffer<float> values,
                           uniform ImageOut out, uniform int width, uniform int height, uniform int n_entries)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  float4 acc = float4(0.0f);
  for(int i = 0; i < n_entries; i++)
    acc += values[i] * in.Areadpixel(clamp(x + offsets_x[i], 0, width - 1),
                                     clamp(y + offsets_y[i], 0, height - 1));

  acc.w = in.Areadpixel(x, y).w;
  out.write(int2(x, y), acc);
}

// the RGB of blurred with the alpha of original, the pipeline mask
[shader("compute")]
[numthreads(16, 16, 1)]
void blurs_restore_alpha(uint3 gid : SV_DispatchThreadID,
                         uniform ImageIn original, uniform ImageIn blurred, uniform ImageOut out,
                         uniform int width, uniform int height)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  float4 px = blurred.Areadpixel(x, y);
  px.w = original.Areadpixel(x, y).w;
  out.write(int2(x, y), px);
}
