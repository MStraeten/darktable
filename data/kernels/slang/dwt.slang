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

// the wavelet decomposition of src/common/dwt.c on images of the Slang
// kernels. It runs the vertical pass first, as dwt_decompose_layer() does,
// where ../dwt.cl runs the rows first.

import common;

// first pass: the weighted sum of the row with the rows vscale above and
// below, reflected at the edges
[shader("compute")]
[numthreads(16, 16, 1)]
void dwt_decompose_vert(uint3 gid : SV_DispatchThreadID,
                        uniform ImageIn in, uniform ImageOut out,
                        uniform int width, uniform int height, uniform int vscale)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const int above = (y > vscale) ? y - vscale : vscale - y;
  const int below = (y + vscale < height) ? y + vscale : 2 * (height - 1) - (y + vscale);

  out.write(int2(x, y), 2.f * in.Areadpixel(x, y) + in.Areadpixel(x, above) + in.Areadpixel(x, below));
}

// second pass: the same over the columns of the first pass, renormalized,
// which gives the coarse scale, and detail = input - coarse
[shader("compute")]
[numthreads(16, 16, 1)]
void dwt_decompose_horiz(uint3 gid : SV_DispatchThreadID,
                         uniform ImageIn vsum, uniform ImageIn in,
                         uniform ImageOut coarse, uniform ImageOut detail,
                         uniform int width, uniform int height, uniform int hscale)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const int left = abs(x - hscale);
  const int right = (x < width - hscale) ? x + hscale : 2 * width - 2 - (x + hscale);

  const float4 hat = (2.f * vsum.Areadpixel(x, y) + vsum.Areadpixel(left, y) + vsum.Areadpixel(right, y)) / 16.f;
  coarse.write(int2(x, y), hat);
  detail.write(int2(x, y), in.Areadpixel(x, y) - hat);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void dwt_add(uint3 gid : SV_DispatchThreadID,
             uniform ImageIn a, uniform ImageIn b, uniform ImageOut out,
             uniform int width, uniform int height)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  out.write(int2(x, y), a.Areadpixel(x, y) + b.Areadpixel(x, y));
}
