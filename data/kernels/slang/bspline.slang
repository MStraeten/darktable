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

// Slang ports of the kernels in ../bspline.cl

import common;

// B spline filter
static const int FSIZE = 5;
static const int FSTART = (FSIZE - 1) / 2;
static const float filter[FSIZE] = { 1.0f / 16.0f, 4.0f / 16.0f, 6.0f / 16.0f, 4.0f / 16.0f, 1.0f / 16.0f };

[shader("compute")]
[numthreads(16, 16, 1)]
void blur_2D_Bspline_vertical(uint3 gid : SV_DispatchThreadID,
                              uniform ImageIn in, uniform ImageOut out,
                              uniform int width, uniform int height, uniform int mult)
{
  // à-trous B-spline interpolation/blur shifted by mult, over lines
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  float4 accumulator = float4(0.f);

  [unroll]
  for(int jj = 0; jj < FSIZE; ++jj)
  {
    const int yy = mult * (jj - FSTART) + y;
    accumulator += filter[jj] * in.Areadpixel(x, clamp(yy, 0, height - 1));
  }

  out.write(int2(x, y), max(accumulator, float4(0.f)));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void blur_2D_Bspline_horizontal(uint3 gid : SV_DispatchThreadID,
                                uniform ImageIn in, uniform ImageOut out,
                                uniform int width, uniform int height, uniform int mult)
{
  // à-trous B-spline interpolation/blur shifted by mult, over columns
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  float4 accumulator = float4(0.f);

  [unroll]
  for(int ii = 0; ii < FSIZE; ++ii)
  {
    const int xx = mult * (ii - FSTART) + x;
    accumulator += filter[ii] * in.Areadpixel(clamp(xx, 0, width - 1), y);
  }

  out.write(int2(x, y), max(accumulator, float4(0.f)));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void wavelets_detail_level(uint3 gid : SV_DispatchThreadID,
                           uniform ImageIn detail, uniform ImageIn LF, uniform ImageOut HF,
                           uniform int width, uniform int height)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float4 d = detail.Areadpixel(x, y);
  const float4 lf = LF.Areadpixel(x, y);

  HF.write(int2(x, y), d - lf);
}
