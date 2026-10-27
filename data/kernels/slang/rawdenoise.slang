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

// Slang port of the Bayer path of src/iop/rawdenoise.c, which has no OpenCL
// code, with dwt_denoise() of src/common/dwt.c. Each of the four CFA
// channels is denoised as a half-size plane: extract, then a vertical and a
// horizontal pass per wavelet band, then scatter back.

import common;

[shader("compute")]
[numthreads(16, 16, 1)]
void rawdenoise_copy(uint3 gid : SV_DispatchThreadID,
                     uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;
  out.write(int2(x, y), in.readA(int2(x, y)));
}

// the sensels of channel c, at rows c & 1 and columns (c & 2) >> 1 modulo 2,
// with sqrt() as a variance-stabilizing transform
[shader("compute")]
[numthreads(16, 16, 1)]
void rawdenoise_extract(uint3 gid : SV_DispatchThreadID,
                        uniform ImageIn in, uniform RWStructuredBuffer<float> fimg,
                        uniform int c, uniform int halfwidth, uniform int halfheight)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= halfwidth || y >= halfheight) return;
  const int row = 2 * y + (c & 1);
  const int col = 2 * x + ((c & 2) >> 1);
  fimg[y * halfwidth + x] = sqrt(max(0.0f, in.Areadsingle(col, row)));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void rawdenoise_scatter(uint3 gid : SV_DispatchThreadID,
                        uniform StructuredBuffer<float> fimg, uniform ImageOut out,
                        uniform int c, uniform int halfwidth, uniform int halfheight)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= halfwidth || y >= halfheight) return;
  const int row = 2 * y + (c & 1);
  const int col = 2 * x + ((c & 2) >> 1);
  const float d = fimg[y * halfwidth + x];
  out.write(int2(col, row), float4(d * d));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void rawdenoise_clear(uint3 gid : SV_DispatchThreadID,
                      uniform RWStructuredBuffer<float> buf, uniform int width, uniform int height)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;
  buf[y * width + x] = 0.0f;
}

// dwt_denoise_vert_1ch(): rows scale apart, reflected at the edges
[shader("compute")]
[numthreads(16, 16, 1)]
void dwt_denoise_vert(uint3 gid : SV_DispatchThreadID,
                      uniform StructuredBuffer<float> in, uniform RWStructuredBuffer<float> out,
                      uniform int width, uniform int height, uniform int vscale)
{
  const int col = gid.x;
  const int row = gid.y;
  if(col >= width || row >= height) return;
  const int below = row + vscale < height ? row + vscale : 2 * (height - 1) - (row + vscale);
  const int above = abs(row - vscale);
  out[row * width + col] = 2.0f * in[row * width + col] + in[above * width + col] + in[below * width + col];
}

// dwt_denoise_horiz_1ch(): leaves the coarse scale in img and adds the
// thresholded details to accum, which goes back into img after the last band
[shader("compute")]
[numthreads(16, 16, 1)]
void dwt_denoise_horiz(uint3 gid : SV_DispatchThreadID,
                       uniform StructuredBuffer<float> coarse, uniform RWStructuredBuffer<float> img,
                       uniform RWStructuredBuffer<float> accum,
                       uniform int width, uniform int height, uniform int hscale,
                       uniform float thold, uniform int last)
{
  const int col = gid.x;
  const int row = gid.y;
  if(col >= width || row >= height) return;
  const int k = row * width;
  const int left = abs(col - hscale);
  const int right = col + hscale < width ? col + hscale : 2 * (width - 1) - (col + hscale);
  const float hat = (2.0f * coarse[k + col] + coarse[k + left] + coarse[k + right]) / 16.0f;
  const float diff = img[k + col] - hat;
  const float sum = accum[k + col] + (max(diff - thold, 0.0f) + min(diff + thold, 0.0f));
  accum[k + col] = sum;
  img[k + col] = last != 0 ? hat + sum : hat;
}
