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

// Slang ports of the kernels in ../locallaplacian.cl

import common;

static const float w[5] = { 1.0f / 16.0f, 4.0f / 16.0f, 6.0f / 16.0f, 4.0f / 16.0f, 1.0f / 16.0f };

[shader("compute")]
[numthreads(16, 16, 1)]
void pad_input(uint3 gid : SV_DispatchThreadID,
               uniform ImageIn input, uniform ImageOut padded,
               uniform int wd,       // dimensions of input
               uniform int ht,
               uniform int max_supp, // size of border
               uniform int wd2,      // padded dimensions
               uniform int ht2)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= wd2 || y >= ht2) return;

  // fill boundary with max_supp px
  const int cx = clamp(x - max_supp, 0, wd - 1);
  const int cy = clamp(y - max_supp, 0, ht - 1);

  const float pixel = input.readsingle(cx, cy);
  padded.write(int2(x, y), float4(pixel * 0.01f));
}

float expand_gaussian(ImageIn coarse, int i, int j)
{
  float c = 0.0f;
  const int cx = i / 2;
  const int cy = j / 2;
  switch((i & 1) + 2 * (j & 1))
  {
    case 0: // both are even, 3x3 stencil
      for(int ii = -1; ii <= 1; ii++)
        for(int jj = -1; jj <= 1; jj++)
          c += coarse.readsingle(cx + ii, cy + jj) * w[2 * jj + 2] * w[2 * ii + 2];
      break;
    case 1: // i is odd, 2x3 stencil
      for(int ii = 0; ii <= 1; ii++)
        for(int jj = -1; jj <= 1; jj++)
          c += coarse.readsingle(cx + ii, cy + jj) * w[2 * jj + 2] * w[2 * ii + 1];
      break;
    case 2: // j is odd, 3x2 stencil
      for(int ii = -1; ii <= 1; ii++)
        for(int jj = 0; jj <= 1; jj++)
          c += coarse.readsingle(cx + ii, cy + jj) * w[2 * jj + 1] * w[2 * ii + 2];
      break;
    default: // both are odd, 2x2 stencil
      for(int ii = 0; ii <= 1; ii++)
        for(int jj = 0; jj <= 1; jj++)
          c += coarse.readsingle(cx + ii, cy + jj) * w[2 * jj + 1] * w[2 * ii + 1];
      break;
  }
  return 4.0f * c;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void gauss_reduce(uint3 gid : SV_DispatchThreadID,
                  uniform ImageIn input,  // fine input buffer
                  uniform ImageOut coarse, // coarse scale, blurred input buf
                  uniform int wd,         // coarse res (also run this kernel on coarse res only)
                  uniform int ht)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= wd || y >= ht) return;

  // fill boundary with 1 px
  const int cx = clamp(x, 1, wd - 2);
  const int cy = clamp(y, 1, ht - 2);

  // blur, store only coarse res: direct 5x5 stencil only on required pixels
  float pixel = 0.0f;
  for(int jj = -2; jj <= 2; jj++)
    for(int ii = -2; ii <= 2; ii++)
      pixel += input.readsingle(2 * cx + ii, 2 * cy + jj) * w[ii + 2] * w[jj + 2];
  coarse.write(int2(x, y), float4(pixel));
}

float laplacian(ImageIn coarse, // coarse res gaussian
                ImageIn fine,   // fine res gaussian
                int i,          // fine index
                int j,
                int ci,         // clamped fine index
                int cj)
{
  const float c = expand_gaussian(coarse, ci, cj);
  return fine.readsingle(i, j) - c;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void laplacian_assemble(uint3 gid : SV_DispatchThreadID,
                        uniform ImageIn input,   // original input buffer, gauss at current fine pyramid level
                        uniform ImageIn output1, // state of reconstruction, coarse output buffer
                        uniform ImageOut output0, // reconstruction, one level finer, run kernel on this dimension
                        uniform ImageIn buf_g0_l0,
                        uniform ImageIn buf_g0_l1,
                        uniform ImageIn buf_g1_l0,
                        uniform ImageIn buf_g1_l1,
                        uniform ImageIn buf_g2_l0,
                        uniform ImageIn buf_g2_l1,
                        uniform ImageIn buf_g3_l0,
                        uniform ImageIn buf_g3_l1,
                        uniform ImageIn buf_g4_l0,
                        uniform ImageIn buf_g4_l1,
                        uniform ImageIn buf_g5_l0,
                        uniform ImageIn buf_g5_l1,
                        uniform int pw, // width and height of the fine buffers (l0)
                        uniform int ph)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= pw || y >= ph) return;

  int i = x;
  int j = y;
  // fill boundary with 1 or 2 px
  if((pw & 1) != 0) { if(x > pw - 2) i = pw - 2; }
  else              { if(x > pw - 3) i = pw - 3; }
  if((ph & 1) != 0) { if(y > ph - 2) j = ph - 2; }
  else              { if(y > ph - 3) j = ph - 3; }
  if(x <= 0) i = 1;
  if(y <= 0) j = 1;

  float pixel = expand_gaussian(output1, i, j);

  const int num_gamma = 6;
  const float v = input.readsingle(x, y);
  int hi = 1;
  // for(; hi < num_gamma - 1 && gamma[hi] <= v; hi++);
  for(; hi < num_gamma - 1 && ((float)hi + .5f) / (float)num_gamma <= v; hi++);
  const int lo = hi - 1;
  const float a = clipf(v * num_gamma - ((float)lo + .5f));

  float l0, l1;
  switch(lo)
  {
    case 0:
      l0 = laplacian(buf_g0_l1, buf_g0_l0, x, y, i, j);
      l1 = laplacian(buf_g1_l1, buf_g1_l0, x, y, i, j);
      break;
    case 1:
      l0 = laplacian(buf_g1_l1, buf_g1_l0, x, y, i, j);
      l1 = laplacian(buf_g2_l1, buf_g2_l0, x, y, i, j);
      break;
    case 2:
      l0 = laplacian(buf_g2_l1, buf_g2_l0, x, y, i, j);
      l1 = laplacian(buf_g3_l1, buf_g3_l0, x, y, i, j);
      break;
    case 3:
      l0 = laplacian(buf_g3_l1, buf_g3_l0, x, y, i, j);
      l1 = laplacian(buf_g4_l1, buf_g4_l0, x, y, i, j);
      break;
    default:
      l0 = laplacian(buf_g4_l1, buf_g4_l0, x, y, i, j);
      l1 = laplacian(buf_g5_l1, buf_g5_l0, x, y, i, j);
      break;
  }
  pixel += l0 * (1.0f - a) + l1 * a;
  output0.write(int2(x, y), float4(pixel));
}

float curve(float x, float g, float sigma, float shadows, float highlights, float clarity)
{
  const float c = x - g;
  float val;
  const float ssigma = c > 0.0f ? sigma : -sigma;
  const float shadhi = c > 0.0f ? shadows : highlights;
  if(abs(c) > 2 * sigma)
  {
    val = g + ssigma + shadhi * (c - ssigma); // linear part
  }
  else
  {
    // blend in via quadratic bezier
    const float t = clipf(c / (2.0f * ssigma));
    const float t2 = t * t;
    const float mt = 1.0f - t;
    val = g + ssigma * 2.0f * mt * t + t2 * (ssigma + ssigma * shadhi);
  }
  // midtone local contrast
  val += clarity * c * dt_fast_expf(-c * c / (2.0f * sigma * sigma / 3.0f));
  return val;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void process_curve(uint3 gid : SV_DispatchThreadID,
                   uniform ImageIn input, uniform ImageOut output,
                   uniform float g, uniform float sigma, uniform float shadows, uniform float highlights,
                   uniform float clarity, uniform int wd, uniform int ht)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= wd || y >= ht) return;

  float4 pixel = input.readpixel(x, y);
  pixel.x = curve(pixel.x, g, sigma, shadows, highlights, clarity);
  output.write(int2(x, y), pixel);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void write_back(uint3 gid : SV_DispatchThreadID,
                uniform ImageIn input, uniform ImageIn processed, uniform ImageOut output,
                uniform int max_supp, uniform int wd, uniform int ht)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= wd || y >= ht) return;

  float4 pixel = input.readpixel(x, y);
  pixel.x = 100.0f * processed.readsingle(x + max_supp, y + max_supp);
  output.write(int2(x, y), pixel);
}
