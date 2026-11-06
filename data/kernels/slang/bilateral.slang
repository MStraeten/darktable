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

// Slang port of the bilateral grid in ../bilateral.cl. The grid is laid out
// as there, x + sizex * (y + sizey * z), and positions are found as in
// src/common/bilateral.c, by multiplying with the reciprocal sigmas.

import common;

// the OpenCL splat scatters every pixel into its 8 grid points with float
// atomics. Here each work item owns one (x, y) column of the grid and gathers
// the pixels that land in it, so that the sum needs no atomics and doesn't
// depend on the order the work items run in. Each pixel is read by the 4
// columns around it.
[shader("compute")]
[numthreads(16, 16, 1)]
void bilateral_splat(uint3 gid : SV_DispatchThreadID,
                     uniform ImageIn in, uniform RWStructuredBuffer<float> grid,
                     uniform int width, uniform int height,
                     uniform int sizex, uniform int sizey, uniform int sizez,
                     uniform float sigma_s, uniform float sigma_s_inv, uniform float sigma_r_inv)
{
  const int gx = gid.x;
  const int gy = gid.y;
  if(gx >= sizex || gy >= sizey) return;

  const int oz = sizex * sizey;
  const int base = gx + sizex * gy;
  for(int z = 0; z < sizez; z++) grid[base + z * oz] = 0.0f;

  // the pixels whose cell starts at gx - 1 or gx, with a pixel of margin for
  // the rounding of i * sigma_s_inv. The last cells take every pixel up to
  // the border, as positions are clamped to them.
  const int i0 = max(0, int(floor((gx - 1) * sigma_s)) - 1);
  const int i1 = gx >= sizex - 2 ? width - 1 : min(width - 1, int(ceil((gx + 1) * sigma_s)) + 1);
  const int j0 = max(0, int(floor((gy - 1) * sigma_s)) - 1);
  const int j1 = gy >= sizey - 2 ? height - 1 : min(height - 1, int(ceil((gy + 1) * sigma_s)) + 1);

  const float contrib = 100.0f / (sigma_s * sigma_s);

  for(int j = j0; j <= j1; j++)
  {
    const float y = clamp(j * sigma_s_inv, 0.0f, sizey - 1.0f);
    const int yi = min(int(y), sizey - 2);
    if(yi != gy && yi != gy - 1) continue;
    const float fy = y - yi;
    const float wy = contrib * (yi == gy ? 1.0f - fy : fy);

    for(int i = i0; i <= i1; i++)
    {
      const float x = clamp(i * sigma_s_inv, 0.0f, sizex - 1.0f);
      const int xi = min(int(x), sizex - 2);
      if(xi != gx && xi != gx - 1) continue;
      const float fx = x - xi;
      const float w = wy * (xi == gx ? 1.0f - fx : fx);

      const float L = in.Areadpixel(i, j).x;
      const float z = clamp(L * sigma_r_inv, 0.0f, sizez - 1.0f);
      const int zi = min(int(z), sizez - 2);
      const float fz = z - zi;
      grid[base + zi * oz] += w * (1.0f - fz);
      grid[base + (zi + 1) * oz] += w * fz;
    }
  }
}

// one line of the separable gaussian, from ibuf into obuf
[shader("compute")]
[numthreads(16, 16, 1)]
void bilateral_blur_line(uint3 gid : SV_DispatchThreadID,
                         uniform StructuredBuffer<float> ibuf, uniform RWStructuredBuffer<float> obuf,
                         uniform int offset1, uniform int offset2, uniform int offset3,
                         uniform int size1, uniform int size2, uniform int size3)
{
  const int k = gid.x;
  const int j = gid.y;
  if(k >= size1 || j >= size2) return;

  const float w0 = 6.0f / 16.0f;
  const float w1 = 4.0f / 16.0f;
  const float w2 = 1.0f / 16.0f;
  int index = k * offset1 + j * offset2;

  float tmp1 = ibuf[index];
  obuf[index] = ibuf[index] * w0 + w1 * ibuf[index + offset3] + w2 * ibuf[index + 2 * offset3];
  index += offset3;
  float tmp2 = ibuf[index];
  obuf[index] = ibuf[index] * w0 + w1 * (ibuf[index + offset3] + tmp1) + w2 * ibuf[index + 2 * offset3];
  index += offset3;
  for(int i = 2; i < size3 - 2; i++)
  {
    const float tmp3 = ibuf[index];
    obuf[index] = ibuf[index] * w0
      + w1 * (ibuf[index + offset3] + tmp2)
      + w2 * (ibuf[index + 2 * offset3] + tmp1);
    index += offset3;
    tmp1 = tmp2;
    tmp2 = tmp3;
  }
  const float tmp3 = ibuf[index];
  obuf[index] = ibuf[index] * w0 + w1 * (ibuf[index + offset3] + tmp2) + w2 * tmp1;
  index += offset3;
  obuf[index] = ibuf[index] * w0 + w1 * tmp3 + w2 * tmp2;
}

// the derivative of the gaussian along z
[shader("compute")]
[numthreads(16, 16, 1)]
void bilateral_blur_line_z(uint3 gid : SV_DispatchThreadID,
                           uniform StructuredBuffer<float> ibuf, uniform RWStructuredBuffer<float> obuf,
                           uniform int offset1, uniform int offset2, uniform int offset3,
                           uniform int size1, uniform int size2, uniform int size3)
{
  const int k = gid.x;
  const int j = gid.y;
  if(k >= size1 || j >= size2) return;

  const float w1 = 4.0f / 16.0f;
  const float w2 = 2.0f / 16.0f;
  int index = k * offset1 + j * offset2;

  float tmp1 = ibuf[index];
  obuf[index] = w1 * ibuf[index + offset3] + w2 * ibuf[index + 2 * offset3];
  index += offset3;
  float tmp2 = ibuf[index];
  obuf[index] = w1 * (ibuf[index + offset3] - tmp1) + w2 * ibuf[index + 2 * offset3];
  index += offset3;
  for(int i = 2; i < size3 - 2; i++)
  {
    const float tmp3 = ibuf[index];
    obuf[index] = w1 * (ibuf[index + offset3] - tmp2)
      + w2 * (ibuf[index + 2 * offset3] - tmp1);
    index += offset3;
    tmp1 = tmp2;
    tmp2 = tmp3;
  }
  const float tmp3 = ibuf[index];
  obuf[index] = w1 * (ibuf[index + offset3] - tmp2) - w2 * tmp1;
  index += offset3;
  obuf[index] = -w1 * tmp3 - w2 * tmp2;
}

// the trilinear lookup of the grid at the position of pixel (x, y) with
// lightness L
float _grid_lookup(StructuredBuffer<float> grid, const int x, const int y, const float L,
                   const int sizex, const int sizey, const int sizez,
                   const float sigma_s_inv, const float sigma_r_inv)
{
  const float px = clamp(x * sigma_s_inv, 0.0f, sizex - 1.0f);
  const float py = clamp(y * sigma_s_inv, 0.0f, sizey - 1.0f);
  const float pz = clamp(L * sigma_r_inv, 0.0f, sizez - 1.0f);
  const int xi = min(int(px), sizex - 2);
  const int yi = min(int(py), sizey - 2);
  const int zi = min(int(pz), sizez - 2);
  const float fx = px - xi;
  const float fy = py - yi;
  const float fz = pz - zi;

  const int ox = 1;
  const int oy = sizex;
  const int oz = sizey * sizex;
  const int gi = xi + sizex * (yi + sizey * zi);
  return grid[gi]                * (1.0f - fx) * (1.0f - fy) * (1.0f - fz)
       + grid[gi + ox]           * (       fx) * (1.0f - fy) * (1.0f - fz)
       + grid[gi + oy]           * (1.0f - fx) * (       fy) * (1.0f - fz)
       + grid[gi + ox + oy]      * (       fx) * (       fy) * (1.0f - fz)
       + grid[gi + oz]           * (1.0f - fx) * (1.0f - fy) * (       fz)
       + grid[gi + ox + oz]      * (       fx) * (1.0f - fy) * (       fz)
       + grid[gi + oy + oz]      * (1.0f - fx) * (       fy) * (       fz)
       + grid[gi + ox + oy + oz] * (       fx) * (       fy) * (       fz);
}

// detail: 0 is leave as is, -1 is bilateral filtered, +1 is contrast boost
[shader("compute")]
[numthreads(16, 16, 1)]
void bilateral_slice(uint3 gid : SV_DispatchThreadID,
                     uniform ImageIn in, uniform ImageOut out, uniform StructuredBuffer<float> grid,
                     uniform int width, uniform int height,
                     uniform int sizex, uniform int sizey, uniform int sizez,
                     uniform float sigma_s_inv, uniform float sigma_r_inv, uniform float sigma_r,
                     uniform float detail)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float norm = -detail * sigma_r * 0.04f;
  float4 pixel = in.Areadpixel(x, y);
  const float Ldiff = _grid_lookup(grid, x, y, pixel.x, sizex, sizey, sizez, sigma_s_inv, sigma_r_inv);
  pixel.x = max(0.0f, pixel.x + norm * Ldiff);
  out.write(int2(x, y), pixel);
}

// like bilateral_slice(), but the lightness of in only finds the grid
// position, and the detail is added to target
[shader("compute")]
[numthreads(16, 16, 1)]
void bilateral_slice_to_output(uint3 gid : SV_DispatchThreadID,
                               uniform ImageIn in, uniform ImageIn target, uniform ImageOut out,
                               uniform StructuredBuffer<float> grid,
                               uniform int width, uniform int height,
                               uniform int sizex, uniform int sizey, uniform int sizez,
                               uniform float sigma_s_inv, uniform float sigma_r_inv, uniform float sigma_r,
                               uniform float detail)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float norm = -detail * sigma_r * 0.04f;
  const float L = in.Areadpixel(x, y).x;
  float4 pixel = target.Areadpixel(x, y);
  const float Ldiff = _grid_lookup(grid, x, y, L, sizex, sizey, sizez, sigma_s_inv, sigma_r_inv);
  pixel.x = max(0.0f, pixel.x + norm * Ldiff);
  out.write(int2(x, y), pixel);
}
