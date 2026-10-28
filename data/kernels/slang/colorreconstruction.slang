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

// Slang port of the bilateral grid of color reconstruction, which holds
// L, a, b and a weight per grid point, x + sizex * (y + sizey * z) as in
// ../colorreconstruction.cl. Positions are found as in
// src/iop/colorreconstruction.c.

import common;

#define COLORRECONSTRUCT_PRECEDENCE_NONE 0
#define COLORRECONSTRUCT_PRECEDENCE_CHROMA 1
#define COLORRECONSTRUCT_PRECEDENCE_HUE 2

// the splat adds every pixel to its nearest grid point. As in bilateral.slang,
// each work item owns one (x, y) column of the grid and gathers the pixels
// that round to it, instead of the float atomics of the OpenCL kernel.
[shader("compute")]
[numthreads(16, 16, 1)]
void colorreconstruction_splat(uint3 gid : SV_DispatchThreadID,
                               uniform ImageIn in, uniform RWStructuredBuffer<float4> grid,
                               uniform int width, uniform int height,
                               uniform int sizex, uniform int sizey, uniform int sizez,
                               uniform float sigma_s, uniform float sigma_r, uniform float threshold,
                               uniform int precedence, uniform float4 params)
{
  const int gx = gid.x;
  const int gy = gid.y;
  if(gx >= sizex || gy >= sizey) return;

  const int oz = sizex * sizey;
  const int base = gx + sizex * gy;
  for(int z = 0; z < sizez; z++) grid[base + z * oz] = float4(0.0f);

  // the pixels that round to gx, with a pixel of margin; the last cell takes
  // every pixel up to the border, as positions are clamped to it
  const int i0 = max(0, int(floor((gx - 0.5f) * sigma_s)) - 1);
  const int i1 = gx >= sizex - 1 ? width - 1 : min(width - 1, int(ceil((gx + 0.5f) * sigma_s)) + 1);
  const int j0 = max(0, int(floor((gy - 0.5f) * sigma_s)) - 1);
  const int j1 = gy >= sizey - 1 ? height - 1 : min(height - 1, int(ceil((gy + 0.5f) * sigma_s)) + 1);

  for(int j = j0; j <= j1; j++)
  {
    const float y = clamp(j / sigma_s, 0.0f, sizey - 1.0f);
    if(clamp(int(round(y)), 0, sizey - 1) != gy) continue;

    for(int i = i0; i <= i1; i++)
    {
      const float x = clamp(i / sigma_s, 0.0f, sizex - 1.0f);
      if(clamp(int(round(x)), 0, sizex - 1) != gx) continue;

      const float4 pixel = in.Areadpixel(i, j);
      // we deliberately ignore pixels above threshold
      if(pixel.x > threshold) continue;

      float weight = 1.0f;
      if(precedence == COLORRECONSTRUCT_PRECEDENCE_CHROMA)
        weight = sqrt(pixel.y * pixel.y + pixel.z * pixel.z);
      else if(precedence == COLORRECONSTRUCT_PRECEDENCE_HUE)
      {
        float m = atan2(pixel.z, pixel.y) - params.x;
        // readjust m into [-pi, +pi] interval
        m = m > M_PI_F ? m - DT_2PI_F : (m < -M_PI_F ? m + DT_2PI_F : m);
        weight = exp(-m * m / params.y);
      }

      const float z = clamp(pixel.x / sigma_r, 0.0f, sizez - 1.0f);
      const int zi = clamp(int(round(z)), 0, sizez - 1);
      grid[base + zi * oz] += weight * float4(pixel.x, pixel.y, pixel.z, 1.0f);
    }
  }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void colorreconstruction_blur_line(uint3 gid : SV_DispatchThreadID,
                                   uniform StructuredBuffer<float4> ibuf, uniform RWStructuredBuffer<float4> obuf,
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

  float4 tmp1 = ibuf[index];
  obuf[index] = ibuf[index] * w0 + w1 * ibuf[index + offset3] + w2 * ibuf[index + 2 * offset3];
  index += offset3;
  float4 tmp2 = ibuf[index];
  obuf[index] = ibuf[index] * w0 + w1 * (ibuf[index + offset3] + tmp1) + w2 * ibuf[index + 2 * offset3];
  index += offset3;
  for(int i = 2; i < size3 - 2; i++)
  {
    const float4 tmp3 = ibuf[index];
    obuf[index] = ibuf[index] * w0
      + w1 * (ibuf[index + offset3] + tmp2)
      + w2 * (ibuf[index + 2 * offset3] + tmp1);
    index += offset3;
    tmp1 = tmp2;
    tmp2 = tmp3;
  }
  const float4 tmp3 = ibuf[index];
  obuf[index] = ibuf[index] * w0 + w1 * (ibuf[index + offset3] + tmp2) + w2 * tmp1;
  index += offset3;
  obuf[index] = ibuf[index] * w0 + w1 * tmp3 + w2 * tmp2;
}

// bxy and roixy are the origins of the grid and of the image, and scale
// takes image positions to those of the grid, which can come from the
// preview pipe
[shader("compute")]
[numthreads(16, 16, 1)]
void colorreconstruction_slice(uint3 gid : SV_DispatchThreadID,
                               uniform ImageIn in, uniform ImageOut out, uniform StructuredBuffer<float4> grid,
                               uniform int width, uniform int height,
                               uniform int sizex, uniform int sizey, uniform int sizez,
                               uniform float sigma_s, uniform float sigma_r, uniform float threshold,
                               uniform int2 bxy, uniform int2 roixy, uniform float scale)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  float4 pixel = in.Areadpixel(x, y);
  const float blend = clamp(20.0f / threshold * pixel.x - 19.0f, 0.0f, 1.0f);
  if(blend == 0.0f)
  {
    out.write(int2(x, y), pixel);
    return;
  }

  const float2 pxy = float2(roixy + int2(x, y)) * scale - float2(bxy);
  const float px = clamp(pxy.x / sigma_s, 0.0f, sizex - 1.0f);
  const float py = clamp(pxy.y / sigma_s, 0.0f, sizey - 1.0f);
  const float pz = clamp(pixel.x / sigma_r, 0.0f, sizez - 1.0f);
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
  const float4 opixel = grid[gi]                * (1.0f - fx) * (1.0f - fy) * (1.0f - fz)
                      + grid[gi + ox]           * (       fx) * (1.0f - fy) * (1.0f - fz)
                      + grid[gi + oy]           * (1.0f - fx) * (       fy) * (1.0f - fz)
                      + grid[gi + ox + oy]      * (       fx) * (       fy) * (1.0f - fz)
                      + grid[gi + oz]           * (1.0f - fx) * (1.0f - fy) * (       fz)
                      + grid[gi + ox + oz]      * (       fx) * (1.0f - fy) * (       fz)
                      + grid[gi + oy + oz]      * (1.0f - fx) * (       fy) * (       fz)
                      + grid[gi + ox + oy + oz] * (       fx) * (       fy) * (       fz);

  const float lout = max(opixel.x, 0.01f);
  if(opixel.w > 0.0f)
    pixel.yz = pixel.yz * (1.0f - blend) + opixel.yz * pixel.x / lout * blend;
  out.write(int2(x, y), pixel);
}
