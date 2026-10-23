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

// Slang port of the recursive gaussian in ../gaussian.cl on buffers of 1, 2
// and 4 channels: the blend path's mask blur uses the first, the guided
// filters of color equalizer the others. The kernels are dispatched on 16x16
// work groups, so the ones the OpenCL code runs over a single row skip the
// work items of the other rows.

import common;

[shader("compute")]
[numthreads(16, 16, 1)]
void gaussian_column_1c(uint3 gid : SV_DispatchThreadID,
                        uniform StructuredBuffer<float> in, uniform RWStructuredBuffer<float> out,
                        uniform int width, uniform int height,
                        uniform float a0, uniform float a1, uniform float a2, uniform float a3,
                        uniform float b1, uniform float b2, uniform float coefp, uniform float coefn,
                        uniform float Labmax, uniform float Labmin)
{
  const int x = gid.x;
  if(x >= width || gid.y != 0) return;

  // forward filter
  float xp = clamp(in[x], Labmin, Labmax);
  float yb = xp * coefp;
  float yp = yb;

  for(int y = 0; y < height; y++)
  {
    const int idx = y * width + x;
    const float xc = clamp(in[idx], Labmin, Labmax);
    const float yc = (a0 * xc) + (a1 * xp) - (b1 * yp) - (b2 * yb);
    xp = xc;
    yb = yp;
    yp = yc;
    out[idx] = yc;
  }

  // backward filter
  float xn = clamp(in[(height - 1) * width + x], Labmin, Labmax);
  float xa = xn;
  float yn = xn * coefn;
  float ya = yn;

  for(int y = height - 1; y > -1; y--)
  {
    const int idx = y * width + x;
    const float xc = clamp(in[idx], Labmin, Labmax);
    const float yc = (a2 * xn) + (a3 * xa) - (b1 * yn) - (b2 * ya);
    xa = xn;
    xn = xc;
    ya = yn;
    yn = yc;
    out[idx] += yc;
  }
}

// the OpenCL kernel stages blocks in local memory to coalesce its writes;
// this one transposes directly
[shader("compute")]
[numthreads(16, 16, 1)]
void gaussian_transpose_1c(uint3 gid : SV_DispatchThreadID,
                           uniform StructuredBuffer<float> in, uniform RWStructuredBuffer<float> out,
                           uniform int width, uniform int height)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;
  out[x * height + y] = in[y * width + x];
}

[shader("compute")]
[numthreads(16, 16, 1)]
void gaussian_column_2c(uint3 gid : SV_DispatchThreadID,
                        uniform StructuredBuffer<float2> in, uniform RWStructuredBuffer<float2> out,
                        uniform int width, uniform int height,
                        uniform float a0, uniform float a1, uniform float a2, uniform float a3,
                        uniform float b1, uniform float b2, uniform float coefp, uniform float coefn,
                        uniform float2 Labmax, uniform float2 Labmin)
{
  const int x = gid.x;
  if(x >= width || gid.y != 0) return;

  // forward filter
  float2 xp = clamp(in[x], Labmin, Labmax);
  float2 yb = xp * coefp;
  float2 yp = yb;

  for(int y = 0; y < height; y++)
  {
    const int idx = y * width + x;
    const float2 xc = clamp(in[idx], Labmin, Labmax);
    const float2 yc = (a0 * xc) + (a1 * xp) - (b1 * yp) - (b2 * yb);
    xp = xc;
    yb = yp;
    yp = yc;
    out[idx] = yc;
  }

  // backward filter
  float2 xn = clamp(in[(height - 1) * width + x], Labmin, Labmax);
  float2 xa = xn;
  float2 yn = xn * coefn;
  float2 ya = yn;

  for(int y = height - 1; y > -1; y--)
  {
    const int idx = y * width + x;
    const float2 xc = clamp(in[idx], Labmin, Labmax);
    const float2 yc = (a2 * xn) + (a3 * xa) - (b1 * yn) - (b2 * ya);
    xa = xn;
    xn = xc;
    ya = yn;
    yn = yc;
    out[idx] += yc;
  }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void gaussian_column_4c(uint3 gid : SV_DispatchThreadID,
                        uniform StructuredBuffer<float4> in, uniform RWStructuredBuffer<float4> out,
                        uniform int width, uniform int height,
                        uniform float a0, uniform float a1, uniform float a2, uniform float a3,
                        uniform float b1, uniform float b2, uniform float coefp, uniform float coefn,
                        uniform float4 Labmax, uniform float4 Labmin)
{
  const int x = gid.x;
  if(x >= width || gid.y != 0) return;

  // forward filter
  float4 xp = clamp(in[x], Labmin, Labmax);
  float4 yb = xp * coefp;
  float4 yp = yb;

  for(int y = 0; y < height; y++)
  {
    const int idx = y * width + x;
    const float4 xc = clamp(in[idx], Labmin, Labmax);
    const float4 yc = (a0 * xc) + (a1 * xp) - (b1 * yp) - (b2 * yb);
    xp = xc;
    yb = yp;
    yp = yc;
    out[idx] = yc;
  }

  // backward filter
  float4 xn = clamp(in[(height - 1) * width + x], Labmin, Labmax);
  float4 xa = xn;
  float4 yn = xn * coefn;
  float4 ya = yn;

  for(int y = height - 1; y > -1; y--)
  {
    const int idx = y * width + x;
    const float4 xc = clamp(in[idx], Labmin, Labmax);
    const float4 yc = (a2 * xn) + (a3 * xa) - (b1 * yn) - (b2 * ya);
    xa = xn;
    xn = xc;
    ya = yn;
    yn = yc;
    out[idx] += yc;
  }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void gaussian_transpose_2c(uint3 gid : SV_DispatchThreadID,
                           uniform StructuredBuffer<float2> in, uniform RWStructuredBuffer<float2> out,
                           uniform int width, uniform int height)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;
  out[x * height + y] = in[y * width + x];
}

[shader("compute")]
[numthreads(16, 16, 1)]
void gaussian_transpose_4c(uint3 gid : SV_DispatchThreadID,
                           uniform StructuredBuffer<float4> in, uniform RWStructuredBuffer<float4> out,
                           uniform int width, uniform int height)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;
  out[x * height + y] = in[y * width + x];
}

// the OpenCL host code copies between image and buffer with
// clEnqueueCopyImageToBuffer and back; Slang has no such call
[shader("compute")]
[numthreads(16, 16, 1)]
void gaussian_image_to_buffer_1c(uint3 gid : SV_DispatchThreadID,
                                 uniform ImageIn in, uniform RWStructuredBuffer<float> out,
                                 uniform int width, uniform int height)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;
  out[y * width + x] = in.Areadsingle(x, y);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void gaussian_buffer_to_image_1c(uint3 gid : SV_DispatchThreadID,
                                 uniform StructuredBuffer<float> in, uniform ImageOut out,
                                 uniform int width, uniform int height)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;
  out.write(int2(x, y), float4(in[y * width + x]));
}
