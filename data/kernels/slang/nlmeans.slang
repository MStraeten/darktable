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


// Slang port of ../nlmeans.cl, following nlmeans_denoise() of
// src/common/nlmeans_core.c where the two differ: the box sums of the patch
// distances count nothing outside the image, where OpenCL repeats the border.
// The accumulation is denoiseprofile_accu of denoiseprofile.slang.

import common;

[shader("compute")]
[numthreads(16, 16, 1)]
void nlmeans_dist(uint3 gid : SV_DispatchThreadID,
                  uniform ImageIn in, uniform RWStructuredBuffer<float> U4,
                  uniform int width, uniform int height, uniform int2 q, uniform float nL2, uniform float nC2)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float4 norm2 = float4(nL2, nC2, nC2, 1.0f);
  const bool inside = x + q.x < width && x + q.x >= 0 && y + q.y < height && y + q.y >= 0;

  // out of bounds positions read pixel 0 and count as no distance, as in OpenCL
  const int xpq = (x + q.x < width && x + q.x >= 0) ? x + q.x : 0;
  const int ypq = (y + q.y < height && y + q.y >= 0) ? y + q.y : 0;

  const float4 p1 = in.readpixel(x, y);
  const float4 p2 = in.readpixel(xpq, ypq);
  const float4 tmp = (p1 - p2) * (p1 - p2) * norm2;

  U4[y * width + x] = inside ? tmp.x + tmp.y + tmp.z : 0.0f;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void nlmeans_horiz(uint3 gid : SV_DispatchThreadID,
                   uniform StructuredBuffer<float> U4_in, uniform RWStructuredBuffer<float> U4_out,
                   uniform int width, uniform int height, uniform int P)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  float distacc = 0.0f;
  for(int pi = max(-P, -x); pi <= min(P, width - 1 - x); pi++)
    distacc += U4_in[y * width + x + pi];

  U4_out[y * width + x] = distacc;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void nlmeans_vert(uint3 gid : SV_DispatchThreadID,
                  uniform StructuredBuffer<float> U4_in, uniform RWStructuredBuffer<float> U4_out,
                  uniform int width, uniform int height, uniform int P, uniform float sharpness)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  float distacc = 0.0f;
  for(int pj = max(-P, -y); pj <= min(P, height - 1 - y); pj++)
    distacc += U4_in[(y + pj) * width + x];

  // make sharpness bigger: less smoothing
  U4_out[y * width + x] = fast_mexp2f(distacc * sharpness);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void nlmeans_finish(uint3 gid : SV_DispatchThreadID,
                    uniform ImageIn in, uniform StructuredBuffer<float4> U2, uniform ImageOut out,
                    uniform int width, uniform int height, uniform float4 weight)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float4 i = in.readpixel(x, y);
  const float4 u2 = U2[y * width + x];
  float4 o = i * (float4(1.0f) - weight) + u2 / u2.w * weight;
  o.w = i.w;
  out.write(int2(x, y), o);
}
