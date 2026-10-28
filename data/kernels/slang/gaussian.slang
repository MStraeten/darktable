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
import color_conversion;

[shader("compute")]
[numthreads(16, 16, 1)]
void gaussian_column_1c(uint3 gid : SV_DispatchThreadID,
                        uniform StructuredBuffer<float> in, uniform RWStructuredBuffer<float> out,
                        uniform int width, uniform int height,
                        uniform float a0, uniform float a1, uniform float a2, uniform float a3,
                        uniform float b1, uniform float b2, uniform float coefp, uniform float coefn,
                        uniform float Labmax, uniform float Labmin)
{
  const int x = gid.y * 16 + gid.x;
  if(x >= width) return;

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
  const int x = gid.y * 16 + gid.x;
  if(x >= width) return;

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
  const int x = gid.y * 16 + gid.x;
  if(x >= width) return;

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

[shader("compute")]
[numthreads(16, 16, 1)]
void gaussian_image_to_buffer_4c(uint3 gid : SV_DispatchThreadID,
                                 uniform ImageIn in, uniform RWStructuredBuffer<float4> out,
                                 uniform int width, uniform int height)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;
  out[y * width + x] = in.Areadpixel(x, y);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void gaussian_buffer_to_image_4c(uint3 gid : SV_DispatchThreadID,
                                 uniform StructuredBuffer<float4> in, uniform ImageOut out,
                                 uniform int width, uniform int height)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;
  out.write(int2(x, y), in[y * width + x]);
}

// the mixing steps that ../gaussian.cl holds for the modules built on the
// blur: lowpass and shadows and highlights

[shader("compute")]
[numthreads(16, 16, 1)]
void lowpass_mix(uint3 gid : SV_DispatchThreadID,
                 uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
                 uniform float saturation, uniform ImageIn ctable, uniform StructuredBuffer<float> ca,
                 uniform ImageIn ltable, uniform StructuredBuffer<float> la, uniform int unbound)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float4 i = in.Areadpixel(x, y);
  const float2 abmin = unbound != 0 ? float2(-float.maxValue) : float2(-128.0f);
  const float2 abmax = unbound != 0 ? float2(float.maxValue) : float2(128.0f);
  const float a_c[3] = { ca[0], ca[1], ca[2] };
  const float a_l[3] = { la[0], la[1], la[2] };

  float4 o;
  o.x = lookup_unbounded(ctable, i.x / 100.0f, a_c);
  o.x = lookup_unbounded(ltable, o.x / 100.0f, a_l);
  o.yz = clamp(i.yz * saturation, abmin, abmax);
  o.w = i.w;
  out.write(int2(x, y), o);
}

#define UNBOUND_L 1
#define UNBOUND_A 2
#define UNBOUND_B 4
#define UNBOUND_SHADOWS_L UNBOUND_L
#define UNBOUND_SHADOWS_A UNBOUND_A
#define UNBOUND_SHADOWS_B UNBOUND_B
#define UNBOUND_HIGHLIGHTS_L (UNBOUND_L << 3)
#define UNBOUND_HIGHLIGHTS_A (UNBOUND_A << 3)
#define UNBOUND_HIGHLIGHTS_B (UNBOUND_B << 3)

// a holds the image, b the mask
float4 _overlay(const float4 in_a, const float4 in_b, const float opacity, const float transform,
                const float ccorrect, const bool4 unbound, const float low_approximation)
{
  const float4 scale = float4(100.0f, 128.0f, 128.0f, 1.0f);
  const float lmin = 0.0f;
  const float lmax = 1.0f;
  const float halfmax = 0.5f;
  const float doublemax = 2.0f;

  float4 a = in_a / scale;
  const float4 b = in_b / scale;

  float opacity2 = opacity * opacity;

  while(opacity2 > 0.0f)
  {
    const float la = unbound.x ? a.x : clamp(a.x, lmin, lmax);
    float lb = (b.x - halfmax) * float(sign(opacity)) * float(sign(lmax - la)) + halfmax;
    lb = unbound.w ? lb : clamp(lb, lmin, lmax);
    const float lref = copysign(abs(la) > low_approximation ? 1.0f / abs(la) : 1.0f / low_approximation, la);
    const float href = copysign(abs(1.0f - la) > low_approximation ? 1.0f / abs(1.0f - la)
                                                                   : 1.0f / low_approximation, 1.0f - la);

    const float chunk = opacity2 > 1.0f ? 1.0f : opacity2;
    const float optrans = chunk * transform;
    opacity2 -= 1.0f;

    a.x = la * (1.0f - optrans)
      + (la > halfmax ? lmax - (lmax - doublemax * (la - halfmax)) * (lmax - lb) : doublemax * la * lb) * optrans;
    a.x = unbound.x ? a.x : clamp(a.x, lmin, lmax);

    const float chroma = (a.x * lref * ccorrect + (1.0f - a.x) * href * (1.0f - ccorrect)) * optrans;
    a.y = a.y * (1.0f - optrans) + (a.y + b.y) * chroma;
    a.y = unbound.y ? a.y : clamp(a.y, -1.0f, 1.0f);

    a.z = a.z * (1.0f - optrans) + (a.z + b.z) * chroma;
    a.z = unbound.z ? a.z : clamp(a.z, -1.0f, 1.0f);
  }
  return a * scale;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void shadows_highlights_mix(uint3 gid : SV_DispatchThreadID,
                            uniform ImageIn in, uniform ImageIn mask, uniform ImageOut out,
                            uniform int width, uniform int height,
                            uniform float shadows, uniform float highlights, uniform float compress,
                            uniform float shadows_ccorrect, uniform float highlights_ccorrect,
                            uniform uint flags, uniform int unbound_mask, uniform float low_approximation,
                            uniform float whitepoint)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  float4 io = in.Areadpixel(x, y);
  const float w = io.w;

  // blurred, inverted and desaturated mask in m
  float4 m = float4(0.0f);
  m.x = 100.0f - mask.Areadsingle(x, y);

  // white point adjustment
  io.x = io.x > 0.0f ? io.x / whitepoint : io.x;
  m.x = m.x > 0.0f ? m.x / whitepoint : m.x;

  // overlay highlights
  float xform = clipf(1.0f - 0.01f * m.x / (1.0f - compress));
  bool4 unbound = bool4((flags & UNBOUND_HIGHLIGHTS_L) != 0, (flags & UNBOUND_HIGHLIGHTS_A) != 0,
                        (flags & UNBOUND_HIGHLIGHTS_B) != 0, unbound_mask != 0);
  io = _overlay(io, m, -highlights, xform, 1.0f - highlights_ccorrect, unbound, low_approximation);

  // overlay shadows
  xform = clipf(0.01f * m.x / (1.0f - compress) - compress / (1.0f - compress));
  unbound = bool4((flags & UNBOUND_SHADOWS_L) != 0, (flags & UNBOUND_SHADOWS_A) != 0,
                  (flags & UNBOUND_SHADOWS_B) != 0, unbound_mask != 0);
  io = _overlay(io, m, shadows, xform, shadows_ccorrect, unbound, low_approximation);

  io.w = w;
  out.write(int2(x, y), io);
}
