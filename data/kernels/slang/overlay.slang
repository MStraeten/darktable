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


// Slang port of ../overlay.cl. The overlay is drawn on the host, as for
// OpenCL, and only its blend runs on the device.

import common;

// a straight-alpha float RGBA overlay in the working RGB, 4 floats per pixel
[shader("compute")]
[numthreads(16, 16, 1)]
void overlay_blend(uint3 gid : SV_DispatchThreadID,
                   uniform ImageIn in, uniform StructuredBuffer<float4> overlay_rgba, uniform ImageOut out,
                   uniform int width, uniform int height, uniform float opacity)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float4 i = in.Areadpixel(x, y);
  const float4 s = overlay_rgba[y * width + x];
  const float a = s.w * opacity;
  float4 o = (1.0f - a) * i + a * s;
  o.w = i.w;
  out.write(int2(x, y), o);
}

// the legacy 8 bit Cairo ARGB32 overlay, bytes B, G, R, A at each pixel with
// stride bytes per row, read as one uint per pixel
[shader("compute")]
[numthreads(16, 16, 1)]
void overlay_blend_legacy(uint3 gid : SV_DispatchThreadID,
                          uniform ImageIn in, uniform StructuredBuffer<uint> overlay_argb, uniform ImageOut out,
                          uniform int width, uniform int height, uniform float opacity, uniform int stride)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float4 i = in.Areadpixel(x, y);
  const uint v = overlay_argb[(y * stride) / 4 + x];
  const float b = float(v & 0xffu) / 255.0f;
  const float g = float((v >> 8) & 0xffu) / 255.0f;
  const float r = float((v >> 16) & 0xffu) / 255.0f;
  const float a = float(v >> 24) / 255.0f * opacity;
  float4 o = (1.0f - a) * i + opacity * float4(r, g, b, 0.0f);
  o.w = i.w;
  out.write(int2(x, y), o);
}
