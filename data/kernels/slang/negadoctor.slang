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

// Slang port of ../negadoctor.cl. Entry point arguments keep the order of the
// OpenCL kernel arguments, so host code can bind both the same way.

import common;

[shader("compute")]
[numthreads(16, 16, 1)]
void negadoctor(uint3 gid : SV_DispatchThreadID,
                uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
                uniform float4 Dmin, uniform float4 wb_high, uniform float4 offset,
                uniform float exposure, uniform float black, uniform float gamma,
                uniform float soft_clip, uniform float soft_clip_comp)
{
  const int x = gid.x;
  const int y = gid.y;

  if(x >= width || y >= height) return;

  const float4 i = in.read(int2(x, y));

  // Convert transmission to density using Dmin as a fulcrum
  float4 o = -dtcl_log10(Dmin / max(i, float4(2.3283064365386963e-10f))); // threshold to -32 EV

  // Correct density in log space
  o = wb_high * o + offset;

  // Print density on paper : ((1 - 10^corrected_de + black) * exposure)^gamma rewritten for FMA
  o = -(float4(exposure) * dtcl_exp10(o) + float4(black));
  o = dtcl_pow(max(o, float4(0.0f)), gamma); // note : this is always > 0

  // Compress highlights and clip negatives. from https://lists.gnu.org/archive/html/openexr-devel/2005-03/msg00009.html
  o = select(o > float4(soft_clip),
             soft_clip + (float4(1.0f) - dtcl_exp(-(o - float4(soft_clip)) / float4(soft_clip_comp))) * float4(soft_clip_comp),
             o);

  // Copy alpha
  o.w = i.w;

  out.write(int2(x, y), o);
}
