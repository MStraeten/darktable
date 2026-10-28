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


// Slang port of ../lut3d.cl, the three OpenCL kernels in one: interpolation
// is 0 for tetrahedral, 1 trilinear and 2 pyramid, as dt_iop_lut3d_interpolation_t

import common;

float4 _clut(StructuredBuffer<float> clut, const int i)
{
  return float4(clut[i], clut[i + 1], clut[i + 2], 0.0f);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void lut3d(uint3 gid : SV_DispatchThreadID,
           uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
           uniform StructuredBuffer<float> clut, uniform int level, uniform int interpolation)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float4 input = clip4(in.Areadpixel(x, y));
  const int level2 = level * level;

  float4 rgbd = input * float(level - 1);
  const int4 rgbi = clamp(int4(rgbd), int4(0), int4(level - 2));
  rgbd -= float4(rgbi);

  // indexes of P000 to P111 in clut
  const int color = rgbi.x + rgbi.y * level + rgbi.z * level2;
  const int i000 = color * 3;
  const int i010 = (color + level) * 3;
  const int i001 = (color + level2) * 3;
  const int i011 = (color + level + level2) * 3;
  const float4 c000 = _clut(clut, i000);
  const float4 c100 = _clut(clut, i000 + 3);
  const float4 c010 = _clut(clut, i010);
  const float4 c110 = _clut(clut, i010 + 3);
  const float4 c001 = _clut(clut, i001);
  const float4 c101 = _clut(clut, i001 + 3);
  const float4 c011 = _clut(clut, i011);
  const float4 c111 = _clut(clut, i011 + 3);

  float4 output;
  if(interpolation == 0)
  {
    if(rgbd.x > rgbd.y)
    {
      if(rgbd.y > rgbd.z)
        output = (1.0f - rgbd.x) * c000 + (rgbd.x - rgbd.y) * c100 + (rgbd.y - rgbd.z) * c110 + rgbd.z * c111;
      else if(rgbd.x > rgbd.z)
        output = (1.0f - rgbd.x) * c000 + (rgbd.x - rgbd.z) * c100 + (rgbd.z - rgbd.y) * c101 + rgbd.y * c111;
      else
        output = (1.0f - rgbd.z) * c000 + (rgbd.z - rgbd.x) * c001 + (rgbd.x - rgbd.y) * c101 + rgbd.y * c111;
    }
    else
    {
      if(rgbd.z > rgbd.y)
        output = (1.0f - rgbd.z) * c000 + (rgbd.z - rgbd.y) * c001 + (rgbd.y - rgbd.x) * c011 + rgbd.x * c111;
      else if(rgbd.z > rgbd.x)
        output = (1.0f - rgbd.y) * c000 + (rgbd.y - rgbd.z) * c010 + (rgbd.z - rgbd.x) * c011 + rgbd.x * c111;
      else
        output = (1.0f - rgbd.y) * c000 + (rgbd.y - rgbd.x) * c010 + (rgbd.x - rgbd.z) * c110 + rgbd.z * c111;
    }
  }
  else if(interpolation == 1)
  {
    const float4 t1 = c000 * (1.0f - rgbd.x) + c100 * rgbd.x;
    const float4 t2 = c010 * (1.0f - rgbd.x) + c110 * rgbd.x;
    const float4 t3 = c001 * (1.0f - rgbd.x) + c101 * rgbd.x;
    const float4 t4 = c011 * (1.0f - rgbd.x) + c111 * rgbd.x;
    output = (t1 * (1.0f - rgbd.y) + t2 * rgbd.y) * (1.0f - rgbd.z)
             + (t3 * (1.0f - rgbd.y) + t4 * rgbd.y) * rgbd.z;
  }
  else
  {
    if(rgbd.y > rgbd.x && rgbd.z > rgbd.x)
      output = c000 + (c111 - c011) * rgbd.x + (c010 - c000) * rgbd.y + (c001 - c000) * rgbd.z
               + (c011 - c001 - c010 + c000) * rgbd.y * rgbd.z;
    else if(rgbd.x > rgbd.y && rgbd.z > rgbd.y)
      output = c000 + (c100 - c000) * rgbd.x + (c111 - c101) * rgbd.y + (c001 - c000) * rgbd.z
               + (c101 - c001 - c100 + c000) * rgbd.x * rgbd.z;
    else
      output = c000 + (c100 - c000) * rgbd.x + (c010 - c000) * rgbd.y + (c111 - c110) * rgbd.z
               + (c110 - c100 - c010 + c000) * rgbd.x * rgbd.y;
  }
  output.w = input.w;
  out.write(int2(x, y), output);
}
