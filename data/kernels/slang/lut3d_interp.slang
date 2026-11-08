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

// Slang port of ../lut3d.h: tetrahedral interpolation in a level^3 color LUT
// of 3 floats per node, the first channel varying fastest. The coordinates in
// rgb must already lie in [0, 1]. Shared by lut3d.slang and the LittleCMS
// path of color in and color out in basic.slang.

module lut3d_interp;

float4 _lut3d_node(StructuredBuffer<float> clut, const int i)
{
  return float4(clut[i], clut[i + 1], clut[i + 2], 0.0f);
}

public float4 dt_lut3d_tetrahedral(const float4 rgb, StructuredBuffer<float> clut, const int level)
{
  const int level2 = level * level;

  float4 rgbd = rgb * float(level - 1);
  const int4 rgbi = clamp(int4(rgbd), int4(0), int4(level - 2));
  rgbd -= float4(rgbi);

  // indexes of P000 to P111 in clut
  const int color = rgbi.x + rgbi.y * level + rgbi.z * level2;
  const int i000 = color * 3;
  const int i010 = (color + level) * 3;
  const int i001 = (color + level2) * 3;
  const int i011 = (color + level + level2) * 3;
  const float4 c000 = _lut3d_node(clut, i000);
  const float4 c100 = _lut3d_node(clut, i000 + 3);
  const float4 c010 = _lut3d_node(clut, i010);
  const float4 c110 = _lut3d_node(clut, i010 + 3);
  const float4 c001 = _lut3d_node(clut, i001);
  const float4 c101 = _lut3d_node(clut, i001 + 3);
  const float4 c011 = _lut3d_node(clut, i011);
  const float4 c111 = _lut3d_node(clut, i011 + 3);

  if(rgbd.x > rgbd.y)
  {
    if(rgbd.y > rgbd.z)
      return (1.0f - rgbd.x) * c000 + (rgbd.x - rgbd.y) * c100 + (rgbd.y - rgbd.z) * c110 + rgbd.z * c111;
    else if(rgbd.x > rgbd.z)
      return (1.0f - rgbd.x) * c000 + (rgbd.x - rgbd.z) * c100 + (rgbd.z - rgbd.y) * c101 + rgbd.y * c111;
    else
      return (1.0f - rgbd.z) * c000 + (rgbd.z - rgbd.x) * c001 + (rgbd.x - rgbd.y) * c101 + rgbd.y * c111;
  }
  else
  {
    if(rgbd.z > rgbd.y)
      return (1.0f - rgbd.z) * c000 + (rgbd.z - rgbd.y) * c001 + (rgbd.y - rgbd.x) * c011 + rgbd.x * c111;
    else if(rgbd.z > rgbd.x)
      return (1.0f - rgbd.y) * c000 + (rgbd.y - rgbd.z) * c010 + (rgbd.z - rgbd.x) * c011 + rgbd.x * c111;
    else
      return (1.0f - rgbd.y) * c000 + (rgbd.y - rgbd.x) * c010 + (rgbd.x - rgbd.z) * c110 + rgbd.z * c111;
  }
}
