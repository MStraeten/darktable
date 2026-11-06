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

// Slang port of the kernels in ../demosaic_other.cl.

import common;

// the one channel in all of r, g, b and a
[shader("compute")]
[numthreads(16, 16, 1)]
void passthrough_monochrome(uint3 gid : SV_DispatchThreadID,
                            uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  out.write(int2(x, y), float4(in.Areadsingle(x, y)));
}

// each photosite in its own color channel
[shader("compute")]
[numthreads(16, 16, 1)]
void passthrough_color(uint3 gid : SV_DispatchThreadID,
                       uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
                       uniform uint filters, uniform StructuredBuffer<uint> xtrans)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float ival = in.Areadsingle(x, y);
  const int c = fcol(y, x, filters, xtrans);

  float4 oval = float4(0.0f);
  if(c == 0)      oval.x = ival;
  else if(c == 1) oval.y = ival;
  else            oval.z = ival;
  out.write(int2(x, y), oval);
}

// downscales a mosaic to a monochrome image of the output size
[shader("compute")]
[numthreads(16, 16, 1)]
void clip_and_zoom_demosaic_passthrough_monochrome(uint3 gid : SV_DispatchThreadID,
                                                   uniform ImageIn in, uniform ImageOut out,
                                                   uniform int width, uniform int height,
                                                   uniform int rin_wd, uniform int rin_ht, uniform float r_scale)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  float4 color = float4(0.0f);
  float weight = 0.0f;

  // pixel footprint on input buffer, and how many pixels fit in it
  const float px_footprint = 1.0f / r_scale;
  const int samples = cl_round(px_footprint);

  const float2 f = float2(x * px_footprint, y * px_footprint);
  const int2 p = int2(int(f.x), int(f.y));
  const float2 d = float2(f.x - p.x, f.y - p.y);

  for(int j = 0; j <= samples + 1; j++)
    for(int i = 0; i <= samples + 1; i++)
    {
      const int xx = p.x + i;
      const int yy = p.y + j;

      const float xfilter = (i == 0) ? 1.0f - d.x : ((i == samples + 1) ? d.x : 1.0f);
      const float yfilter = (j == 0) ? 1.0f - d.y : ((j == samples + 1) ? d.y : 1.0f);

      const float px = in.readsingle(xx, yy);
      color += yfilter * xfilter * float4(px, px, px, 0.0f);
      weight += yfilter * xfilter;
    }
  color = (weight > 0.0f) ? color / weight : float4(0.0f);
  out.write(int2(x, y), color);
}
