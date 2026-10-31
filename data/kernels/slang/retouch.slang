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

// Slang port of ../retouch.cl, following the CPU code in src/iop/retouch.c
// where the two differ. A kernel can't read and write the same image, so the
// masked blends write the rectangle under the mask to an image of its own,
// and retouch_rect_to_layer() puts it back into the layer.

import common;

// rt_copy_in_to_out(): out(x, y) = in(x + xoffs, y + yoffs), or the rotated
// and bilinear-sampled source around the pivot (cx, cy) when rotate is set
[shader("compute")]
[numthreads(16, 16, 1)]
void retouch_copy_in_to_out(uint3 gid : SV_DispatchThreadID,
                            uniform ImageIn in, uniform ImageOut out,
                            uniform int width, uniform int height,
                            uniform int in_width, uniform int in_height,
                            uniform int xoffs, uniform int yoffs, uniform int rotate,
                            uniform float angle, uniform float cx, uniform float cy)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  if(rotate == 0)
  {
    out.write(int2(x, y), in.read(int2(x + xoffs, y + yoffs)));
    return;
  }

  const float c = cos(angle);
  const float s = sin(angle);
  const float cx_source = cx + float(xoffs);
  const float cy_source = cy + float(yoffs);

  const float rx = float(x + xoffs) - cx_source;
  const float ry = float(y + yoffs) - cy_source;
  const float ix = cx_source + rx * c - ry * s;
  const float iy = cy_source + rx * s + ry * c;

  // edge-clamp (replicate) the samples outside roi_in, as the CPU does
  const float ixc = clamp(ix, 0.0f, float(in_width - 1));
  const float iyc = clamp(iy, 0.0f, float(in_height - 1));
  const int x0 = clamp(int(ixc), 0, in_width - 2);
  const int y0 = clamp(int(iyc), 0, in_height - 2);
  const float dx0 = ixc - float(x0);
  const float dy0 = iyc - float(y0);

  const float4 in00 = in.read(int2(x0, y0));
  const float4 in10 = in.read(int2(x0 + 1, y0));
  const float4 in01 = in.read(int2(x0, y0 + 1));
  const float4 in11 = in.read(int2(x0 + 1, y0 + 1));

  out.write(int2(x, y), in00 * (1.0f - dx0) * (1.0f - dy0) + in10 * dx0 * (1.0f - dy0)
                        + in01 * (1.0f - dx0) * dy0 + in11 * dx0 * dy0);
}

// rt_copy_image_masked(): the layer blended with src under the mask, all four
// channels, for the rectangle at (ox, oy) in the layer
[shader("compute")]
[numthreads(16, 16, 1)]
void retouch_blend_masked(uint3 gid : SV_DispatchThreadID,
                          uniform ImageIn layer, uniform ImageIn src, uniform ImageIn mask,
                          uniform ImageOut out, uniform int width, uniform int height,
                          uniform int ox, uniform int oy, uniform float opacity)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float f = mask.Areadpixel(x, y).x * opacity;
  out.write(int2(x, y), layer.read(int2(x + ox, y + oy)) * (1.0f - f) + src.Areadpixel(x, y) * f);
}

// _retouch_fill(). The CPU fills the alpha channel with 0 too, where
// ../retouch.cl keeps it
[shader("compute")]
[numthreads(16, 16, 1)]
void retouch_fill_masked(uint3 gid : SV_DispatchThreadID,
                         uniform ImageIn layer, uniform ImageIn mask, uniform ImageOut out,
                         uniform int width, uniform int height, uniform int ox, uniform int oy,
                         uniform float opacity, uniform float4 color)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float f = mask.Areadpixel(x, y).x * opacity;
  const float4 d = layer.read(int2(x + ox, y + oy));
  out.write(int2(x, y), d * (1.0f - f) + float4(color.xyz, 0.0f) * f);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void retouch_rect_to_layer(uint3 gid : SV_DispatchThreadID,
                           uniform ImageIn rect, uniform ImageOut layer,
                           uniform int width, uniform int height, uniform int ox, uniform int oy)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  layer.write(int2(x + ox, y + oy), rect.Areadpixel(x, y));
}

// rt_copy_mask_to_alpha(): the mask, for the mask view, raised into the
// alpha channel of the rectangle at (ox, oy) in the layer
[shader("compute")]
[numthreads(16, 16, 1)]
void retouch_mask_to_alpha(uint3 gid : SV_DispatchThreadID,
                           uniform ImageIn layer, uniform ImageIn mask, uniform ImageOut out,
                           uniform int width, uniform int height, uniform int ox, uniform int oy,
                           uniform float opacity)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float f = mask.Areadpixel(x, y).x * opacity;
  float4 d = layer.read(int2(x + ox, y + oy));
  if(f > d.w) d.w = f;
  out.write(int2(x, y), d);
}

// the color of rgb with the alpha channel of alpha, or 0 where clear is set
[shader("compute")]
[numthreads(16, 16, 1)]
void retouch_set_alpha(uint3 gid : SV_DispatchThreadID,
                       uniform ImageIn rgb, uniform ImageIn alpha, uniform ImageOut out,
                       uniform int width, uniform int height, uniform int clear)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float4 d = rgb.read(int2(x, y));
  out.write(int2(x, y), float4(d.xyz, clear != 0 ? 0.0f : alpha.read(int2(x, y)).w));
}
