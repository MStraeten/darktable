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

// the heal solver of src/common/heal.c, as retouch_heal_* in ../retouch.cl.
// The image difference is split into 'red' and 'black' checkerboard pixels,
// each in its own buffer of subwidth pixels per row plus a zero padding row
// at the top and bottom. A pixel of split row r (1..height) and column j sits
// at original row r-1 and column 2j + (cs ^ ((r-1) & 1)), with cs 1 for red
// and 0 for black.
int _heal_col(int j, int r, int cs)
{
  return 2 * j + (cs ^ ((r - 1) & 1));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void retouch_heal_sub(uint3 gid : SV_DispatchThreadID,
                      uniform ImageIn top, uniform ImageIn bottom, uniform RWStructuredBuffer<float4> split,
                      uniform int width, uniform int height, uniform int subwidth, uniform int cs)
{
  const int j = gid.x;
  const int r = gid.y;
  if(j >= subwidth || r >= height + 2) return;

  float4 val = float4(0.0f);
  if(r > 0 && r <= height)
  {
    const int col = _heal_col(j, r, cs);
    if(col < width) val = top.Areadpixel(col, r - 1) - bottom.Areadpixel(col, r - 1);
  }
  split[r * subwidth + j] = val;
}

groupshared float heal_sums[256];

// one half of a Gauss-Seidel iteration with over-relaxation, as
// _heal_laplace_iteration(): updates the masked pixels of one color and
// writes the work group's sum of squared changes. Does nothing once
// retouch_heal_converged() has set the flag.
[shader("compute")]
[numthreads(16, 16, 1)]
void retouch_heal_iterate(uint3 gid : SV_DispatchThreadID, uint3 lid : SV_GroupThreadID,
                          uint3 grp : SV_GroupID,
                          uniform RWStructuredBuffer<float4> active, uniform RWStructuredBuffer<float4> neighbor,
                          uniform ImageIn mask, uniform int width, uniform int height, uniform int subwidth,
                          uniform int cs, uniform float w, uniform RWStructuredBuffer<int> converged,
                          uniform RWStructuredBuffer<float> partial)
{
  if(converged[0] != 0) return;

  const int j = gid.x;
  const int r = gid.y + 1;
  const int l = lid.y * 16 + lid.x;

  float err = 0.0f;
  if(j < subwidth && r <= height)
  {
    const int col = _heal_col(j, r, cs);
    if(col < width && mask.Areadpixel(col, r - 1).x != 0.0f)
    {
      const int idx = r * subwidth + j;
      const int lro = cs ^ ((r - 1) & 1);
      float a = 4.0f;
      if(r == 1) a -= 1.0f;
      if(r == height) a -= 1.0f;
      // the first and last pixel of a row have no left or right neighbor
      float4 left = float4(0.0f);
      float4 right = float4(0.0f);
      if(j > 0 || lro != 0)
        left = neighbor[idx - 1 + lro];
      else
        a -= 1.0f;
      if(j + 1 < subwidth || lro == 0)
        right = neighbor[idx + lro];
      else
        a -= 1.0f;

      const float4 act = active[idx];
      const float4 diff = w * ((a * act) - (neighbor[idx - subwidth] + neighbor[idx + subwidth] + left + right));
      active[idx] = act - diff;
      err = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
    }
  }

  heal_sums[l] = err;
  GroupMemoryBarrierWithGroupSync();
  for(int offset = 128; offset > 0; offset >>= 1)
  {
    if(l < offset) heal_sums[l] += heal_sums[l + offset];
    GroupMemoryBarrierWithGroupSync();
  }
  if(l == 0) partial[grp.y * ((subwidth + 15) / 16) + grp.x] = heal_sums[0];
}

// adds up the partial sums of both colors in one 16x16 work group and sets
// the flag when the iteration has converged
[shader("compute")]
[numthreads(16, 16, 1)]
void retouch_heal_converged(uint3 lid : SV_GroupThreadID,
                            uniform RWStructuredBuffer<float> partial_black,
                            uniform RWStructuredBuffer<float> partial_red,
                            uniform int groups, uniform float err_exit,
                            uniform RWStructuredBuffer<int> converged)
{
  if(converged[0] != 0) return;

  const int l = lid.y * 16 + lid.x;
  float sum = 0.0f;
  for(int k = l; k < groups; k += 256)
    sum += partial_black[k] + partial_red[k];

  heal_sums[l] = sum;
  GroupMemoryBarrierWithGroupSync();
  for(int offset = 128; offset > 0; offset >>= 1)
  {
    if(l < offset) heal_sums[l] += heal_sums[l + offset];
    GroupMemoryBarrierWithGroupSync();
  }
  if(l == 0 && heal_sums[0] < err_exit) converged[0] = 1;
}

// _heal_add(): the solution plus bottom, back in the original layout
[shader("compute")]
[numthreads(16, 16, 1)]
void retouch_heal_add(uint3 gid : SV_DispatchThreadID,
                      uniform RWStructuredBuffer<float4> red, uniform RWStructuredBuffer<float4> black,
                      uniform ImageIn bottom, uniform ImageOut dest,
                      uniform int width, uniform int height, uniform int subwidth)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  // the pixel's color: red sits at odd columns of even rows
  const int idx = (y + 1) * subwidth + x / 2;
  const int is_red = (x & 1) ^ (y & 1);
  dest.write(int2(x, y), (is_red != 0 ? red[idx] : black[idx]) + bottom.Areadpixel(x, y));
}
