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

// Slang port of the VNG kernels in ../demosaic_vng.cl. Like the OpenCL
// kernels, these stage the neighborhood of their work group in group shared
// memory, filled through the clamping sampler.

import common;

// the low and high 16 bits of a packed offset, both signed
int _off_x(int offset) { return (offset << 16) >> 16; }
int _off_y(int offset) { return offset >> 16; }

// v in channel c, zero in the others
float4 _channel(int c, float v)
{
  return float4(c == 0 ? v : 0.0f, c == 1 ? v : 0.0f, c == 2 ? v : 0.0f, c == 3 ? v : 0.0f);
}

// p with channel c set to v
float4 _with_channel(float4 p, int c, float v)
{
  return float4(c == 0 ? v : p.x, c == 1 ? v : p.y, c == 2 ? v : p.z, c == 3 ? v : p.w);
}

// a 16x16 work group with a margin of 1 (linear) or 2 pixels
groupshared float vng_lin_tile[18 * 18];
groupshared float4 vng_tile[20 * 20];

[shader("compute")]
[numthreads(16, 16, 1)]
void vng_lin_interpolate(uint3 gid : SV_DispatchThreadID, uint3 lid : SV_GroupThreadID,
                         uniform ImageIn in, uniform ImageOut out,
                         uniform int width, uniform int height, uniform int border,
                         uniform uint filters, uniform StructuredBuffer<uint> xtrans,
                         uniform StructuredBuffer<int> lookup, uniform int only_linear)
{
  const int x = gid.x;
  const int y = gid.y;

  // with a border, a work group inside it has nothing to do. The test is the
  // same for the whole group, which must reach the barrier together
  const int x0 = x - int(lid.x);
  const int y0 = y - int(lid.y);
  if(border > 0 && x0 > border && y0 > border && x0 + 15 <= width - border && y0 + 15 <= height - border)
    return;

  for(int k = int(lid.y * 16 + lid.x); k < 18 * 18; k += 256)
    vng_lin_tile[k] = in.readsingle(x0 - 1 + k % 18, y0 - 1 + k / 18);
  GroupMemoryBarrierWithGroupSync();

  if(x >= width || y >= height) return;

  if(border > 0 && x > border && y > border && x <= width - border && y <= height - border)
    return;

  const bool is_xtrans = filters == 9;
  const int colors = is_xtrans ? 3 : 4;
  const int size = is_xtrans ? 6 : 16;

  float sum[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
  float o[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

  // approximate outermost single pixels
  if(x < 1 || x >= width-1 || y < 1 || y >= height-1)
  {
    int count[4] = { 0, 0, 0, 0 };
    for(int j = y-1; j <= y+1; j++)
    {
      for(int i = x-1; i <= x+1; i++)
      {
        if(j >= 0 && i >= 0 && j < height && i < width)
        {
          const int f = fcol(j, i, filters, xtrans);
          sum[f] += max(0.0f, in.readsingle(i, j));
          count[f]++;
        }
      }
    }
    const int f = fcol(y, x, filters, xtrans);
    // for current cell, copy the current sensor's color data,
    // interpolate the other two colors from surrounding pixels of
    // their color
    for(int c = 0; c < colors; c++)
    {
      if(c != f && count[c] != 0)
        o[c] = fp32_div(sum[c], float(count[c]));
      else
        o[c] = max(0.0f, in.readsingle(x, y));
    }
  }
  else // do the bilinear stuff
  {
    // lookup[16][16][32]
    int ip = ((y % size) * 16 + (x % size)) * 32;
    const int num_pixels = lookup[ip];
    ip++;

    // for each adjoining pixel not of this pixel's color, sum up its weighted
    // values. The sums are a vector: an array indexed by table entries would
    // live in scratch memory on Metal, which made this kernel slow there
    float4 sums = float4(0.0f);
    for(int i = 0; i < num_pixels; i++, ip += 3)
    {
      const int offset = lookup[ip];
      const float v = max(0.0f, vng_lin_tile[(int(lid.y) + 1 + _off_y(offset)) * 18 + int(lid.x) + 1 + _off_x(offset)]);
      sums += _channel(lookup[ip + 2], v * lookup[ip + 1]);
    }
    // for each interpolated color, load it into the pixel
    float4 out4 = float4(0.0f);
    for(int i = 0; i < colors - 1; i++, ip += 2)
    {
      const int c = lookup[ip];
      out4 = _with_channel(out4, c, fp32_div(sums[c], float(lookup[ip + 1])));
    }
    out4 = _with_channel(out4, lookup[ip], max(0.0f, vng_lin_tile[(int(lid.y) + 1) * 18 + int(lid.x) + 1]));
    o[0] = out4.x;
    o[1] = out4.y;
    o[2] = out4.z;
    o[3] = out4.w;
  }

  if(only_linear != 0 && !is_xtrans)
  {
    o[1] = 0.5f * (o[1] + o[3]);
    o[3] = 0.0f;
  }
  out.write(int2(x, y), max(float4(0.0f), float4(o[0], o[1], o[2], o[3])));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void vng_interpolate(uint3 gid : SV_DispatchThreadID, uint3 lid : SV_GroupThreadID,
                     uniform ImageIn in, uniform ImageOut out,
                     uniform int width, uniform int height,
                     uniform uint filters, uniform StructuredBuffer<uint> xtrans,
                     uniform StructuredBuffer<int> ips, uniform StructuredBuffer<int> code)
{
  const int x = gid.x;
  const int y = gid.y;

  const int x0 = x - int(lid.x);
  const int y0 = y - int(lid.y);
  for(int k = int(lid.y * 16 + lid.x); k < 20 * 20; k += 256)
    vng_tile[k] = in.readpixel(x0 - 2 + k % 20, y0 - 2 + k / 20);
  GroupMemoryBarrierWithGroupSync();

  if(x >= width || y >= height) return;

  // the pixel at offset dx, dy from this one
  const int t0 = (int(lid.y) + 2) * 20 + int(lid.x) + 2;
#define PX(dx, dy) vng_tile[t0 + (dy) * 20 + (dx)]

  const bool is_xtrans = filters == 9;
  // we don't touch data at the outermost 2 pixels
  if(x < 2 || x >= width-2 || y < 2 || y >= height-2)
  {
    float4 val = PX(0, 0);
    if(!is_xtrans)
    {
      val.y = 0.5f * (val.y + val.w);
      val.w = 0.0f;
    }
    out.write(int2(x, y), max(float4(0.0f), val));
    return;
  }

  const int colors = is_xtrans ? 3 : 4;
  const int prow = is_xtrans ? 6 : 8;
  const int pcol = is_xtrans ? 6 : 2;

  // the gradients 0-3 and 4-7. Arrays indexed by table entries would live in
  // scratch memory on Metal, which made this kernel slow there
  float4 ga = float4(0.0f);
  float4 gb = float4(0.0f);

  // get byte code, code[16][16]
  int ip = code[(y % prow) * 16 + (x % pcol)];

  // the eight neighbors after the gradient terms, three ints each. They are
  // read inside the loop: lavapipe (Mesa 25.2, LLVM 20) crashes running the
  // shader when ips is read again after it
  int nb[24];
  for(int k = 0; k < 24; k++) nb[k] = 0;

  // calculate gradients, at most 64 terms as the host builds the table
  for(int t = 0; t < 64; t++)
  {
    const int offset0 = ips[ip];
    if(offset0 == 0x7fffffff)
    {
      for(int k = 0; k < 24; k++) nb[k] = ips[ip + 1 + k];
      break;
    }
    const int offset1 = ips[ip + 1];
    const int weight = _off_x(ips[ip + 2]);
    const int color = _off_y(ips[ip + 2]);

    const float4 p0 = PX(_off_x(offset0), _off_y(offset0));
    const float4 p1 = PX(_off_x(offset1), _off_y(offset1));
    const float diff = abs(p0[color] - p1[color]) * weight;

    // the gradients this term adds to, each once, up to a -1
    uint mask = 1u << uint(ips[ip + 3]);
    ip += 4;
    for(int k = 0; k < 8; k++)
    {
      const int gi = ips[ip++];
      if(gi == -1) break;
      mask |= 1u << uint(gi);
    }
    ga += float4((mask & 1u) != 0u ? diff : 0.0f, (mask & 2u) != 0u ? diff : 0.0f,
                 (mask & 4u) != 0u ? diff : 0.0f, (mask & 8u) != 0u ? diff : 0.0f);
    gb += float4((mask & 16u) != 0u ? diff : 0.0f, (mask & 32u) != 0u ? diff : 0.0f,
                 (mask & 64u) != 0u ? diff : 0.0f, (mask & 128u) != 0u ? diff : 0.0f);
  }
  const float gval[8] = { ga.x, ga.y, ga.z, ga.w, gb.x, gb.y, gb.z, gb.w };

  // chose a threshold
  float gmin = gval[0];
  float gmax = gval[0];
  [ForceUnroll]
  for(int g = 1; g < 8; g++)
  {
    if(gmin > gval[g]) gmin = gval[g];
    if(gmax < gval[g]) gmax = gval[g];
  }

  const float4 center = PX(0, 0);
  float o[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
  if(gmax == 0.0f)
  {
    o[0] = center.x;
    o[1] = is_xtrans ? center.y : 0.5f * (center.y + center.w);
    o[2] = center.z;
    o[3] = 0.0f;
    out.write(int2(x, y), max(float4(0.0f), float4(o[0], o[1], o[2], o[3])));
    return;
  }

  const float thold = gmin + (gmax * 0.5f);
  float4 sum = float4(0.0f);
  const int color = fcol(y, x, filters, xtrans);
  int num = 0;

  // average the neighbors
  [ForceUnroll]
  for(int g = 0; g < 8; g++)
  {
    if(gval[g] <= thold)
    {
      const int offset0 = nb[3 * g];
      const int offset1 = nb[3 * g + 1];
      const int c1 = nb[3 * g + 2];
      const int x1 = _off_x(offset1);
      const int y1 = _off_y(offset1);
      const float4 p0 = PX(_off_x(offset0), _off_y(offset0));
      const float4 p1 = PX(x1, y1);

      [ForceUnroll]
      for(int c = 0; c < 4; c++)
      {
        if(c >= colors) continue;
        // OpenCL tests the local buffer offset idx1 + c1 for zero, which
        // only the center pixel's first channel gives
        if(c == color && (x1 != 0 || y1 != 0 || c1 != 0))
          sum[c] += (center[c] + p1[c1]) * 0.5f;
        else
          sum[c] += p0[c];
      }
      num++;
    }
  }

  // save to output
  [ForceUnroll]
  for(int c = 0; c < 4; c++)
  {
    if(c >= colors) continue;
    float tot = center[color];
    if(c != color) tot += fp32_div(sum[c] - sum[color], float(num));
    o[c] = tot;
  }

  o[1] = is_xtrans ? o[1] : 0.5f * (o[1] + o[3]);
  o[3] = 0.0f;
  out.write(int2(x, y), max(float4(0.0f), float4(o[0], o[1], o[2], 0.0f)));
#undef PX
}
