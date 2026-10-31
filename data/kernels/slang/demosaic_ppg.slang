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

// Slang port of the kernels in ../demosaic_ppg.cl, which PPG, the borders of
// RCD and the low quality thumbnails use. The OpenCL kernels stage their neighborhood in local memory; these
// read it from the image, through the same clamped sampler, which gives the
// same values and leaves the caching to the texture units.

import common;

[shader("compute")]
[numthreads(16, 16, 1)]
void ppg_demosaic_green(uint3 gid : SV_DispatchThreadID,
                        uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
                        uniform uint filters, uniform int border)
{
  const int col = gid.x;
  const int row = gid.y;

  if(col >= width - 3 || col < 3 || row >= height - 3 || row < 3) return;
  if(col >= border && col < width - border && row >= border && row < height - border) return;

  // process all non-green pixels
  const int c = FC(row, col, filters);
  float4 color = float4(0.0f); // output color

  const float pc = in.readsingle(col, row);

  if     (c == RED) color.x = pc;
  else if(c == GREEN) color.y = pc;
  else if(c == BLUE) color.z = pc;
  else            color.y = pc; // green2

  // fill green layer for red and blue pixels:
  if(c == RED || c == BLUE)
  {
    // look up horizontal and vertical neighbours, sharpened weight:
    const float pym  = in.readsingle(col, row - 1);
    const float pym2 = in.readsingle(col, row - 2);
    const float pym3 = in.readsingle(col, row - 3);
    const float pyM  = in.readsingle(col, row + 1);
    const float pyM2 = in.readsingle(col, row + 2);
    const float pyM3 = in.readsingle(col, row + 3);
    const float pxm  = in.readsingle(col - 1, row);
    const float pxm2 = in.readsingle(col - 2, row);
    const float pxm3 = in.readsingle(col - 3, row);
    const float pxM  = in.readsingle(col + 1, row);
    const float pxM2 = in.readsingle(col + 2, row);
    const float pxM3 = in.readsingle(col + 3, row);
    const float guessx = (pxm + pc + pxM) * 2.0f - pxM2 - pxm2;
    const float diffx  = (abs(pxm2 - pc) +
                          abs(pxM2 - pc) +
                          abs(pxm  - pxM)) * 3.0f +
                         (abs(pxM3 - pxM) + abs(pxm3 - pxm)) * 2.0f;
    const float guessy = (pym + pc + pyM) * 2.0f - pyM2 - pym2;
    const float diffy  = (abs(pym2 - pc) +
                          abs(pyM2 - pc) +
                          abs(pym  - pyM)) * 3.0f +
                         (abs(pyM3 - pyM) + abs(pym3 - pym)) * 2.0f;
    if(diffx > diffy)
    {
      // use guessy
      const float m = min(pym, pyM);
      const float M = max(pym, pyM);
      color.y = max(min(guessy*0.25f, M), m);
    }
    else
    {
      const float m = min(pxm, pxM);
      const float M = max(pxm, pxM);
      color.y = max(min(guessx*0.25f, M), m);
    }
  }
  out.write(int2(col, row), max(float4(0.0f), color));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void ppg_demosaic_redblue(uint3 gid : SV_DispatchThreadID,
                          uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
                          uniform uint filters, uniform int border)
{
  // image in contains full green and sparse r b
  const int col = gid.x;
  const int row = gid.y;

  if(col >= width || row >= height) return;
  if(col >= border && col < width - border && row >= border && row < height - border) return;

  const int c = FC(row, col, filters);
  float4 color = in.readpixel(col, row);
  if(row > 0 && col > 0 && col < width - 1 && row < height - 1)
  {
    if(c == GREEN || c == 3)
    { // calculate red and blue for green pixels:
      // need 4-nbhood:
      const float4 nt = in.readpixel(col, row - 1);
      const float4 nb = in.readpixel(col, row + 1);
      const float4 nl = in.readpixel(col - 1, row);
      const float4 nr = in.readpixel(col + 1, row);
      if(FC(row, col+1, filters) == RED) // red nb in same row
      {
        color.z = (nt.z + nb.z + 2.0f*color.y - nt.y - nb.y)*0.5f;
        color.x = (nl.x + nr.x + 2.0f*color.y - nl.y - nr.y)*0.5f;
      }
      else
      { // blue nb
        color.x = (nt.x + nb.x + 2.0f*color.y - nt.y - nb.y)*0.5f;
        color.z = (nl.z + nr.z + 2.0f*color.y - nl.y - nr.y)*0.5f;
      }
    }
    else
    {
      // get 4-star-nbhood:
      const float4 ntl = in.readpixel(col - 1, row - 1);
      const float4 ntr = in.readpixel(col + 1, row - 1);
      const float4 nbl = in.readpixel(col - 1, row + 1);
      const float4 nbr = in.readpixel(col + 1, row + 1);

      if(c == RED)
      { // red pixel, fill blue:
        const float diff1  = abs(ntl.z - nbr.z) + abs(ntl.y - color.y) + abs(nbr.y - color.y);
        const float guess1 = ntl.z + nbr.z + 2.0f*color.y - ntl.y - nbr.y;
        const float diff2  = abs(ntr.z - nbl.z) + abs(ntr.y - color.y) + abs(nbl.y - color.y);
        const float guess2 = ntr.z + nbl.z + 2.0f*color.y - ntr.y - nbl.y;
        if     (diff1 > diff2) color.z = guess2 * 0.5f;
        else if(diff1 < diff2) color.z = guess1 * 0.5f;
        else color.z = (guess1 + guess2)*0.25f;
      }
      else // c == 2, blue pixel, fill red:
      {
        const float diff1  = abs(ntl.x - nbr.x) + abs(ntl.y - color.y) + abs(nbr.y - color.y);
        const float guess1 = ntl.x + nbr.x + 2.0f*color.y - ntl.y - nbr.y;
        const float diff2  = abs(ntr.x - nbl.x) + abs(ntr.y - color.y) + abs(nbl.y - color.y);
        const float guess2 = ntr.x + nbl.x + 2.0f*color.y - ntr.y - nbl.y;
        if     (diff1 > diff2) color.x = guess2 * 0.5f;
        else if(diff1 < diff2) color.x = guess1 * 0.5f;
        else color.x = (guess1 + guess2)*0.25f;
      }
    }
  }
  out.write(int2(col, row), max(float4(0.0f), color));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void border_interpolate(uint3 gid : SV_DispatchThreadID,
                        uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
                        uniform uint filters)
{
  const int x = gid.x;
  const int y = gid.y;

  if(x >= width || y >= height) return;
  if(x >= 3 && x < width-3 && y >= 3 && y < height-3) return;

  float4 o = float4(0.0f);
  float sum[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
  int count[4] = { 0, 0, 0, 0 };

  for(int j=y-1; j<=y+1; j++) for (int i=x-1; i<=x+1; i++)
  {
    if(j>=0 && i>=0 && j<height && i<width)
    {
      const int f = FC(j,i,filters);
      sum[f] += in.Areadsingle(i, j);
      count[f]++;
    }
  }

  const float i = max(0.0f, in.Areadsingle(x, y));
  o.x = count[0] > 0 ? fp32_div(sum[0], float(count[0])) : i;
  o.y = count[1]+count[3] > 0 ? fp32_div(sum[1]+sum[3], float(count[1]+count[3])) : i;
  o.z = count[2] > 0 ? fp32_div(sum[2], float(count[2])) : i;

  const int f = FC(y,x,filters);

  if     (f == 0) o.x = i;
  else if(f == 1) o.y = i;
  else if(f == 2) o.z = i;
  else            o.y = i;

  out.write(int2(x, y), max(float4(0.0f), o));
}

// green equilibration and color smoothing, from the same file. They stage
// their neighborhood in group shared memory like the OpenCL kernels, with a
// 16x16 work group

groupshared float lavg_tile[20 * 20];

[shader("compute")]
[numthreads(16, 16, 1)]
void green_equilibration_lavg(uint3 gid : SV_DispatchThreadID, uint3 lid : SV_GroupThreadID,
                              uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
                              uniform uint filters, uniform float thr)
{
  const int x = gid.x;
  const int y = gid.y;

  // the work group with a border of 2
  const int xul = x - int(lid.x) - 2;
  const int yul = y - int(lid.y) - 2;
  for(int k = int(lid.y * 16 + lid.x); k < 20 * 20; k += 256)
    lavg_tile[k] = in.readsingle(xul + k % 20, yul + k / 20);
  GroupMemoryBarrierWithGroupSync();

  if(x >= width || y >= height) return;

  const int t0 = (int(lid.y) + 2) * 20 + int(lid.x) + 2;
#define BUF(dx, dy) lavg_tile[t0 + (dy) * 20 + (dx)]

  const float maximum = 1.0f;
  float o = BUF(0, 0);

  // check lines to be used as we do in cpu code
  int oj = 2, oi = 2;
  if(FC(oj, oi, filters) != GREEN) oj++;
  if(FC(oj, oi, filters) != GREEN) oi++;
  if(FC(oj, oi, filters) != GREEN) oj--;

  const bool row_ok = (oj & 1) == (y & 1);
  const bool col_ok = (oi & 1) == (x & 1);
  if(row_ok && col_ok && x >= 2 && y >= 2 && x < width - 2 && y < height - 2)
  {
    const float o1_1 = BUF(-1, -1);
    const float o1_2 = BUF(1, -1);
    const float o1_3 = BUF(-1, 1);
    const float o1_4 = BUF(1, 1);
    const float o2_1 = BUF(0, -2);
    const float o2_2 = BUF(0, 2);
    const float o2_3 = BUF(-2, 0);
    const float o2_4 = BUF(2, 0);

    const float m1 = (o1_1 + o1_2 + o1_3 + o1_4) / 4.0f;
    const float m2 = (o2_1 + o2_2 + o2_3 + o2_4) / 4.0f;

    if((m2 > 0.0f) && (m1 > 0.0f) && (m1 / m2 < maximum * 2.0f))
    {
      const float c1 = (abs(o1_1 - o1_2) + abs(o1_1 - o1_3) + abs(o1_1 - o1_4) + abs(o1_2 - o1_3)
                        + abs(o1_3 - o1_4) + abs(o1_2 - o1_4)) / 6.0f;
      const float c2 = (abs(o2_1 - o2_2) + abs(o2_1 - o2_3) + abs(o2_1 - o2_4) + abs(o2_2 - o2_3)
                        + abs(o2_3 - o2_4) + abs(o2_2 - o2_4)) / 6.0f;

      if((o < maximum * 0.95f) && (c1 < maximum * thr) && (c2 < maximum * thr))
        o = max(0.0f, o * m1 / m2);
    }
  }
#undef BUF
  out.write(int2(x, y), float4(o));
}

groupshared float2 favg_sums[256];

// the sums of the greens on even (x) and odd (y) rows, per work group, into
// accu[group], with the groups numbered row by row
[shader("compute")]
[numthreads(16, 16, 1)]
void green_equilibration_favg_reduce_first(uint3 gid : SV_DispatchThreadID, uint3 lid : SV_GroupThreadID,
                                           uint3 grp : SV_GroupID,
                                           uniform ImageIn in, uniform int width, uniform int height,
                                           uniform RWStructuredBuffer<float2> accu, uniform uint filters)
{
  const int x = gid.x;
  const int y = gid.y;
  const int l = int(lid.y * 16 + lid.x);

  const int c = FC(y, x, filters);

  // make sure we add same number of greens for even and odd lines
  const bool isinimage = x < 2 * (width / 2) && y < 2 * (height / 2);
  const bool isgreen1 = c == GREEN && (y & 1) == 0;
  const bool isgreen2 = c == GREEN && (y & 1) != 0;

  const float pixel = in.readsingle(x, y);

  favg_sums[l] = float2(isinimage && isgreen1 ? pixel : 0.0f, isinimage && isgreen2 ? pixel : 0.0f);
  GroupMemoryBarrierWithGroupSync();

  for(int offset = 128; offset > 0; offset >>= 1)
  {
    if(l < offset) favg_sums[l] += favg_sums[l + offset];
    GroupMemoryBarrierWithGroupSync();
  }

  if(l == 0) accu[grp.y * ((width + 15) / 16) + grp.x] = favg_sums[0];
}

// adds up the length sums of green_equilibration_favg_reduce_first() in a
// single 16x16 work group, into result[0]. OpenCL leaves the last step to the
// host, which green_equilibration_favg_apply() does here
[shader("compute")]
[numthreads(16, 16, 1)]
void green_equilibration_favg_reduce_second(uint3 lid : SV_GroupThreadID,
                                            uniform RWStructuredBuffer<float2> input,
                                            uniform RWStructuredBuffer<float2> result, uniform int length)
{
  const int l = int(lid.y * 16 + lid.x);
  float m1 = 0.0f;
  float m2 = 0.0f;
  float sum1 = 0.0f;
  float sum2 = 0.0f;
  for(int k = l; k < length; k += 256)
  {
    Kahan_sum(sum1, m1, input[k].x);
    Kahan_sum(sum2, m2, input[k].y);
  }
  favg_sums[l] = float2(sum1, sum2);
  GroupMemoryBarrierWithGroupSync();

  for(int offset = 128; offset > 0; offset >>= 1)
  {
    if(l < offset) favg_sums[l] += favg_sums[l + offset];
    GroupMemoryBarrierWithGroupSync();
  }

  if(l == 0) result[0] = favg_sums[0];
}

// sums holds the two sums of green_equilibration_favg_reduce_second(), from
// which OpenCL's host computes gr_ratio
[shader("compute")]
[numthreads(16, 16, 1)]
void green_equilibration_favg_apply(uint3 gid : SV_DispatchThreadID,
                                    uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
                                    uniform uint filters, uniform RWStructuredBuffer<float2> sums)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float2 s = sums[0];
  const float gr_ratio = (s.x > 0.0f && s.y > 0.0f) ? s.y / s.x : 1.0f;

  float pixel = in.readsingle(x, y);

  const int c = FC(y, x, filters);

  const bool isgreen1 = c == GREEN && (y & 1) == 0; // on even lines
  if(isgreen1) pixel = max(0.0f, pixel * gr_ratio);
  out.write(int2(x, y), float4(pixel));
}

// the compare-and-swap step of the sorting network below
void _cas(inout float a, inout float b)
{
  const float x = a;
  const bool c = a > b;
  a = c ? b : a;
  b = c ? x : b;
}

// the median of nine values, by the sorting network of OpenCL's color_smoothing()
float _median9(float s[9])
{
  _cas(s[1], s[2]);
  _cas(s[4], s[5]);
  _cas(s[7], s[8]);
  _cas(s[0], s[1]);
  _cas(s[3], s[4]);
  _cas(s[6], s[7]);
  _cas(s[1], s[2]);
  _cas(s[4], s[5]);
  _cas(s[7], s[8]);
  _cas(s[0], s[3]);
  _cas(s[5], s[8]);
  _cas(s[4], s[7]);
  _cas(s[3], s[6]);
  _cas(s[1], s[4]);
  _cas(s[2], s[5]);
  _cas(s[4], s[7]);
  _cas(s[4], s[2]);
  _cas(s[6], s[4]);
  _cas(s[4], s[2]);
  return s[4];
}

groupshared float4 smooth_tile[18 * 18];

// 3x3 median of red and blue, each relative to green
[shader("compute")]
[numthreads(16, 16, 1)]
void color_smoothing(uint3 gid : SV_DispatchThreadID, uint3 lid : SV_GroupThreadID,
                     uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height)
{
  const int x = gid.x;
  const int y = gid.y;

  const int xul = x - int(lid.x) - 1;
  const int yul = y - int(lid.y) - 1;
  for(int k = int(lid.y * 16 + lid.x); k < 18 * 18; k += 256)
    smooth_tile[k] = in.readpixel(xul + k % 18, yul + k / 18);
  GroupMemoryBarrierWithGroupSync();

  if(x >= width || y >= height) return;

  const int t0 = (int(lid.y) + 1) * 18 + int(lid.x) + 1;
  float4 o = smooth_tile[t0];

  // don't touch outermost 1 pixels, just copy
  if(x < 1 || x >= width - 1 || y < 1 || y >= height - 1)
  {
    out.write(int2(x, y), o);
    return;
  }

  float r[9], b[9];
  for(int j = 0; j < 3; j++)
    for(int i = 0; i < 3; i++)
    {
      const float4 p = smooth_tile[t0 + (j - 1) * 18 + i - 1];
      r[j * 3 + i] = p.x - p.y;
      b[j * 3 + i] = p.z - p.y;
    }

  o.x = max(0.0f, _median9(r) + o.y);
  o.z = max(0.0f, _median9(b) + o.y);

  out.write(int2(x, y), o);
}

// the median prefilter of PPG on the green pixels, as pre_median_b() in
// src/iop/demosaicing/basics.c: unlike the OpenCL kernel it leaves the
// 3 pixel border unfiltered
[shader("compute")]
[numthreads(16, 16, 1)]
void pre_median(uint3 gid : SV_DispatchThreadID,
                uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
                uniform uint filters, uniform float threshold)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float pc = in.readsingle(x, y);
  const int c = FC(y, x, filters);
  if(x < 3 || x >= width - 3 || y < 3 || y >= height - 3 || (c & 1) == 0)
  {
    out.write(int2(x, y), float4(pc));
    return;
  }

  const int lim[5] = { 0, 1, 2, 1, 0 };
  float med[9];
  int cnt = 0;
  int k = 0;
  for(int i = 0; i < 5; i++)
  {
    for(int j = -lim[i]; j <= lim[i]; j += 2)
    {
      const float v = in.readsingle(x + j, y + i - 2);
      if(abs(v - pc) < threshold)
      {
        med[k++] = v;
        cnt++;
      }
      else
        med[k++] = 64.0f + v;
    }
  }

  for(int i = 0; i < 8; i++)
    for(int ii = i + 1; ii < 9; ii++)
      if(med[i] > med[ii])
      {
        const float tmp = med[ii];
        med[ii] = med[i];
        med[i] = tmp;
      }

  out.write(int2(x, y), float4(cnt == 1 ? med[4] - 64.0f : med[(cnt - 1) / 2]));
}

// downscales a Bayer mosaic to the output size, demosaicing each 2x2 block
[shader("compute")]
[numthreads(16, 16, 1)]
void clip_and_zoom_demosaic_half_size(uint3 gid : SV_DispatchThreadID,
                                      uniform ImageIn in, uniform ImageOut out,
                                      uniform int width, uniform int height,
                                      uniform int rin_wd, uniform int rin_ht, uniform float r_scale,
                                      uniform uint filters)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  float4 color = float4(0.0f);
  float weight = 0.0f;

  // pixel footprint on input buffer, and how many 2x2 blocks fit in it
  const float px_footprint = 1.0f / r_scale;
  const int samples = cl_round(px_footprint / 2.0f);

  int trggbx = 0, trggby = 0;
  if(FC(trggby, trggbx + 1, filters) != 1) trggbx++;
  if(FC(trggby, trggbx, filters) != 0)
  {
    trggbx = (trggbx + 1) & 1;
    trggby++;
  }
  const int2 rggb = int2(trggbx, trggby);

  // upper left corner, moved to an rggb block
  const float2 f = float2(x * px_footprint, y * px_footprint);
  int2 p = int2(int(f.x) & ~1, int(f.y) & ~1);
  const float2 d = float2((f.x - p.x) / 2.0f, (f.y - p.y) / 2.0f);
  p += rggb;

  for(int j = 0; j <= samples + 1; j++)
    for(int i = 0; i <= samples + 1; i++)
    {
      const int xx = p.x + 2 * i;
      const int yy = p.y + 2 * j;

      if(xx + 1 >= rin_wd || yy + 1 >= rin_ht) continue;

      const float xfilter = (i == 0) ? 1.0f - d.x : ((i == samples + 1) ? d.x : 1.0f);
      const float yfilter = (j == 0) ? 1.0f - d.y : ((j == samples + 1) ? d.y : 1.0f);

      const float p1 = max(0.0f, in.readsingle(xx, yy));
      const float p2 = max(0.0f, in.readsingle(xx + 1, yy));
      const float p3 = max(0.0f, in.readsingle(xx, yy + 1));
      const float p4 = max(0.0f, in.readsingle(xx + 1, yy + 1));
      color += yfilter * xfilter * float4(p1, (p2 + p3) * 0.5f, p4, 0.0f);
      weight += yfilter * xfilter;
    }
  color = (weight > 0.0f) ? color / weight : float4(0.0f);
  out.write(int2(x, y), color);
}
