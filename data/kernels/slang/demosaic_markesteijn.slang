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

// Slang port of the Markesteijn kernels in ../demosaic_markesteijn.cl.
//
// The OpenCL kernels stage their neighborhood in local memory, filled by
// linear index: a pixel whose index y * width + x falls outside the buffer
// reads as zero, any other one as stored, even when x is off the row. _rgb()
// and friends read the buffers the same way.
//
// The OpenCL kernels pass each step through a full size buffer, and the
// steps after the green interpolation run once per direction. Here steps are
// merged where they work per pixel, which saves most of that memory traffic.
// The homogeneity maps, bytes in OpenCL, hold four directions per word.

import common;

static const float HOMOGENITY_EPS = 1e-10f;

// directions along the both axes and both diagonals
static const int2 dir[4] = { int2(1, 0), int2(0, 1), int2(1, 1), int2(-1, 1) };

// all possible offsets of a neighboring pixel in x/y directions
static const int2 nboff[4] = { int2(1, 0), int2(0, 1), int2(-1, 0), int2(0, -1) };

float4 _rgb(RWStructuredBuffer<float4> buf, int x, int y, int width, int height)
{
  const int i = y * width + x;
  return (i >= 0 && i < width * height) ? buf[i] : float4(0.0f);
}

float _drv(RWStructuredBuffer<float> buf, int x, int y, int width, int height)
{
  const int i = y * width + x;
  return (i >= 0 && i < width * height) ? buf[i] : 0.0f;
}

// allhex is the host's signed char [3][3][8][2], read as words
int _sbyte(StructuredBuffer<uint> buf, int i)
{
  return (int(buf[i >> 2] << uint(24 - (i & 3) * 8))) >> 24;
}

int2 _hex(StructuredBuffer<uint> allhex, int row, int col, int c)
{
  const int i = ((row * 3 + col) * 8 + c) * 2;
  return int2(_sbyte(allhex, i), _sbyte(allhex, i + 1));
}

// the raw value at x, y as OpenCL's markesteijn_initial_copy() stores it in
// the rgb buffer: in the channel of its color, the others zero
float4 _cfa_pixel(ImageIn in, int x, int y, StructuredBuffer<uint> xtrans)
{
  const int f = FCxtrans(y, x, xtrans);
  const float p = max(0.0f, in.Areadsingle(x, y));
  return float4(f == 0 ? p : 0.0f, f == 1 ? p : 0.0f, f == 2 ? p : 0.0f, 0.0f);
}

// _cfa_pixel() read by linear index, as _rgb() reads the buffer
float4 _cfa(ImageIn in, int x, int y, int width, int height, StructuredBuffer<uint> xtrans)
{
  const int i = y * width + x;
  return (i >= 0 && i < width * height) ? _cfa_pixel(in, i % width, i / width, xtrans) : float4(0.0f);
}

// the YPbPr conversion of markesteijn_convert_yuv()
float4 _yuv(float4 p)
{
  const float Y = 0.2627f * p.x + 0.6780f * p.y + 0.0593f * p.z;
  return float4(Y, (p.z - Y) * 0.56433f, (p.x - Y) * 0.67815f, 0.0f);
}

uint _word(RWStructuredBuffer<uint> buf, int x, int y, int width, int height)
{
  const int i = y * width + x;
  return (i >= 0 && i < width * height) ? buf[i] : 0u;
}

// clEnqueueCopyBuffer() for the rgb buffers
[shader("compute")]
[numthreads(16, 16, 1)]
void markesteijn_copy(uint3 gid : SV_DispatchThreadID,
                      uniform RWStructuredBuffer<float4> in, uniform RWStructuredBuffer<float4> out,
                      uniform int width, uniform int height)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;
  out[y * width + x] = in[y * width + x];
}

// find minimum and maximum allowed green values of red/blue pixel pairs,
// from the raw data that OpenCL copies to the first rgb buffer first
[shader("compute")]
[numthreads(16, 16, 1)]
void markesteijn_green_minmax(uint3 gid : SV_DispatchThreadID,
                              uniform ImageIn in, uniform RWStructuredBuffer<float2> gminmax,
                              uniform int width, uniform int height, uniform int border,
                              uniform StructuredBuffer<uint> xtrans, uniform StructuredBuffer<uint> allhex)
{
  const int x = gid.x;
  const int y = gid.y;

  // take sufficient border into account
  if(x < border || x >= width-border || y < border || y >= height-border) return;

  // the color of this pixel
  const int f = FCxtrans(y, x, xtrans);

  // we only work on non-green pixels
  if(f == GREEN) return;

  // get min/max of *this* pixel
  float gmin = 3.402823466e+38f;
  float gmax = 0.0f;
  for(int c = 0; c < 6; c++)
  {
    const int2 hex = _hex(allhex, y % 3, x % 3, c);
    const float val = _cfa(in, x + hex.x, y + hex.y, width, height, xtrans).y;
    if(gmin > val) gmin = val;
    if(gmax < val) gmax = val;
  }

  // find the *one* neighboring non-green pixel in four directions
  int2 p2nd = int2(x, y);
  for(int n = 0; n < 4; n++)
  {
    if(FCxtrans(y + nboff[n].y, x + nboff[n].x, xtrans) == 1) continue; // this is a green pixel
    p2nd = int2(x + nboff[n].x, y + nboff[n].y);
  }

  // we have found the second pixel: now include it into min/max calculation
  for(int c = 0; c < 6; c++)
  {
    const int2 hex = _hex(allhex, p2nd.y % 3, p2nd.x % 3, c);
    const float val = _cfa(in, p2nd.x + hex.x, p2nd.y + hex.y, width, height, xtrans).y;
    if(gmin > val) gmin = val;
    if(gmax < val) gmax = val;
  }

  gminmax[y * width + x] = float2(gmin, gmax);
}

// fill the four rgb buffers from the raw data and interpolate green
// horizontally, vertically, and along both diagonals. This merges OpenCL's
// markesteijn_initial_copy(), the copies of its result and
// markesteijn_interpolate_green(), which reads the greens from the raw data
// as copied, so each buffer is written once
[shader("compute")]
[numthreads(16, 16, 1)]
void markesteijn_initial_green(uint3 gid : SV_DispatchThreadID,
                               uniform ImageIn in,
                               uniform RWStructuredBuffer<float4> rgb_0, uniform RWStructuredBuffer<float4> rgb_1,
                               uniform RWStructuredBuffer<float4> rgb_2, uniform RWStructuredBuffer<float4> rgb_3,
                               uniform RWStructuredBuffer<float2> gminmax,
                               uniform int width, uniform int height, uniform int border,
                               uniform int2 sgreen, uniform StructuredBuffer<uint> xtrans,
                               uniform StructuredBuffer<uint> allhex)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float4 center = _cfa_pixel(in, x, y, xtrans);
  float4 o[4] = { center, center, center, center };

  // the color of this pixel
  const int f = FCxtrans(y, x, xtrans);

  // only non-green pixels within a sufficient border get interpolated
  if(f != GREEN && x >= border && x < width-border && y >= border && y < height-border)
  {
    // receive min/max of this pixel
    const float2 mm = gminmax[y * width + x];
    const float gmin = mm.x;
    const float gmax = mm.y;

    int2 hex[6];
    for(int c = 0; c < 6; c++) hex[c] = _hex(allhex, y % 3, x % 3, c);

    // the raw data at k times a hexagon offset from this pixel
#define AT(k, h) _cfa(in, x + (k) * (h).x, y + (k) * (h).y, width, height, xtrans)

    float color[4];

    color[0] = 0.6796875f * (AT(1, hex[1]).y + AT(1, hex[0]).y)
               - 0.1796875f * (AT(2, hex[1]).y + AT(2, hex[0]).y);

    color[1] = 0.87109375f * AT(1, hex[3]).y
               + 0.13f * AT(1, hex[2]).y
               + 0.359375f * (center[f] - AT(-1, hex[2])[f]);

    for(int c = 0; c < 2; c++)
    {
      color[2 + c] = 0.640625f * AT(1, hex[4 + c]).y
                     + 0.359375f * AT(-2, hex[4 + c]).y
                     + 0.12890625f * (2.0f * center[f]
                                      - AT(3, hex[4 + c])[f]
                                      - AT(-3, hex[4 + c])[f]);
    }
#undef AT

    const int d = (((y - sgreen.y) % 3) == 0) ? 1 : 0;
    for(int c = 0; c < 4; c++) o[c ^ d].y = clamp(color[c], gmin, gmax);
  }

  const int glidx = y * width + x;
  rgb_0[glidx] = o[0];
  rgb_1[glidx] = o[1];
  rgb_2[glidx] = o[2];
  rgb_3[glidx] = o[3];
}

// interpolate red and blue values for solitary green pixels
[shader("compute")]
[numthreads(16, 16, 1)]
void markesteijn_solitary_green(uint3 gid : SV_DispatchThreadID,
                                uniform RWStructuredBuffer<float4> rgb, uniform RWStructuredBuffer<float4> aux,
                                uniform int width, uniform int height, uniform int border,
                                uniform int d, uniform int2 dir, uniform int hcomp,
                                uniform int2 sgreen, uniform StructuredBuffer<uint> xtrans)
{
  // the work items cover only the solitary green pixels, every third one
  // in both directions
  const int x = sgreen.x + 3 * int(gid.x);
  const int y = sgreen.y + 3 * int(gid.y);

  // take sufficient border into account
  if(x < border || x >= width-border || y < border || y >= height-border) return;

  // the color of the pixel right to this one
  int h = FCxtrans(y, x + 1, xtrans);

  // the complement according to this run
  h ^= hcomp;

  const int glidx = y * width + x;

  float color[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
  float ocolor[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

  if(d > 0)
  {
    const float4 a = aux[glidx];
    for(int c = 0; c < 4; c++) ocolor[c] = color[c] = a[c];
  }

  float diff = 0.0f;

  const float4 center = _rgb(rgb, x, y, width, height);

  for(int c = 0; c < 2; c++, h ^= 2)
  {
    const int k = 1 << c;
    const float4 pp = _rgb(rgb, x + k * dir.x, y + k * dir.y, width, height);
    const float4 pm = _rgb(rgb, x - k * dir.x, y - k * dir.y, width, height);
    const float g = 2.0f * center.y - pp.y - pm.y;
    color[h] = g + pp[h] + pm[h];
    diff += (d > 1) ? fsquare(pp.y - pm.y - pp[h] + pm[h]) + fsquare(g) : 0.0f;
  }

  color[3] = diff;

  if((d > 1) && (d & 1) != 0)
    for(int c = 0; c < 2; c++) color[c * 2] = (ocolor[3] < diff) ? ocolor[c * 2] : color[c * 2];

  if((d < 2) || (d & 1) != 0)
  {
    float4 out = rgb[glidx];
    out.x = color[0] / 2.0f;
    out.z = color[2] / 2.0f;
    rgb[glidx] = out;
  }

  aux[glidx] = float4(color[0], color[1], color[2], color[3]);
}

// recalculate green from interpolated values of closer pixels.
[shader("compute")]
[numthreads(16, 16, 1)]
void markesteijn_recalculate_green(uint3 gid : SV_DispatchThreadID,
                                   uniform RWStructuredBuffer<float4> rgb_0, uniform RWStructuredBuffer<float4> rgb_1,
                                   uniform RWStructuredBuffer<float4> rgb_2, uniform RWStructuredBuffer<float4> rgb_3,
                                   uniform RWStructuredBuffer<float2> gminmax,
                                   uniform int width, uniform int height, uniform int border,
                                   uniform int2 sgreen, uniform StructuredBuffer<uint> xtrans,
                                   uniform StructuredBuffer<uint> allhex)
{
  const int x = gid.x;
  const int y = gid.y;

  // take sufficient border into account
  if(x < border || x >= width-border || y < border || y >= height-border) return;

  // the color of this pixel
  const int f = FCxtrans(y, x, xtrans);

  // we only work on non-green pixels
  if(f == GREEN) return;

  // receive min/max of this pixel
  const float2 mm = gminmax[y * width + x];

  const int glidx = y * width + x;

  for(int d = 3; d < 6; d++)
  {
    const int2 hex = _hex(allhex, y % 3, x % 3, d);
    const int id = (d - 2) ^ ((((y - sgreen.y) % 3) == 0) ? 1 : 0);
    // OpenCL reads the whole buffer here, with no bounds test
    const int ip = glidx + hex.y * width + hex.x;
    const int im = glidx - 2 * (hex.y * width + hex.x);
    float4 c0, cp, cm;
    switch(id)
    {
      case 0: c0 = rgb_0[glidx]; cp = rgb_0[ip]; cm = rgb_0[im]; break;
      case 1: c0 = rgb_1[glidx]; cp = rgb_1[ip]; cm = rgb_1[im]; break;
      case 2: c0 = rgb_2[glidx]; cp = rgb_2[ip]; cm = rgb_2[im]; break;
      default: c0 = rgb_3[glidx]; cp = rgb_3[ip]; cm = rgb_3[im]; break;
    }
    const float val = cm.y + 2.0f * cp.y - cm[f] - 2.0f * cp[f] + 3.0f * c0[f];
    const float g = clamp(fp32_div(val, 3.0f), mm.x, mm.y);
    switch(id)
    {
      case 0: rgb_0[glidx].y = g; break;
      case 1: rgb_1[glidx].y = g; break;
      case 2: rgb_2[glidx].y = g; break;
      default: rgb_3[glidx].y = g; break;
    }
  }
}

// interpolate red for blue pixels and vice versa
[shader("compute")]
[numthreads(16, 16, 1)]
void markesteijn_red_and_blue(uint3 gid : SV_DispatchThreadID,
                              uniform RWStructuredBuffer<float4> rgb, uniform int width, uniform int height,
                              uniform int border, uniform int d, uniform int2 sgreen,
                              uniform StructuredBuffer<uint> xtrans)
{
  const int x = gid.x;
  const int y = gid.y;

  // take sufficient border into account
  if(x < border || x >= width-border || y < border || y >= height-border) return;

  // the "other" color relative to this pixel's one
  const int f = 2 - FCxtrans(y, x, xtrans);

  // we don't work on green pixels
  if(f == GREEN) return;

  // in which direction to sample for the second pixel of this pair (horizontally or vertically)
  const bool horiz = (y - sgreen.y) % 3 == 0;
  // c is one pixel along the pair, h three pixels across it
  const int2 c = horiz ? int2(1, 0) : int2(0, 1);
  const int2 h = horiz ? int2(0, 3) : int2(3, 0);

  const float4 b0 = _rgb(rgb, x, y, width, height);
  const float4 bcp = _rgb(rgb, x + c.x, y + c.y, width, height);
  const float4 bcm = _rgb(rgb, x - c.x, y - c.y, width, height);
  const float4 bhp = _rgb(rgb, x + h.x, y + h.y, width, height);
  const float4 bhm = _rgb(rgb, x - h.x, y - h.y, width, height);

  const bool use_c = d > 1 || (d == 0 && horiz) || (d == 1 && !horiz)
                     || abs(b0.y - bcp.y) + abs(b0.y - bcm.y) < 2.0f * (abs(b0.y - bhp.y) + abs(b0.y - bhm.y));
  const float4 op = use_c ? bcp : bhp;
  const float4 om = use_c ? bcm : bhm;

  const int glidx = y * width + x;
  float4 out = rgb[glidx];
  out[f] = (op[f] + om[f] + 2.0f * b0.y - op.y - om.y) / 2.0f;
  rgb[glidx] = out;
}

// interpolate red and blue for 2x2 blocks of green
[shader("compute")]
[numthreads(16, 16, 1)]
void markesteijn_interpolate_twoxtwo(uint3 gid : SV_DispatchThreadID,
                                     uniform RWStructuredBuffer<float4> rgb, uniform int width, uniform int height,
                                     uniform int border, uniform int d, uniform int2 sgreen,
                                     uniform StructuredBuffer<uint> allhex)
{
  const int x = gid.x;
  const int y = gid.y;

  // take sufficient border into account
  if(x < border || x >= width-border || y < border || y >= height-border) return;

  // we only work on pixels within an 2x2 block of green.
  // for all other pixels which are in the same row or column
  // as the solitary green pixel we skip
  if((y - sgreen.y) % 3 == 0 || (x - sgreen.x) % 3 == 0) return;

  // get hexagon of surrounding pixels
  const int2 h0 = _hex(allhex, y % 3, x % 3, d);
  const int2 h1 = _hex(allhex, y % 3, x % 3, d + 1);

  float m[5];
  // OpenCL tests the sum of both local buffer offsets for zero
  if(any(h0 + h1 != int2(0)))
  {
    m[0] = 3.0f;
    m[1] = -2.0f;
    m[2] = -1.0f;
    m[3] = 2.0f;
    m[4] = 1.0f / 3.0f;
  }
  else
  {
    m[0] = 2.0f;
    m[1] = -1.0f;
    m[2] = -1.0f;
    m[3] = 1.0f;
    m[4] = 1.0f / 2.0f;
  }

  const float4 b0 = _rgb(rgb, x, y, width, height);
  const float4 b1 = _rgb(rgb, x + h0.x, y + h0.y, width, height);
  const float4 b2 = _rgb(rgb, x + h1.x, y + h1.y, width, height);

  const float g = m[0] * b0.y + m[1] * b1.y + m[2] * b2.y;

  const int glidx = y * width + x;
  float4 out = rgb[glidx];
  out.x = (g + m[3] * b1.x + b2.x) * m[4];
  out.z = (g + m[3] * b1.z + b2.z) * m[4];
  rgb[glidx] = out;
}

// convert to perceptual YPbPr colorspace and differentiate in direction d,
// which OpenCL does in two kernels through a yuv buffer
[shader("compute")]
[numthreads(16, 16, 1)]
void markesteijn_differentiate(uint3 gid : SV_DispatchThreadID,
                               uniform RWStructuredBuffer<float4> rgb, uniform RWStructuredBuffer<float> drv,
                               uniform int width, uniform int height, uniform int border, uniform int d)
{
  const int x = gid.x;
  const int y = gid.y;

  // take sufficient border into account
  if(x < border || x >= width-border || y < border || y >= height-border) return;

  const int2 p = dir[d & 3];
  const float4 b0 = _yuv(_rgb(rgb, x, y, width, height));
  const float4 bp = _yuv(_rgb(rgb, x + p.x, y + p.y, width, height));
  const float4 bm = _yuv(_rgb(rgb, x - p.x, y - p.y, width, height));

  drv[y * width + x] = fsquare(2.0f * b0.x - bp.x - bm.x)
                       + fsquare(2.0f * b0.y - bp.y - bm.y)
                       + fsquare(2.0f * b0.z - bp.z - bm.z);
}

// SPIR-V cannot pick a buffer through a pointer, so branch on its index
float _drv_of(int d, int x, int y, int width, int height,
              RWStructuredBuffer<float> drv_0, RWStructuredBuffer<float> drv_1,
              RWStructuredBuffer<float> drv_2, RWStructuredBuffer<float> drv_3,
              RWStructuredBuffer<float> drv_4, RWStructuredBuffer<float> drv_5,
              RWStructuredBuffer<float> drv_6, RWStructuredBuffer<float> drv_7)
{
  switch(d)
  {
    case 0: return _drv(drv_0, x, y, width, height);
    case 1: return _drv(drv_1, x, y, width, height);
    case 2: return _drv(drv_2, x, y, width, height);
    case 3: return _drv(drv_3, x, y, width, height);
    case 4: return _drv(drv_4, x, y, width, height);
    case 5: return _drv(drv_5, x, y, width, height);
    case 6: return _drv(drv_6, x, y, width, height);
    default: return _drv(drv_7, x, y, width, height);
  }
}

// set the homogeneity maps of all ndir directions, with the threshold OpenCL
// keeps in a buffer. Byte d & 3 of homo_lo (d < 4) or homo_hi holds the value
// of direction d
[shader("compute")]
[numthreads(16, 16, 1)]
void markesteijn_homo_set(uint3 gid : SV_DispatchThreadID,
                          uniform RWStructuredBuffer<float> drv_0, uniform RWStructuredBuffer<float> drv_1,
                          uniform RWStructuredBuffer<float> drv_2, uniform RWStructuredBuffer<float> drv_3,
                          uniform RWStructuredBuffer<float> drv_4, uniform RWStructuredBuffer<float> drv_5,
                          uniform RWStructuredBuffer<float> drv_6, uniform RWStructuredBuffer<float> drv_7,
                          uniform RWStructuredBuffer<uint> homo_lo, uniform RWStructuredBuffer<uint> homo_hi,
                          uniform int width, uniform int height, uniform int border, uniform int ndir)
{
  const int x = gid.x;
  const int y = gid.y;

  // take sufficient border into account
  if(x < border || x >= width-border || y < border || y >= height-border) return;

#define DRV(d, xx, yy) _drv_of((d), (xx), (yy), width, height, drv_0, drv_1, drv_2, drv_3, drv_4, drv_5, drv_6, drv_7)

  // the threshold is the smallest derivative of this pixel
  float tr = 3.402823466e+38f;
  for(int d = 0; d < ndir; d++)
  {
    const float deriv = DRV(d, x, y);
    tr = (tr > deriv) ? deriv : tr;
  }
  tr = max(8.0f * tr, HOMOGENITY_EPS);

  uint lo = 0u, hi = 0u;
  for(int d = 0; d < ndir; d++)
  {
    uint accu = 0u;
    for(int v = -1; v <= 1; v++)
      for(int h = -1; h <= 1; h++)
        accu += (DRV(d, x + h, y + v) <= tr) ? 1u : 0u;
    if(d < 4)
      lo |= accu << uint(d * 8);
    else
      hi |= accu << uint((d - 4) * 8);
  }
#undef DRV

  const int glidx = y * width + x;
  homo_lo[glidx] = lo;
  if(ndir > 4) homo_hi[glidx] = hi;
}

// build the 5x5 sums of the homogeneity maps. A value is at most 9 and a sum
// at most 225, so the packed words add without carries between the bytes
[shader("compute")]
[numthreads(16, 16, 1)]
void markesteijn_homo_sum(uint3 gid : SV_DispatchThreadID,
                          uniform RWStructuredBuffer<uint> homo_lo, uniform RWStructuredBuffer<uint> homo_hi,
                          uniform RWStructuredBuffer<uint> homosum_lo, uniform RWStructuredBuffer<uint> homosum_hi,
                          uniform int width, uniform int height, uniform int border, uniform int ndir)
{
  const int x = gid.x;
  const int y = gid.y;

  // take sufficient border into account
  if(x < border || x >= width-border || y < border || y >= height-border) return;

  uint lo = 0u, hi = 0u;
  for(int v = -2; v <= 2; v++)
    for(int h = -2; h <= 2; h++)
    {
      lo += _word(homo_lo, x + h, y + v, width, height);
      if(ndir > 4) hi += _word(homo_hi, x + h, y + v, width, height);
    }

  const int glidx = y * width + x;
  homosum_lo[glidx] = lo;
  if(ndir > 4) homosum_hi[glidx] = hi;
}

float4 _rgb_of(int d, int i,
               RWStructuredBuffer<float4> rgb_0, RWStructuredBuffer<float4> rgb_1,
               RWStructuredBuffer<float4> rgb_2, RWStructuredBuffer<float4> rgb_3,
               RWStructuredBuffer<float4> rgb_4, RWStructuredBuffer<float4> rgb_5,
               RWStructuredBuffer<float4> rgb_6, RWStructuredBuffer<float4> rgb_7)
{
  switch(d)
  {
    case 0: return rgb_0[i];
    case 1: return rgb_1[i];
    case 2: return rgb_2[i];
    case 3: return rgb_3[i];
    case 4: return rgb_4[i];
    case 5: return rgb_5[i];
    case 6: return rgb_6[i];
    default: return rgb_7[i];
  }
}

// average the directions whose homogeneity is near the best one, and write
// the result. This merges OpenCL's markesteijn_homo_max(), _homo_max_corr(),
// _homo_quench(), _accu() and _final(), in their order
[shader("compute")]
[numthreads(16, 16, 1)]
void markesteijn_accu_final(uint3 gid : SV_DispatchThreadID,
                            uniform RWStructuredBuffer<float4> rgb_0, uniform RWStructuredBuffer<float4> rgb_1,
                            uniform RWStructuredBuffer<float4> rgb_2, uniform RWStructuredBuffer<float4> rgb_3,
                            uniform RWStructuredBuffer<float4> rgb_4, uniform RWStructuredBuffer<float4> rgb_5,
                            uniform RWStructuredBuffer<float4> rgb_6, uniform RWStructuredBuffer<float4> rgb_7,
                            uniform RWStructuredBuffer<uint> homosum_lo, uniform RWStructuredBuffer<uint> homosum_hi,
                            uniform ImageOut out, uniform int width, uniform int height, uniform int border,
                            uniform int ndir)
{
  const int x = gid.x;
  const int y = gid.y;

  // take sufficient border into account
  if(x < border || x >= width-border || y < border || y >= height-border) return;

  const int glidx = y * width + x;

  const uint lo = homosum_lo[glidx];
  const uint hi = ndir > 4 ? homosum_hi[glidx] : 0u;
  int hs[8];
  for(int d = 0; d < 4; d++)
  {
    hs[d] = int((lo >> uint(d * 8)) & 0xffu);
    hs[d + 4] = int((hi >> uint(d * 8)) & 0xffu);
  }

  int hmax = 0;
  for(int d = 0; d < ndir; d++) hmax = (hmax < hs[d]) ? hs[d] : hmax;
  hmax -= hmax >> 3;

  // for Markesteijn-3: only use one of two directions if one is better than the other
  if(ndir > 4)
    for(int d = 0; d < 4; d++)
    {
      const int hmi1 = hs[d];
      const int hmi2 = hs[d + 4];
      hs[d] = (hmi1 < hmi2) ? 0 : hmi1;
      hs[d + 4] = (hmi1 > hmi2) ? 0 : hmi2;
    }

  float4 pixel = float4(0.0f);
  for(int d = 0; d < ndir; d++)
  {
    float4 add = _rgb_of(d, glidx, rgb_0, rgb_1, rgb_2, rgb_3, rgb_4, rgb_5, rgb_6, rgb_7);
    add.w = 1.0f;
    pixel += (hs[d] >= hmax) ? add : float4(0.0f);
  }

  pixel = (pixel.w > 0.0f) ? pixel / pixel.w : float4(0.0f);
  pixel.w = 0.0f;

  out.write(int2(x, y), max(float4(0.0f), pixel));
}
