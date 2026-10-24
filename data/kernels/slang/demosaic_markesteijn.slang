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
// and friends read the buffers the same way. Byte-sized buffer elements are
// not portable, so the homogeneity maps pack four bytes per word.

import common;

static const float HOMOGENITY_EPS = 1e-10f;
static const float SOLITARY_EPS = 1e-12f;

// directions along the both axes and both diagonals
static const int2 dir[4] = { int2(1, 0), int2(0, 1), int2(1, 1), int2(-1, 1) };

// all possible offsets of a neighboring pixel in x/y directions
static const int2 nboff[4] = { int2(1, 0), int2(0, 1), int2(-1, 0), int2(0, -1) };

float4 _rgb(RWStructuredBuffer<float4> buf, int x, int y, int width, int height)
{
  const int i = y * width + x;
  return (i >= 0 && i < width * height) ? buf[i] : float4(0.0f);
}

float4 _rgb_ro(StructuredBuffer<float4> buf, int x, int y, int width, int height)
{
  const int i = y * width + x;
  return (i >= 0 && i < width * height) ? buf[i] : float4(0.0f);
}

float _green(RWStructuredBuffer<float4> buf, int x, int y, int width, int height)
{
  return _rgb(buf, x, y, width, height).y;
}

float _drv(RWStructuredBuffer<float> buf, int x, int y, int width, int height)
{
  const int i = y * width + x;
  return (i >= 0 && i < width * height) ? buf[i] : 0.0f;
}

int _byte_get(RWStructuredBuffer<uint> buf, int i)
{
  return int((buf[i >> 2] >> uint((i & 3) * 8)) & 0xffu);
}

// each byte has one writer, but its neighbors in the word do not
void _byte_put(RWStructuredBuffer<uint> buf, int i, int val)
{
  const uint shift = uint((i & 3) * 8);
  InterlockedAnd(buf[i >> 2], ~(0xffu << shift));
  InterlockedOr(buf[i >> 2], uint(val & 0xff) << shift);
}

int _homo(RWStructuredBuffer<uint> buf, int x, int y, int width, int height)
{
  const int i = y * width + x;
  return (i >= 0 && i < width * height) ? _byte_get(buf, i) : 0;
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

// copy image from image object to buffer.
[shader("compute")]
[numthreads(16, 16, 1)]
void markesteijn_initial_copy(uint3 gid : SV_DispatchThreadID,
                              uniform ImageIn in, uniform RWStructuredBuffer<float4> rgb,
                              uniform int width, uniform int height, uniform StructuredBuffer<uint> xtrans)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const int f = FCxtrans(y, x, xtrans);
  const float p = max(0.0f, in.Areadsingle(x, y));

  // OpenCL writes three floats and leaves the fourth as allocated
  float4 pix = rgb[y * width + x];
  pix.x = f == 0 ? p : 0.0f;
  pix.y = f == 1 ? p : 0.0f;
  pix.z = f == 2 ? p : 0.0f;
  rgb[y * width + x] = pix;
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

// find minimum and maximum allowed green values of red/blue pixel pairs
[shader("compute")]
[numthreads(16, 16, 1)]
void markesteijn_green_minmax(uint3 gid : SV_DispatchThreadID,
                              uniform RWStructuredBuffer<float4> rgb, uniform RWStructuredBuffer<float2> gminmax,
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
  float gmax = 1.175494351e-38f;
  for(int c = 0; c < 6; c++)
  {
    const int2 hex = _hex(allhex, y % 3, x % 3, c);
    const float val = _green(rgb, x + hex.x, y + hex.y, width, height);
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
    const float val = _green(rgb, p2nd.x + hex.x, p2nd.y + hex.y, width, height);
    if(gmin > val) gmin = val;
    if(gmax < val) gmax = val;
  }

  gminmax[y * width + x] = float2(gmin, gmax);
}

// Interpolate green horizontally, vertically, and along both diagonals
[shader("compute")]
[numthreads(16, 16, 1)]
void markesteijn_interpolate_green(uint3 gid : SV_DispatchThreadID,
                                   uniform StructuredBuffer<float4> rgb_in,
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
  const float gmin = mm.x;
  const float gmax = mm.y;

  int2 hex[6];
  for(int c = 0; c < 6; c++) hex[c] = _hex(allhex, y % 3, x % 3, c);

  // rgb_0 at k times a hexagon offset from this pixel. The writes below
  // reach the green of pixels twice an offset away, which the OpenCL kernel
  // reads from its local copy, as it was before; rgb_in is that copy
#define AT(k, h) _rgb_ro(rgb_in, x + (k) * (h).x, y + (k) * (h).y, width, height)

  const float4 center = AT(0, hex[0]);
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

  const int glidx = y * width + x;
  const int d = (((y - sgreen.y) % 3) == 0) ? 1 : 0;
  for(int c = 0; c < 4; c++)
  {
    const float g = clamp(color[c], gmin, gmax);
    // SPIR-V cannot pick a buffer through a pointer, so branch on its index
    switch(c ^ d)
    {
      case 0: rgb_0[glidx].y = g; break;
      case 1: rgb_1[glidx].y = g; break;
      case 2: rgb_2[glidx].y = g; break;
      default: rgb_3[glidx].y = g; break;
    }
  }
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
  const int x = gid.x;
  const int y = gid.y;

  // take sufficient border into account
  if(x < border || x >= width-border || y < border || y >= height-border) return;

  // we only work on solitary green pixels
  if((x - sgreen.x) % 3 != 0 || (y - sgreen.y) % 3 != 0) return;

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
    for(int c = 0; c < 2; c++) color[c * 2] = (ocolor[3] + SOLITARY_EPS < diff) ? ocolor[c * 2] : color[c * 2];

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
    const float g = clamp(val / 3.0f, mm.x, mm.y);
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

// Convert to perceptual YPbPr colorspace
[shader("compute")]
[numthreads(16, 16, 1)]
void markesteijn_convert_yuv(uint3 gid : SV_DispatchThreadID,
                             uniform RWStructuredBuffer<float4> rgb, uniform RWStructuredBuffer<float4> yuv,
                             uniform int width, uniform int height, uniform int border)
{
  const int x = gid.x;
  const int y = gid.y;

  // take sufficient border into account
  if(x < border || x >= width-border || y < border || y >= height-border) return;

  const int idx = y * width + x;
  const float4 p = rgb[idx];

  const float Y = 0.2627f * p.x + 0.6780f * p.y + 0.0593f * p.z;

  // OpenCL leaves the fourth float as it was
  float4 o = yuv[idx];
  o.x = Y;
  o.y = (p.z - Y) * 0.56433f;
  o.z = (p.x - Y) * 0.67815f;
  yuv[idx] = o;
}

// differentiate in all directions
[shader("compute")]
[numthreads(16, 16, 1)]
void markesteijn_differentiate(uint3 gid : SV_DispatchThreadID,
                               uniform RWStructuredBuffer<float4> yuv, uniform RWStructuredBuffer<float> drv,
                               uniform int width, uniform int height, uniform int border, uniform int d)
{
  const int x = gid.x;
  const int y = gid.y;

  // take sufficient border into account
  if(x < border || x >= width-border || y < border || y >= height-border) return;

  const int2 p = dir[d & 3];
  const float4 b0 = _rgb(yuv, x, y, width, height);
  const float4 bp = _rgb(yuv, x + p.x, y + p.y, width, height);
  const float4 bm = _rgb(yuv, x - p.x, y - p.y, width, height);

  drv[y * width + x] = fsquare(2.0f * b0.x - bp.x - bm.x)
                       + fsquare(2.0f * b0.y - bp.y - bm.y)
                       + fsquare(2.0f * b0.z - bp.z - bm.z);
}

// Get threshold for homogeneity maps
[shader("compute")]
[numthreads(16, 16, 1)]
void markesteijn_homo_threshold(uint3 gid : SV_DispatchThreadID,
                                uniform RWStructuredBuffer<float> drv, uniform RWStructuredBuffer<float> thresh,
                                uniform int width, uniform int height, uniform int border, uniform int d)
{
  const int x = gid.x;
  const int y = gid.y;

  // take sufficient border into account
  if(x < border || x >= width-border || y < border || y >= height-border) return;

  const int glidx = y * width + x;

  const float deriv = drv[glidx];

  float tr = (d == 0) ? 3.402823466e+38f : thresh[glidx];

  tr = (tr > deriv) ? deriv : tr;

  thresh[glidx] = tr;
}

// set homogeneity maps
[shader("compute")]
[numthreads(16, 16, 1)]
void markesteijn_homo_set(uint3 gid : SV_DispatchThreadID,
                          uniform RWStructuredBuffer<float> drv, uniform RWStructuredBuffer<float> thresh,
                          uniform RWStructuredBuffer<uint> homo,
                          uniform int width, uniform int height, uniform int border)
{
  const int x = gid.x;
  const int y = gid.y;

  // take sufficient border into account
  if(x < border || x >= width-border || y < border || y >= height-border) return;

  const int glidx = y * width + x;

  const float tr = max(8.0f * thresh[glidx], HOMOGENITY_EPS);

  int accu = 0;
  for(int v = -1; v <= 1; v++)
    for(int h = -1; h <= 1; h++)
      accu += (_drv(drv, x + h, y + v, width, height) <= tr) ? 1 : 0;

  _byte_put(homo, glidx, accu);
}

// build 5x5 sum of homogeneity maps
[shader("compute")]
[numthreads(16, 16, 1)]
void markesteijn_homo_sum(uint3 gid : SV_DispatchThreadID,
                          uniform RWStructuredBuffer<uint> homo, uniform RWStructuredBuffer<uint> homosum,
                          uniform int width, uniform int height, uniform int border)
{
  const int x = gid.x;
  const int y = gid.y;

  // take sufficient border into account
  if(x < border || x >= width-border || y < border || y >= height-border) return;

  // a uchar in OpenCL, which wraps; at most 9 * 25 = 225 here
  int accu = 0;
  for(int v = -2; v <= 2; v++)
    for(int h = -2; h <= 2; h++)
      accu += _homo(homo, x + h, y + v, width, height);

  _byte_put(homosum, y * width + x, accu);
}

// get maximum value for homosum
[shader("compute")]
[numthreads(16, 16, 1)]
void markesteijn_homo_max(uint3 gid : SV_DispatchThreadID,
                          uniform RWStructuredBuffer<uint> homosum, uniform RWStructuredBuffer<uint> maxval,
                          uniform int width, uniform int height, uniform int border, uniform int d)
{
  const int x = gid.x;
  const int y = gid.y;

  // take sufficient border into account
  if(x < border || x >= width-border || y < border || y >= height-border) return;

  const int glidx = y * width + x;

  const int hm = _byte_get(homosum, glidx);

  int hmax = (d == 0) ? 0 : _byte_get(maxval, glidx);

  hmax = (hmax < hm) ? hm : hmax;

  _byte_put(maxval, glidx, hmax);
}

// adjust maximum value for homosum
[shader("compute")]
[numthreads(16, 16, 1)]
void markesteijn_homo_max_corr(uint3 gid : SV_DispatchThreadID,
                               uniform RWStructuredBuffer<uint> maxval, uniform int width, uniform int height,
                               uniform int border)
{
  const int x = gid.x;
  const int y = gid.y;

  // take sufficient border into account
  if(x < border || x >= width-border || y < border || y >= height-border) return;

  const int glidx = y * width + x;

  int hmax = _byte_get(maxval, glidx);

  hmax -= hmax >> 3;

  _byte_put(maxval, glidx, hmax);
}

// for Markesteijn-3: only use one of two directions if one is better than the other
[shader("compute")]
[numthreads(16, 16, 1)]
void markesteijn_homo_quench(uint3 gid : SV_DispatchThreadID,
                             uniform RWStructuredBuffer<uint> homosum1, uniform RWStructuredBuffer<uint> homosum2,
                             uniform int width, uniform int height, uniform int border)
{
  const int x = gid.x;
  const int y = gid.y;

  // take sufficient border into account
  if(x < border || x >= width-border || y < border || y >= height-border) return;

  const int glidx = y * width + x;

  const int hmi1 = _byte_get(homosum1, glidx);
  const int hmi2 = _byte_get(homosum2, glidx);

  _byte_put(homosum1, glidx, (hmi1 < hmi2) ? 0 : hmi1);
  _byte_put(homosum2, glidx, (hmi1 > hmi2) ? 0 : hmi2);
}

// zero the inner part of the output, as dt_opencl_fill_image() does
[shader("compute")]
[numthreads(16, 16, 1)]
void markesteijn_zero(uint3 gid : SV_DispatchThreadID,
                      uniform ImageOut out, uniform int width, uniform int height, uniform int border)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x < border || x >= width-border || y < border || y >= height-border) return;
  out.write(int2(x, y), float4(0.0f));
}

// accumulate contributions of all directions into output image
[shader("compute")]
[numthreads(16, 16, 1)]
void markesteijn_accu(uint3 gid : SV_DispatchThreadID,
                      uniform ImageIn in, uniform ImageOut out, uniform RWStructuredBuffer<float4> rgb,
                      uniform RWStructuredBuffer<uint> homosum, uniform RWStructuredBuffer<uint> maxval,
                      uniform int width, uniform int height, uniform int border)
{
  const int x = gid.x;
  const int y = gid.y;

  // take sufficient border into account
  if(x < border || x >= width-border || y < border || y >= height-border) return;

  const int glidx = y * width + x;

  float4 pixel = in.Areadpixel(x, y);
  float4 add = rgb[glidx];
  add.w = 1.0f;

  pixel += (_byte_get(homosum, glidx) >= _byte_get(maxval, glidx)) ? add : float4(0.0f);

  out.write(int2(x, y), pixel);
}

// process the final image
[shader("compute")]
[numthreads(16, 16, 1)]
void markesteijn_final(uint3 gid : SV_DispatchThreadID,
                       uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
                       uniform int border)
{
  const int x = gid.x;
  const int y = gid.y;

  // take sufficient border into account
  if(x < border || x >= width-border || y < border || y >= height-border) return;

  float4 pixel = in.Areadpixel(x, y);

  pixel = (pixel.w > 0.0f) ? pixel / pixel.w : float4(0.0f);
  pixel.w = 0.0f;

  out.write(int2(x, y), max(float4(0.0f), pixel));
}
