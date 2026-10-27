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

// Slang port of process() in src/iop/cacorrect.c, which has no OpenCL code.
//
// The CPU code works on 128x128 tiles that overlap by 16 sensels, and each
// sensel is counted and corrected by exactly one of them, the one whose inner
// 112x112 part holds it. Inside a tile, every value it reads equals the same
// value computed over the whole image extended by 8 sensels on each side, so
// the kernels work on that extended image (the "E" buffers, 16 sensels wider
// and higher, with image row r at E row r + 8). The extension copies the
// tiles' border fill, including its corners, which are not mirrored the way
// the edges are.
//
// The per-tile sums that the CPU accumulates go to the host, which fits the
// polynomial in double precision as the CPU code does and returns the shift
// of each tile.

import common;

static const float eps = 1e-5f;
static const int TILE = 112; // ts - 2 * border in cacorrect.c

float _sqrf(float x) { return x * x; }
float _interpolatef(float a, float b, float c) { return a * (b - c) + c; }

[shader("compute")]
[numthreads(16, 16, 1)]
void cacorrect_load(uint3 gid : SV_DispatchThreadID,
                    uniform ImageIn in, uniform RWStructuredBuffer<float> out,
                    uniform int width, uniform int height, uniform float scale)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;
  out[y * width + x] = in.Areadsingle(x, y) * scale;
}

// the cropped output, scaled back
[shader("compute")]
[numthreads(16, 16, 1)]
void cacorrect_store(uint3 gid : SV_DispatchThreadID,
                     uniform StructuredBuffer<float> in, uniform ImageOut out,
                     uniform int iwidth, uniform int owidth, uniform int oheight,
                     uniform int dx, uniform int dy, uniform float scale)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= owidth || y >= oheight) return;
  out.write(int2(x, y), float4(in[(y + dy) * iwidth + x + dx] * scale));
}

// the sensel the tiles' border fill puts at image position (row, col), for
// row in [-8, height + 8) and col in [-8, width + 8)
int2 _source(int row, int col, int width, int height)
{
  const bool top = row < 0;
  const bool bottom = row >= height;
  const bool left = col < 0;
  const bool right = col >= width;
  int r = row;
  int c = col;
  if(bottom) r = 2 * height - 2 - row;
  if(right) c = 2 * width - 2 - col;
  if(top) r = left || right ? 8 - row : -row;
  if(left) c = top || bottom ? 8 - col : -col;
  return int2(c, r);
}

// the extended image of src, which is width x height with its first sensel
// at origin and stride sensels per row
[shader("compute")]
[numthreads(16, 16, 1)]
void cacorrect_extend(uint3 gid : SV_DispatchThreadID,
                      uniform StructuredBuffer<float> src, uniform RWStructuredBuffer<float> ext,
                      uniform int width, uniform int height, uniform int origin, uniform int stride)
{
  const int x = gid.x;
  const int y = gid.y;
  const int ewidth = width + 16;
  if(x >= ewidth || y >= height + 16) return;
  const int2 s = _source(y - 8, x - 8, width, height);
  ext[y * ewidth + x] = src[origin + s.y * stride + s.x];
}

// green at the red and blue sensels, a directionally weighted average; the
// tiles leave it at 0 within 3 sensels of their edges
[shader("compute")]
[numthreads(16, 16, 1)]
void cacorrect_green(uint3 gid : SV_DispatchThreadID,
                     uniform StructuredBuffer<float> raw, uniform RWStructuredBuffer<float> green,
                     uniform int ewidth, uniform int eheight, uniform uint filters)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= ewidth || y >= eheight) return;
  const int i = y * ewidth + x;
  const int c = FC(y, x, filters);
  if(c == 1)
  {
    green[i] = raw[i];
    return;
  }
  if(x < 3 || y < 3 || x >= ewidth - 3 || y >= eheight - 3)
  {
    green[i] = 0.0f;
    return;
  }
  const int v1 = ewidth;
  const int v2 = 2 * ewidth;
  const int v3 = 3 * ewidth;
  const float wtu = 1.0f / _sqrf(eps + abs(raw[i + v1] - raw[i - v1])
                                     + abs(raw[i] - raw[i - v2])
                                     + abs(raw[i - v1] - raw[i - v3]));
  const float wtd = 1.0f / _sqrf(eps + abs(raw[i - v1] - raw[i + v1])
                                     + abs(raw[i] - raw[i + v2])
                                     + abs(raw[i + v1] - raw[i + v3]));
  const float wtl = 1.0f / _sqrf(eps + abs(raw[i + 1] - raw[i - 1])
                                     + abs(raw[i] - raw[i - 2])
                                     + abs(raw[i - 1] - raw[i - 3]));
  const float wtr = 1.0f / _sqrf(eps + abs(raw[i - 1] - raw[i + 1])
                                     + abs(raw[i] - raw[i + 2])
                                     + abs(raw[i + 1] - raw[i + 3]));
  green[i] = (wtu * raw[i - v1] + wtd * raw[i + v1] + wtl * raw[i - 1] + wtr * raw[i + 1])
             / (wtu + wtd + wtl + wtr);
}

// the six high and low pass filters at the red and blue sensels, in planes of
// eheight x hwidth with one value per pair of columns:
// 0 rbhpfv, 1 rbhpfh, 2 rblpfv, 3 rblpfh, 4 grblpfv, 5 grblpfh
[shader("compute")]
[numthreads(16, 16, 1)]
void cacorrect_filters(uint3 gid : SV_DispatchThreadID,
                       uniform StructuredBuffer<float> raw, uniform StructuredBuffer<float> green,
                       uniform RWStructuredBuffer<float> filt,
                       uniform int ewidth, uniform int eheight, uniform int hwidth, uniform uint filters)
{
  const int y = gid.y;
  if(y >= eheight) return;
  const int x = 2 * int(gid.x) + (FC(y, 0, filters) & 1);
  if(x >= ewidth) return;
  const int plane = eheight * hwidth;
  const int h = y * hwidth + (x >> 1);
  if(x < 4 || y < 4 || x >= ewidth - 4 || y >= eheight - 4)
  {
    for(int p = 0; p < 6; p++) filt[p * plane + h] = 0.0f;
    return;
  }
  const int i = y * ewidth + x;
  const int v2 = 2 * ewidth;
  const int v4 = 4 * ewidth;
  filt[h] = abs(abs((green[i] - raw[i]) - (green[i + v4] - raw[i + v4]))
                + abs((green[i - v4] - raw[i - v4]) - (green[i] - raw[i]))
                - abs((green[i - v4] - raw[i - v4]) - (green[i + v4] - raw[i + v4])));
  filt[plane + h] = abs(abs((green[i] - raw[i]) - (green[i + 4] - raw[i + 4]))
                        + abs((green[i - 4] - raw[i - 4]) - (green[i] - raw[i]))
                        - abs((green[i - 4] - raw[i - 4]) - (green[i + 4] - raw[i + 4])));
  const float glpfv = 0.25f * (2.0f * green[i] + green[i + v2] + green[i - v2]);
  const float glpfh = 0.25f * (2.0f * green[i] + green[i + 2] + green[i - 2]);
  const float rblv = 0.25f * (2.0f * raw[i] + raw[i + v2] + raw[i - v2]);
  const float rblh = 0.25f * (2.0f * raw[i] + raw[i + 2] + raw[i - 2]);
  filt[2 * plane + h] = eps + abs(glpfv - rblv);
  filt[3 * plane + h] = eps + abs(glpfh - rblh);
  filt[4 * plane + h] = glpfv + rblv;
  filt[5 * plane + h] = glpfh + rblh;
}

groupshared float coeff_sums[12][256];

// the colour variance coefficients of one tile per 16x16 work group, into
// sums[tile * 12 + dir * 6 + k * 2 + color], as coeff[dir][k][color] in
// cacorrect.c
[shader("compute")]
[numthreads(16, 16, 1)]
void cacorrect_tile_sums(uint3 lid : SV_GroupThreadID, uint3 grp : SV_GroupID,
                         uniform StructuredBuffer<float> raw, uniform StructuredBuffer<float> green,
                         uniform StructuredBuffer<float> filt, uniform RWStructuredBuffer<float> sums,
                         uniform int width, uniform int height, uniform int hwidth, uniform int htiles,
                         uniform uint filters)
{
  const int l = int(lid.y * 16 + lid.x);
  const int ewidth = width + 16;
  const int plane = (height + 16) * hwidth;
  const int row0 = int(grp.y) * TILE;
  const int col0 = int(grp.x) * TILE;
  const int row1 = min(row0 + TILE, height);
  const int col1 = min(col0 + TILE, width);

  float s[12];
  for(int k = 0; k < 12; k++) s[k] = 0.0f;

  for(int row = row0 + int(lid.y); row < row1; row += 16)
  {
    const int y = row + 8;
    for(int col = col0 + (FC(row, 0, filters) & 1) + 2 * int(lid.x); col < col1; col += 32)
    {
      const int x = col + 8;
      const int i = y * ewidth + x;
      const int h = y * hwidth + (x >> 1);
      const int hv = 2 * hwidth; // two rows up or down in the filter planes
      const int ci = FC(row, col, filters) >> 1;

      // vertical
      float gdiff = 0.3125f * (green[i + ewidth] - green[i - ewidth])
                    + 0.09375f * (green[i + ewidth + 1] - green[i - ewidth + 1]
                                  + green[i + ewidth - 1] - green[i - ewidth - 1]);
      const float deltgrb = raw[i] - green[i];
      float gradwt = abs(0.25f * filt[h] + 0.125f * (filt[h + 1] + filt[h - 1]))
                     * (filt[4 * plane + h - hv] + filt[4 * plane + h + hv])
                     / (eps + 0.1f * (filt[4 * plane + h - hv] + filt[4 * plane + h + hv])
                        + filt[2 * plane + h - hv] + filt[2 * plane + h + hv]);
      s[ci] += gradwt * deltgrb * deltgrb;
      s[2 + ci] += gradwt * gdiff * deltgrb;
      s[4 + ci] += gradwt * gdiff * gdiff;

      // horizontal
      gdiff = 0.3125f * (green[i + 1] - green[i - 1])
              + 0.09375f * (green[i + 1 + ewidth] - green[i - 1 + ewidth]
                            + green[i + 1 - ewidth] - green[i - 1 - ewidth]);
      gradwt = abs(0.25f * filt[plane + h] + 0.125f * (filt[plane + h + hv] + filt[plane + h - hv]))
               * (filt[5 * plane + h - 1] + filt[5 * plane + h + 1])
               / (eps + 0.1f * (filt[5 * plane + h - 1] + filt[5 * plane + h + 1])
                  + filt[3 * plane + h - 1] + filt[3 * plane + h + 1]);
      s[6 + ci] += gradwt * deltgrb * deltgrb;
      s[8 + ci] += gradwt * gdiff * deltgrb;
      s[10 + ci] += gradwt * gdiff * gdiff;
    }
  }

  for(int k = 0; k < 12; k++) coeff_sums[k][l] = s[k];
  GroupMemoryBarrierWithGroupSync();
  for(int offset = 128; offset > 0; offset >>= 1)
  {
    if(l < offset)
      for(int k = 0; k < 12; k++) coeff_sums[k][l] += coeff_sums[k][l + offset];
    GroupMemoryBarrierWithGroupSync();
  }
  if(l < 12) sums[(int(grp.y) * htiles + int(grp.x)) * 12 + l] = coeff_sums[l][0];
}

struct Shift
{
  int vfloor, vceil, hfloor, hceil;
  float vfrac, hfrac;
};

// green at the CA shift point of extended sensel i
float _gint(StructuredBuffer<float> green, int i, int ewidth, Shift s)
{
  const float gfloor = _interpolatef(s.hfrac, green[i + s.vfloor * ewidth + s.hceil],
                                     green[i + s.vfloor * ewidth + s.hfloor]);
  const float gceil = _interpolatef(s.hfrac, green[i + s.vceil * ewidth + s.hceil],
                                    green[i + s.vceil * ewidth + s.hfloor]);
  return _interpolatef(s.vfrac, gceil, gfloor);
}

// the correction pass: red and blue from the colour differences at their
// shifted optical positions, green copied. shifts holds the vertical and
// horizontal shift of red and then of blue for each tile.
[shader("compute")]
[numthreads(16, 16, 1)]
void cacorrect_correct(uint3 gid : SV_DispatchThreadID,
                       uniform StructuredBuffer<float> raw, uniform StructuredBuffer<float> green,
                       uniform StructuredBuffer<float> shifts, uniform RWStructuredBuffer<float> out,
                       uniform int width, uniform int height, uniform int htiles, uniform uint filters)
{
  const int col = gid.x;
  const int row = gid.y;
  if(col >= width || row >= height) return;
  const int ewidth = width + 16;
  const int i = (row + 8) * ewidth + col + 8;
  const int c = FC(row, col, filters);
  if(c == 1)
  {
    out[row * width + col] = raw[i];
    return;
  }

  const int tile = (row / TILE) * htiles + col / TILE;
  const float lbv = shifts[tile * 4 + (c >> 1) * 2];
  const float lbh = shifts[tile * 4 + (c >> 1) * 2 + 1];

  Shift s;
  s.vfloor = int(floor(lbv));
  s.vceil = int(ceil(lbv));
  if(lbv < 0.0f)
  {
    const int t = s.vfloor;
    s.vfloor = s.vceil;
    s.vceil = t;
  }
  s.vfrac = abs(lbv - float(s.vfloor));
  s.hfloor = int(floor(lbh));
  s.hceil = int(ceil(lbh));
  if(lbh < 0.0f)
  {
    const int t = s.hfloor;
    s.hfloor = s.hceil;
    s.hceil = t;
  }
  s.hfrac = abs(lbh - float(s.hfloor));
  const int dirv = lbv > 0.0f ? 2 : -2;
  const int dirh = lbh > 0.0f ? 2 : -2;

  // the colour differences and shifted greens at this sensel (0), dirh
  // columns to the left (1), dirv rows up (2) and both (3)
  const int k[4] = { i, i - dirh, i - dirv * ewidth, i - dirv * ewidth - dirh };
  float grbdiff[4];
  float gshift[4];
  for(int n = 0; n < 4; n++)
  {
    gshift[n] = _gint(green, k[n], ewidth, s);
    grbdiff[n] = gshift[n] - raw[k[n]];
  }

  const float hfrac = s.hfrac / 2.0f;
  const float vfrac = s.vfrac / 2.0f;
  const float g = green[i];
  float o = raw[i];
  const float grbdiffold = g - o;
  const float grbdiffinthfloor = _interpolatef(hfrac, grbdiff[1], grbdiff[0]);
  const float grbdiffinthceil = _interpolatef(hfrac, grbdiff[3], grbdiff[2]);
  float grbdiffint = _interpolatef(vfrac, grbdiffinthceil, grbdiffinthfloor);

  const float RBint = g - grbdiffint;
  if(abs(RBint - o) < 0.25f * (RBint + o))
  {
    if(abs(grbdiffold) > abs(grbdiffint)) o = RBint;
  }
  else
  {
    const float p0 = 1.0f / (eps + abs(g - gshift[0]));
    const float p1 = 1.0f / (eps + abs(g - gshift[1]));
    const float p2 = 1.0f / (eps + abs(g - gshift[2]));
    const float p3 = 1.0f / (eps + abs(g - gshift[3]));
    grbdiffint = (p0 * grbdiff[0] + p1 * grbdiff[1] + p2 * grbdiff[2] + p3 * grbdiff[3])
                 / (p0 + p1 + p2 + p3);
    if(abs(grbdiffold) > abs(grbdiffint)) o = g - grbdiffint;
  }

  // if colour difference interpolation overshot the correction, just desaturate
  if(grbdiffold * grbdiffint < 0.0f) o = g - 0.5f * (grbdiffold + grbdiffint);
  out[row * width + col] = o;
}

// the colorshift avoidance: per red and blue sensel, the ratio of its value
// before and after the correction, in half-size planes. (prow, pcol) is the
// parity of the plane's sensels, and the odd last row and column of the
// planes repeat the ones before them, as in process()
[shader("compute")]
[numthreads(16, 16, 1)]
void cacorrect_factors(uint3 gid : SV_DispatchThreadID,
                       uniform StructuredBuffer<float> before, uniform StructuredBuffer<float> after,
                       uniform ImageOut factors, uniform int width, uniform int height,
                       uniform int hwidth, uniform int hheight, uniform int prow, uniform int pcol,
                       uniform int fixrow, uniform int fixcol)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= hwidth || y >= hheight) return;
  const int fx = fixcol != 0 && x == hwidth - 1 ? x - 1 : x;
  const int fy = fixrow != 0 && y == hheight - 1 ? y - 1 : y;
  const int row = 2 * fy + prow;
  const int col = 2 * fx + pcol;
  float f = 0.0f;
  if(row < height && col < width)
  {
    const int i = row * width + col;
    // CLAMPF() turns the NaN of 0 / 0 into the lower bound
    const float q = before[i] / after[i];
    f = q >= 0.5f ? min(q, 2.0f) : 0.5f;
  }
  factors.write(int2(x, y), float4(f));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void cacorrect_apply_factors(uint3 gid : SV_DispatchThreadID,
                             uniform RWStructuredBuffer<float> buf, uniform ImageIn red, uniform ImageIn blue,
                             uniform int width, uniform int height, uniform uint filters)
{
  const int col = gid.x;
  const int row = gid.y;
  if(col >= width - 2 || row < 2 || row >= height - 2) return;
  const int c = FC(row, col, filters);
  if(c == 1) return;
  const float f = c == 0 ? red.Areadsingle(col / 2, row / 2) : blue.Areadsingle(col / 2, row / 2);
  buf[row * width + col] *= f;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void cacorrect_copy(uint3 gid : SV_DispatchThreadID,
                    uniform StructuredBuffer<float> in, uniform RWStructuredBuffer<float> out,
                    uniform int width, uniform int height)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;
  out[y * width + x] = in[y * width + x];
}
