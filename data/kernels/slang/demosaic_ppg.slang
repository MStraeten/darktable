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

// Slang port of the kernels in ../demosaic_ppg.cl that RCD uses for the image
// borders. The OpenCL kernels stage their neighborhood in local memory; these
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
      sum[f] += in.readsingle(i, j);
      count[f]++;
    }
  }

  const float i = max(0.0f, in.readsingle(x, y));
  o.x = count[0] > 0 ? sum[0]/count[0] : i;
  o.y = count[1]+count[3] > 0 ? (sum[1]+sum[3])/(count[1]+count[3]) : i;
  o.z = count[2] > 0 ? sum[2]/count[2] : i;

  const int f = FC(y,x,filters);

  if     (f == 0) o.x = i;
  else if(f == 1) o.y = i;
  else if(f == 2) o.z = i;
  else            o.y = i;

  out.write(int2(x, y), max(float4(0.0f), o));
}
