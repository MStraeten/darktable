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

// the statistics of dt_color_picker_helper() in src/common/color_picker.c,
// reduced on the device. OpenCL has no such kernel: it reads the box back
// and runs the CPU code on it.
//
// Each work item reduces one tile of the box and writes 16 floats: the sums,
// minima, maxima and pixel counts of the four channels. The host adds up the
// tiles, so only those come back, never the image.

import color_conversion;

static const float PICK_FLT_MAX = 3.402823466e+38f;
static const int PICK_PLAIN = 0;
static const int PICK_LCH = 1;
static const int PICK_HSL = 2;
static const int PICK_JZCZHZ = 3;

static const float filter[5] = { 1.0f / 16.0f, 4.0f / 16.0f, 6.0f / 16.0f, 4.0f / 16.0f, 1.0f / 16.0f };

struct Stats
{
  float4 sum;
  float4 low;
  float4 high;
  float4 count;

  [mutating]
  void init()
  {
    sum = float4(0.0f);
    low = float4(PICK_FLT_MAX);
    high = float4(-PICK_FLT_MAX);
    count = float4(0.0f);
  }

  [mutating]
  void add(float4 v)
  {
    sum += v;
    low = min(low, v);
    high = max(high, v);
    count += float4(1.0f);
  }

  [mutating]
  void add_channel(int c, float v)
  {
    sum[c] += v;
    low[c] = min(low[c], v);
    high[c] = max(high[c], v);
    count[c] += 1.0f;
  }

  void store(RWStructuredBuffer<float> stats, int item)
  {
    for(int k = 0; k < 4; k++)
    {
      stats[item * 16 + k] = sum[k];
      stats[item * 16 + 4 + k] = low[k];
      stats[item * 16 + 8 + k] = high[k];
      stats[item * 16 + 12 + k] = count[k];
    }
  }
}

// blur_2D_Bspline() at one scale and without clipping, as the picker's
// denoise does over the whole image: first over columns, then over rows,
// with the edges clamped to the image
float4 _denoised(ImageIn in, int x, int y, int width, int height)
{
  float4 acc = float4(0.0f);
  for(int ii = 0; ii < 5; ii++)
  {
    const int xx = clamp(x + ii - 2, 0, width - 1);
    float4 v = float4(0.0f);
    for(int jj = 0; jj < 5; jj++)
      v += filter[jj] * in.Areadpixel(xx, clamp(y + jj - 2, 0, height - 1));
    acc += filter[ii] * v;
  }
  return acc;
}

// dt_Lab_2_LCH()
float4 _lch(float4 Lab)
{
  const float h = atan2(Lab.z, Lab.y);
  const float H = h > 0.0f ? h / DT_2PI_F : 1.0f - abs(h) / DT_2PI_F;
  return float4(Lab.x, sqrt(Lab.y * Lab.y + Lab.z * Lab.z), H, 0.0f);
}

// dt_RGB_2_HSL()
float4 _hsl(float4 RGB)
{
  const float mn = min(RGB.x, min(RGB.y, RGB.z));
  const float mx = max(RGB.x, max(RGB.y, RGB.z));
  const float delta = mx - mn;
  const float L = (mx + mn) / 2.0f;
  float H = 0.0f;
  float S = 0.0f;
  if(abs(mx) > 1e-6f && abs(delta) > 1e-6f)
  {
    S = L < 0.5f ? delta / (mx + mn) : delta / (2.0f - mx - mn);
    float hue;
    if(RGB.x == mx)      hue = (RGB.y - RGB.z) / delta;
    else if(RGB.y == mx) hue = 2.0f + (RGB.z - RGB.x) / delta;
    else                 hue = 4.0f + (RGB.x - RGB.y) / delta;
    hue /= 6.0f;
    const float h = hue - floor(hue);
    H = h < 1.0f ? h : 0.0f;
  }
  return float4(H, S, L, 0.0f);
}

// rgb_to_JzCzhz() of color_picker.c: the profile's matrix to XYZ D50, then
// the Bradford adaptation to D65
float4 _jzczhz(float4 rgb, ProfileInfo info, ImageIn lut)
{
  const float4 D50 = rgb_matrix_to_xyz(rgb, info, info.matrix_in, lut);
  const float4 D65 = float4(0.9555766f * D50.x - 0.0230393f * D50.y + 0.0631636f * D50.z,
                            -0.0282895f * D50.x + 1.0099416f * D50.y + 0.0210077f * D50.z,
                            0.0122982f * D50.x - 0.0204830f * D50.y + 1.3299098f * D50.z,
                            0.0f);
  const float4 JzAzBz = XYZ_to_JzAzBz(D65);
  const float h = atan2(JzAzBz.z, JzAzBz.y) / DT_2PI_F;
  return float4(JzAzBz.x, sqrt(JzAzBz.y * JzAzBz.y + JzAzBz.z * JzAzBz.z), h >= 0.0f ? h : 1.0f + h, 0.0f);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void color_picker_stats_4ch(uint3 gid : SV_DispatchThreadID,
                            uniform ImageIn in, uniform RWStructuredBuffer<float> stats,
                            uniform int width, uniform int height, uniform int4 box,
                            uniform int2 tile, uniform int2 tiles, uniform int mode, uniform int denoise,
                            uniform StructuredBuffer<ProfileInfo> profile_info, uniform ImageIn lut)
{
  const int tx = gid.x;
  const int ty = gid.y;
  if(tx >= tiles.x || ty >= tiles.y) return;

  const ProfileInfo info = profile_info[0];
  const int x0 = box.x + tx * tile.x;
  const int y0 = box.y + ty * tile.y;
  const int x1 = min(x0 + tile.x, box.z);
  const int y1 = min(y0 + tile.y, box.w);

  Stats s;
  s.init();
  for(int y = y0; y < y1; y++)
    for(int x = x0; x < x1; x++)
    {
      const float4 p = denoise != 0 ? _denoised(in, x, y, width, height) : in.Areadpixel(x, y);
      if(mode == PICK_PLAIN)
        s.add(p);
      else
      {
        float4 pick = mode == PICK_LCH ? _lch(p) : (mode == PICK_HSL ? _hsl(p) : _jzczhz(p, info, lut));
        // the hue shifted by half a turn, for sensible minima and maxima,
        // as _update_stats_4ch() does
        pick.w = pick.z < 0.5f ? pick.z + 0.5f : pick.z - 0.5f;
        s.add(pick);
      }
    }
  s.store(stats, ty * tiles.x + tx);
}

// a Bayer or X-Trans mosaic, per CFA color. filters is 9 for X-Trans, whose
// 6x6 pattern is in xtrans
[shader("compute")]
[numthreads(16, 16, 1)]
void color_picker_stats_mosaic(uint3 gid : SV_DispatchThreadID,
                               uniform ImageIn in, uniform RWStructuredBuffer<float> stats,
                               uniform int4 box, uniform int2 tile, uniform int2 tiles,
                               uniform int2 offset, uniform uint filters,
                               uniform StructuredBuffer<int> xtrans)
{
  const int tx = gid.x;
  const int ty = gid.y;
  if(tx >= tiles.x || ty >= tiles.y) return;

  const int x0 = box.x + tx * tile.x;
  const int y0 = box.y + ty * tile.y;
  const int x1 = min(x0 + tile.x, box.z);
  const int y1 = min(y0 + tile.y, box.w);

  Stats s;
  s.init();
  for(int y = y0; y < y1; y++)
    for(int x = x0; x < x1; x++)
    {
      // FC() and FCxtrans() in src/develop/imageop_math.h
      const uint row = uint(y + offset.y);
      const uint col = uint(x + offset.x);
      const int c = filters == 9u
        ? xtrans[((y + 600 + offset.y) % 6) * 6 + (x + 600 + offset.x) % 6]
        : int((filters >> ((((row << 1) & 14u) + (col & 1u)) << 1)) & 3u);
      s.add_channel(c, in.Areadpixel(x, y).x);
    }
  s.store(stats, ty * tiles.x + tx);
}
