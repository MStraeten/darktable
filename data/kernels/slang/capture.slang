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

// Slang port of ../capture.cl, the capture sharpening of demosaic

import common;
import capture_common;

static const int CAPTURE_KERNEL_ALIGN = 32;
static const float CAPTURE_YMIN = 0.001f;
static const int CAPTURE_SMALL_ULIM = 66;

// the table holds one byte per pixel, packed four to a word
int _table(StructuredBuffer<uint> table, int i)
{
  return int((table[i >> 2] >> uint((i & 3) * 8)) & 0xffu);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void kernel_9x9_mul(uint3 gid : SV_DispatchThreadID,
                    uniform StructuredBuffer<float> in, uniform RWStructuredBuffer<float> out,
                    uniform StructuredBuffer<float> blend, uniform StructuredBuffer<float> kernels,
                    uniform StructuredBuffer<uint> table, uniform int w1, uniform int height)
{
  const int col = gid.x;
  const int row = gid.y;
  if(col >= w1 || row >= height) return;

  const int i = row * w1 + col;
  if(blend[i] <= 0.0f) return;

  const int t = _table(table, i);
  out[i] *= sum_9x9(in, kernels, CAPTURE_KERNEL_ALIGN * t, col, row, w1, height, t < CAPTURE_SMALL_ULIM);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void kernel_9x9_div(uint3 gid : SV_DispatchThreadID,
                    uniform StructuredBuffer<float> in, uniform RWStructuredBuffer<float> out,
                    uniform StructuredBuffer<float> luminance, uniform StructuredBuffer<float> blend,
                    uniform StructuredBuffer<float> kernels, uniform StructuredBuffer<uint> table,
                    uniform int w1, uniform int height)
{
  const int col = gid.x;
  const int row = gid.y;
  if(col >= w1 || row >= height) return;

  const int i = row * w1 + col;
  if(blend[i] <= 0.0f) return;

  const int t = _table(table, i);
  const float val = sum_9x9(in, kernels, CAPTURE_KERNEL_ALIGN * t, col, row, w1, height, t < CAPTURE_SMALL_ULIM);
  out[i] = luminance[i] / max(val, CAPTURE_YMIN);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void prepare_blend(uint3 gid : SV_DispatchThreadID,
                   uniform ImageIn cfa, uniform ImageIn dev_out, uniform uint filters,
                   uniform StructuredBuffer<uint> xtrans, uniform RWStructuredBuffer<float> mask,
                   uniform RWStructuredBuffer<float> Yold, uniform float4 wb, uniform int w, uniform int height)
{
  const int col = gid.x;
  const int row = gid.y;
  if(col >= w || row >= height) return;

  const float whites[4] = { wb.x, wb.y, wb.z, wb.w };

  float4 rgb = dev_out.Areadpixel(col, row);
  // Photometric/digital ITU BT.709
  rgb *= float4(0.212671f, 0.715160f, 0.072169f, 0.0f);
  const float Y = max(0.0f, rgb.x + rgb.y + rgb.z);
  const int k = row * w + col;
  Yold[k] = Y;

  if(row > 1 && col > 1 && (row < height - 2) && (col < w - 2))
  {
    const int w2 = 2 * w;
    const int color = fcol(row, col, filters, xtrans);
    const float val = cfa.Areadsingle(col, row);
    // neighboring work items write the same zeros, so the overlap is harmless
    if(val > whites[color] || Y < CAPTURE_YMIN)
    {
      mask[k-w2-1] = 0.0f; mask[k-w2] = 0.0f; mask[k-w2+1] = 0.0f;
      mask[k-w-2] = 0.0f; mask[k-w-1] = 0.0f; mask[k-w] = 0.0f; mask[k-w+1] = 0.0f; mask[k-w+2] = 0.0f;
      mask[k-2] = 0.0f; mask[k-1] = 0.0f; mask[k] = 0.0f; mask[k+1] = 0.0f; mask[k+2] = 0.0f;
      mask[k+w-2] = 0.0f; mask[k+w-1] = 0.0f; mask[k+w] = 0.0f; mask[k+w+1] = 0.0f; mask[k+w+2] = 0.0f;
      mask[k+w2-1] = 0.0f; mask[k+w2] = 0.0f; mask[k+w2+1] = 0.0f;
    }
  }
  else
    mask[k] = 0.0f;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void modify_blend(uint3 gid : SV_DispatchThreadID,
                  uniform RWStructuredBuffer<float> blend, uniform StructuredBuffer<float> Yold,
                  uniform RWStructuredBuffer<float> luminance, uniform float dthresh,
                  uniform int width, uniform int height)
{
  const int icol = gid.x;
  const int irow = gid.y;
  if(icol >= width || irow >= height) return;

  const int row = clamp(irow, 2, height - 3);
  const int col = clamp(icol, 2, width - 3);

  const float threshold = 0.6f * fsquare(dthresh);
  const float tscale = 200.0f;
  const float offset = -2.5f + tscale * threshold / 2.0f;

  float sum = 0.0f;
  float sum_sq = 0.0f;
  for(int y = row - 1; y < row + 2; y++)
  {
    for(int x = col - 2; x < col + 3; x++)
    {
      sum += Yold[y * width + x];
      sum_sq += fsquare(Yold[y * width + x]);
    }
  }
  for(int x = col - 1; x < col + 2; x++)
  {
    sum += Yold[(row - 2) * width + x];
    sum_sq += fsquare(Yold[(row - 2) * width + x]);
    sum += Yold[(row + 2) * width + x];
    sum_sq += fsquare(Yold[(row + 2) * width + x]);
  }

  const int k = irow * width + icol;
  const float sum_of_squares = max(0.0f, sum_sq - fsquare(sum) / 21.0f);
  const float std_deviation = dtcl_sqrt(sum_of_squares / 21.0f);
  const float modified_coef_variation = std_deviation / dtcl_sqrt(max(NORM_MIN, sum / 21.0f));
  const float t = dtcl_log(1.0f + modified_coef_variation);
  const float weight = 1.0f / (1.0f + dtcl_exp(offset - tscale * t));
  blend[k] = clipf(blend[k] * 1.01011f * (weight - 0.01f));
  luminance[k] = Yold[k];
}

// a line kernel, laid out 16 wide as dt_slang_gpu_run_1d_args() dispatches it
[shader("compute")]
[numthreads(16, 16, 1)]
void final_blend(uint3 gid : SV_DispatchThreadID,
                 uniform RWStructuredBuffer<float> blendmask, uniform StructuredBuffer<float> unblurred,
                 uniform int pixels)
{
  const int k = gid.y * 16 + gid.x;
  if(k >= pixels) return;

  const float diff = unblurred[k] - blendmask[k];
  const float w_tmp2 = 1.0f / (1.0f + dtcl_exp(5.0f - 10.0f * diff));
  blendmask[k] = clipf(w_tmp2 * unblurred[k] + (1.0f - w_tmp2) * blendmask[k]);
}

// the blend or sigma mask in the alpha channel, for the mask views
[shader("compute")]
[numthreads(16, 16, 1)]
void show_blend_mask(uint3 gid : SV_DispatchThreadID,
                     uniform ImageIn in, uniform ImageOut out, uniform StructuredBuffer<float> blend_mask,
                     uniform StructuredBuffer<uint> sigma_mask, uniform int width, uniform int height,
                     uniform int blender)
{
  const int col = gid.x;
  const int row = gid.y;
  if(col >= width || row >= height) return;

  float4 pix = max(float4(0.0f), in.Areadpixel(col, row));
  const int k = row * width + col;
  pix.w = blender != 0 ? blend_mask[k] : float(_table(sigma_mask, k)) / 255.0f;
  out.write(int2(col, row), pix);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void capture_result(uint3 gid : SV_DispatchThreadID,
                    uniform ImageIn in, uniform ImageOut out, uniform StructuredBuffer<float> blendmask,
                    uniform StructuredBuffer<float> luminance, uniform StructuredBuffer<float> tmp,
                    uniform int width, uniform int height)
{
  const int col = gid.x;
  const int row = gid.y;
  if(col >= width || row >= height) return;

  float4 pix = in.Areadpixel(col, row);
  const int k = row * width + col;

  if(blendmask[k] > 0.0f)
  {
    const float mixer = clipf(blendmask[k]);
    const float luminance_new = lerp(luminance[k], tmp[k], mixer);
    pix *= float4(luminance_new / max(luminance[k], CAPTURE_YMIN));
  }
  out.write(int2(col, row), pix);
}

// auto radius, see capture_radius in ../capture.cl: the sensor value at
// (row, col) of the tested area, white balance undone
float _radius_val(ImageIn in, StructuredBuffer<uint> xtrans, uint filters, float4 coeff,
                  int dx, int dy, int row, int col)
{
  const int c = fcol(row, col, filters, xtrans);
  const float f = c == 0 ? coeff.x : (c == 1 ? coeff.y : (c == 2 ? coeff.z : coeff.w));
  return f * in.readsingle(col + dx, row + dy);
}

static const float RAWEPS = 0.005f;
static const float lowerLimit = 0.01f;
static const float upperLimit = 0.9f;

#define V(r, c) _radius_val(in, xtrans, filters, coeff, dx, dy, row + (r), col + (c))

// largest ratio of neighbouring greens at one site, same tests as _calcRadiusBayer()
float _radius_bayer(ImageIn in, StructuredBuffer<uint> xtrans, uint filters, float4 coeff,
                    int dx, int dy, int row, int col)
{
  float maxRatio = 1.0f;
  const float val00 = V(0, 0);
  if(val00 <= RAWEPS) return maxRatio;

  const float val1m1 = V(1, -1);
  const float val1p1 = V(1, 1);
  const float maxVal0 = max(val00, val1m1);
  if(val1m1 > RAWEPS && maxVal0 > lowerLimit)
  {
    const float minVal = min(val00, val1m1);
    if(maxVal0 > maxRatio * minVal)
    {
      const float clip = maxVal0 == val00
                       ? max(max(V(-1, -1), V(-1, 1)), val1p1)
                       : max(max(max(V(0, -2), val00), V(2, -2)), V(2, 0));
      if(clip < upperLimit)
        maxRatio = maxVal0 / minVal;
    }
  }

  const float maxVal1 = max(val00, val1p1);
  if(val1p1 > RAWEPS && maxVal1 > lowerLimit)
  {
    const float minVal = min(val00, val1p1);
    if(maxVal1 > maxRatio * minVal)
    {
      const float clip = maxVal1 == val00
                       ? max(max(V(-1, -1), V(-1, 1)), val1p1)
                       : max(max(max(val00, V(0, 2)), V(2, 0)), V(2, 2));
      if(clip < upperLimit)
        maxRatio = maxVal1 / minVal;
    }
  }
  return maxRatio;
}

float _radius_pair(float a, float b, float maxRatio)
{
  const float maxVal = max(a, b);
  const float minVal = min(a, b);
  return maxVal > lowerLimit && maxVal > maxRatio * minVal ? maxVal / minVal : maxRatio;
}

// same tests as _calcRadiusXtrans()
float _radius_xtrans(ImageIn in, StructuredBuffer<uint> xtrans, uint filters, float4 coeff,
                     int dx, int dy, int row, int col)
{
  float maxRatio = 1.0f;
  const float valp1p1 = V(1, 1);
  const bool squareClipped = max(max(max(valp1p1, V(1, 2)), V(2, 1)), V(2, 2)) >= upperLimit;
  const float greenSolitary = V(0, 0);
  if(greenSolitary > RAWEPS && max(V(-1, -1), V(-1, 1)) < upperLimit)
  {
    if(greenSolitary < upperLimit)
    {
      const float valp1m1 = V(1, -1);
      if(valp1m1 > RAWEPS && max(max(max(V(1, -2), valp1m1), V(2, -2)), V(1, -1)) < upperLimit)
        maxRatio = _radius_pair(greenSolitary, valp1m1, maxRatio);
      if(valp1p1 > RAWEPS && !squareClipped)
        maxRatio = _radius_pair(greenSolitary, valp1p1, maxRatio);
    }
  }

  if(!squareClipped)
  {
    const float valp2p2 = V(2, 2);
    if(valp2p2 > RAWEPS)
    {
      if(valp1p1 > RAWEPS)
        maxRatio = _radius_pair(valp1p1, valp2p2, maxRatio);
      const float greenSolitaryRight = V(3, 3);
      if(max(max(greenSolitaryRight, V(4, 2)), V(4, 4)) < upperLimit && greenSolitaryRight > RAWEPS)
        maxRatio = _radius_pair(greenSolitaryRight, valp2p2, maxRatio);
    }
    const float valp1p2 = V(1, 2);
    const float valp2p1 = V(2, 1);
    if(valp2p1 > RAWEPS)
    {
      if(valp1p2 > RAWEPS)
        maxRatio = _radius_pair(valp1p2, valp2p1, maxRatio);
      const float greenSolitaryLeft = V(3, 0);
      if(max(max(greenSolitaryLeft, V(4, -1)), V(4, 1)) < upperLimit && greenSolitaryLeft > RAWEPS)
        maxRatio = _radius_pair(greenSolitaryLeft, valp2p1, maxRatio);
    }
  }
  return maxRatio;
}

#undef V

groupshared float radius_max[256];

// one work item per tested site, each 16x16 work group writes the largest
// ratio it found to partial
[shader("compute")]
[numthreads(16, 16, 1)]
void capture_radius(uint3 gid : SV_DispatchThreadID, uint3 lid : SV_GroupThreadID, uint3 grp : SV_GroupID,
                    uniform ImageIn in, uniform StructuredBuffer<uint> xtrans, uniform uint filters,
                    uniform float4 coeff, uniform int dx, uniform int dy, uniform int width, uniform int height,
                    uniform int startx, uniform int starty, uniform int groups_x,
                    uniform RWStructuredBuffer<float> partial)
{
  const int x = gid.x;
  const int y = gid.y;
  const int l = int(lid.y * 16 + lid.x);

  float maxRatio = 1.0f;
  if(filters == 9u)
  {
    const int row = starty + 2 + 3 * y;
    const int col = startx + 2 + 3 * x;
    if(row < height - 4 && col < width - 4)
      maxRatio = _radius_xtrans(in, xtrans, filters, coeff, dx, dy, row, col);
  }
  else
  {
    const int row = 4 + y;
    const int col = (filters != 0u ? 5 + (FC(row & 1, 0, filters) & 1) : 5) + 2 * x;
    if(row < height - 4 && col < width - 4)
      maxRatio = _radius_bayer(in, xtrans, filters, coeff, dx, dy, row, col);
  }

  radius_max[l] = maxRatio;
  GroupMemoryBarrierWithGroupSync();

  for(int offset = 128; offset > 0; offset >>= 1)
  {
    if(l < offset) radius_max[l] = max(radius_max[l], radius_max[l + offset]);
    GroupMemoryBarrierWithGroupSync();
  }

  if(l == 0) partial[grp.y * groups_x + grp.x] = radius_max[0];
}

// the largest of the length partial maxima of capture_radius(), into
// result[0], in a single 16x16 work group. OpenCL leaves this step to the host
[shader("compute")]
[numthreads(16, 16, 1)]
void capture_radius_reduce(uint3 lid : SV_GroupThreadID,
                           uniform StructuredBuffer<float> partial, uniform RWStructuredBuffer<float> result,
                           uniform int length)
{
  const int l = int(lid.y * 16 + lid.x);
  float m = 1.0f;
  for(int k = l; k < length; k += 256)
    m = max(m, partial[k]);
  radius_max[l] = m;
  GroupMemoryBarrierWithGroupSync();

  for(int offset = 128; offset > 0; offset >>= 1)
  {
    if(l < offset) radius_max[l] = max(radius_max[l], radius_max[l + offset]);
    GroupMemoryBarrierWithGroupSync();
  }

  if(l == 0) result[0] = radius_max[0];
}
