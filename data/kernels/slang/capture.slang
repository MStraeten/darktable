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

// Slang port of ../capture.cl, the capture sharpening of demosaic. The
// sigma mask view (show_blend_mask) stays on the CPU.

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
