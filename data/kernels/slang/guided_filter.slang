/*
    This file is part of darktable,
    copyright (c) 2019 Heiko Bauke
    Copyright (C) 2023-2026 darktable developers.

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

// Slang port of ../guided_filter.cl. The box means run one work item per row
// or column in OpenCL, which leaves most of a GPU idle; here each work item
// takes a chunk of a row or column, and starts its running sum anew.

import common;

static const float FLT_EPSILON = 1.19209290e-7f;

[shader("compute")]
[numthreads(16, 16, 1)]
void guided_filter_split_rgb_image(uint3 gid : SV_DispatchThreadID,
                                   uniform int width, uniform int height, uniform int first,
                                   uniform ImageIn guide,
                                   uniform ImageOut out_r, uniform ImageOut out_g, uniform ImageOut out_b,
                                   uniform float guide_weight)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float4 weight = float4(guide_weight, guide_weight, guide_weight, 0.0f);
  const float4 pixel = weight * min(float4(100.0f), guide.readpixel(x, y + first));
  out_r.write(int2(x, y), float4(pixel.x));
  out_g.write(int2(x, y), float4(pixel.y));
  out_b.write(int2(x, y), float4(pixel.z));
}

// the means over the 2w+1 windows, clipped to the line of n values, at
// positions i0 to i1 - 1 along x (along_x) or y
void _box_mean(ImageIn in, ImageOut out, int line, int n, int i0, int i1, int w, bool along_x)
{
  float m = 0.f, n_box = 0.f, c = 0.f;
  for(int j = max(i0 - w, 0), j_end = min(i0 + w + 1, n); j < j_end; j++)
  {
    Kahan_sum(m, c, in.readsingle(along_x ? j : line, along_x ? line : j));
    n_box += 1.f;
  }
  for(int i = i0; i < i1; i++)
  {
    out.write(along_x ? int2(i, line) : int2(line, i), float4(m / n_box));
    if(i - w >= 0)
    {
      Kahan_sum(m, c, -in.readsingle(along_x ? i - w : line, along_x ? line : i - w));
      n_box -= 1.f;
    }
    if(i + w + 1 < n)
    {
      Kahan_sum(m, c, in.readsingle(along_x ? i + w + 1 : line, along_x ? line : i + w + 1));
      n_box += 1.f;
    }
  }
}

// work item x, y takes chunk values of row y from x * chunk on
[shader("compute")]
[numthreads(16, 16, 1)]
void guided_filter_box_mean_x(uint3 gid : SV_DispatchThreadID,
                              uniform int width, uniform int height,
                              uniform ImageIn in, uniform ImageOut out, uniform int w, uniform int chunk)
{
  const int x0 = gid.x * chunk;
  const int y = gid.y;
  if(x0 >= width || y >= height) return;
  _box_mean(in, out, y, width, x0, min(x0 + chunk, width), w, true);
}

// work item x, y takes chunk values of column x from y * chunk on
[shader("compute")]
[numthreads(16, 16, 1)]
void guided_filter_box_mean_y(uint3 gid : SV_DispatchThreadID,
                              uniform int width, uniform int height,
                              uniform ImageIn in, uniform ImageOut out, uniform int w, uniform int chunk)
{
  const int x = gid.x;
  const int y0 = gid.y * chunk;
  if(x >= width || y0 >= height) return;
  _box_mean(in, out, x, height, y0, min(y0 + chunk, height), w, false);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void guided_filter_covariances(uint3 gid : SV_DispatchThreadID,
                               uniform int width, uniform int height, uniform int first,
                               uniform ImageIn guide, uniform ImageIn img,
                               uniform ImageOut cov_imgg_img_r, uniform ImageOut cov_imgg_img_g,
                               uniform ImageOut cov_imgg_img_b,
                               uniform float guide_weight)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float4 weight = float4(guide_weight, guide_weight, guide_weight, 0.0f);
  const float img_ = img.Areadsingle(x, y);
  const float4 imgv = float4(img_, img_, img_, 0.0f);
  const float4 pixel = imgv * weight * min(float4(100.0f), guide.readpixel(x, y + first));
  cov_imgg_img_r.write(int2(x, y), float4(pixel.x));
  cov_imgg_img_g.write(int2(x, y), float4(pixel.y));
  cov_imgg_img_b.write(int2(x, y), float4(pixel.z));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void guided_filter_variances(uint3 gid : SV_DispatchThreadID,
                             uniform int width, uniform int height, uniform int first,
                             uniform ImageIn guide,
                             uniform ImageOut var_imgg_rr, uniform ImageOut var_imgg_rg,
                             uniform ImageOut var_imgg_rb, uniform ImageOut var_imgg_gg,
                             uniform ImageOut var_imgg_gb, uniform ImageOut var_imgg_bb,
                             uniform float guide_weight)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float4 weight = float4(guide_weight, guide_weight, guide_weight, 0.0f);
  const float4 pixel = weight * min(float4(100.0f), guide.readpixel(x, y + first));
  var_imgg_rr.write(int2(x, y), float4(pixel.x * pixel.x));
  var_imgg_rg.write(int2(x, y), float4(pixel.x * pixel.y));
  var_imgg_rb.write(int2(x, y), float4(pixel.x * pixel.z));
  var_imgg_gg.write(int2(x, y), float4(pixel.y * pixel.y));
  var_imgg_gb.write(int2(x, y), float4(pixel.y * pixel.z));
  var_imgg_bb.write(int2(x, y), float4(pixel.z * pixel.z));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void guided_filter_update_covariance(uint3 gid : SV_DispatchThreadID,
                                     uniform int width, uniform int height,
                                     uniform ImageIn in, uniform ImageOut out,
                                     uniform ImageIn a, uniform ImageIn b,
                                     uniform float eps)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float i_val = in.Areadsingle(x, y);
  const float a_val = a.Areadsingle(x, y);
  const float b_val = b.Areadsingle(x, y);
  out.write(int2(x, y), float4(i_val - a_val * b_val + eps));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void guided_filter_solve(uint3 gid : SV_DispatchThreadID,
                         uniform int width, uniform int height,
                         uniform ImageIn img_mean,
                         uniform ImageIn imgg_mean_r, uniform ImageIn imgg_mean_g, uniform ImageIn imgg_mean_b,
                         uniform ImageIn cov_imgg_img_r, uniform ImageIn cov_imgg_img_g,
                         uniform ImageIn cov_imgg_img_b,
                         uniform ImageIn var_imgg_rr, uniform ImageIn var_imgg_rg, uniform ImageIn var_imgg_rb,
                         uniform ImageIn var_imgg_gg, uniform ImageIn var_imgg_gb, uniform ImageIn var_imgg_bb,
                         uniform ImageOut a_r, uniform ImageOut a_g, uniform ImageOut a_b, uniform ImageOut b)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float Sigma_0_0 = var_imgg_rr.Areadsingle(x, y);
  const float Sigma_0_1 = var_imgg_rg.Areadsingle(x, y);
  const float Sigma_0_2 = var_imgg_rb.Areadsingle(x, y);
  const float Sigma_1_1 = var_imgg_gg.Areadsingle(x, y);
  const float Sigma_1_2 = var_imgg_gb.Areadsingle(x, y);
  const float Sigma_2_2 = var_imgg_bb.Areadsingle(x, y);
  const float cov_imgg_img[3] = { cov_imgg_img_r.Areadsingle(x, y),
                                  cov_imgg_img_g.Areadsingle(x, y),
                                  cov_imgg_img_b.Areadsingle(x, y) };
  const float det0 = Sigma_0_0 * (Sigma_1_1 * Sigma_2_2 - Sigma_1_2 * Sigma_1_2)
                     - Sigma_0_1 * (Sigma_0_1 * Sigma_2_2 - Sigma_0_2 * Sigma_1_2)
                     + Sigma_0_2 * (Sigma_0_1 * Sigma_1_2 - Sigma_0_2 * Sigma_1_1);
  float a_r_ = 0.0f;
  float a_g_ = 0.0f;
  float a_b_ = 0.0f;
  float b_ = img_mean.Areadsingle(x, y);
  if(abs(det0) > 4.f * FLT_EPSILON)
  {
    const float det1 = cov_imgg_img[0] * (Sigma_1_1 * Sigma_2_2 - Sigma_1_2 * Sigma_1_2)
                       - Sigma_0_1 * (cov_imgg_img[1] * Sigma_2_2 - cov_imgg_img[2] * Sigma_1_2)
                       + Sigma_0_2 * (cov_imgg_img[1] * Sigma_1_2 - cov_imgg_img[2] * Sigma_1_1);
    const float det2 = Sigma_0_0 * (cov_imgg_img[1] * Sigma_2_2 - cov_imgg_img[2] * Sigma_1_2)
                       - cov_imgg_img[0] * (Sigma_0_1 * Sigma_2_2 - Sigma_0_2 * Sigma_1_2)
                       + Sigma_0_2 * (Sigma_0_1 * cov_imgg_img[2] - Sigma_0_2 * cov_imgg_img[1]);
    const float det3 = Sigma_0_0 * (Sigma_1_1 * cov_imgg_img[2] - Sigma_1_2 * cov_imgg_img[1])
                       - Sigma_0_1 * (Sigma_0_1 * cov_imgg_img[2] - Sigma_0_2 * cov_imgg_img[1])
                       + cov_imgg_img[0] * (Sigma_0_1 * Sigma_1_2 - Sigma_0_2 * Sigma_1_1);
    a_r_ = det1 / det0;
    a_g_ = det2 / det0;
    a_b_ = det3 / det0;
    b_ = b_
         - a_r_ * imgg_mean_r.Areadsingle(x, y)
         - a_g_ * imgg_mean_g.Areadsingle(x, y)
         - a_b_ * imgg_mean_b.Areadsingle(x, y);
  }

  a_r.write(int2(x, y), float4(a_r_));
  a_g.write(int2(x, y), float4(a_g_));
  a_b.write(int2(x, y), float4(a_b_));
  b.write(int2(x, y), float4(b_));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void guided_filter_generate_result(uint3 gid : SV_DispatchThreadID,
                                   uniform int width, uniform int height, uniform int first,
                                   uniform ImageIn guide,
                                   uniform ImageIn a_r, uniform ImageIn a_g, uniform ImageIn a_b, uniform ImageIn b,
                                   uniform ImageOut res,
                                   uniform float guide_weight, uniform float minval, uniform float maxval)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float4 pixel = min(float4(100.0f), guide.readpixel(x, y + first));
  const float a_r_ = pixel.x * a_r.Areadsingle(x, y);
  const float a_g_ = pixel.y * a_g.Areadsingle(x, y);
  const float a_b_ = pixel.z * a_b.Areadsingle(x, y);
  const float b_ = b.Areadsingle(x, y);
  const float res_ = guide_weight * (a_r_ + a_g_ + a_b_) + b_;
  res.write(int2(x, y), float4(min(maxval, max(minval, res_))));
}
