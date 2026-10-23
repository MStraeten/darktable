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

// Slang port of ../toneequal.cl. The box means are dispatched over a single
// row like the OpenCL ones, so the work items of the other rows return early.

import common;

// keep in sync with src/common/luminance_mask.h and src/iop/toneequal.c
static const float TONEEQ_MIN_FLOAT = 1.52587890625e-05f; // exp2f(-16.0f)
static const float TONEEQ_MIN_EV = -8.0f;
static const float TONEEQ_MAX_EV = 0.0f;

// dt_iop_luminance_mask_method_t
static const int TONEEQ_MEAN = 0;
static const int TONEEQ_LIGHTNESS = 1;
static const int TONEEQ_VALUE = 2;
static const int TONEEQ_NORM_1 = 3;
static const int TONEEQ_NORM_2 = 4;
static const int TONEEQ_NORM_POWER = 5;
static const int TONEEQ_GEOMEAN = 6;

// dt_iop_guided_filter_blending_t
static const int TONEEQ_BLENDING_LINEAR = 0;
static const int TONEEQ_BLENDING_GEOMEAN = 1;

// the sampling of ../bilinear.h, which bilinear.slang inlines in its kernels
vector<float, N> _bilinear_sample<let N : int>(StructuredBuffer<vector<float, N>> in,
                                               int width_in, int height_in, int x, int y,
                                               int width_out, int height_out)
{
  const float x_in = (float(x) / float(width_out)) * float(width_in);
  const float y_in = (float(y) / float(height_out)) * float(height_in);

  int x_prev = int(floor(x_in));
  int x_next = x_prev + 1;
  int y_prev = int(floor(y_in));
  int y_next = y_prev + 1;
  x_prev = (x_prev < width_in) ? x_prev : width_in - 1;
  x_next = (x_next < width_in) ? x_next : width_in - 1;
  y_prev = (y_prev < height_in) ? y_prev : height_in - 1;
  y_next = (y_next < height_in) ? y_next : height_in - 1;

  const vector<float, N> Q_NW = in[y_prev * width_in + x_prev];
  const vector<float, N> Q_NE = in[y_prev * width_in + x_next];
  const vector<float, N> Q_SE = in[y_next * width_in + x_next];
  const vector<float, N> Q_SW = in[y_next * width_in + x_prev];

  // spatial differences between nodes, next - prev = 1
  const float Dy_next = float(y_next) - y_in;
  const float Dy_prev = 1.0f - Dy_next;
  const float Dx_next = float(x_next) - x_in;
  const float Dx_prev = 1.0f - Dx_next;

  return Dy_prev * (Q_SW * Dx_next + Q_SE * Dx_prev)
       + Dy_next * (Q_NW * Dx_next + Q_NE * Dx_prev);
}

float _linear_contrast(float pixel, float fulcrum, float contrast)
{
  // increase the slope of the value around a fulcrum value
  return max((pixel - fulcrum) * contrast + fulcrum, TONEEQ_MIN_FLOAT);
}

// flatten an RGB image into a luminance map, applying exposure and contrast
// compensation on the fly. Mirrors luminance_mask() from common/luminance_mask.h
[shader("compute")]
[numthreads(16, 16, 1)]
void toneequal_luminance_mask(uint3 gid : SV_DispatchThreadID,
                              uniform ImageIn in, uniform RWStructuredBuffer<float> luminance,
                              uniform int width, uniform int height, uniform int method,
                              uniform float exposure_boost, uniform float fulcrum,
                              uniform float contrast_boost)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float4 pixel = in.Areadpixel(x, y);
  float lum;

  switch(method)
  {
    case TONEEQ_LIGHTNESS:
    {
      // (max(RGB) + min(RGB)) / 2 is equivalent to HSL lightness
      const float max_rgb = max(max(pixel.x, pixel.y), pixel.z);
      const float min_rgb = min(min(pixel.x, pixel.y), pixel.z);
      lum = exposure_boost * (max_rgb + min_rgb) / 2.0f;
      break;
    }

    case TONEEQ_VALUE:
      // max(RGB) is equivalent to HSV value
      lum = exposure_boost * max(max(pixel.x, pixel.y), pixel.z);
      break;

    case TONEEQ_NORM_1:
      // vector norm L1
      lum = exposure_boost * (abs(pixel.x) + abs(pixel.y) + abs(pixel.z));
      break;

    case TONEEQ_NORM_2:
      // vector norm L2: euclidean norm
      lum = exposure_boost
        * dtcl_sqrt(pixel.x * pixel.x + pixel.y * pixel.y + pixel.z * pixel.z);
      break;

    case TONEEQ_NORM_POWER:
    {
      const float4 value = abs(pixel);
      const float4 RGB_square = value * value;
      const float4 RGB_cubic = RGB_square * value;
      const float numerator = RGB_cubic.x + RGB_cubic.y + RGB_cubic.z;
      const float denominator = RGB_square.x + RGB_square.y + RGB_square.z;
      lum = exposure_boost * numerator / denominator;
      break;
    }

    case TONEEQ_GEOMEAN:
    {
      // geometric_mean(RGB)
      const float product = abs(pixel.x) * abs(pixel.y) * abs(pixel.z);
      lum = exposure_boost * dtcl_pow(product, 1.0f / 3.0f);
      break;
    }

    case TONEEQ_MEAN:
    default:
      // mean(RGB) is the intensity
      lum = exposure_boost * (pixel.x + pixel.y + pixel.z) / 3.0f;
      break;
  }

  luminance[y * width + x] = _linear_contrast(lum, fulcrum, contrast_boost);
}

// the correction as the sum of the contribution of each luminance channel,
// weighted by the gaussian of the radial distance in EV
float _pixel_correction(float exposure, float4 factors_low, float4 factors_high, float gauss_denom)
{
  // radial distances of the exposure octaves, see centers_ops[] in toneequal.c
  const float4 centers_low = float4(-56.0f / 7.0f, -48.0f / 7.0f, -40.0f / 7.0f, -32.0f / 7.0f);
  const float4 centers_high = float4(-24.0f / 7.0f, -16.0f / 7.0f, -8.0f / 7.0f, 0.0f / 7.0f);

  const float4 radius_low = exposure - centers_low;
  const float4 radius_high = exposure - centers_high;

  const float4 gauss_low = dtcl_exp(-radius_low * radius_low / gauss_denom) * factors_low;
  const float4 gauss_high = dtcl_exp(-radius_high * radius_high / gauss_denom) * factors_high;

  const float4 result = gauss_low + gauss_high;

  // the user-set correction is expected in [-2;+2] EV, so is the interpolated one
  return clamp(result.x + result.y + result.z + result.w, 0.25f, 4.0f);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void toneequal_apply(uint3 gid : SV_DispatchThreadID,
                     uniform ImageIn in, uniform StructuredBuffer<float> luminance,
                     uniform ImageOut out, uniform int width, uniform int height,
                     uniform int in_width, uniform int offset_x, uniform int offset_y,
                     uniform float4 factors_low, uniform float4 factors_high,
                     uniform float gauss_denom)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const int xin = x + offset_x;
  const int yin = y + offset_y;

  // the radial-basis interpolation is valid in [-8; 0] EV and can quickly
  // diverge outside
  const float exposure = clamp(dtcl_log2(luminance[yin * in_width + xin]),
                               TONEEQ_MIN_EV, TONEEQ_MAX_EV);
  const float correction = _pixel_correction(exposure, factors_low, factors_high, gauss_denom);

  const float4 pixel = in.Areadpixel(xin, yin);
  out.write(int2(x, y), correction * pixel);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void toneequal_display_mask(uint3 gid : SV_DispatchThreadID,
                            uniform ImageIn in, uniform StructuredBuffer<float> luminance,
                            uniform ImageOut out, uniform int width, uniform int height,
                            uniform int in_width, uniform int offset_x, uniform int offset_y)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const int xin = x + offset_x;
  const int yin = y + offset_y;

  // normalize the mask intensity between -8 EV and 0 EV for clarity,
  // and add a "gamma" 2.0 for better legibility in shadows
  const float lum = luminance[yin * in_width + xin];
  const float intensity = dtcl_sqrt(min(max(lum - 0.00390625f, 0.0f) / 0.99609375f, 1.0f));

  // set gray level for the mask, copy alpha channel
  const float4 pixel = in.Areadpixel(xin, yin);
  out.write(int2(x, y), float4(intensity, intensity, intensity, pixel.w));
}

// quantize in exposure levels evenly spaced in log by sampling.
// Mirrors quantize() from common/fast_guided_filter.h
[shader("compute")]
[numthreads(16, 16, 1)]
void toneequal_quantize(uint3 gid : SV_DispatchThreadID,
                        uniform StructuredBuffer<float> in, uniform RWStructuredBuffer<float> out,
                        uniform int width, uniform int height, uniform float sampling,
                        uniform float clip_min, uniform float clip_max)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const int k = y * width + x;
  const float value = in[k];

  if(sampling == 0.0f)
    out[k] = value;
  else if(sampling == 1.0f)
    out[k] = clamp(dtcl_exp2(floor(dtcl_log2(value))), clip_min, clip_max);
  else
    out[k] = clamp(dtcl_exp2(floor(dtcl_log2(value) / sampling) * sampling), clip_min, clip_max);
}

// separable box average over a window of size 2*radius + 1, mirroring
// dt_box_mean() from common/box_filters.cc. The window is clipped at the
// image borders and the average is taken over the samples actually seen,
// so input and output buffers must be distinct
void _box_mean_line<let N : int>(StructuredBuffer<vector<float, N>> in,
                                 RWStructuredBuffer<vector<float, N>> out,
                                 int base, int length, int stride, int radius)
{
  vector<float, N> sum = vector<float, N>(0.0f);
  int hits = 0;
  int i;

  // add up the left half of the window
  for(i = 0; i < min(radius, length); i++)
  {
    sum += in[base + i * stride];
    hits++;
  }

  // up to the point where we start removing values from the average
  for(i = 0; i <= radius && i + radius < length; i++)
  {
    sum += in[base + (i + radius) * stride];
    hits++;
    out[base + i * stride] = sum / float(hits);
  }

  // if radius > length/2 we can neither add nor remove values
  for(; i <= radius && i < length; i++)
    out[base + i * stride] = sum / float(hits);

  // the bulk of the line
  for(; i + radius < length; i++)
  {
    sum -= in[base + (i - radius - 1) * stride];
    sum += in[base + (i + radius) * stride];
    out[base + i * stride] = sum / float(hits);
  }

  // the end of the line, no more values to add to the average
  for(; i < length; i++)
  {
    sum -= in[base + (i - radius - 1) * stride];
    hits--;
    out[base + i * stride] = sum / float(hits);
  }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void toneequal_box_mean_x_2c(uint3 gid : SV_DispatchThreadID,
                             uniform StructuredBuffer<float2> input, uniform RWStructuredBuffer<float2> output,
                             uniform int width, uniform int height, uniform int radius)
{
  const int y = gid.x;
  if(y >= height || gid.y != 0) return;
  _box_mean_line<2>(input, output, y * width, width, 1, radius);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void toneequal_box_mean_y_2c(uint3 gid : SV_DispatchThreadID,
                             uniform StructuredBuffer<float2> input, uniform RWStructuredBuffer<float2> output,
                             uniform int width, uniform int height, uniform int radius)
{
  const int x = gid.x;
  if(x >= width || gid.y != 0) return;
  _box_mean_line<2>(input, output, x, height, width, radius);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void toneequal_box_mean_x_4c(uint3 gid : SV_DispatchThreadID,
                             uniform StructuredBuffer<float4> input, uniform RWStructuredBuffer<float4> output,
                             uniform int width, uniform int height, uniform int radius)
{
  const int y = gid.x;
  if(y >= height || gid.y != 0) return;
  _box_mean_line<4>(input, output, y * width, width, 1, radius);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void toneequal_box_mean_y_4c(uint3 gid : SV_DispatchThreadID,
                             uniform StructuredBuffer<float4> input, uniform RWStructuredBuffer<float4> output,
                             uniform int width, uniform int height, uniform int radius)
{
  const int x = gid.x;
  if(x >= width || gid.y != 0) return;
  _box_mean_line<4>(input, output, x, height, width, radius);
}

// guided filter: pack guide and mask so that the box average of all four
// terms takes a single pass. Mirrors the first loop of variance_analyse()
// in common/fast_guided_filter.h
[shader("compute")]
[numthreads(16, 16, 1)]
void toneequal_gf_pack(uint3 gid : SV_DispatchThreadID,
                       uniform StructuredBuffer<float> guide, uniform StructuredBuffer<float> mask,
                       uniform RWStructuredBuffer<float4> packed,
                       uniform int width, uniform int height)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const int k = y * width + x;
  const float g = guide[k];
  const float m = mask[k];
  packed[k] = float4(g, m, g * g, g * m);
}

// guided filter: turn the averaged terms into the linear blending parameters
// a and b such that mask = a * I + b
[shader("compute")]
[numthreads(16, 16, 1)]
void toneequal_gf_ab(uint3 gid : SV_DispatchThreadID,
                     uniform StructuredBuffer<float4> packed, uniform RWStructuredBuffer<float2> ab,
                     uniform int width, uniform int height, uniform float feathering)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const int k = y * width + x;
  const float4 avg = packed[k];

  // avoid division by 0
  const float d = max((avg.z - avg.x * avg.x) + feathering, 1e-15f);
  const float a = (avg.w - avg.x * avg.y) / d;
  const float b = avg.y - a * avg.x;
  ab[k] = float2(a, b);
}

// guided filter: blend one pixel with its a and b parameters.
// Mirrors apply_linear_blending[_w_geomean]() in common/fast_guided_filter.h
float _gf_blend(float pixel, float2 blend, int filter)
{
  // image[k] is positive at the outside of the luminance mask
  const float blended = max(pixel * blend.x + blend.y, TONEEQ_MIN_FLOAT);
  return (filter == TONEEQ_BLENDING_GEOMEAN) ? dtcl_sqrt(pixel * blended) : blended;
}

// guided filter: blend the intermediate iterations, where the parameters
// are at the same (downscaled) size as the image
[shader("compute")]
[numthreads(16, 16, 1)]
void toneequal_gf_blend(uint3 gid : SV_DispatchThreadID,
                        uniform RWStructuredBuffer<float> image, uniform StructuredBuffer<float2> ab,
                        uniform int width, uniform int height, uniform int filter)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const int k = y * width + x;
  image[k] = _gf_blend(image[k], ab[k], filter);
}

// guided filter: final blend of the full size image, sampling the a and b
// parameters at the downscaled size
[shader("compute")]
[numthreads(16, 16, 1)]
void toneequal_gf_blend_upsample(uint3 gid : SV_DispatchThreadID,
                                 uniform RWStructuredBuffer<float> image,
                                 uniform int width, uniform int height,
                                 uniform StructuredBuffer<float2> ds_ab,
                                 uniform int ds_width, uniform int ds_height, uniform int filter)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const int k = y * width + x;
  const float2 blend = _bilinear_sample<2>(ds_ab, ds_width, ds_height, x, y, width, height);
  image[k] = _gf_blend(image[k], blend, filter);
}

// exposure independent guided filter: pack guide and mask so that a single
// gaussian blur yields the averages of guide, guide^2, mask and mask * guide.
// Mirrors eigf_variance_analysis() in common/eigf.h
[shader("compute")]
[numthreads(16, 16, 1)]
void toneequal_eigf_pack_4c(uint3 gid : SV_DispatchThreadID,
                            uniform StructuredBuffer<float> guide, uniform StructuredBuffer<float> mask,
                            uniform RWStructuredBuffer<float4> packed,
                            uniform int width, uniform int height)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const int k = y * width + x;
  const float g = guide[k];
  const float m = mask[k];
  packed[k] = float4(g, g * g, m, m * g);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void toneequal_eigf_finish_4c(uint3 gid : SV_DispatchThreadID,
                              uniform RWStructuredBuffer<float4> av,
                              uniform int width, uniform int height)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const int k = y * width + x;
  const float4 avg = av[k];

  // turn the averages of the squares into variance and covariance
  av[k] = float4(avg.x, avg.y - avg.x * avg.x, avg.z, avg.w - avg.x * avg.z);
}

// same as above, specialized for the case where guide == mask
[shader("compute")]
[numthreads(16, 16, 1)]
void toneequal_eigf_pack_2c(uint3 gid : SV_DispatchThreadID,
                            uniform StructuredBuffer<float> guide, uniform RWStructuredBuffer<float2> packed,
                            uniform int width, uniform int height)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const int k = y * width + x;
  const float g = guide[k];
  packed[k] = float2(g, g * g);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void toneequal_eigf_finish_2c(uint3 gid : SV_DispatchThreadID,
                              uniform RWStructuredBuffer<float2> av,
                              uniform int width, uniform int height)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const int k = y * width + x;
  const float2 avg = av[k];
  av[k] = float2(avg.x, avg.y - avg.x * avg.x);
}

// exposure independent guided filter: blending step. Mirrors eigf_blending()
// in common/eigf.h
float _eigf_blend(float pixel, float mask, float4 avg, int filter, float feathering)
{
  const float avg_g = avg.x;
  const float var_g = avg.y;
  const float avg_m = avg.z;
  const float covar_mg = avg.w;

  const float norm_g = max(avg_g * pixel, 1E-6f);
  const float norm_m = max(avg_m * mask, 1E-6f);
  const float normalized_var_guide = var_g / norm_g;
  const float normalized_covar = covar_mg / dtcl_sqrt(norm_g * norm_m);
  const float a = normalized_covar / (normalized_var_guide + feathering);
  const float b = avg_m - a * avg_g;

  const float blended = max(pixel * a + b, TONEEQ_MIN_FLOAT);
  return (filter == TONEEQ_BLENDING_GEOMEAN) ? dtcl_sqrt(pixel * blended) : blended;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void toneequal_eigf_blend(uint3 gid : SV_DispatchThreadID,
                          uniform RWStructuredBuffer<float> image, uniform StructuredBuffer<float> mask,
                          uniform int width, uniform int height,
                          uniform StructuredBuffer<float4> ds_av,
                          uniform int ds_width, uniform int ds_height,
                          uniform int filter, uniform float feathering)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const int k = y * width + x;
  const float4 avg = _bilinear_sample<4>(ds_av, ds_width, ds_height, x, y, width, height);
  image[k] = _eigf_blend(image[k], mask[k], avg, filter, feathering);
}

// same as above, specialized for the case where guide == mask
float _eigf_blend_no_mask(float pixel, float2 avg, int filter, float feathering)
{
  const float avg_g = avg.x;
  const float var_g = avg.y;

  const float norm_g = max(avg_g * pixel, 1E-6f);
  const float normalized_var_guide = var_g / norm_g;
  const float a = normalized_var_guide / (normalized_var_guide + feathering);
  const float b = avg_g - a * avg_g;

  const float blended = max(pixel * a + b, TONEEQ_MIN_FLOAT);
  return (filter == TONEEQ_BLENDING_GEOMEAN) ? dtcl_sqrt(pixel * blended) : blended;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void toneequal_eigf_blend_no_mask(uint3 gid : SV_DispatchThreadID,
                                  uniform RWStructuredBuffer<float> image,
                                  uniform int width, uniform int height,
                                  uniform StructuredBuffer<float2> ds_av,
                                  uniform int ds_width, uniform int ds_height,
                                  uniform int filter, uniform float feathering)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const int k = y * width + x;
  const float2 avg = _bilinear_sample<2>(ds_av, ds_width, ds_height, x, y, width, height);
  image[k] = _eigf_blend_no_mask(image[k], avg, filter, feathering);
}
