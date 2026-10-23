/*
    This file is part of darktable,
    Copyright (C) 2025-2026 darktable developers.

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

// Slang port of ../agx.cl

import common;
import colorspace;

static const float _epsilon = 1E-6f;

// must match hdr_params_t in agx.c
struct dt_iop_agx_hdr_params_t
{
  int enabled;
  float headroom;
  float purity;
  float log_min;
  float pivot_x;
  float pivot_y;
  float toe_scale;
  float shoulder_scale;
};

// must match tone_mapping_params_t in agx.c
struct dt_iop_agx_tone_mapping_params_t
{
  float min_ev;
  float max_ev;
  float range_in_ev;
  float curve_gamma;
  float pivot_x;
  float pivot_y;
  float target_black;
  float toe_power;
  float toe_transition_x;
  float toe_transition_y;
  float toe_scale;
  int need_convex_toe;
  float toe_fallback_coefficient;
  float toe_fallback_power;
  float slope;
  float intercept;
  float target_white;
  float shoulder_power;
  float shoulder_transition_x;
  float shoulder_transition_y;
  float shoulder_scale;
  int need_concave_shoulder;
  float shoulder_fallback_coefficient;
  float shoulder_fallback_power;
  float look_offset;
  float look_slope;
  float look_power;
  float look_saturation;
  float look_original_hue_mix_ratio;
  int look_tuned;
  int restore_hue;
  dt_iop_agx_hdr_params_t hdr;
};

float4 _agx_compress_into_gamut(float4 pixel)
{
  const float3 luminance_coeffs = float3(0.2658180370250449f, 0.59846986045365f, 0.1357121025213052f);
  const float input_y = pixel.x * luminance_coeffs[0] + pixel.y * luminance_coeffs[1] + pixel.z * luminance_coeffs[2];
  const float max_rgb = max(pixel.x, max(pixel.y, pixel.z));

  const float4 opponent_rgb = max_rgb - pixel;
  const float opponent_y = opponent_rgb.x * luminance_coeffs[0] + opponent_rgb.y * luminance_coeffs[1] + opponent_rgb.z * luminance_coeffs[2];
  const float max_opponent = max(opponent_rgb.x, max(opponent_rgb.y, opponent_rgb.z));
  const float y_compensate_negative = max_opponent - opponent_y + input_y;

  const float min_rgb = min(pixel.x, min(pixel.y, pixel.z));
  const float offset = max(-min_rgb, 0.0f);
  const float4 rgb_offset = pixel + float4(offset, offset, offset, 0.0f);

  const float max_of_rgb_offset = max(rgb_offset.x, max(rgb_offset.y, rgb_offset.z));
  const float4 opponent_rgb_offset = max_of_rgb_offset - rgb_offset;

  const float max_inverse_rgb_offset = max(opponent_rgb_offset.x, max(opponent_rgb_offset.y, opponent_rgb_offset.z));
  const float y_inverse_rgb_offset = opponent_rgb_offset.x * luminance_coeffs[0] + opponent_rgb_offset.y * luminance_coeffs[1] + opponent_rgb_offset.z * luminance_coeffs[2];
  float y_new = rgb_offset.x * luminance_coeffs[0] + rgb_offset.y * luminance_coeffs[1] + rgb_offset.z * luminance_coeffs[2];
  y_new = max_inverse_rgb_offset - y_inverse_rgb_offset + y_new;

  const float luminance_ratio =
    (y_new > y_compensate_negative && y_new > _epsilon)
    ? y_compensate_negative / y_new
    : 1.f;
  return luminance_ratio * rgb_offset;
}

float _agx_apply_log_encoding(const float x, const float range_in_ev, const float min_ev)
{
  const float x_relative = max(_epsilon, x / 0.18f);
  const float mapped = (log2(max(x_relative, 0.0f)) - min_ev) / range_in_ev;
  return clipf(mapped);
}

float _agx_sigmoid(const float x, const float power)
{
  return x / dtcl_pow(1.0f + dtcl_pow(x, power), 1.0f / power);
}

float _agx_scaled_sigmoid(const float x, const float scale, const float slope, const float power, const float transition_x, const float transition_y)
{
  return scale * _agx_sigmoid(slope * (x - transition_x) / scale, power) + transition_y;
}

float _agx_fallback_toe(const float x, const dt_iop_agx_tone_mapping_params_t params)
{
  return x < 0.0f ? params.target_black : params.target_black + max(0.0f, params.toe_fallback_coefficient * dtcl_pow(x, params.toe_fallback_power));
}

float _agx_fallback_shoulder(const float x, const dt_iop_agx_tone_mapping_params_t params)
{
  return x >= 1.0f ? params.target_white : params.target_white - max(0.0f, params.shoulder_fallback_coefficient * dtcl_pow(1.0f - x, params.shoulder_fallback_power));
}

float _agx_apply_curve(const float x, const dt_iop_agx_tone_mapping_params_t params)
{
  float result = 0.0f;
  if(x < params.toe_transition_x)
  {
    result = params.need_convex_toe != 0 ? _agx_fallback_toe(x, params) : _agx_scaled_sigmoid(x, params.toe_scale, params.slope, params.toe_power, params.toe_transition_x, params.toe_transition_y);
  }
  else if(x <= params.shoulder_transition_x)
  {
    result = params.slope * x + params.intercept;
  }
  else
  {
    result = params.need_concave_shoulder != 0 ? _agx_fallback_shoulder(x, params) : _agx_scaled_sigmoid(x, params.shoulder_scale, params.slope, params.shoulder_power, params.shoulder_transition_x, params.shoulder_transition_y);
  }
  return clamp(result, params.target_black, params.target_white);
}

float _agx_apply_slope_offset(const float x, const float slope, const float offset)
{
  const float m = slope / (1.0f + offset);
  const float b = offset * m;
  return m * x + b;
}

float4 _agx_look(float4 pixel, const dt_iop_agx_tone_mapping_params_t params, StructuredBuffer<float> rendering_to_xyz)
{
  const float slope = params.look_slope;
  const float offset = params.look_offset;
  const float power = params.look_power;
  const float sat = params.look_saturation;

  float4 temp;
  temp.x = _agx_apply_slope_offset(pixel.x, slope, offset);
  temp.y = _agx_apply_slope_offset(pixel.y, slope, offset);
  temp.z = _agx_apply_slope_offset(pixel.z, slope, offset);

  pixel.x = temp.x > 0.0f ? dtcl_pow(temp.x, power) : temp.x;
  pixel.y = temp.y > 0.0f ? dtcl_pow(temp.y, power) : temp.y;
  pixel.z = temp.z > 0.0f ? dtcl_pow(temp.z, power) : temp.z;

  const float luma = matrix_product_float4(pixel, rendering_to_xyz).y;

  pixel.x = luma + sat * (pixel.x - luma);
  pixel.y = luma + sat * (pixel.y - luma);
  pixel.z = luma + sat * (pixel.z - luma);
  return pixel;
}

float _agx_lerp_hue(const float original_hue, const float processed_hue, const float mix)
{
  const float shortest_distance = processed_hue - original_hue - rint(processed_hue - original_hue);
  const float mixed_hue = (1.0f - mix) * shortest_distance + original_hue;
  return mixed_hue - floor(mixed_hue);
}

// see _hdr_darken() in agx.c
float _agx_hdr_darken(const float x, const dt_iop_agx_hdr_params_t hdr)
{
  const float log_x = clamp(1.0f - dtcl_log2(max(x, 1e-12f)) / hdr.log_min, 0.0f, 1.0f);
  const float u = log_x - hdr.pivot_x;
  const float log_y = u < 0.0f
    ? _agx_scaled_sigmoid(log_x, hdr.toe_scale, 1.0f, 3.0f, hdr.pivot_x, hdr.pivot_y)
    : hdr.pivot_y + hdr.shoulder_scale * u / (hdr.shoulder_scale + u);
  return x > 0.0f ? dtcl_exp2(hdr.log_min * (1.0f - log_y)) : 0.0f;
}

// see _agx_hdr() in agx.c
float4 _agx_hdr(const float4 pixel, const dt_iop_agx_hdr_params_t hdr)
{
  const float4 hsv_before = RGB_2_HSV(pixel);
  float4 darkened = pixel;
  darkened.x = _agx_hdr_darken(pixel.x, hdr);
  darkened.y = _agx_hdr_darken(pixel.y, hdr);
  darkened.z = _agx_hdr_darken(pixel.z, hdr);
  float4 hsv_after = RGB_2_HSV(darkened);

  hsv_after.x = _agx_lerp_hue(hsv_before.x, hsv_after.x, 1.0f - hdr.purity);
  hsv_after.y = hsv_before.y + hdr.purity * (hsv_after.y - hsv_before.y);
  darkened = HSV_2_RGB(hsv_after);

  return float4(hdr.headroom * darkened.x, hdr.headroom * darkened.y, hdr.headroom * darkened.z, pixel.w);
}

float4 _agx_tone_mapping(const float4 rgb_in, const dt_iop_agx_tone_mapping_params_t params, StructuredBuffer<float> rendering_to_xyz)
{
  float4 hsv_pixel = float4(0.0f);
  if(params.restore_hue != 0)
  {
    hsv_pixel = RGB_2_HSV(rgb_in);
  }
  const float h_before = hsv_pixel.x;

  float4 transformed_pixel;
  transformed_pixel.x = _agx_apply_curve(_agx_apply_log_encoding(rgb_in.x, params.range_in_ev, params.min_ev), params);
  transformed_pixel.y = _agx_apply_curve(_agx_apply_log_encoding(rgb_in.y, params.range_in_ev, params.min_ev), params);
  transformed_pixel.z = _agx_apply_curve(_agx_apply_log_encoding(rgb_in.z, params.range_in_ev, params.min_ev), params);
  transformed_pixel.w = rgb_in.w;

  if(params.look_tuned != 0)
  {
    transformed_pixel = _agx_look(transformed_pixel, params, rendering_to_xyz);
  }

  transformed_pixel.x = dtcl_pow(max(0.0f, transformed_pixel.x), params.curve_gamma);
  transformed_pixel.y = dtcl_pow(max(0.0f, transformed_pixel.y), params.curve_gamma);
  transformed_pixel.z = dtcl_pow(max(0.0f, transformed_pixel.z), params.curve_gamma);

  if(params.hdr.enabled != 0)
  {
    transformed_pixel = _agx_hdr(transformed_pixel, params.hdr);
  }

  if(params.restore_hue != 0)
  {
    hsv_pixel = RGB_2_HSV(transformed_pixel);
    hsv_pixel.x = _agx_lerp_hue(h_before, hsv_pixel.x, params.look_original_hue_mix_ratio);
    return HSV_2_RGB(hsv_pixel);
  }
  return transformed_pixel;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void kernel_agx(uint3 gid : SV_DispatchThreadID,
                uniform ImageIn input, uniform ImageOut output,
                uniform int width, uniform int height,
                uniform dt_iop_agx_tone_mapping_params_t params,
                uniform StructuredBuffer<float> pipe_to_base,
                uniform StructuredBuffer<float> base_to_rendering,
                uniform StructuredBuffer<float> rendering_to_pipe,
                uniform StructuredBuffer<float> rendering_to_xyz,
                uniform int base_working_same_profile)
{
  const int i = gid.x;
  const int j = gid.y;
  if(i >= width || j >= height) return;

  float4 in_pixel = input.read(int2(i, j));

  // sanitize input range and get rid of NaNs
  in_pixel = select(isnan(in_pixel), float4(0.0f), clamp(in_pixel, -1e6f, 1e6f));

  float4 base_rgb = base_working_same_profile != 0 ? in_pixel : matrix_product_float4(in_pixel, pipe_to_base);

  base_rgb = _agx_compress_into_gamut(base_rgb);

  float4 rendering_rgb = matrix_product_float4(base_rgb, base_to_rendering);

  rendering_rgb = _agx_tone_mapping(rendering_rgb, params, rendering_to_xyz);

  float4 out_pixel = matrix_product_float4(rendering_rgb, rendering_to_pipe);

  out_pixel.w = in_pixel.w;
  output.write(int2(i, j), out_pixel);
}
