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

// Slang port of ../contrastntexture.cl.

import common;

static const float INVERSE_SQRT_3 = 0.5773502691896258f;

[shader("compute")]
[numthreads(16, 16, 1)]
void contrastntexture_luminance(uint3 gid : SV_DispatchThreadID,
                                uniform ImageIn in, uniform RWStructuredBuffer<float> log_luminance,
                                uniform int width, uniform int height, uniform float noise_bias)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float4 pixel = in.Areadpixel(x, y);
  const float norm = INVERSE_SQRT_3 * dtcl_sqrt(pixel.x * pixel.x + pixel.y * pixel.y + pixel.z * pixel.z)
                     + noise_bias;
  log_luminance[y * width + x] = dtcl_log2(max(norm, NORM_MIN));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void contrastntexture_guided_filter_pack(uint3 gid : SV_DispatchThreadID,
                                         uniform StructuredBuffer<float> log_luminance,
                                         uniform RWStructuredBuffer<float2> moments,
                                         uniform int width, uniform int height, uniform float center)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const int k = y * width + x;
  const float v = log_luminance[k] - center;
  moments[k] = float2(v, v * v);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void contrastntexture_details(uint3 gid : SV_DispatchThreadID,
                              uniform RWStructuredBuffer<float> log_luminance,
                              uniform StructuredBuffer<float2> moments,
                              uniform RWStructuredBuffer<float> corrections,
                              uniform int width, uniform int height, uniform float feathering,
                              uniform float gain, uniform float noise_bias, uniform float next_bias_step,
                              uniform int accumulate, uniform int last_level, uniform float center)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const int k = y * width + x;
  const float log_pixel = log_luminance[k];
  const float2 m = moments[k];
  const float variance = max(m.y - m.x * m.x, 0.0f);
  const float a = fp32_div(variance, variance + feathering);
  const float log_smoothed = a * log_pixel + (1.0f - a) * (m.x + center);
  const float smoothed = dtcl_exp2(log_smoothed);

  const float noise_power = noise_bias * noise_bias;
  const float combined_power = smoothed * smoothed;
  const float wiener_gain = fp32_div(max(combined_power - noise_power, 0.0f), max(combined_power, NORM_MIN));
  const float detail = gain * (wiener_gain * clamp(log_pixel - log_smoothed, -5.0f, 5.0f));
  corrections[k] = accumulate != 0 ? corrections[k] + detail : detail;

  log_luminance[k] = last_level != 0 ? log_smoothed
                                     : dtcl_log2(max(smoothed - next_bias_step, NORM_MIN));
}

float _shadows_highlights(float log_luminance_lowpass, float slope_shadows, float slope_highlights,
                          float midtones_width, float4 polynomial, float pivot_offset)
{
  const float normalized_ev = log_luminance_lowpass - pivot_offset;
  float correction_ev;
  if(normalized_ev <= -midtones_width)
    correction_ev = slope_shadows * normalized_ev;
  else if(normalized_ev >= midtones_width)
    correction_ev = slope_highlights * normalized_ev;
  else
  {
    const float shifted_ev = normalized_ev + midtones_width;
    correction_ev = polynomial.x + polynomial.y * shifted_ev + polynomial.z * shifted_ev * shifted_ev;
  }
  return correction_ev - normalized_ev;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void contrastntexture_apply(uint3 gid : SV_DispatchThreadID,
                            uniform ImageIn in, uniform ImageOut out,
                            uniform StructuredBuffer<float> lowpass,
                            uniform StructuredBuffer<float> corrections,
                            uniform int width, uniform int height,
                            uniform float slope_shadows, uniform float slope_highlights,
                            uniform float midtones_width, uniform float4 polynomial,
                            uniform float pivot_offset)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const int k = y * width + x;
  const float4 pixel = in.Areadpixel(x, y);
  const float lowpass_correction = _shadows_highlights(lowpass[k], slope_shadows, slope_highlights,
                                                       midtones_width, polynomial, pivot_offset);
  const float multiplier = dtcl_exp2(corrections[k] + lowpass_correction);
  float4 o = pixel * multiplier;
  o.w = pixel.w;
  out.write(int2(x, y), o);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void contrastntexture_display_details(uint3 gid : SV_DispatchThreadID,
                                      uniform ImageOut out,
                                      uniform StructuredBuffer<float> corrections,
                                      uniform int width, uniform int height)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float local_ev = corrections[y * width + x];
  const float intensity = local_ev / dtcl_sqrt(local_ev * local_ev + 1.0f) * 0.5f + 0.5f;
  out.write(int2(x, y), float4(intensity, intensity, intensity, 1.0f));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void contrastntexture_display_lowpass(uint3 gid : SV_DispatchThreadID,
                                      uniform ImageOut out,
                                      uniform StructuredBuffer<float> lowpass,
                                      uniform int width, uniform int height,
                                      uniform float pivot_offset)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float l = dtcl_exp2(lowpass[y * width + x]);
  const float intensity = l / (pivot_offset + l);
  float saturation = (intensity - 0.5f) * 2.0f;
  saturation *= saturation;
  const float4 tint = intensity < 0.5f ? float4(0.0f, 0.0f, 1.0f, 0.0f)
                                       : float4(1.0f, 1.0f, 0.0f, 0.0f);
  float4 o = intensity * ((1.0f - saturation) + saturation * tint);
  o.w = 1.0f;
  out.write(int2(x, y), o);
}
