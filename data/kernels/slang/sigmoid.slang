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

// Slang port of ../sigmoid.cl

import colorspace;

struct ValueOrder
{
  int min;
  int mid;
  int max;
};

ValueOrder _pixel_channel_order(float pix_in[3])
{
  ValueOrder o;
  if(pix_in[0] >= pix_in[1])
  {
    if(pix_in[1] > pix_in[2])
    { // Case 1: r >= g >  b
      o.max = 0; o.mid = 1; o.min = 2;
    }
    else if(pix_in[2] > pix_in[0])
    { // Case 2: b >  r >= g
      o.max = 2; o.mid = 0; o.min = 1;
    }
    else if(pix_in[2] > pix_in[1])
    { // Case 3: r >= b >  g
      o.max = 0; o.mid = 2; o.min = 1;
    }
    else
    { // Case 4: r == g == b
      // No change of the middle value, just assign something.
      o.max = 0; o.mid = 1; o.min = 2;
    }
  }
  else
  {
    if(pix_in[0] >= pix_in[2])
    { // Case 5: g >  r >= b
      o.max = 1; o.mid = 0; o.min = 2;
    }
    else if(pix_in[2] > pix_in[1])
    { // Case 6: b >  g >  r
      o.max = 2; o.mid = 1; o.min = 0;
    }
    else
    { // Case 7: g >= b >  r
      o.max = 1; o.mid = 2; o.min = 0;
    }
  }
  return o;
}

float4 _desaturate_negative_values(float4 i)
{
  const float pixel_average = max((i.x + i.y + i.z) / 3.0f, 0.0f);
  const float min_value = min(min(i.x, i.y), i.z);
  const float saturation_factor = min_value < 0.0f ? -pixel_average / (min_value - pixel_average) : 1.0f;

  return pixel_average + saturation_factor * (i - pixel_average);
}

float _generalized_loglogistic_sigmoid_scalar(float value, float magnitude, float paper_exp,
                                              float film_fog, float film_power, float paper_power)
{
  const float clamped_value = max(value, 0.0f);
  // The following equation can be derived as a model for film + paper but it has a pole at 0
  // magnitude * powf(1.0 + paper_exp * powf(film_fog + value, -film_power), -paper_power);
  // Rewritten on a stable around zero form:
  const float film_response = pow(film_fog + clamped_value, film_power);
  const float paper_response = magnitude * pow(film_response / (paper_exp + film_response), paper_power);

  // Safety check for very large floats that cause numerical errors
  return isnan(paper_response) ? magnitude : paper_response;
}

float4 _generalized_loglogistic_sigmoid_vector(float4 i, float magnitude, float paper_exp,
                                               float film_fog, float film_power, float paper_power)
{
  // clamped_value
  float4 io = max(i, float4(0.0f));

  // film_response
  io = pow(film_fog + io, float4(film_power));
  // paper_response
  io = magnitude * pow(io / (paper_exp + io), float4(paper_power));

  // Safety check for very large floats that cause numerical errors
  return isnan(io) ? float4(magnitude) : io;
}

// Linear interpolation of hue that also preserve sum of channels
// Assumes hue_preservation strictly in range [0, 1]
void _preserve_hue_and_energy(inout float pix_io[3], float per_channel[3], ValueOrder order,
                              float hue_preservation)
{
  // Naive Hue correction of the middle channel
  const float chroma = pix_io[order.max] - pix_io[order.min];
  const float midscale = chroma != 0.f ? (pix_io[order.mid] - pix_io[order.min]) / chroma : 0.f;
  const float full_hue_correction =
    per_channel[order.min] + (per_channel[order.max] - per_channel[order.min]) * midscale;
  const float naive_hue_mid =
    (1.0f - hue_preservation) * per_channel[order.mid] + hue_preservation * full_hue_correction;

  const float per_channel_energy = per_channel[0] + per_channel[1] + per_channel[2];
  const float naive_hue_energy = per_channel[order.min] + naive_hue_mid + per_channel[order.max];
  const float pix_in_min_plus_mid = pix_io[order.min] + pix_io[order.mid];
  const float blend_factor = pix_in_min_plus_mid != 0.f ? 2.0f * pix_io[order.min] / pix_in_min_plus_mid : 0.f;
  const float energy_target = blend_factor * per_channel_energy + (1.0f - blend_factor) * naive_hue_energy;

  // Preserve hue constrained to maintain the same energy as the per channel result
  if(naive_hue_mid <= per_channel[order.mid])
  {
    const float corrected_mid =
      ((1.0f - hue_preservation) * per_channel[order.mid] + hue_preservation *
      (midscale * per_channel[order.max] + (1.0f - midscale) * (energy_target - per_channel[order.max]))) /
      (1.0f + hue_preservation * (1.0f - midscale));
    pix_io[order.min] = energy_target - per_channel[order.max] - corrected_mid;
    pix_io[order.mid] = corrected_mid;
    pix_io[order.max] = per_channel[order.max];
  }
  else
  {
    const float corrected_mid =
      ((1.0f - hue_preservation) * per_channel[order.mid] +
      hue_preservation * (per_channel[order.min] * (1.0f - midscale) +
      midscale * (energy_target - per_channel[order.min]))) / (1.0f + hue_preservation * midscale);
    pix_io[order.min] = per_channel[order.min];
    pix_io[order.mid] = corrected_mid;
    pix_io[order.max] = energy_target - per_channel[order.min] - corrected_mid;
  }
}

// the host passes 4x4 dt_colormatrix_t, of which matrix_product_float4 reads 3 rows
float4 _matrix_product(float4 v, StructuredBuffer<float> m)
{
  float a[12];
  for(int k = 0; k < 12; k++) a[k] = m[k];
  return matrix_product_float4(v, a);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void sigmoid_loglogistic_per_channel(uint3 gid : SV_DispatchThreadID,
                                     uniform ImageIn in, uniform ImageOut out,
                                     uniform int width, uniform int height,
                                     uniform float white_target, uniform float paper_exp,
                                     uniform float film_fog, uniform float contrast_power,
                                     uniform float skew_power, uniform float hue_preservation,
                                     uniform StructuredBuffer<float> pipe_to_base,
                                     uniform StructuredBuffer<float> base_to_rendering,
                                     uniform StructuredBuffer<float> rendering_to_pipe)
{
  const int x = gid.x;
  const int y = gid.y;

  if(x >= width || y >= height) return;

  float4 i = in.readA(int2(x, y));
  const float alpha = i.w;

  i = _matrix_product(i, pipe_to_base);

  // Force negative values to zero
  i = _desaturate_negative_values(i);

  // Convert to rendering primaries
  i = _matrix_product(i, base_to_rendering);
  float pix_array[3] = { i.x, i.y, i.z };

  i = _generalized_loglogistic_sigmoid_vector(i, white_target, paper_exp, film_fog, contrast_power, skew_power);
  const float per_channel[3] = { i.x, i.y, i.z };

  // Hue correction by scaling the middle value relative to the max and min values.
  const ValueOrder pixel_value_order = _pixel_channel_order(pix_array);
  _preserve_hue_and_energy(pix_array, per_channel, pixel_value_order, hue_preservation);

  i.xyz = float3(pix_array[0], pix_array[1], pix_array[2]);
  i = _matrix_product(i, rendering_to_pipe);

  // Copy over the alpha channel
  i.w = alpha;

  out.write(int2(x, y), i);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void sigmoid_loglogistic_rgb_ratio(uint3 gid : SV_DispatchThreadID,
                                   uniform ImageIn in, uniform ImageOut out,
                                   uniform int width, uniform int height,
                                   uniform float white_target, uniform float black_target,
                                   uniform float paper_exp, uniform float film_fog,
                                   uniform float contrast_power, uniform float skew_power)
{
  const int x = gid.x;
  const int y = gid.y;

  if(x >= width || y >= height) return;

  float4 i = in.readA(int2(x, y));
  const float alpha = i.w;

  // Force negative values to zero
  i = _desaturate_negative_values(i);

  // Preserve color ratios by applying the tone curve on a luma estimate and then scale the RGB tripplet uniformly
  const float luma = (i.x + i.y + i.z) / 3.0f;
  const float mapped_luma =
    _generalized_loglogistic_sigmoid_scalar(luma, white_target, paper_exp, film_fog, contrast_power, skew_power);

  if(luma > 1e-9f)
  {
    const float scaling_factor = mapped_luma / luma;
    i = scaling_factor * i;
  }
  else
  {
    i = float4(mapped_luma);
  }

  // get min an max;
  const float pixel_min = min(min(i.x, i.y), i.z);
  const float pixel_max = max(max(i.x, i.y), i.z);

  // Chroma relative display gamut and scene "mapping" gamut.

  const float epsilon = 1e-6f;
  // "Distance" to max channel = white_target
  const float display_border_vs_chroma_white = (white_target - mapped_luma) / (pixel_max - mapped_luma + epsilon);
  // "Distance" to min_channel = black_target
  const float display_border_vs_chroma_black = (black_target - mapped_luma) / (pixel_min - mapped_luma - epsilon);

  const float display_border_vs_chroma = min(display_border_vs_chroma_white, display_border_vs_chroma_black);
  // "Distance" to min channel = 0.0
  const float chroma_vs_mapping_border = (mapped_luma - pixel_min) / (mapped_luma + epsilon);

  // Hyperbolic gamut compression
  // Small chroma values, i.e. colors close to the acromatic axis are preserved while large chroma values are compressed.

  const float pixel_chroma_adjustment = 1.0f / (chroma_vs_mapping_border * display_border_vs_chroma + epsilon);
  const float hyperbolic_chroma =
    2.0f * chroma_vs_mapping_border /
    (1.0f - chroma_vs_mapping_border * chroma_vs_mapping_border + epsilon) * pixel_chroma_adjustment;

  const float hyperbolic_z = sqrt(hyperbolic_chroma * hyperbolic_chroma + 1.0f);
  const float chroma_factor = hyperbolic_chroma / (1.0f + hyperbolic_z) * display_border_vs_chroma;

  i = mapped_luma + chroma_factor * (i - mapped_luma);

  // Copy over the alpha channel
  i.w = alpha;

  out.write(int2(x, y), i);
}
