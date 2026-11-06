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


// Slang port of ../colorharmonizer.cl

import common;
import colorspace;

float _wrap_hue(float h)
{
  h = fmod(h, 1.0f);
  if(h < 0.0f) h += 1.0f;
  return h;
}

// pull the pixel hue toward the nearest harmony node, scaled by gaussian
// proximity; also returns the winning node and its weight
float _get_weighted_hue_shift(const float px_hue, StructuredBuffer<float> nodes, const int num_nodes,
                              const float zone_width_factor, out int winning_idx, out float max_w)
{
  winning_idx = 0;
  max_w = 0.0f;
  if(num_nodes <= 0) return 0.0f;

  const float sigma = zone_width_factor * 0.5f / float(num_nodes);
  const float inv_2sigma2 = 1.0f / (2.0f * sigma * sigma);

  float diff_winning = 0.0f;
  for(int i = 0; i < num_nodes; i++)
  {
    float d = abs(px_hue - nodes[i]);
    if(d > 0.5f) d = 1.0f - d;

    const float w = exp(-d * d * inv_2sigma2);
    float diff = nodes[i] - px_hue;
    if(diff > 0.5f) diff -= 1.0f;
    else if(diff < -0.5f) diff += 1.0f;

    if(w > max_w)
    {
      max_w = w;
      winning_idx = i;
      diff_winning = diff;
    }
  }
  return diff_winning * max_w;
}

// the per-pixel corrections (hue_delta, sat_delta), and the forward JCH
// conversion as (J, chroma, normalized hue, alpha) for the apply kernel
[shader("compute")]
[numthreads(16, 16, 1)]
void colorharmonizer_map(uint3 gid : SV_DispatchThreadID,
                         uniform ImageIn in, uniform RWStructuredBuffer<float2> p_out,
                         uniform RWStructuredBuffer<float4> jch_out, uniform int width, uniform int height,
                         uniform StructuredBuffer<float> matrix_in, uniform StructuredBuffer<float> nodes,
                         uniform int num_nodes, uniform float zone_width,
                         uniform StructuredBuffer<float> node_saturation, uniform float L_white)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float4 pix_in = in.Areadpixel(x, y);
  const float4 XYZ_D65 = matrix_product_float4(max(0.0f, pix_in), matrix_in);
  const float4 xyY = dt_D65_XYZ_to_xyY(XYZ_D65);
  const float4 JCH = xyY_to_dt_UCS_JCH(xyY, L_white);

  const float hue = (JCH.z + M_PI_F) / DT_2PI_F;

  const int idx = y * width + x;
  jch_out[idx] = float4(JCH.x, JCH.y, hue, pix_in.w);

  int winning_idx;
  float max_weight;
  const float hue_shift = _get_weighted_hue_shift(hue, nodes, num_nodes, zone_width, winning_idx, max_weight);
  p_out[idx] = float2(hue_shift, (node_saturation[winning_idx] - 1.0f) * max_weight);
}

// the smoothed corrections applied to the cached JCH
[shader("compute")]
[numthreads(16, 16, 1)]
void colorharmonizer_apply(uint3 gid : SV_DispatchThreadID,
                           uniform ImageOut out, uniform int width, uniform int height,
                           uniform StructuredBuffer<float> matrix_out, uniform StructuredBuffer<float4> jch_in,
                           uniform StructuredBuffer<float2> corrections, uniform float effect_strength,
                           uniform float protect_neutral, uniform float L_white)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const int k = y * width + x;
  const float4 cached = jch_in[k];
  const float chroma = cached.y;
  const float2 corr = corrections[k];

  const float t = protect_neutral;
  const float cutoff = t * t * t * 0.03f;
  const float chroma_weight = chroma / (chroma + cutoff + 1.0e-5f);

  const float4 JCH = float4(cached.x,
                            max(chroma * (1.0f + corr.y * chroma_weight), 0.0f),
                            _wrap_hue(cached.z + corr.x * effect_strength * chroma_weight) * DT_2PI_F - M_PI_F,
                            0.0f);

  const float4 xyY = dt_UCS_JCH_to_xyY(JCH, L_white);
  float4 pix_out = matrix_product_float4(dt_xyY_to_XYZ(xyY), matrix_out);
  pix_out.w = cached.w;
  out.write(int2(x, y), pix_out);
}
