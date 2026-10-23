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

// Slang ports of the kernels in ../colorequal.cl

import common;
import colorspace;

// dt_iop_colorequal_channel_t
static const int HUE = 0;
static const int SATURATION = 1;
static const int BRIGHTNESS = 2;
static const int GRAD_SWITCH = 4;
static const int SATURATION_GRAD = SATURATION + GRAD_SWITCH;
static const int BRIGHTNESS_GRAD = BRIGHTNESS + GRAD_SWITCH;

static const float SAT_EFFECT = 2.0f;
static const float BRIGHT_EFFECT = 8.0f;
static const float SATSIZE = 4096.0f;
static const float FLT_EPSILON = 1.19209290e-7f;

// the colorspace.h helpers colorspace.slang lacks
float4 dt_UCS_HSB_to_XYZ(float4 HSB, float L_w)
{
  const float4 JCH = dt_UCS_HSB_to_JCH(HSB);
  const float4 xyY = dt_UCS_JCH_to_xyY(JCH, L_w);
  return dt_xyY_to_XYZ(xyY);
}

float4 dt_UCS_LUV_to_JCH(float L_star, float L_white, float2 UV_star_prime)
{
  const float M2 = UV_star_prime.x * UV_star_prime.x + UV_star_prime.y * UV_star_prime.y; // square of colorfulness M
  return float4(L_star / L_white,
                15.932993652962535f * dtcl_pow(L_star, 0.6523997524738018f) * dtcl_pow(M2, 0.6007557017508491f)
                  / L_white,
                atan2(UV_star_prime.y, UV_star_prime.x),
                0.0f);
}

float _get_satweight(float sat, StructuredBuffer<float> weights)
{
  const float isat = SATSIZE * (1.0f + clamp(sat, -1.0f, 1.0f - (1.0f / SATSIZE)));
  const float base = floor(isat);
  const int i = int(base);
  return weights[i] + (isat - base) * (weights[i + 1] - weights[i]);
}

float gamut_map_HSB(float4 HSB, StructuredBuffer<float> gamut_LUT, float L_white)
{
  const float4 JCH = dt_UCS_HSB_to_JCH(HSB);
  const float max_colorfulness = lookup_gamut(gamut_LUT, JCH.z);
  const float max_chroma = 15.932993652962535f * dtcl_pow(JCH.x * L_white, 0.6523997524738018f)
                           * dtcl_pow(max_colorfulness, 0.6007557017508491f) / L_white;
  const float4 JCH_gamut_boundary = float4(JCH.x, max_chroma, JCH.z, 0.0f);
  const float4 HSB_gamut_boundary = dt_UCS_JCH_to_HSB(JCH_gamut_boundary);

  // soft-clip the current pixel saturation at constant brightness
  return soft_clip(HSB.y, 0.8f * HSB_gamut_boundary.y, HSB_gamut_boundary.y);
}

float4 _matrix_dot(float4 v, StructuredBuffer<float4> mat)
{
  float4 M[3] = { mat[0], mat[1], mat[2] };
  return matrix_dot(v, M);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void init_covariance(uint3 gid : SV_DispatchThreadID,
                     uniform RWStructuredBuffer<float4> covariance, uniform StructuredBuffer<float2> uv,
                     uniform int width, uniform int height)
{
  const int col = int(gid.x);
  const int row = int(gid.y);
  if(col >= width || row >= height) return;

  const int k = row * width + col;
  const float2 UV = uv[k];
  covariance[k] = float4(UV.x * UV.x, UV.x * UV.y, UV.x * UV.y, UV.y * UV.y);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void finish_covariance(uint3 gid : SV_DispatchThreadID,
                       uniform RWStructuredBuffer<float4> covariance, uniform StructuredBuffer<float2> uv,
                       uniform int width, uniform int height)
{
  const int col = int(gid.x);
  const int row = int(gid.y);
  if(col >= width || row >= height) return;

  const int k = row * width + col;
  const float2 UV = uv[k];
  covariance[k] -= float4(UV.x * UV.x, UV.x * UV.y, UV.x * UV.y, UV.y * UV.y);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void prepare_prefilter(uint3 gid : SV_DispatchThreadID,
                       uniform StructuredBuffer<float2> uv, uniform StructuredBuffer<float4> covariance,
                       uniform RWStructuredBuffer<float4> a, uniform RWStructuredBuffer<float2> b,
                       uniform float eps, uniform int width, uniform int height)
{
  const int col = int(gid.x);
  const int row = int(gid.y);
  if(col >= width || row >= height) return;

  const int k = row * width + col;
  const float4 cov = covariance[k];
  const float4 sigma = float4(cov.x + eps, cov.y, cov.z, cov.w + eps);
  const float det = sigma.x * sigma.w - sigma.y * sigma.z;

  float4 ak = float4(0.0f);
  if(abs(det) > 4.0f * FLT_EPSILON)
  {
    const float4 sigma_inv = float4(sigma.w / det, -sigma.y / det, -sigma.z / det, sigma.x / det);
    ak.x = cov.x * sigma_inv.x + cov.y * sigma_inv.y;
    ak.y = cov.x * sigma_inv.z + cov.y * sigma_inv.w;
    ak.z = cov.z * sigma_inv.x + cov.w * sigma_inv.y;
    ak.w = cov.z * sigma_inv.z + cov.w * sigma_inv.w;
  }
  a[k] = ak;

  const float2 UV = uv[k];
  b[k] = float2(UV.x - ak.x * UV.x - ak.y * UV.y,
                UV.y - ak.z * UV.x - ak.w * UV.y);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void apply_prefilter(uint3 gid : SV_DispatchThreadID,
                     uniform RWStructuredBuffer<float2> uv, uniform StructuredBuffer<float> saturation,
                     uniform StructuredBuffer<float4> a, uniform StructuredBuffer<float2> b,
                     uniform StructuredBuffer<float> weights,
                     uniform float sat_shift, uniform int width, uniform int height)
{
  const int col = int(gid.x);
  const int row = int(gid.y);
  if(col >= width || row >= height) return;

  const int k = row * width + col;
  const float2 UV = uv[k];
  const float4 ak = a[k];
  const float2 bk = b[k];
  const float2 cv = float2(ak.x * UV.x + ak.y * UV.y + bk.x,
                           ak.z * UV.x + ak.w * UV.y + bk.y);

  const float satweight = _get_satweight(saturation[k] - sat_shift, weights);
  uv[k] = float2(lerp(UV.x, cv.x, satweight), lerp(UV.y, cv.y, satweight));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void prepare_correlations(uint3 gid : SV_DispatchThreadID,
                          uniform StructuredBuffer<float2> corrections, uniform StructuredBuffer<float> b_corrections,
                          uniform StructuredBuffer<float2> uv, uniform RWStructuredBuffer<float4> correlations,
                          uniform int width, uniform int height)
{
  const int col = int(gid.x);
  const int row = int(gid.y);
  if(col >= width || row >= height) return;

  const int k = row * width + col;
  const float2 UV = uv[k];
  correlations[k] = float4(UV.x * corrections[k].y, UV.y * corrections[k].y,
                           UV.x * b_corrections[k], UV.y * b_corrections[k]);
}

// also writes the covariance
[shader("compute")]
[numthreads(16, 16, 1)]
void finish_correlations(uint3 gid : SV_DispatchThreadID,
                         uniform StructuredBuffer<float2> corrections, uniform StructuredBuffer<float> b_corrections,
                         uniform StructuredBuffer<float2> uv, uniform RWStructuredBuffer<float4> correlations,
                         uniform RWStructuredBuffer<float4> covariance,
                         uniform int width, uniform int height)
{
  const int col = int(gid.x);
  const int row = int(gid.y);
  if(col >= width || row >= height) return;

  const int k = row * width + col;
  const float2 UV = uv[k];
  covariance[k] -= float4(UV.x * UV.x, UV.x * UV.y, UV.x * UV.y, UV.y * UV.y);
  correlations[k] -= float4(UV.x * corrections[k].y, UV.y * corrections[k].y,
                            UV.x * b_corrections[k], UV.y * b_corrections[k]);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void final_guide(uint3 gid : SV_DispatchThreadID,
                 uniform StructuredBuffer<float4> covariance, uniform StructuredBuffer<float4> correlations,
                 uniform StructuredBuffer<float2> corrections, uniform StructuredBuffer<float> b_corrections,
                 uniform StructuredBuffer<float2> uv,
                 uniform RWStructuredBuffer<float4> a, uniform RWStructuredBuffer<float2> b,
                 uniform float eps, uniform int width, uniform int height)
{
  const int col = int(gid.x);
  const int row = int(gid.y);
  if(col >= width || row >= height) return;

  const int k = row * width + col;
  const float4 cov = covariance[k];
  const float4 corr = correlations[k];
  const float4 sigma = float4(cov.x + eps, cov.y, cov.z, cov.w + eps);
  const float det = sigma.x * sigma.w - sigma.y * sigma.z;

  float4 ak = float4(0.0f);
  if(abs(det) > 4.0f * FLT_EPSILON)
  {
    const float4 sigma_inv = float4(sigma.w / det, -sigma.y / det, -sigma.z / det, sigma.x / det);
    ak.x = corr.x * sigma_inv.x + corr.y * sigma_inv.y;
    ak.y = corr.x * sigma_inv.z + corr.y * sigma_inv.w;
    ak.z = corr.z * sigma_inv.x + corr.w * sigma_inv.y;
    ak.w = corr.z * sigma_inv.z + corr.w * sigma_inv.w;
  }
  a[k] = ak;

  const float2 UV = uv[k];
  b[k] = float2(corrections[k].y - ak.x * UV.x - ak.y * UV.y,
                b_corrections[k] - ak.z * UV.x - ak.w * UV.y);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void apply_guided(uint3 gid : SV_DispatchThreadID,
                  uniform StructuredBuffer<float2> uv, uniform StructuredBuffer<float> saturation,
                  uniform StructuredBuffer<float> scharr, uniform StructuredBuffer<float4> a,
                  uniform StructuredBuffer<float2> b, uniform RWStructuredBuffer<float2> corrections,
                  uniform RWStructuredBuffer<float> b_corrections, uniform StructuredBuffer<float> weights,
                  uniform float sat_shift, uniform float bright_shift, uniform int width, uniform int height)
{
  const int col = int(gid.x);
  const int row = int(gid.y);
  if(col >= width || row >= height) return;

  const int k = row * width + col;
  const float2 UV = uv[k];
  const float4 ak = a[k];
  const float2 bk = b[k];
  const float2 CV = float2(ak.x * UV.x + ak.y * UV.y + bk.x,
                           ak.z * UV.x + ak.w * UV.y + bk.y);

  corrections[k].y = 1.0f + (CV.x - 1.0f) * _get_satweight(saturation[k] - sat_shift, weights);
  const float gradient_weight = 1.0f - clipf(scharr[k]);
  b_corrections[k] = CV.y * gradient_weight * _get_satweight(saturation[k] - bright_shift, weights);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void sample_input(uint3 gid : SV_DispatchThreadID,
                  uniform ImageIn dev_in, uniform RWStructuredBuffer<float> saturation,
                  uniform RWStructuredBuffer<float> lum, uniform RWStructuredBuffer<float2> uv,
                  uniform RWStructuredBuffer<float4> pix_out, uniform StructuredBuffer<float4> mat,
                  uniform int width, uniform int height)
{
  const int col = int(gid.x);
  const int row = int(gid.y);
  if(col >= width || row >= height) return;

  const int k = row * width + col;
  const float4 pix_in = dev_in.Areadpixel(col, row);

  // saturation from the input data
  const float dmin = min(pix_in.x, min(pix_in.y, pix_in.z));
  const float dmax = max(pix_in.x, max(pix_in.y, pix_in.z));
  const float delta = dmax - dmin;
  saturation[k] = (dmax > NORM_MIN && delta > NORM_MIN) ? delta / dmax : 0.0f;

  const float4 XYZ_D65 = _matrix_dot(pix_in, mat);
  const float4 xyY = dt_D65_XYZ_to_xyY(XYZ_D65);

  lum[k] = Y_to_dt_UCS_L_star(xyY.z);
  uv[k] = xyY_to_dt_UCS_UV(xyY);
  pix_out[k].w = pix_in.w;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void write_output(uint3 gid : SV_DispatchThreadID,
                  uniform ImageOut dev_out, uniform RWStructuredBuffer<float4> pix_out,
                  uniform StructuredBuffer<float2> corrections, uniform StructuredBuffer<float> b_corrections,
                  uniform StructuredBuffer<float4> mat, uniform StructuredBuffer<float> gamut_LUT,
                  uniform float white, uniform int width, uniform int height)
{
  const int col = int(gid.x);
  const int row = int(gid.y);
  if(col >= width || row >= height) return;

  const int k = row * width + col;
  float4 p = pix_out[k];
  p.x += corrections[k].x;
  p.y = max(0.0f, p.y * (1.0f + SAT_EFFECT * (corrections[k].y - 1.0f)));
  p.z = max(0.0f, p.z * (1.0f + BRIGHT_EFFECT * b_corrections[k]));
  p.y = gamut_map_HSB(p, gamut_LUT, white);
  pix_out[k] = p;

  const float4 XYZ_D65 = dt_UCS_HSB_to_XYZ(p, white);
  dev_out.write(int2(col, row), _matrix_dot(XYZ_D65, mat));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void write_visual(uint3 gid : SV_DispatchThreadID,
                  uniform ImageOut dev_out, uniform StructuredBuffer<float4> pixout,
                  uniform StructuredBuffer<float2> corrections, uniform StructuredBuffer<float> b_corrections,
                  uniform StructuredBuffer<float> saturation, uniform StructuredBuffer<float> scharr,
                  uniform StructuredBuffer<float> weights,
                  uniform float bright_shift, uniform float sat_shift, uniform float white, uniform int mode,
                  uniform int width, uniform int height)
{
  const int col = int(gid.x);
  const int row = int(gid.y);
  if(col >= width || row >= height) return;

  const int k = row * width + col;
  const float val = dtcl_sqrt(max(0.0f, pixout[k].z * white));
  float corr = 0.0f;
  switch(mode)
  {
    case BRIGHTNESS:
      corr = BRIGHT_EFFECT * b_corrections[k];
      break;
    case SATURATION:
      corr = SAT_EFFECT * (corrections[k].y - 1.0f);
      break;
    case BRIGHTNESS_GRAD:
      corr = _get_satweight(saturation[k] - bright_shift, weights) - 0.5f;
      break;
    case SATURATION_GRAD:
      corr = _get_satweight(saturation[k] - sat_shift, weights) - 0.5f;
      break;
    default: // HUE
      corr = 0.2f * corrections[k].x;
      break;
  }

  const bool neg = corr < 0.0f;
  corr = abs(corr);
  corr = corr < 2e-3f ? 0.0f : corr;

  float4 pout = float4(max(0.0f, neg ? val - corr : val),
                       max(0.0f, val - corr),
                       max(0.0f, neg ? val : val - corr),
                       0.0f);

  if(mode == BRIGHTNESS && scharr[k] > 0.1f)
  {
    pout.x = pout.z = 0.0f;
    pout.y = scharr[k];
  }
  dev_out.write(int2(col, row), pout);
}

// a 1D kernel over the columns in OpenCL; here the other rows of the 16x16
// work groups do nothing
[shader("compute")]
[numthreads(16, 16, 1)]
void draw_weight(uint3 gid : SV_DispatchThreadID,
                 uniform ImageOut dev_out, uniform StructuredBuffer<float> weights,
                 uniform float bright_shift, uniform float sat_shift, uniform int mode,
                 uniform int width, uniform int height)
{
  const int col = int(gid.y * 16 + gid.x);
  if(col >= width) return;

  const float eps = 0.5f / float(height);
  const float shift = (mode == SATURATION_GRAD) ? sat_shift : bright_shift;
  const float4 mark = float4(0.0f, 1.0f, 0.0f, 0.0f);
  for(int i = 0; i < 16; i++)
  {
    const float pos = float(16 * col + i) / float(16 * width);
    const float weight = _get_satweight(pos - shift, weights);
    if(weight > eps && weight < 1.0f - eps)
    {
      const int row = int((1.0f - weight) * float(height - 1));
      dev_out.write(int2(col, row), mark);
    }
  }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void process_data(uint3 gid : SV_DispatchThreadID,
                  uniform StructuredBuffer<float2> uv, uniform RWStructuredBuffer<float> Lscharr,
                  uniform StructuredBuffer<float> saturation, uniform RWStructuredBuffer<float2> corrections,
                  uniform RWStructuredBuffer<float> b_corrections, uniform RWStructuredBuffer<float4> pixout,
                  uniform StructuredBuffer<float> LUT_saturation, uniform StructuredBuffer<float> LUT_hue,
                  uniform StructuredBuffer<float> LUT_brightness,
                  uniform float white, uniform float gradient_amp, uniform int guiding,
                  uniform int width, uniform int height)
{
  const int col = int(gid.x);
  const int row = int(gid.y);
  if(col >= width || row >= height) return;

  const int k = row * width + col;
  const float4 JCH = dt_UCS_LUV_to_JCH(Lscharr[k], white, uv[k]);
  const float4 HSB = dt_UCS_JCH_to_HSB(JCH);

  const float hue = HSB.x;
  const float sat = HSB.y;
  pixout[k].x = hue;
  pixout[k].y = sat;
  pixout[k].z = HSB.z;

  if(guiding != 0)
  {
    const int kk = clamp(row, 1, height - 2) * width + clamp(col, 1, width - 2);
    const float kscharr = max(0.0f, scharr_gradient(saturation, kk, width) - 0.02f);
    Lscharr[k] = gradient_amp * kscharr * kscharr;
  }

  if(JCH.y > NORM_MIN)
  {
    corrections[k] = float2(lookup_gamut(LUT_hue, hue), lookup_gamut(LUT_saturation, hue));
    b_corrections[k] = sat * (lookup_gamut(LUT_brightness, hue) - 1.0f);
  }
  else
  {
    corrections[k] = float2(0.0f, 1.0f);
    b_corrections[k] = 0.0f;
  }
}
