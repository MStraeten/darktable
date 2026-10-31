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

// Slang ports of the kernels in ../satcurve.cl

import common;
import colorspace;

static const int DT_IOP_SATCURVE_RES = 256;
static const int DT_IOP_SATCURVE_HIST_RES = 256;

// dt_iop_satcurve_formula_t
static const int DT_IOP_SATCURVE_JZAZBZ = 0;
static const int DT_IOP_SATCURVE_DTUCS = 1;

static const float FLT_MIN = 1.175494350822287507969e-38f;

// periodic lookup in the hue-indexed gamut LUT, unlike lookup_gamut() of
// colorspace.slang with the bin wrap of satcurve.cl
float satcurve_lookup_gamut(StructuredBuffer<float> gamut_lut, float h)
{
  const float position = (h + M_PI_F) * float(LUT_ELEM) / (2.0f * M_PI_F);
  const float position_floor = floor(position);
  const int bin0 = int(position_floor) % LUT_ELEM;
  const int bin1 = (bin0 + 1) % LUT_ELEM;
  const float f = position - position_floor;
  return gamut_lut[bin0] * (1.f - f) + gamut_lut[bin1] * f;
}

float satcurve_soft_clip(float x, float knee, float maximum)
{
  if(x <= knee) return x;
  const float range = maximum - knee;
  if(range <= 0.f) return maximum;
  return knee + range * (1.f - dtcl_exp(-(x - knee) / range));
}

// linear interpolation into the 256-entry curve LUT
float satcurve_lookup_lut(StructuredBuffer<float> lut, float x)
{
  const float position = clamp(x, 0.f, 1.f) * float(DT_IOP_SATCURVE_RES - 1);
  const int i = min(int(position), DT_IOP_SATCURVE_RES - 2);
  return lut[i] + (position - float(i)) * (lut[i + 1] - lut[i]);
}

// clips Cz so that the JzAzBz -> XYZ back-transform never produces negative LMS
float clip_jz_chroma(float Jz, float Cz, float ch, float sh)
{
  const float d0 = 1.6295499532821566e-11f;
  const float dd = -0.56f;
  float Iz = (Jz + d0) / (1.f + dd - dd * (Jz + d0));
  Iz = max(Iz, 0.f);

  const float AI_trans_1[3] = { 0.1386050432715393f, -0.1386050432715393f, -0.0960192420263190f };
  const float AI_trans_2[3] = { 0.0580473161561189f, -0.0580473161561189f, -0.8118918960560390f };

  const float4 izab = float4(Iz, Cz * ch, Cz * sh, 0.f);
  float max_c = Cz;
  for(int k = 0; k < 3; k++)
  {
    const float lms = dot(float4(1.0f, AI_trans_1[k], AI_trans_2[k], 0.0f), izab);
    if(lms < 0.f) max_c = min(max_c, -Iz / (AI_trans_1[k] * ch + AI_trans_2[k] * sh));
  }
  return max(max_c, 0.f);
}

float curve_to_factor(float c)
{
  return max(2.f * c, 0.f);
}

// maximum HSB saturation reachable at the gamut boundary for J, h in dt UCS
float satcurve_ucs_gamut_saturation(float J, float h, float L_white, StructuredBuffer<float> gamut_lut)
{
  const float max_colorfulness = max(satcurve_lookup_gamut(gamut_lut, h), FLT_MIN);
  const float max_chroma = 15.932993652962535f * dtcl_pow(J / L_white, 0.6523997524738018f)
                           * dtcl_pow(max_colorfulness, 0.6007557017508491f) / L_white;
  const float4 HSB_gamut_boundary = dt_UCS_JCH_to_HSB(float4(J, max_chroma, h, 0.f));
  return max(HSB_gamut_boundary.y, FLT_MIN);
}

float satcurve_s_in_norm(float4 rgb_in, StructuredBuffer<float> matrix_in, StructuredBuffer<float> gamut_lut,
                         int formula, float L_white)
{
  const float4 rgb = max(rgb_in, float4(0.f));
  const float4 xyz = matrix_product_float4(rgb, matrix_in);

  if(formula == DT_IOP_SATCURVE_DTUCS)
  {
    const float4 xyY = dt_D65_XYZ_to_xyY(xyz);
    const float4 JCH = xyY_to_dt_UCS_JCH(xyY, L_white);
    const float4 HCB = dt_UCS_JCH_to_HCB(JCH);
    const float gamut_s = satcurve_ucs_gamut_saturation(JCH.x, JCH.z, L_white, gamut_lut);
    const float saturation = HCB.z > 0.f ? HCB.y / HCB.z : 0.f;
    return saturation / gamut_s;
  }
  else
  {
    const float4 jab = XYZ_to_JzAzBz(xyz);
    const float Jz = max(jab.x, 0.f);
    const float Cz = dt_hypot(jab.y, jab.z);
    const float h = atan2(jab.z, jab.y);
    const float gamut = max(satcurve_lookup_gamut(gamut_lut, h), FLT_MIN);
    return (Jz > 0.f ? Cz / Jz : 0.f) / gamut;
  }
}

float smoothstep01(float edge0, float edge1, float x)
{
  const float t = clamp((x - edge0) / max(edge1 - edge0, 1e-6f), 0.0f, 1.0f);
  return t * t * (3.0f - 2.0f * t);
}

// the histogram of s_in_norm, so that only its bins are read back to the
// host. hist_bins has to be zeroed before.
[shader("compute")]
[numthreads(16, 16, 1)]
void satcurve_histogram(uint3 gid : SV_DispatchThreadID,
                        uniform ImageIn in, uniform int width, uniform int height,
                        uniform StructuredBuffer<float> matrix_in, uniform StructuredBuffer<float> gamut_lut,
                        uniform int formula, uniform float L_white, uniform RWStructuredBuffer<int> hist_bins)
{
  const int x = int(gid.x);
  const int y = int(gid.y);
  if(x >= width || y >= height) return;

  const float s_in_norm = satcurve_s_in_norm(in.Areadpixel(x, y), matrix_in, gamut_lut, formula, L_white);
  const int bin = clamp(int(s_in_norm * float(DT_IOP_SATCURVE_HIST_RES - 1)), 0, DT_IOP_SATCURVE_HIST_RES - 1);
  InterlockedAdd(hist_bins[bin], 1);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void satcurvergb(uint3 gid : SV_DispatchThreadID,
                 uniform ImageIn in, uniform ImageOut out,
                 uniform ImageIn sat_mask_in, uniform ImageIn bri_mask_in,
                 uniform ImageIn noise_confidence, uniform int use_mask,
                 uniform int width, uniform int height,
                 uniform StructuredBuffer<float> matrix_in, uniform StructuredBuffer<float> matrix_out,
                 uniform StructuredBuffer<float> sat_lut, uniform StructuredBuffer<float> bri_lut,
                 uniform StructuredBuffer<float> gamut_lut,
                 uniform int formula, uniform float L_white, uniform float noise_protection)
{
  const int x = int(gid.x);
  const int y = int(gid.y);
  if(x >= width || y >= height) return;

  const float4 pix_in = in.Areadpixel(x, y);
  const float4 rgb = max(pix_in, float4(0.f));
  float4 xyz = matrix_product_float4(rgb, matrix_in);

  // Metal drops resources a kernel never reads, which would shift the
  // bindings, so the masks are read whether or not use_mask is set
  const float sat_mask = clamp(sat_mask_in.Areadsingle(x, y), 0.f, 1.f);
  const float bri_mask = clamp(bri_mask_in.Areadsingle(x, y), 0.f, 1.f);
  const float noise_conf = use_mask != 0 ? clamp(noise_confidence.Areadsingle(x, y), 0.f, 1.f) : 0.f;
  const float effect = 1.f - clamp(noise_protection * noise_conf, 0.f, 1.f);

  if(formula == DT_IOP_SATCURVE_DTUCS)
  {
    float4 xyY = dt_D65_XYZ_to_xyY(xyz);
    float4 JCH = xyY_to_dt_UCS_JCH(xyY, L_white);
    float4 HCB = dt_UCS_JCH_to_HCB(JCH);

    float4 HSB = float4(HCB.x, HCB.z > 0.f ? HCB.y / HCB.z : 0.f, HCB.z, 0.f);

    const float gamut_s = satcurve_ucs_gamut_saturation(JCH.x, JCH.z, L_white, gamut_lut);
    const float raw_s_in = HSB.y / gamut_s;

    const float s_in_sat = use_mask != 0 ? sat_mask : raw_s_in;
    const float s_in_bri = use_mask != 0 ? bri_mask : raw_s_in;

    const float sat_c = clamp(satcurve_lookup_lut(sat_lut, s_in_sat), 0.f, 1.f);
    const float bri_c = clamp(satcurve_lookup_lut(bri_lut, s_in_bri), 0.f, 1.f);
    const float sat_factor = 1.f + effect * (curve_to_factor(sat_c) - 1.f);
    const float bri_factor = 1.f + effect * (curve_to_factor(bri_c) - 1.f);

    HSB.y = max(HSB.y * sat_factor, 0.f);
    HSB.y = satcurve_soft_clip(HSB.y, 0.8f * gamut_s, gamut_s);

    JCH = dt_UCS_HSB_to_JCH(HSB);
    HCB = dt_UCS_JCH_to_HCB(JCH);

    HCB.y = max(HCB.y * bri_factor, 0.f);
    HCB.z = max(HCB.z * bri_factor, 0.f);

    JCH = dt_UCS_HCB_to_JCH(HCB);

    const float gamut_s_out = satcurve_ucs_gamut_saturation(JCH.x, JCH.z, L_white, gamut_lut);

    float4 HSB_out = float4(HCB.x, HCB.z > 0.f ? HCB.y / HCB.z : 0.f, HCB.z, 0.f);
    HSB_out.y = satcurve_soft_clip(HSB_out.y, 0.8f * gamut_s_out, gamut_s_out);

    JCH = dt_UCS_HSB_to_JCH(HSB_out);
    xyY = dt_UCS_JCH_to_xyY(JCH, L_white);
    xyz = dt_xyY_to_XYZ(xyY);
  }
  else
  {
    float4 jab = XYZ_to_JzAzBz(xyz);

    const float Jz = max(jab.x, 0.f);
    const float Cz = dt_hypot(jab.y, jab.z);
    const float h = atan2(jab.z, jab.y);
    const float ch = dtcl_cos(h), sh = dtcl_sin(h);
    const float gamut = max(satcurve_lookup_gamut(gamut_lut, h), FLT_MIN);
    const float raw_s_in = (Jz > 0.f ? Cz / Jz : 0.f) / gamut;

    const float s_in_sat = use_mask != 0 ? sat_mask : raw_s_in;
    const float s_in_bri = use_mask != 0 ? bri_mask : raw_s_in;

    const float sat_c = clamp(satcurve_lookup_lut(sat_lut, s_in_sat), 0.f, 1.f);
    const float bri_c = clamp(satcurve_lookup_lut(bri_lut, s_in_bri), 0.f, 1.f);
    const float sat_factor = 1.f + effect * (curve_to_factor(sat_c) - 1.f);
    const float bri_factor = 1.f + effect * (curve_to_factor(bri_c) - 1.f);

    const float s_out = satcurve_soft_clip(max(s_in_sat * sat_factor, 0.f), 0.8f, 1.f) * gamut;

    const float r = dt_hypot(Jz, Cz);
    const float inv_norm = 1.f / sqrt(1.f + s_out * s_out);

    float Jz_tmp = r * inv_norm;
    float Cz_tmp = clip_jz_chroma(Jz_tmp, r * s_out * inv_norm, ch, sh);

    Jz_tmp *= bri_factor;
    Cz_tmp = clip_jz_chroma(Jz_tmp, Cz_tmp * bri_factor, ch, sh);

    jab.x = Jz_tmp;
    jab.y = Cz_tmp * ch;
    jab.z = Cz_tmp * sh;
    xyz = JzAzBz_2_XYZ(jab);
  }

  float4 pixout = max(matrix_product_float4(xyz, matrix_out), float4(0.f));
  pixout.w = pix_in.w;
  out.write(int2(x, y), pixout);
}

// the saturation mask for the GUI: white at zero, magenta {0.5, 0, 0.5} at full saturation
[shader("compute")]
[numthreads(16, 16, 1)]
void satcurve_mask(uint3 gid : SV_DispatchThreadID,
                   uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
                   uniform StructuredBuffer<float> matrix_in, uniform StructuredBuffer<float> gamut_lut,
                   uniform int formula, uniform float L_white)
{
  const int x = int(gid.x);
  const int y = int(gid.y);
  if(x >= width || y >= height) return;

  const float4 pix_in = in.Areadpixel(x, y);
  const float s_in_norm = satcurve_s_in_norm(pix_in, matrix_in, gamut_lut, formula, L_white);
  const float t = sqrt(clamp(s_in_norm, 0.f, 1.f));
  out.write(int2(x, y), float4(1.0f - 0.5f * t, 1.0f - t, 1.0f - 0.5f * t, pix_in.w));
}

// the single channel saturation mask fed to the guided filter
[shader("compute")]
[numthreads(16, 16, 1)]
void satcurve_prepare_scalar_mask(uint3 gid : SV_DispatchThreadID,
                                  uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
                                  uniform StructuredBuffer<float> matrix_in, uniform StructuredBuffer<float> gamut_lut,
                                  uniform int formula, uniform float L_white)
{
  const int x = int(gid.x);
  const int y = int(gid.y);
  if(x >= width || y >= height) return;

  const float s_in_norm = satcurve_s_in_norm(in.Areadpixel(x, y), matrix_in, gamut_lut, formula, L_white);
  out.write(int2(x, y), float4(max(s_in_norm, 0.f)));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void satcurve_prepare_perceptual_guide(uint3 gid : SV_DispatchThreadID,
                                       uniform ImageIn in, uniform ImageOut out, uniform int width,
                                       uniform int height, uniform StructuredBuffer<float> matrix_in,
                                       uniform float L_white)
{
  const int x = int(gid.x);
  const int y = int(gid.y);
  if(x >= width || y >= height) return;

  const float4 rgb = max(in.Areadpixel(x, y), float4(0.f));
  const float4 xyz = matrix_product_float4(rgb, matrix_in);
  const float4 xyY = dt_D65_XYZ_to_xyY(xyz);
  const float4 JCH = xyY_to_dt_UCS_JCH(xyY, L_white);
  const float norm = max(L_white, 1e-6f);
  out.write(int2(x, y), float4(JCH.x / norm, JCH.y * dtcl_cos(JCH.z) / norm, JCH.y * dtcl_sin(JCH.z) / norm, 0.f));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void satcurve_prepare_filter_confidence(uint3 gid : SV_DispatchThreadID,
                                        uniform ImageIn raw, uniform ImageIn filtered, uniform ImageIn guide,
                                        uniform ImageOut control, uniform ImageOut confidence,
                                        uniform int width, uniform int height,
                                        uniform float protect_from, uniform float protect_to)
{
  const int x = int(gid.x);
  const int y = int(gid.y);
  if(x >= width || y >= height) return;

  const float raw_center = raw.Areadsingle(x, y);
  const float filtered_center = filtered.Areadsingle(x, y);
  const float4 guide_center = guide.Areadpixel(x, y);
  float residual_energy = 0.f, edge_energy = 0.f;
  for(int dy = -1; dy <= 1; dy++)
    for(int dx = -1; dx <= 1; dx++)
    {
      const int nx = clamp(x + dx, 0, width - 1);
      const int ny = clamp(y + dy, 0, height - 1);
      const float residual = raw.Areadsingle(nx, ny) - filtered.Areadsingle(nx, ny);
      const float4 delta = guide.Areadpixel(nx, ny) - guide_center;
      residual_energy += residual * residual;
      edge_energy += dot(delta, delta);
    }
  residual_energy /= 9.f;
  edge_energy /= 27.f;
  const float noisy = smoothstep(0.0005f, 0.01f, residual_energy);
  const float edge_safe = 1.f - smoothstep(0.0005f, 0.01f, edge_energy);
  confidence.write(int2(x, y), float4(noisy * edge_safe));

  // only the neutral region weight, as protection_weight() with no noise protection
  const float neutral_w = smoothstep01(protect_from, protect_to, raw_center);
  control.write(int2(x, y), float4(clamp(raw_center + neutral_w * (filtered_center - raw_center), 0.f, 1.f)));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void satcurve_mask_from_control(uint3 gid : SV_DispatchThreadID,
                                uniform ImageIn raw, uniform ImageIn filtered, uniform ImageIn confidence,
                                uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
                                uniform float protect_from, uniform float protect_to,
                                uniform float noise_protection)
{
  const int x = int(gid.x);
  const int y = int(gid.y);
  if(x >= width || y >= height) return;

  const float raw_value = raw.Areadsingle(x, y);
  const float filtered_value = filtered.Areadsingle(x, y);
  const float conf = clamp(confidence.Areadsingle(x, y), 0.f, 1.f);

  const float neutral_w = smoothstep01(protect_from, protect_to, raw_value);
  const float noise_w = 1.f - clamp(noise_protection * conf, 0.f, 1.f);
  const float weight = neutral_w * noise_w;

  const float value = clamp(raw_value + weight * (filtered_value - raw_value), 0.f, 1.f);
  const float t = sqrt(value);
  out.write(int2(x, y), float4(1.f - .5f * t, 1.f - t, 1.f - .5f * t, in.Areadpixel(x, y).w));
}

// the module with neutral curves passes its input through, as process_cl()
// does with a copy
[shader("compute")]
[numthreads(16, 16, 1)]
void satcurve_copy(uint3 gid : SV_DispatchThreadID,
                   uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height)
{
  const int x = int(gid.x);
  const int y = int(gid.y);
  if(x >= width || y >= height) return;
  out.write(int2(x, y), in.Areadpixel(x, y));
}
