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

// Slang ports of kernels in ../extended.cl

import colorspace;
import color_conversion;

float4 opacity_masks(float x, float shadows_weight, float highlights_weight,
                     float midtones_weight, float mask_grey_fulcrum)
{
  float4 res;
  const float x_offset = (x - mask_grey_fulcrum);
  const float x_offset_norm = x_offset / mask_grey_fulcrum;
  const float alpha = 1.f / (1.f + dtcl_exp(x_offset_norm * shadows_weight));    // opacity of shadows
  const float beta = 1.f / (1.f + dtcl_exp(-x_offset_norm * highlights_weight)); // opacity of highlights
  const float gamma = dtcl_exp(-fsquare(x_offset) * midtones_weight / 4.f) * fsquare(1.f - alpha)
                      * fsquare(1.f - beta) * 8.f; // opacity of midtones

  res.x = alpha;
  res.y = gamma;
  res.z = beta;
  res.w = 0.f;
  return res;
}

// dt_iop_colorbalancrgb_saturation_t
static const int DT_COLORBALANCE_SATURATION_JZAZBZ = 0;

[shader("compute")]
[numthreads(16, 16, 1)]
void colorbalancergb(uint3 gid : SV_DispatchThreadID,
                     uniform ImageIn in, uniform ImageOut out,
                     uniform int width, uniform int height,
                     uniform StructuredBuffer<ProfileInfo> profile_info,
                     uniform StructuredBuffer<float> matrix_in, uniform StructuredBuffer<float> matrix_out,
                     uniform StructuredBuffer<float> gamut_lut,
                     uniform float shadows_weight, uniform float highlights_weight, uniform float midtones_weight,
                     uniform float mask_grey_fulcrum,
                     uniform float hue_angle, uniform float chroma_global, uniform float chroma[4], uniform float vibrance,
                     uniform float global_offset[4], uniform float shadows[4], uniform float highlights[4],
                     uniform float midtones[4],
                     uniform float white_fulcrum, uniform float midtones_Y,
                     uniform float grey_fulcrum, uniform float contrast,
                     uniform float brilliance_global, uniform float brilliance[4],
                     uniform float saturation_global, uniform float saturation[4],
                     uniform int mask_display, uniform int mask_type, uniform int checker_1, uniform int checker_2,
                     uniform float checker_color_1[4], uniform float checker_color_2[4], uniform float L_white,
                     uniform int saturation_formula,
                     uniform StructuredBuffer<float> hue_rotation_matrix)
{
  // float4 arguments are passed as float[4]: a float4 aligns to 16 bytes and
  // the padding pushes the parameters past 256 bytes of push constants
  const float4 chroma4 = float4(chroma[0], chroma[1], chroma[2], chroma[3]);
  const float4 global_offset4 = float4(global_offset[0], global_offset[1], global_offset[2], global_offset[3]);
  const float4 shadows4 = float4(shadows[0], shadows[1], shadows[2], shadows[3]);
  const float4 highlights4 = float4(highlights[0], highlights[1], highlights[2], highlights[3]);
  const float4 midtones4 = float4(midtones[0], midtones[1], midtones[2], midtones[3]);
  const float4 brilliance4 = float4(brilliance[0], brilliance[1], brilliance[2], brilliance[3]);
  const float4 saturation4 = float4(saturation[0], saturation[1], saturation[2], saturation[3]);
  const float4 checker_rgba_1 = float4(checker_color_1[0], checker_color_1[1], checker_color_1[2], checker_color_1[3]);
  const float4 checker_rgba_2 = float4(checker_color_2[0], checker_color_2[1], checker_color_2[2], checker_color_2[3]);
  const int x = gid.x;
  const int y = gid.y;
  // the kernel doesn't use the profile or the hue angle, but Metal drops
  // parameters nothing reads
  if(width < 0) out.write(int2(0), float4(profile_info[0].grey, hue_angle, 0.f, 0.f));
  if(x >= width || y >= height) return;
  // we clip pipeline RGB while reading; this also ensures a proper alpha
  const float4 pix_in = max(float4(0.0f), in.readpixel(x, y));

  float4 XYZ_D65 = float4(0.f);
  float4 LMS = float4(0.f);
  float4 RGB = float4(0.f);
  float4 Yrg = float4(0.f);
  float4 Ych = float4(0.f);

  RGB = pix_in;

  // go to CIE 2006 LMS D65
  LMS = matrix_product_float4(RGB, matrix_in);

  // go to Filmlight Yrg
  Yrg = LMS_to_Yrg(LMS);

  // go to Ych
  Ych = Yrg_to_Ych(Yrg);

  // sanitize input: no negative luminance
  Ych.x = max(Ych.x, 0.f);
  const float4 opacities = opacity_masks(dtcl_pow(Ych.x, 0.4101205819200422f), // center middle grey in 50 %
                                         shadows_weight, highlights_weight, midtones_weight, mask_grey_fulcrum);
  const float4 opacities_comp = float4(1.f) - opacities;

  // hue shift, now because we need the gamut limit at output hue right
  // after. The hue rotation is a matrix multiplication.
  const float cos_h = Ych.z;
  const float sin_h = Ych.w;
  Ych.z = hue_rotation_matrix[0] * cos_h + hue_rotation_matrix[1] * sin_h;
  Ych.w = hue_rotation_matrix[2] * cos_h + hue_rotation_matrix[3] * sin_h;

  // linear chroma: distance to achromatic at constant luminance in scene-referred
  const float chroma_boost = chroma_global + dot(opacities, chroma4);
  const float vib = vibrance * (1.0f - dtcl_pow(Ych.y, abs(vibrance)));
  const float chroma_factor = max(1.f + chroma_boost + vib, 0.f);
  Ych.y *= chroma_factor;

  // clip chroma at constant Y and hue
  Ych = gamut_check_Yrg(Ych);

  // go to Yrg for real
  Yrg = Ych_to_Yrg(Ych);

  // go to LMS
  LMS = Yrg_to_LMS(Yrg);

  // go to Filmlight RGB
  RGB = LMS_to_gradingRGB(LMS);

  // color balance

  // global: offset
  RGB += global_offset4;

  // highlights, shadows: 2 slopes with masking
  RGB *= opacities_comp.z * (opacities_comp.x + opacities.x * shadows4) + opacities.z * highlights4;

  // midtones: power with sign preservation
  RGB = sign(RGB) * dtcl_pow(abs(RGB) / white_fulcrum, midtones4) * white_fulcrum;

  // for the non-linear ops we need to go in Yrg again because RGB doesn't preserve color
  LMS = gradingRGB_to_LMS(RGB);
  Yrg = LMS_to_Yrg(LMS);

  // Y midtones power (gamma)
  Yrg.x = dtcl_pow(max(Yrg.x / white_fulcrum, 0.f), midtones_Y) * white_fulcrum;

  // Y fulcrumed contrast
  Yrg.x = grey_fulcrum * dtcl_pow(Yrg.x / grey_fulcrum, contrast);

  LMS = Yrg_to_LMS(Yrg);
  XYZ_D65 = LMS_to_XYZ(LMS);

  // perceptual color adjustments
  if(saturation_formula == DT_COLORBALANCE_SATURATION_JZAZBZ)
  {
    // go to JzAzBz for perceptual saturation
    float4 Jab = XYZ_to_JzAzBz(XYZ_D65);

    // convert to JCh
    float JC[2] = { Jab.x, dt_hypot(Jab.y, Jab.z) }; // brightness/chroma vector
    const float h = atan2(Jab.z, Jab.y);              // hue: (a, b) angle

    // project JC onto S, the saturation eigenvector, with orthogonal vector O
    const float T = atan2(JC[1], JC[0]); // angle of the eigenvector over the hue plane
    const float sin_T = dtcl_sin(T);
    const float cos_T = dtcl_cos(T);
    const float M_rot_dir[2][2] = { { cos_T, sin_T },
                                    { -sin_T, cos_T } };
    const float M_rot_inv[2][2] = { { cos_T, -sin_T },
                                    { sin_T, cos_T } };
    float SO[2];

    // brilliance & saturation: mix of chroma and luminance
    const float boosts[2] = { 1.f + brilliance_global + dot(opacities, brilliance4), // move in S direction
                              saturation_global + dot(opacities, saturation4) };     // move in O direction

    SO[0] = JC[0] * M_rot_dir[0][0] + JC[1] * M_rot_dir[0][1];
    SO[1] = SO[0] * clamp(T * boosts[1], -T, M_PI_F / 2.f - T);
    SO[0] = max(SO[0] * boosts[0], 0.f);

    // project back to JCh, that is rotate back of -T angle
    JC[0] = max(SO[0] * M_rot_inv[0][0] + SO[1] * M_rot_inv[0][1], 0.f);
    JC[1] = max(SO[0] * M_rot_inv[1][0] + SO[1] * M_rot_inv[1][1], 0.f);

    // gamut mapping
    const float out_max_sat_h = lookup_gamut(gamut_lut, h);
    // if JC[0] == 0, the saturation / luminance ratio is infinite: assign the largest practical value we have
    const float sat = (JC[0] > 0.f) ? soft_clip(JC[1] / JC[0], 0.8f * out_max_sat_h, out_max_sat_h)
                                    : out_max_sat_h;
    const float max_C_at_sat = JC[0] * sat;
    // if sat == 0, the chroma is zero: assign the original luminance because there's no need to gamut map
    const float max_J_at_sat = (sat > 0.f) ? JC[1] / sat : JC[0];
    JC[0] = (JC[0] + max_J_at_sat) / 2.f;
    JC[1] = (JC[1] + max_C_at_sat) / 2.f;

    // gamut-clip in Jch at constant hue and lightness, e.g. find the max
    // chroma available at current hue that doesn't yield negative L'M'S'
    // values, which will need to be clipped during conversion
    const float cos_H = dtcl_cos(h);
    const float sin_H = dtcl_sin(h);

    const float d0 = 1.6295499532821566e-11f;
    const float d = -0.56f;
    float Iz = JC[0] + d0;
    Iz /= (1.f + d - d * Iz);
    Iz = max(Iz, 0.f);

    const float4 AI[3] = { float4(1.0f, 0.1386050432715393f, 0.0580473161561189f, 0.0f),
                           float4(1.0f, -0.1386050432715393f, -0.0580473161561189f, 0.0f),
                           float4(1.0f, -0.0960192420263190f, -0.8118918960560390f, 0.0f) };

    // do a test conversion to L'M'S'
    const float4 IzAzBz = float4(Iz, JC[1] * cos_H, JC[1] * sin_H, 0.f);
    LMS.x = dot(AI[0], IzAzBz);
    LMS.y = dot(AI[1], IzAzBz);
    LMS.z = dot(AI[2], IzAzBz);

    // clip chroma
    float max_C = JC[1];
    if(LMS.x < 0.f)
      max_C = min(-Iz / (AI[0].y * cos_H + AI[0].z * sin_H), max_C);

    if(LMS.y < 0.f)
      max_C = min(-Iz / (AI[1].y * cos_H + AI[1].z * sin_H), max_C);

    if(LMS.z < 0.f)
      max_C = min(-Iz / (AI[2].y * cos_H + AI[2].z * sin_H), max_C);

    // project back to JzAzBz for real
    Jab.x = JC[0];
    Jab.y = max_C * cos_H;
    Jab.z = max_C * sin_H;

    XYZ_D65 = JzAzBz_2_XYZ(Jab);
  }
  else
  {
    float4 xyY = dt_D65_XYZ_to_xyY(XYZ_D65);
    float4 JCH = xyY_to_dt_UCS_JCH(xyY, L_white);
    float4 HCB = dt_UCS_JCH_to_HCB(JCH);

    const float radius = dt_hypot(HCB.y, HCB.z);
    const float sin_T = (radius > 0.f) ? HCB.y / radius : 0.f;
    const float cos_T = (radius > 0.f) ? HCB.z / radius : 0.f;
    const float M_rot_inv[2][2] = { { cos_T, sin_T }, { -sin_T, cos_T } };

    const float P = max(1.175494350822287507969e-38f, HCB.y);
    const float W = sin_T * HCB.y + cos_T * HCB.z;

    float a = max(1.f + saturation_global + dot(opacities, saturation4), 0.f);
    const float b = max(1.f + brilliance_global + dot(opacities, brilliance4), 0.f);

    const float max_a = dt_hypot(P, W) / P;
    a = soft_clip(a, 0.5f * max_a, max_a);

    const float P_prime = (a - 1.f) * P;
    const float W_prime = dtcl_sqrt(fsquare(P) * (1.f - fsquare(a)) + fsquare(W)) * b;

    HCB.y = max(M_rot_inv[0][0] * P_prime + M_rot_inv[0][1] * W_prime, 0.f);
    HCB.z = max(M_rot_inv[1][0] * P_prime + M_rot_inv[1][1] * W_prime, 0.f);

    JCH = dt_UCS_HCB_to_JCH(HCB);

    // gamut mapping
    const float max_colorfulness = lookup_gamut(gamut_lut, JCH.z); // WARNING: this is M²
    const float max_chroma = 15.932993652962535f * dtcl_pow(JCH.x * L_white, 0.6523997524738018f)
                             * dtcl_pow(max_colorfulness, 0.6007557017508491f) / L_white;
    const float4 JCH_gamut_boundary = float4(JCH.x, max_chroma, JCH.z, 0.f);
    const float4 HSB_gamut_boundary = dt_UCS_JCH_to_HSB(JCH_gamut_boundary);

    // clip saturation at constant brightness
    float4 HSB = float4(HCB.x, (HCB.z > 0.f) ? HCB.y / HCB.z : 0.f, HCB.z, 0.f);
    HSB.y = soft_clip(HSB.y, 0.8f * HSB_gamut_boundary.y, HSB_gamut_boundary.y);

    JCH = dt_UCS_HSB_to_JCH(HSB);
    xyY = dt_UCS_JCH_to_xyY(JCH, L_white);
    XYZ_D65 = dt_xyY_to_XYZ(xyY);
  }

  // project back to D50 pipeline RGB
  RGB = matrix_product_float4(XYZ_D65, matrix_out);

  if(mask_display != 0)
  {
    // draw checkerboard
    float4 color;
    if(x % checker_1 < x % checker_2)
    {
      if(y % checker_1 < y % checker_2) color = checker_rgba_2;
      else color = checker_rgba_1;
    }
    else
    {
      if(y % checker_1 < y % checker_2) color = checker_rgba_1;
      else color = checker_rgba_2;
    }
    const float opacity = opacities[mask_type];
    const float opacity_comp = 1.0f - opacity;

    RGB = opacity_comp * color + opacity * max(RGB, float4(0.f));
    RGB.w = 1.0f; // alpha is opaque, we need to preview it
  }
  else
  {
    RGB = max(RGB, float4(0.f));
    RGB.w = pix_in.w; // alpha copy
  }

  out.write(int2(x, y), RGB);
}

// TEA encryption as a cheap per pixel random number
void _encrypt_tea(inout uint2 v)
{
  const uint key[4] = { 0xa341316cu, 0xc8013ea4u, 0xad90777du, 0x7e95761eu };
  const uint delta = 0x9e3779b9u;
  uint sum = 0u;
  for(int i = 0; i < 8; i++)
  {
    sum += delta;
    v.x += ((v.y << 4) + key[0]) ^ (v.y + sum) ^ ((v.y >> 5) + key[1]);
    v.y += ((v.x << 4) + key[2]) ^ (v.x + sum) ^ ((v.x >> 5) + key[3]);
  }
}

// triangular probability density function in [-1, 1]
float _tpdf(uint urandom)
{
  const float frandom = float(urandom) / float(0xFFFFFFFFu);
  return frandom < 0.5f ? (sqrt(2.0f * frandom) - 1.0f) : (1.0f - sqrt(2.0f * (1.0f - frandom)));
}

/* kernel for the vignette plugin. */
[shader("compute")]
[numthreads(16, 16, 1)]
void vignette(uint3 gid : SV_DispatchThreadID,
              uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
              uniform float2 scale, uniform float2 roi_center_scaled, uniform float2 expt,
              uniform float dscale, uniform float fscale, uniform float brightness, uniform float saturation,
              uniform float dither, uniform int unbound)
{
  const int x = gid.x;
  const int y = gid.y;

  if(x >= width || y >= height) return;

  uint2 tea_state = uint2(uint(y * width + x), 0u);
  _encrypt_tea(tea_state);

  const float2 pv = abs(float2(x, y) * scale - roi_center_scaled);

  const float cplen = dtcl_pow(dtcl_pow(pv.x, expt.x) + dtcl_pow(pv.y, expt.x), expt.y);

  float weight = 0.0f;
  float dith = 0.0f;

  if(cplen >= dscale)
  {
    weight = ((cplen - dscale) / fscale);

    dith = (weight <= 1.0f && weight >= 0.0f) ? dither * _tpdf(tea_state.x) : 0.0f;

    weight = weight >= 1.0f ? 1.0f : (weight <= 0.0f ? 0.0f : 0.5f - dtcl_cos(M_PI_F * weight) / 2.0f);
  }

  float4 pixel = in.read(int2(x, y));

  if(weight > 0.0f)
  {
    const float falloff = brightness < 0.0f ? 1.0f + (weight * brightness) : weight * brightness;

    pixel.xyz = (brightness < 0.0f ? pixel * falloff + dith : pixel + falloff + dith).xyz;

    pixel.xyz = unbound != 0 ? pixel.xyz : clamp(pixel, float4(0.0f), float4(1.0f)).xyz;

    const float mv = (pixel.x + pixel.y + pixel.z) / 3.0f;
    const float wss = weight * saturation;

    pixel.xyz = (pixel - (mv - pixel) * wss).xyz;

    pixel.xyz = unbound != 0 ? pixel.xyz : clamp(pixel, float4(0.0f), float4(1.0f)).xyz;
  }

  out.write(int2(x, y), pixel);
}
