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

// Slang ports of the mask and blend kernels in ../blendop.cl. The channel
// display kernel has no port: the Slang path leaves mask display to the CPU.

import color_conversion;

static const uint DEVELOP_BLEND_LIGHTEN = 0x02;
static const uint DEVELOP_BLEND_DARKEN = 0x03;
static const uint DEVELOP_BLEND_MULTIPLY = 0x04;
static const uint DEVELOP_BLEND_AVERAGE = 0x05;
static const uint DEVELOP_BLEND_ADD = 0x06;
static const uint DEVELOP_BLEND_SUBTRACT = 0x07;
static const uint DEVELOP_BLEND_DIFFERENCE = 0x08;
static const uint DEVELOP_BLEND_SCREEN = 0x09;
static const uint DEVELOP_BLEND_OVERLAY = 0x0A;
static const uint DEVELOP_BLEND_SOFTLIGHT = 0x0B;
static const uint DEVELOP_BLEND_HARDLIGHT = 0x0C;
static const uint DEVELOP_BLEND_VIVIDLIGHT = 0x0D;
static const uint DEVELOP_BLEND_LINEARLIGHT = 0x0E;
static const uint DEVELOP_BLEND_PINLIGHT = 0x0F;
static const uint DEVELOP_BLEND_LIGHTNESS = 0x10;
static const uint DEVELOP_BLEND_CHROMA = 0x11;
static const uint DEVELOP_BLEND_HUE = 0x12;
static const uint DEVELOP_BLEND_COLOR = 0x13;
static const uint DEVELOP_BLEND_COLORADJUST = 0x16;
static const uint DEVELOP_BLEND_DIFFERENCE2 = 0x17;
static const uint DEVELOP_BLEND_NORMAL2 = 0x18;
static const uint DEVELOP_BLEND_BOUNDED = 0x19;
static const uint DEVELOP_BLEND_LAB_LIGHTNESS = 0x1A;
static const uint DEVELOP_BLEND_LAB_COLOR = 0x1B;
static const uint DEVELOP_BLEND_HSV_LIGHTNESS = 0x1C;
static const uint DEVELOP_BLEND_HSV_COLOR = 0x1D;
static const uint DEVELOP_BLEND_LAB_L = 0x1E;
static const uint DEVELOP_BLEND_LAB_A = 0x1F;
static const uint DEVELOP_BLEND_LAB_B = 0x20;
static const uint DEVELOP_BLEND_RGB_R = 0x21;
static const uint DEVELOP_BLEND_RGB_G = 0x22;
static const uint DEVELOP_BLEND_RGB_B = 0x23;
static const uint DEVELOP_BLEND_SUBTRACT_INVERSE = 0x25;
static const uint DEVELOP_BLEND_DIVIDE = 0x26;
static const uint DEVELOP_BLEND_DIVIDE_INVERSE = 0x27;
static const uint DEVELOP_BLEND_GEOMETRIC_MEAN = 0x28;
static const uint DEVELOP_BLEND_HARMONIC_MEAN = 0x29;
static const uint DEVELOP_BLEND_REVERSE = 0x80000000;
static const uint DEVELOP_BLEND_MODE_MASK = 0xFF;

static const uint DEVELOP_MASK_CONDITIONAL = 0x04;

static const uint DEVELOP_COMBINE_INV = 0x01;
static const uint DEVELOP_COMBINE_INCL = 0x02;

static const int DEVELOP_BLENDIF_L_in = 0;
static const int DEVELOP_BLENDIF_A_in = 1;
static const int DEVELOP_BLENDIF_B_in = 2;
static const int DEVELOP_BLENDIF_L_out = 4;
static const int DEVELOP_BLENDIF_A_out = 5;
static const int DEVELOP_BLENDIF_B_out = 6;
static const int DEVELOP_BLENDIF_GRAY_in = 0;
static const int DEVELOP_BLENDIF_RED_in = 1;
static const int DEVELOP_BLENDIF_GREEN_in = 2;
static const int DEVELOP_BLENDIF_BLUE_in = 3;
static const int DEVELOP_BLENDIF_GRAY_out = 4;
static const int DEVELOP_BLENDIF_RED_out = 5;
static const int DEVELOP_BLENDIF_GREEN_out = 6;
static const int DEVELOP_BLENDIF_BLUE_out = 7;
static const int DEVELOP_BLENDIF_C_in = 8;
static const int DEVELOP_BLENDIF_h_in = 9;
static const int DEVELOP_BLENDIF_C_out = 12;
static const int DEVELOP_BLENDIF_h_out = 13;
static const int DEVELOP_BLENDIF_H_in = 8;
static const int DEVELOP_BLENDIF_S_in = 9;
static const int DEVELOP_BLENDIF_l_in = 10;
static const int DEVELOP_BLENDIF_H_out = 12;
static const int DEVELOP_BLENDIF_S_out = 13;
static const int DEVELOP_BLENDIF_l_out = 14;
static const int DEVELOP_BLENDIF_Jz_in = 8;
static const int DEVELOP_BLENDIF_Cz_in = 9;
static const int DEVELOP_BLENDIF_hz_in = 10;
static const int DEVELOP_BLENDIF_Jz_out = 12;
static const int DEVELOP_BLENDIF_Cz_out = 13;
static const int DEVELOP_BLENDIF_hz_out = 14;
static const int DEVELOP_BLENDIF_MAX = 14;
static const int DEVELOP_BLENDIF_SIZE = 16;
static const uint DEVELOP_BLENDIF_Lab_MASK = 0x3377;
static const uint DEVELOP_BLENDIF_RGB_MASK = 0x77FF;

float4 RGB_2_HSL(float4 RGB)
{
  float H = 0.0f;
  float S = 0.0f;

  // assumes that each channel is scaled to [0; 1]
  const float R = RGB.x;
  const float G = RGB.y;
  const float B = RGB.z;

  const float var_Min = min(R, min(G, B));
  const float var_Max = max(R, max(G, B));
  const float del_Max = var_Max - var_Min;

  const float L = (var_Max + var_Min) / 2.0f;

  if(abs(var_Max) > 1e-6f && abs(del_Max) > 1e-6f)
  {
    if(L < 0.5f) S = del_Max / (var_Max + var_Min);
    else         S = del_Max / (2.0f - var_Max - var_Min);

    const float del_R = (((var_Max - R) / 6.0f) + (del_Max / 2.0f)) / del_Max;
    const float del_G = (((var_Max - G) / 6.0f) + (del_Max / 2.0f)) / del_Max;
    const float del_B = (((var_Max - B) / 6.0f) + (del_Max / 2.0f)) / del_Max;

    if(R == var_Max)      H = del_B - del_G;
    else if(G == var_Max) H = (1.0f / 3.0f) + del_R - del_B;
    else if(B == var_Max) H = (2.0f / 3.0f) + del_G - del_R;

    H -= floor(H);
  }

  return float4(H, S, L, RGB.w);
}

float Hue_2_RGB(float v1, float v2, float vH)
{
  vH = vH - floor(vH);
  if((6.0f * vH) < 1.0f) return (v1 + (v2 - v1) * 6.0f * vH);
  if((2.0f * vH) < 1.0f) return (v2);
  if((3.0f * vH) < 2.0f) return (v1 + (v2 - v1) * ((2.0f / 3.0f) - vH) * 6.0f);
  return (v1);
}

float4 HSL_2_RGB(float4 HSL)
{
  float R, G, B;

  const float H = HSL.x;
  const float S = HSL.y;
  const float L = HSL.z;

  if(S < 1e-6f)
  {
    R = B = G = L;
  }
  else
  {
    const float var_2 = L < 0.5f ? L * (1.0f + S) : (L + S) - (S * L);
    const float var_1 = 2.0f * L - var_2;

    R = Hue_2_RGB(var_1, var_2, H + (1.0f / 3.0f));
    G = Hue_2_RGB(var_1, var_2, H);
    B = Hue_2_RGB(var_1, var_2, H - (1.0f / 3.0f));
  }

  // returns RGB scaled to [0; 1] for each channel
  return float4(R, G, B, HSL.w);
}

float4 JzAzBz_to_JzCzhz(float4 JzAzBz)
{
  const float h = atan2(JzAzBz.z, JzAzBz.y) / DT_2PI_F;
  float4 JzCzhz;
  JzCzhz.x = JzAzBz.x;
  JzCzhz.y = dt_fast_hypot(JzAzBz.y, JzAzBz.z);
  JzCzhz.z = (h >= 0.0f) ? h : 1.0f + h;
  JzCzhz.w = JzAzBz.w;
  return JzCzhz;
}

// the product of the per-channel factors of the parametric mask, shared by
// the blendif_factor_* functions of the OpenCL code
float _blendif_combine(float scaled[DEVELOP_BLENDIF_SIZE], uint blendif, StructuredBuffer<float> parameters,
                       uint mask_combine, uint cs_mask)
{
  const uint invert_mask = (blendif >> 16) ^ ((mask_combine & DEVELOP_COMBINE_INCL) != 0 ? cs_mask : 0);

  float result = 1.0f;
  for(int ch = 0; ch <= DEVELOP_BLENDIF_MAX; ch++)
  {
    if((cs_mask & (1u << ch)) == 0) continue; // skip blendif channels not used in this color space

    float factor;
    if((blendif & (1u << ch)) == 0) // deal with channels where sliders span the whole range
      factor = 1.0f;
    else if(result <= 0.000001f)
      break; // no need to continue if we are already close to or at zero
    else if(scaled[ch] <= parameters[6 * ch + 0])
      factor = 0.0f;
    else if(scaled[ch] < parameters[6 * ch + 1])
      factor = (scaled[ch] - parameters[6 * ch + 0]) * parameters[6 * ch + 4];
    else if(scaled[ch] <= parameters[6 * ch + 2])
      factor = 1.0f;
    else if(scaled[ch] < parameters[6 * ch + 3])
      factor = 1.0f - (scaled[ch] - parameters[6 * ch + 2]) * parameters[6 * ch + 5];
    else
      factor = 0.0f;
    result *= (invert_mask & (1u << ch)) != 0 ? 1.0f - factor : factor; // inverted channel?
  }

  return (mask_combine & DEVELOP_COMBINE_INCL) != 0 ? 1.0f - result : result;
}

float blendif_factor_Lab(float4 input, float4 output, uint blendif, StructuredBuffer<float> parameters,
                         uint mask_mode, uint mask_combine)
{
  if((mask_mode & DEVELOP_MASK_CONDITIONAL) == 0) return (mask_combine & DEVELOP_COMBINE_INCL) != 0 ? 0.0f : 1.0f;

  float scaled[DEVELOP_BLENDIF_SIZE];
  for(int k = 0; k < DEVELOP_BLENDIF_SIZE; k++) scaled[k] = 0.0f;

  scaled[DEVELOP_BLENDIF_L_in] = input.x / 100.0f;  // L scaled to 0..1
  scaled[DEVELOP_BLENDIF_A_in] = input.y / 256.0f;  // a scaled to -0.5..0.5
  scaled[DEVELOP_BLENDIF_B_in] = input.z / 256.0f;  // b scaled to -0.5..0.5

  scaled[DEVELOP_BLENDIF_L_out] = output.x / 100.0f; // L scaled to 0..1
  scaled[DEVELOP_BLENDIF_A_out] = output.y / 256.0f; // a scaled to -0.5..0.5
  scaled[DEVELOP_BLENDIF_B_out] = output.z / 256.0f; // b scaled to -0.5..0.5

  if((blendif & 0x7f00) != 0) // do we need to consider LCh ?
  {
    const float4 LCH_input = Lab_2_LCH(input);
    const float4 LCH_output = Lab_2_LCH(output);

    scaled[DEVELOP_BLENDIF_C_in] = LCH_input.y / (128.0f * M_SQRT2_F);  // C scaled to 0..1
    scaled[DEVELOP_BLENDIF_h_in] = LCH_input.z;                         // h scaled to 0..1

    scaled[DEVELOP_BLENDIF_C_out] = LCH_output.y / (128.0f * M_SQRT2_F); // C scaled to 0..1
    scaled[DEVELOP_BLENDIF_h_out] = LCH_output.z;                        // h scaled to 0..1
  }

  return _blendif_combine(scaled, blendif, parameters, mask_combine, DEVELOP_BLENDIF_Lab_MASK);
}

float blendif_factor_rgb_hsl(float4 input, float4 output, uint blendif, StructuredBuffer<float> parameters,
                             uint mask_mode, uint mask_combine, ProfileInfo profile_info, ImageIn profile_lut,
                             int use_work_profile)
{
  if((mask_mode & DEVELOP_MASK_CONDITIONAL) == 0) return (mask_combine & DEVELOP_COMBINE_INCL) != 0 ? 0.0f : 1.0f;

  float scaled[DEVELOP_BLENDIF_SIZE];
  for(int k = 0; k < DEVELOP_BLENDIF_SIZE; k++) scaled[k] = 0.0f;

  if(use_work_profile == 0)
  {
    scaled[DEVELOP_BLENDIF_GRAY_in] = 0.3f * input.x + 0.59f * input.y + 0.11f * input.z;    // Gray scaled to 0..1
    scaled[DEVELOP_BLENDIF_GRAY_out] = 0.3f * output.x + 0.59f * output.y + 0.11f * output.z; // Gray scaled to 0..1
  }
  else
  {
    scaled[DEVELOP_BLENDIF_GRAY_in] = get_rgb_matrix_luminance(input, profile_info, profile_info.matrix_in, profile_lut);
    scaled[DEVELOP_BLENDIF_GRAY_out] = get_rgb_matrix_luminance(output, profile_info, profile_info.matrix_in, profile_lut);
  }

  scaled[DEVELOP_BLENDIF_RED_in] = input.x;
  scaled[DEVELOP_BLENDIF_GREEN_in] = input.y;
  scaled[DEVELOP_BLENDIF_BLUE_in] = input.z;

  scaled[DEVELOP_BLENDIF_RED_out] = output.x;
  scaled[DEVELOP_BLENDIF_GREEN_out] = output.y;
  scaled[DEVELOP_BLENDIF_BLUE_out] = output.z;

  if((blendif & 0x7f00) != 0) // do we need to consider HSL ?
  {
    const float4 HSL_input = RGB_2_HSL(input);
    const float4 HSL_output = RGB_2_HSL(output);

    scaled[DEVELOP_BLENDIF_H_in] = clipf(HSL_input.x); // H scaled to 0..1
    scaled[DEVELOP_BLENDIF_S_in] = clipf(HSL_input.y); // S scaled to 0..1
    scaled[DEVELOP_BLENDIF_l_in] = clipf(HSL_input.z); // L scaled to 0..1

    scaled[DEVELOP_BLENDIF_H_out] = clipf(HSL_output.x); // H scaled to 0..1
    scaled[DEVELOP_BLENDIF_S_out] = clipf(HSL_output.y); // S scaled to 0..1
    scaled[DEVELOP_BLENDIF_l_out] = clipf(HSL_output.z); // L scaled to 0..1
  }

  return _blendif_combine(scaled, blendif, parameters, mask_combine, DEVELOP_BLENDIF_RGB_MASK);
}

float4 rgb_to_JzCzhz(float4 rgb, ProfileInfo profile_info, ImageIn profile_lut)
{
  // use the matrix_out of the hacked profile for blending to use the
  // conversion from RGB to XYZ D65 (instead of XYZ D50)
  const float4 xyz = rgb_matrix_to_xyz(rgb, profile_info, profile_info.matrix_out, profile_lut);
  return JzAzBz_to_JzCzhz(XYZ_to_JzAzBz(xyz));
}

float blendif_factor_rgb_jzczhz(float4 input, float4 output, uint blendif, StructuredBuffer<float> parameters,
                                uint mask_mode, uint mask_combine, ProfileInfo profile_info,
                                ImageIn profile_lut)
{
  if((mask_mode & DEVELOP_MASK_CONDITIONAL) == 0) return (mask_combine & DEVELOP_COMBINE_INCL) != 0 ? 0.0f : 1.0f;

  float scaled[DEVELOP_BLENDIF_SIZE];
  for(int k = 0; k < DEVELOP_BLENDIF_SIZE; k++) scaled[k] = 0.0f;

  scaled[DEVELOP_BLENDIF_GRAY_in] = get_rgb_matrix_luminance(input, profile_info, profile_info.matrix_in, profile_lut);
  scaled[DEVELOP_BLENDIF_GRAY_out] = get_rgb_matrix_luminance(output, profile_info, profile_info.matrix_in, profile_lut);

  scaled[DEVELOP_BLENDIF_RED_in] = input.x;
  scaled[DEVELOP_BLENDIF_GREEN_in] = input.y;
  scaled[DEVELOP_BLENDIF_BLUE_in] = input.z;

  scaled[DEVELOP_BLENDIF_RED_out] = output.x;
  scaled[DEVELOP_BLENDIF_GREEN_out] = output.y;
  scaled[DEVELOP_BLENDIF_BLUE_out] = output.z;

  if((blendif & 0x7f00) != 0) // do we need to consider JzCzhz ?
  {
    const float4 JzCzhz_input = rgb_to_JzCzhz(input, profile_info, profile_lut);
    const float4 JzCzhz_output = rgb_to_JzCzhz(output, profile_info, profile_lut);

    scaled[DEVELOP_BLENDIF_Jz_in] = JzCzhz_input.x;
    scaled[DEVELOP_BLENDIF_Cz_in] = JzCzhz_input.y;
    scaled[DEVELOP_BLENDIF_hz_in] = JzCzhz_input.z;

    scaled[DEVELOP_BLENDIF_Jz_out] = JzCzhz_output.x;
    scaled[DEVELOP_BLENDIF_Cz_out] = JzCzhz_output.y;
    scaled[DEVELOP_BLENDIF_hz_out] = JzCzhz_output.z;
  }

  return _blendif_combine(scaled, blendif, parameters, mask_combine, DEVELOP_BLENDIF_RGB_MASK);
}

float _combine_form(float form, float conditional, uint mask_combine, float gopacity)
{
  float opacity = (mask_combine & DEVELOP_COMBINE_INCL) != 0
    ? 1.0f - (1.0f - form) * (1.0f - conditional)
    : form * conditional;
  opacity = (mask_combine & DEVELOP_COMBINE_INV) != 0 ? 1.0f - opacity : opacity;
  return gopacity * opacity;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void blendop_mask_Lab(uint3 gid : SV_DispatchThreadID,
                      uniform ImageIn in_a, uniform ImageIn in_b, uniform ImageIn mask_in, uniform ImageOut mask,
                      uniform int width, uniform int height,
                      uniform float gopacity, uniform int blendif, uniform StructuredBuffer<float> blendif_parameters,
                      uniform uint mask_mode, uniform uint mask_combine, uniform int2 offs,
                      uniform StructuredBuffer<ProfileInfo> profile_info, uniform ImageIn profile_lut,
                      uniform int use_work_profile)
{
  const int x = gid.x;
  const int y = gid.y;
  // Lab masks take no profile, as in OpenCL, but Metal drops parameters nothing reads
  if(width < 0)
    mask.write(int2(0), float4(profile_info[0].grey + profile_lut.Areadsingle(0, 0) + float(use_work_profile)));
  if(x >= width || y >= height) return;

  const float4 a = in_a.readpixel(x + offs.x, y + offs.y);
  const float4 b = in_b.Areadpixel(x, y);
  const float form = mask_in.Areadsingle(x, y);

  const float conditional = blendif_factor_Lab(a, b, uint(blendif), blendif_parameters, mask_mode, mask_combine);
  mask.write(int2(x, y), float4(_combine_form(form, conditional, mask_combine, gopacity)));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void blendop_mask_RAW(uint3 gid : SV_DispatchThreadID,
                      uniform ImageIn in_a, uniform ImageIn in_b, uniform ImageIn mask_in, uniform ImageOut mask,
                      uniform int width, uniform int height,
                      uniform float gopacity, uniform int blendif, uniform StructuredBuffer<float> blendif_parameters,
                      uniform uint mask_mode, uniform uint mask_combine, uniform int2 offs,
                      uniform StructuredBuffer<ProfileInfo> profile_info, uniform ImageIn profile_lut,
                      uniform int use_work_profile)
{
  const int x = gid.x;
  const int y = gid.y;
  // raw masks have no parametric part, as in OpenCL, but Metal drops parameters nothing reads
  if(width < 0)
    mask.write(int2(0), float4(in_a.Areadsingle(0, 0) + in_b.Areadsingle(0, 0) + float(blendif + offs.x + offs.y)
                               + blendif_parameters[0] + float(mask_mode) + profile_info[0].grey
                               + profile_lut.Areadsingle(0, 0) + float(use_work_profile)));
  if(x >= width || y >= height) return;

  const float form = mask_in.Areadsingle(x, y);
  const float opacity = (mask_combine & DEVELOP_COMBINE_INV) != 0 ? 1.0f - form : form;
  mask.write(int2(x, y), float4(gopacity * opacity));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void blendop_mask_rgb_hsl(uint3 gid : SV_DispatchThreadID,
                          uniform ImageIn in_a, uniform ImageIn in_b, uniform ImageIn mask_in, uniform ImageOut mask,
                          uniform int width, uniform int height,
                          uniform float gopacity, uniform int blendif, uniform StructuredBuffer<float> blendif_parameters,
                          uniform uint mask_mode, uniform uint mask_combine, uniform int2 offs,
                          uniform StructuredBuffer<ProfileInfo> profile_info, uniform ImageIn profile_lut,
                          uniform int use_work_profile)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float4 a = in_a.readpixel(x + offs.x, y + offs.y);
  const float4 b = in_b.Areadpixel(x, y);
  const float form = mask_in.Areadsingle(x, y);

  const float conditional = blendif_factor_rgb_hsl(a, b, uint(blendif), blendif_parameters, mask_mode, mask_combine,
                                                   profile_info[0], profile_lut, use_work_profile);
  mask.write(int2(x, y), float4(_combine_form(form, conditional, mask_combine, gopacity)));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void blendop_mask_rgb_jzczhz(uint3 gid : SV_DispatchThreadID,
                             uniform ImageIn in_a, uniform ImageIn in_b, uniform ImageIn mask_in, uniform ImageOut mask,
                             uniform int width, uniform int height,
                             uniform float gopacity, uniform int blendif, uniform StructuredBuffer<float> blendif_parameters,
                             uniform uint mask_mode, uniform uint mask_combine, uniform int2 offs,
                             uniform StructuredBuffer<ProfileInfo> profile_info, uniform ImageIn profile_lut,
                             uniform int use_work_profile)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height || use_work_profile == 0) return;

  const float4 a = in_a.readpixel(x + offs.x, y + offs.y);
  const float4 b = in_b.Areadpixel(x, y);
  const float form = mask_in.Areadsingle(x, y);

  const float conditional = blendif_factor_rgb_jzczhz(a, b, uint(blendif), blendif_parameters, mask_mode,
                                                      mask_combine, profile_info[0], profile_lut);
  mask.write(int2(x, y), float4(_combine_form(form, conditional, mask_combine, gopacity)));
}

// a and b of the Lab blend modes that scale the chroma along with L
float2 _lab_ab_follow_L(float4 a, float4 b, float4 o, float opacity, float weight, float4 vmin, float4 vmax)
{
  const float ratio = o.x / (a.x > 0.01f ? a.x : 0.01f);
  return float2(clamp(a.y * (1.0f - opacity) + weight * (a.y + b.y) * ratio * opacity, vmin.y, vmax.y),
                clamp(a.z * (1.0f - opacity) + weight * (a.z + b.z) * ratio * opacity, vmin.z, vmax.z));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void blendop_Lab(uint3 gid : SV_DispatchThreadID,
                 uniform ImageIn in_a, uniform ImageIn in_b, uniform ImageIn mask, uniform ImageOut out,
                 uniform int width, uniform int height,
                 uniform int blend_mode, uniform float blend_parameter, uniform int2 offs, uniform int mask_display)
{
  const int x = gid.x;
  const int y = gid.y;
  // blend_parameter is unused, as in OpenCL, but Metal drops parameters nothing reads
  if(width < 0) out.write(int2(0), float4(blend_parameter));
  if(x >= width || y >= height) return;

  float4 a, b, o;
  float4 ta, tb, to;
  float d, s;

  const uint mode = uint(blend_mode);
  if((mode & DEVELOP_BLEND_REVERSE) == DEVELOP_BLEND_REVERSE)
  {
    b = in_a.readpixel(x + offs.x, y + offs.y);
    a = in_b.Areadpixel(x, y);
  }
  else
  {
    a = in_a.readpixel(x + offs.x, y + offs.y);
    b = in_b.Areadpixel(x, y);
  }
  const float opacity = mask.Areadsingle(x, y);

  // scale L down to [0; 1] and a,b to [-1; 1]
  const float4 scale = float4(100.0f, 128.0f, 128.0f, 1.0f);
  a /= scale;
  b /= scale;

  const float4 vmin = float4(0.0f, -1.0f, -1.0f, 0.0f);
  const float4 vmax = float4(1.0f, 1.0f, 1.0f, 1.0f);
  const float4 lmin = float4(0.0f, 0.0f, 0.0f, 1.0f);
  const float4 lmax = float4(1.0f, 2.0f, 2.0f, 1.0f);      // vmax + abs(vmin)
  const float4 halfmax = float4(0.5f, 1.0f, 1.0f, 0.5f);   // lmax / 2.0f
  const float4 doublemax = float4(2.0f, 4.0f, 4.0f, 2.0f); // lmax * 2.0f
  const float opacity2 = fsquare(opacity);

  const float4 la = clamp(a + abs(vmin), lmin, lmax);
  const float4 lb = clamp(b + abs(vmin), lmin, lmax);

  // select the blend operator
  switch(mode & DEVELOP_BLEND_MODE_MASK)
  {
    case DEVELOP_BLEND_LIGHTEN:
      o = clamp(a * (1.0f - opacity) + select(a > b, a, b) * opacity, vmin, vmax);
      o.y = clamp(a.y * (1.0f - abs(o.x - a.x)) + 0.5f * (a.y + b.y) * abs(o.x - a.x), vmin.y, vmax.y);
      o.z = clamp(a.z * (1.0f - abs(o.x - a.x)) + 0.5f * (a.z + b.z) * abs(o.x - a.x), vmin.z, vmax.z);
      break;

    case DEVELOP_BLEND_DARKEN:
      o = clamp(a * (1.0f - opacity) + select(a < b, a, b) * opacity, vmin, vmax);
      o.y = clamp(a.y * (1.0f - abs(o.x - a.x)) + 0.5f * (a.y + b.y) * abs(o.x - a.x), vmin.y, vmax.y);
      o.z = clamp(a.z * (1.0f - abs(o.x - a.x)) + 0.5f * (a.z + b.z) * abs(o.x - a.x), vmin.z, vmax.z);
      break;

    case DEVELOP_BLEND_MULTIPLY:
      o = clamp(a * (1.0f - opacity) + a * b * opacity, vmin, vmax);
      o.yz = _lab_ab_follow_L(a, b, o, opacity, 1.0f, vmin, vmax);
      break;

    case DEVELOP_BLEND_AVERAGE:
      o = clamp(a * (1.0f - opacity) + (a + b) / 2.0f * opacity, vmin, vmax);
      break;

    case DEVELOP_BLEND_ADD:
      o = clamp(a * (1.0f - opacity) + (a + b) * opacity, vmin, vmax);
      break;

    case DEVELOP_BLEND_SUBTRACT:
      o = clamp(a * (1.0f - opacity) + (b + a - abs(vmin + vmax)) * opacity, vmin, vmax);
      break;

    case DEVELOP_BLEND_DIFFERENCE:
      o = clamp(la * (1.0f - opacity) + abs(la - lb) * opacity, lmin, lmax) - abs(vmin);
      break;

    case DEVELOP_BLEND_DIFFERENCE2:
      to = abs(a - b) / abs(vmax - vmin);
      to.x = max(to.x, max(to.y, to.z));
      o = clamp(la * (1.0f - opacity) + to * opacity, lmin, lmax);
      o.y = o.z = 0.0f;
      break;

    case DEVELOP_BLEND_SCREEN:
      o = clamp(la * (1.0f - opacity) + (lmax - (lmax - la) * (lmax - lb)) * opacity, lmin, lmax) - abs(vmin);
      o.yz = _lab_ab_follow_L(a, b, o, opacity, 0.5f, vmin, vmax);
      break;

    case DEVELOP_BLEND_OVERLAY:
      o = clamp(la * (1.0f - opacity2)
                + select(la > halfmax, lmax - (lmax - doublemax * (la - halfmax)) * (lmax - lb), doublemax * la * lb)
                  * opacity2, lmin, lmax) - abs(vmin);
      o.yz = _lab_ab_follow_L(a, b, o, opacity2, 1.0f, vmin, vmax);
      break;

    case DEVELOP_BLEND_SOFTLIGHT:
      o = clamp(la * (1.0f - opacity2)
                + select(lb > halfmax, lmax - (lmax - la) * (lmax - (lb - halfmax)), la * (lb + halfmax))
                  * opacity2, lmin, lmax) - abs(vmin);
      o.yz = _lab_ab_follow_L(a, b, o, opacity2, 1.0f, vmin, vmax);
      break;

    case DEVELOP_BLEND_HARDLIGHT:
      o = clamp(la * (1.0f - opacity2)
                + select(lb > halfmax, lmax - (lmax - doublemax * (la - halfmax)) * (lmax - lb), doublemax * la * lb)
                  * opacity2, lmin, lmax) - abs(vmin);
      o.yz = _lab_ab_follow_L(a, b, o, opacity2, 1.0f, vmin, vmax);
      break;

    case DEVELOP_BLEND_VIVIDLIGHT:
      o = clamp(la * (1.0f - opacity2)
                + select(lb > halfmax,
                         select(lb >= lmax, lmax, la / (doublemax * (lmax - lb))),
                         select(lb <= lmin, lmin, lmax - (lmax - la) / (doublemax * lb)))
                  * opacity2, lmin, lmax) - abs(vmin);
      o.yz = _lab_ab_follow_L(a, b, o, opacity2, 1.0f, vmin, vmax);
      break;

    case DEVELOP_BLEND_LINEARLIGHT:
      o = clamp(la * (1.0f - opacity2) + (la + doublemax * lb - lmax) * opacity2, lmin, lmax) - abs(vmin);
      o.yz = _lab_ab_follow_L(a, b, o, opacity2, 1.0f, vmin, vmax);
      break;

    case DEVELOP_BLEND_PINLIGHT:
      o = clamp(la * (1.0f - opacity2)
                + select(lb > halfmax, max(la, doublemax * (lb - halfmax)), min(la, doublemax * lb)) * opacity2,
                lmin, lmax) - abs(vmin);
      o.y = clamp(a.y, vmin.y, vmax.y);
      o.z = clamp(a.z, vmin.z, vmax.z);
      break;

    case DEVELOP_BLEND_LIGHTNESS:
      // no need to transfer to LCH as we only work on L, which is the same as in Lab
      o.x = clamp((a.x * (1.0f - opacity)) + (b.x * opacity), vmin.x, vmax.x);
      o.y = clamp(a.y, vmin.y, vmax.y);
      o.z = clamp(a.z, vmin.z, vmax.z);
      break;

    case DEVELOP_BLEND_CHROMA:
      ta = Lab_2_LCH(clamp(a, vmin, vmax));
      tb = Lab_2_LCH(clamp(b, vmin, vmax));
      to.x = ta.x;
      to.y = (ta.y * (1.0f - opacity)) + (tb.y * opacity);
      to.z = ta.z;
      to.w = ta.w;
      o = clamp(LCH_2_Lab(to), vmin, vmax);
      break;

    case DEVELOP_BLEND_HUE:
      ta = Lab_2_LCH(clamp(a, vmin, vmax));
      tb = Lab_2_LCH(clamp(b, vmin, vmax));
      to.x = ta.x;
      to.y = ta.y;
      d = abs(ta.z - tb.z);
      s = d > 0.5f ? -opacity * (1.0f - d) / d : opacity;
      to.z = fmod((ta.z * (1.0f - s)) + (tb.z * s) + 1.0f, 1.0f);
      to.w = ta.w;
      o = clamp(LCH_2_Lab(to), vmin, vmax);
      break;

    case DEVELOP_BLEND_COLOR:
      ta = Lab_2_LCH(clamp(a, vmin, vmax));
      tb = Lab_2_LCH(clamp(b, vmin, vmax));
      to.x = ta.x;
      to.y = (ta.y * (1.0f - opacity)) + (tb.y * opacity);
      d = abs(ta.z - tb.z);
      s = d > 0.5f ? -opacity * (1.0f - d) / d : opacity;
      to.z = fmod((ta.z * (1.0f - s)) + (tb.z * s) + 1.0f, 1.0f);
      to.w = ta.w;
      o = clamp(LCH_2_Lab(to), vmin, vmax);
      break;

    case DEVELOP_BLEND_COLORADJUST:
      ta = Lab_2_LCH(clamp(a, vmin, vmax));
      tb = Lab_2_LCH(clamp(b, vmin, vmax));
      to.x = tb.x;
      to.y = (ta.y * (1.0f - opacity)) + (tb.y * opacity);
      d = abs(ta.z - tb.z);
      s = d > 0.5f ? -opacity * (1.0f - d) / d : opacity;
      to.z = fmod((ta.z * (1.0f - s)) + (tb.z * s) + 1.0f, 1.0f);
      to.w = ta.w;
      o = clamp(LCH_2_Lab(to), vmin, vmax);
      break;

    case DEVELOP_BLEND_BOUNDED:
      o = clamp((a * (1.0f - opacity)) + (b * opacity), vmin, vmax);
      break;

    case DEVELOP_BLEND_LAB_LIGHTNESS:
    case DEVELOP_BLEND_LAB_L:
      o.x = (a.x * (1.0f - opacity)) + (b.x * opacity);
      o.y = a.y;
      o.z = a.z;
      break;

    case DEVELOP_BLEND_LAB_A:
      o.x = a.x;
      o.y = (a.y * (1.0f - opacity)) + (b.y * opacity);
      o.z = a.z;
      break;

    case DEVELOP_BLEND_LAB_B:
      o.x = a.x;
      o.y = a.y;
      o.z = (a.z * (1.0f - opacity)) + (b.z * opacity);
      break;

    case DEVELOP_BLEND_LAB_COLOR:
      o.x = a.x;
      o.y = (a.y * (1.0f - opacity)) + (b.y * opacity);
      o.z = (a.z * (1.0f - opacity)) + (b.z * opacity);
      break;

    case DEVELOP_BLEND_HSV_LIGHTNESS:
    case DEVELOP_BLEND_HSV_COLOR:
    case DEVELOP_BLEND_RGB_R:
    case DEVELOP_BLEND_RGB_G:
    case DEVELOP_BLEND_RGB_B:
      o = a; // noop for Lab (without clamping)
      break;

    // fallback to normal blend
    case DEVELOP_BLEND_NORMAL2:
    default:
      o = (a * (1.0f - opacity)) + (b * opacity);
      break;
  }

  // scale L back to [0; 100] and a,b to [-128; 128]
  o *= scale;

  // we transfer alpha channel of input if mask_display is set, else we save opacity into alpha channel
  o.w = mask_display != 0 ? a.w : opacity;

  out.write(int2(x, y), o);
}

// the raw blend modes, for one channel or four: the opacity is the same for
// each channel
vector<float, N> _blend_raw<let N : int>(vector<float, N> a, vector<float, N> b, float opacity, uint mode)
{
  const vector<float, N> vmin = vector<float, N>(0.0f);
  const vector<float, N> vmax = vector<float, N>(1.0f);
  const vector<float, N> lmin = vector<float, N>(0.0f);
  const vector<float, N> lmax = vector<float, N>(1.0f);      // vmax + abs(vmin)
  const vector<float, N> halfmax = vector<float, N>(0.5f);   // lmax / 2.0f
  const vector<float, N> doublemax = vector<float, N>(2.0f); // lmax * 2.0f
  const float opacity2 = opacity * opacity;

  const vector<float, N> la = clamp(a + abs(vmin), lmin, lmax);
  const vector<float, N> lb = clamp(b + abs(vmin), lmin, lmax);

  vector<float, N> o;
  switch(mode & DEVELOP_BLEND_MODE_MASK)
  {
    case DEVELOP_BLEND_LIGHTEN:
      o = clamp(a * (1.0f - opacity) + max(a, b) * opacity, vmin, vmax);
      break;

    case DEVELOP_BLEND_DARKEN:
      o = clamp(a * (1.0f - opacity) + min(a, b) * opacity, vmin, vmax);
      break;

    case DEVELOP_BLEND_MULTIPLY:
      o = clamp(a * (1.0f - opacity) + a * b * opacity, vmin, vmax);
      break;

    case DEVELOP_BLEND_AVERAGE:
      o = clamp(a * (1.0f - opacity) + (a + b) / 2.0f * opacity, vmin, vmax);
      break;

    case DEVELOP_BLEND_ADD:
      o = clamp(a * (1.0f - opacity) + (a + b) * opacity, vmin, vmax);
      break;

    case DEVELOP_BLEND_SUBTRACT:
      o = clamp(a * (1.0f - opacity) + (b + a - abs(vmin + vmax)) * opacity, vmin, vmax);
      break;

    case DEVELOP_BLEND_DIFFERENCE:
    case DEVELOP_BLEND_DIFFERENCE2:
      o = clamp(la * (1.0f - opacity) + abs(la - lb) * opacity, lmin, lmax) - abs(vmin);
      break;

    case DEVELOP_BLEND_SCREEN:
      o = clamp(la * (1.0f - opacity) + (lmax - (lmax - la) * (lmax - lb)) * opacity, lmin, lmax) - abs(vmin);
      break;

    case DEVELOP_BLEND_OVERLAY:
      o = clamp(la * (1.0f - opacity2)
                + select(la > halfmax, lmax - (lmax - doublemax * (la - halfmax)) * (lmax - lb), doublemax * la * lb)
                  * opacity2, lmin, lmax) - abs(vmin);
      break;

    case DEVELOP_BLEND_SOFTLIGHT:
      o = clamp(la * (1.0f - opacity2)
                + select(lb > halfmax, lmax - (lmax - la) * (lmax - (lb - halfmax)), la * (lb + halfmax))
                  * opacity2, lmin, lmax) - abs(vmin);
      break;

    case DEVELOP_BLEND_HARDLIGHT:
      o = clamp(la * (1.0f - opacity2)
                + select(lb > halfmax, lmax - (lmax - doublemax * (la - halfmax)) * (lmax - lb), doublemax * la * lb)
                  * opacity2, lmin, lmax) - abs(vmin);
      break;

    case DEVELOP_BLEND_VIVIDLIGHT:
      // the raw kernels divide where the RGB ones clip, as in OpenCL
      o = clamp(la * (1.0f - opacity2)
                + select(lb > halfmax,
                         la / select(lb >= lmax, lmax, doublemax * (lmax - lb)),
                         select(lb <= lmin, lmin, lmax - (lmax - la) / (doublemax * lb)))
                  * opacity2, lmin, lmax) - abs(vmin);
      break;

    case DEVELOP_BLEND_LINEARLIGHT:
      o = clamp(la * (1.0f - opacity2) + (la + doublemax * lb - lmax) * opacity2, lmin, lmax) - abs(vmin);
      break;

    case DEVELOP_BLEND_PINLIGHT:
      o = clamp(la * (1.0f - opacity2)
                + select(lb > halfmax, max(la, doublemax * (lb - halfmax)), min(la, doublemax * lb)) * opacity2,
                lmin, lmax) - abs(vmin);
      break;

    case DEVELOP_BLEND_LIGHTNESS:
    case DEVELOP_BLEND_CHROMA:
    case DEVELOP_BLEND_HUE:
    case DEVELOP_BLEND_COLOR:
    case DEVELOP_BLEND_COLORADJUST:
      o = clamp(a, vmin, vmax); // noop for raw
      break;

    case DEVELOP_BLEND_BOUNDED:
      o = clamp((a * (1.0f - opacity)) + (b * opacity), vmin, vmax);
      break;

    case DEVELOP_BLEND_LAB_LIGHTNESS:
    case DEVELOP_BLEND_LAB_COLOR:
    case DEVELOP_BLEND_LAB_L:
    case DEVELOP_BLEND_LAB_A:
    case DEVELOP_BLEND_LAB_B:
    case DEVELOP_BLEND_HSV_LIGHTNESS:
    case DEVELOP_BLEND_HSV_COLOR:
    case DEVELOP_BLEND_RGB_R:
    case DEVELOP_BLEND_RGB_G:
    case DEVELOP_BLEND_RGB_B:
      o = a; // noop for raw (without clamping)
      break;

    // fallback to normal blend
    case DEVELOP_BLEND_NORMAL2:
    default:
      o = (a * (1.0f - opacity)) + (b * opacity);
      break;
  }
  return o;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void blendop_RAW(uint3 gid : SV_DispatchThreadID,
                 uniform ImageIn in_a, uniform ImageIn in_b, uniform ImageIn mask, uniform ImageOut out,
                 uniform int width, uniform int height,
                 uniform int blend_mode, uniform float blend_parameter, uniform int2 offs, uniform int mask_display)
{
  const int x = gid.x;
  const int y = gid.y;
  // blend_parameter and mask_display are unused, as in OpenCL, but Metal drops parameters nothing reads
  if(width < 0) out.write(int2(0), float4(blend_parameter + float(mask_display)));
  if(x >= width || y >= height) return;

  const uint mode = uint(blend_mode);
  const float pa = in_a.readsingle(x + offs.x, y + offs.y);
  const float pb = in_b.Areadsingle(x, y);
  const bool reverse = (mode & DEVELOP_BLEND_REVERSE) == DEVELOP_BLEND_REVERSE;
  const float a = reverse ? pb : pa;
  const float b = reverse ? pa : pb;
  const float opacity = mask.Areadsingle(x, y);

  const float1 o = _blend_raw<1>(float1(a), float1(b), opacity, mode);
  out.write(int2(x, y), float4(o.x));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void blendop_RAW4(uint3 gid : SV_DispatchThreadID,
                  uniform ImageIn in_a, uniform ImageIn in_b, uniform ImageIn mask, uniform ImageOut out,
                  uniform int width, uniform int height,
                  uniform int blend_mode, uniform float blend_parameter, uniform int2 offs, uniform int mask_display)
{
  const int x = gid.x;
  const int y = gid.y;
  // blend_parameter and mask_display are unused, as in OpenCL, but Metal drops parameters nothing reads
  if(width < 0) out.write(int2(0), float4(blend_parameter + float(mask_display)));
  if(x >= width || y >= height) return;

  const uint mode = uint(blend_mode);
  const float4 pa = in_a.readpixel(x + offs.x, y + offs.y);
  const float4 pb = in_b.Areadpixel(x, y);
  const bool reverse = (mode & DEVELOP_BLEND_REVERSE) == DEVELOP_BLEND_REVERSE;
  const float4 a = reverse ? pb : pa;
  const float4 b = reverse ? pa : pb;
  const float opacity = mask.Areadsingle(x, y);

  out.write(int2(x, y), _blend_raw<4>(a, b, opacity, mode));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void blendop_rgb_hsl(uint3 gid : SV_DispatchThreadID,
                     uniform ImageIn in_a, uniform ImageIn in_b, uniform ImageIn mask, uniform ImageOut out,
                     uniform int width, uniform int height,
                     uniform int blend_mode, uniform float blend_parameter, uniform int2 offs, uniform int mask_display)
{
  const int x = gid.x;
  const int y = gid.y;
  // blend_parameter is unused, as in OpenCL, but Metal drops parameters nothing reads
  if(width < 0) out.write(int2(0), float4(blend_parameter));
  if(x >= width || y >= height) return;

  float4 a, b, o;
  float4 ta, tb, to;
  float d, s;

  const uint mode = uint(blend_mode);
  if((mode & DEVELOP_BLEND_REVERSE) == DEVELOP_BLEND_REVERSE)
  {
    b = in_a.readpixel(x + offs.x, y + offs.y);
    a = in_b.Areadpixel(x, y);
  }
  else
  {
    a = in_a.readpixel(x + offs.x, y + offs.y);
    b = in_b.Areadpixel(x, y);
  }
  const float opacity = mask.Areadsingle(x, y);

  const float4 vmin = float4(0.0f, 0.0f, 0.0f, 1.0f);
  const float4 vmax = float4(1.0f, 1.0f, 1.0f, 1.0f);
  const float4 lmin = float4(0.0f, 0.0f, 0.0f, 1.0f);
  const float4 lmax = float4(1.0f, 1.0f, 1.0f, 1.0f);      // vmax + abs(vmin)
  const float4 halfmax = float4(0.5f, 0.5f, 0.5f, 1.0f);   // lmax / 2.0f
  const float4 doublemax = float4(2.0f, 2.0f, 2.0f, 1.0f); // lmax * 2.0f
  const float opacity2 = fsquare(opacity);

  const float4 la = clamp(a + abs(vmin), lmin, lmax);
  const float4 lb = clamp(b + abs(vmin), lmin, lmax);

  // select the blend operator
  switch(mode & DEVELOP_BLEND_MODE_MASK)
  {
    case DEVELOP_BLEND_LIGHTEN:
      o = clamp(a * (1.0f - opacity) + max(a, b) * opacity, vmin, vmax);
      break;

    case DEVELOP_BLEND_DARKEN:
      o = clamp(a * (1.0f - opacity) + min(a, b) * opacity, vmin, vmax);
      break;

    case DEVELOP_BLEND_MULTIPLY:
      o = clamp(a * (1.0f - opacity) + a * b * opacity, vmin, vmax);
      break;

    case DEVELOP_BLEND_AVERAGE:
      o = clamp(a * (1.0f - opacity) + (a + b) / 2.0f * opacity, vmin, vmax);
      break;

    case DEVELOP_BLEND_ADD:
      o = clamp(a * (1.0f - opacity) + (a + b) * opacity, vmin, vmax);
      break;

    case DEVELOP_BLEND_SUBTRACT:
      o = clamp(a * (1.0f - opacity) + (b + a - abs(vmin + vmax)) * opacity, vmin, vmax);
      break;

    case DEVELOP_BLEND_DIFFERENCE:
    case DEVELOP_BLEND_DIFFERENCE2:
      o = clamp(la * (1.0f - opacity) + abs(la - lb) * opacity, lmin, lmax) - abs(vmin);
      break;

    case DEVELOP_BLEND_SCREEN:
      o = clamp(la * (1.0f - opacity) + (lmax - (lmax - la) * (lmax - lb)) * opacity, lmin, lmax) - abs(vmin);
      break;

    case DEVELOP_BLEND_OVERLAY:
      o = clamp(la * (1.0f - opacity2)
                + select(la > halfmax, lmax - (lmax - doublemax * (la - halfmax)) * (lmax - lb), doublemax * la * lb)
                  * opacity2, lmin, lmax) - abs(vmin);
      break;

    case DEVELOP_BLEND_SOFTLIGHT:
      o = clamp(la * (1.0f - opacity2)
                + select(lb > halfmax, lmax - (lmax - la) * (lmax - (lb - halfmax)), la * (lb + halfmax))
                  * opacity2, lmin, lmax) - abs(vmin);
      break;

    case DEVELOP_BLEND_HARDLIGHT:
      o = clamp(la * (1.0f - opacity2)
                + select(lb > halfmax, lmax - (lmax - doublemax * (la - halfmax)) * (lmax - lb), doublemax * la * lb)
                  * opacity2, lmin, lmax) - abs(vmin);
      break;

    case DEVELOP_BLEND_VIVIDLIGHT:
      o = clamp(la * (1.0f - opacity2)
                + select(lb > halfmax,
                         select(lb >= lmax, lmax, la / (doublemax * (lmax - lb))),
                         select(lb <= lmin, lmin, lmax - (lmax - la) / (doublemax * lb)))
                  * opacity2, lmin, lmax) - abs(vmin);
      break;

    case DEVELOP_BLEND_LINEARLIGHT:
      o = clamp(la * (1.0f - opacity2) + (la + doublemax * lb - lmax) * opacity2, lmin, lmax) - abs(vmin);
      break;

    case DEVELOP_BLEND_PINLIGHT:
      o = clamp(la * (1.0f - opacity2)
                + select(lb > halfmax, max(la, doublemax * (lb - halfmax)), min(la, doublemax * lb)) * opacity2,
                lmin, lmax) - abs(vmin);
      break;

    case DEVELOP_BLEND_LIGHTNESS:
      ta = RGB_2_HSL(clamp(a, vmin, vmax));
      tb = RGB_2_HSL(clamp(b, vmin, vmax));
      to.x = ta.x;
      to.y = ta.y;
      to.z = (ta.z * (1.0f - opacity)) + (tb.z * opacity);
      to.w = ta.w;
      o = clamp(HSL_2_RGB(to), vmin, vmax);
      break;

    case DEVELOP_BLEND_CHROMA:
      ta = RGB_2_HSL(clamp(a, vmin, vmax));
      tb = RGB_2_HSL(clamp(b, vmin, vmax));
      to.x = ta.x;
      to.y = (ta.y * (1.0f - opacity)) + (tb.y * opacity);
      to.z = ta.z;
      to.w = ta.w;
      o = clamp(HSL_2_RGB(to), vmin, vmax);
      break;

    case DEVELOP_BLEND_HUE:
      ta = RGB_2_HSL(clamp(a, vmin, vmax));
      tb = RGB_2_HSL(clamp(b, vmin, vmax));
      d = abs(ta.x - tb.x);
      s = d > 0.5f ? -opacity * (1.0f - d) / d : opacity;
      to.x = fmod((ta.x * (1.0f - s)) + (tb.x * s) + 1.0f, 1.0f);
      to.y = ta.y;
      to.z = ta.z;
      to.w = ta.w;
      o = clamp(HSL_2_RGB(to), vmin, vmax);
      break;

    case DEVELOP_BLEND_COLOR:
      ta = RGB_2_HSL(clamp(a, vmin, vmax));
      tb = RGB_2_HSL(clamp(b, vmin, vmax));
      d = abs(ta.x - tb.x);
      s = d > 0.5f ? -opacity * (1.0f - d) / d : opacity;
      to.x = fmod((ta.x * (1.0f - s)) + (tb.x * s) + 1.0f, 1.0f);
      to.y = (ta.y * (1.0f - opacity)) + (tb.y * opacity);
      to.z = ta.z;
      to.w = ta.w;
      o = clamp(HSL_2_RGB(to), vmin, vmax);
      break;

    case DEVELOP_BLEND_COLORADJUST:
      ta = RGB_2_HSL(clamp(a, vmin, vmax));
      tb = RGB_2_HSL(clamp(b, vmin, vmax));
      d = abs(ta.x - tb.x);
      s = d > 0.5f ? -opacity * (1.0f - d) / d : opacity;
      to.x = fmod((ta.x * (1.0f - s)) + (tb.x * s) + 1.0f, 1.0f);
      to.y = (ta.y * (1.0f - opacity)) + (tb.y * opacity);
      to.z = tb.z;
      to.w = ta.w;
      o = clamp(HSL_2_RGB(to), vmin, vmax);
      break;

    case DEVELOP_BLEND_BOUNDED:
      o = clamp((a * (1.0f - opacity)) + (b * opacity), vmin, vmax);
      break;

    case DEVELOP_BLEND_LAB_LIGHTNESS:
    case DEVELOP_BLEND_LAB_COLOR:
    case DEVELOP_BLEND_LAB_L:
    case DEVELOP_BLEND_LAB_A:
    case DEVELOP_BLEND_LAB_B:
      o = a; // noop for RGB (without clamping)
      break;

    case DEVELOP_BLEND_HSV_LIGHTNESS:
      ta = RGB_2_HSV(a);
      tb = RGB_2_HSV(b);
      to.x = ta.x;
      to.y = ta.y;
      to.z = (ta.z * (1.0f - opacity)) + (tb.z * opacity);
      to.w = ta.w;
      o = HSV_2_RGB(to);
      break;

    case DEVELOP_BLEND_HSV_COLOR:
      ta = RGB_2_HSV(a);
      tb = RGB_2_HSV(b);
      // blend color vectors of input and output
      d = ta.y * dtcl_cos(DT_2PI_F * ta.x) * (1.0f - opacity) + tb.y * dtcl_cos(DT_2PI_F * tb.x) * opacity;
      s = ta.y * dtcl_sin(DT_2PI_F * ta.x) * (1.0f - opacity) + tb.y * dtcl_sin(DT_2PI_F * tb.x) * opacity;
      to.x = fmod(atan2(s, d) / DT_2PI_F + 1.0f, 1.0f);
      to.y = dt_fast_hypot(s, d);
      to.z = ta.z;
      to.w = ta.w;
      o = HSV_2_RGB(to);
      break;

    case DEVELOP_BLEND_RGB_R:
      o.x = (a.x * (1.0f - opacity)) + (b.x * opacity);
      o.y = a.y;
      o.z = a.z;
      break;

    case DEVELOP_BLEND_RGB_G:
      o.x = a.x;
      o.y = (a.y * (1.0f - opacity)) + (b.y * opacity);
      o.z = a.z;
      break;

    case DEVELOP_BLEND_RGB_B:
      o.x = a.x;
      o.y = a.y;
      o.z = (a.z * (1.0f - opacity)) + (b.z * opacity);
      break;

    // fallback to normal blend
    case DEVELOP_BLEND_NORMAL2:
    default:
      o = (a * (1.0f - opacity)) + (b * opacity);
      break;
  }

  // we transfer alpha channel of input if mask_display is set, else we save opacity into alpha channel
  o.w = mask_display != 0 ? a.w : opacity;

  out.write(int2(x, y), o);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void blendop_rgb_jzczhz(uint3 gid : SV_DispatchThreadID,
                        uniform ImageIn in_a, uniform ImageIn in_b, uniform ImageIn mask, uniform ImageOut out,
                        uniform int width, uniform int height,
                        uniform int blend_mode, uniform float blend_parameter, uniform int2 offs,
                        uniform int mask_display)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  float4 a, b, o;

  const uint mode = uint(blend_mode);
  if((mode & DEVELOP_BLEND_REVERSE) == DEVELOP_BLEND_REVERSE)
  {
    b = in_a.readpixel(x + offs.x, y + offs.y);
    a = in_b.Areadpixel(x, y);
  }
  else
  {
    a = in_a.readpixel(x + offs.x, y + offs.y);
    b = in_b.Areadpixel(x, y);
  }
  const float opacity = mask.Areadsingle(x, y);
  float norm_a;
  float norm_b;

  // select the blend operator
  switch(mode & DEVELOP_BLEND_MODE_MASK)
  {
    case DEVELOP_BLEND_MULTIPLY:
      o = a * (1.0f - opacity) + a * b * blend_parameter * opacity;
      break;

    case DEVELOP_BLEND_AVERAGE:
      o = a * (1.0f - opacity) + 0.5f * (a + b) * opacity;
      break;

    case DEVELOP_BLEND_ADD:
      o = a * (1.0f - opacity) + (a + blend_parameter * b) * opacity;
      break;

    case DEVELOP_BLEND_SUBTRACT:
      o = a * (1.0f - opacity) + max(a - blend_parameter * b, 0.0f) * opacity;
      break;

    case DEVELOP_BLEND_SUBTRACT_INVERSE:
      o = a * (1.0f - opacity) + max(b - blend_parameter * a, 0.0f) * opacity;
      break;

    case DEVELOP_BLEND_DIFFERENCE:
    case DEVELOP_BLEND_DIFFERENCE2:
      o = a * (1.0f - opacity) + abs(a - b) * opacity;
      break;

    case DEVELOP_BLEND_LIGHTNESS:
      norm_a = max(dtcl_sqrt(a.x * a.x + a.y * a.y + a.z * a.z), 1e-6f);
      norm_b = max(dtcl_sqrt(b.x * b.x + b.y * b.y + b.z * b.z), 1e-6f);
      o = a * (1.0f - opacity) + a * norm_b / norm_a * opacity;
      break;

    case DEVELOP_BLEND_CHROMA:
      norm_a = max(dtcl_sqrt(a.x * a.x + a.y * a.y + a.z * a.z), 1e-6f);
      norm_b = max(dtcl_sqrt(b.x * b.x + b.y * b.y + b.z * b.z), 1e-6f);
      o = a * (1.0f - opacity) + b * norm_a / norm_b * opacity;
      break;

    case DEVELOP_BLEND_RGB_R:
      o.x = (a.x * (1.0f - opacity)) + (blend_parameter * b.x * opacity);
      o.y = a.y;
      o.z = a.z;
      break;

    case DEVELOP_BLEND_RGB_G:
      o.x = a.x;
      o.y = (a.y * (1.0f - opacity)) + (blend_parameter * b.y * opacity);
      o.z = a.z;
      break;

    case DEVELOP_BLEND_RGB_B:
      o.x = a.x;
      o.y = a.y;
      o.z = (a.z * (1.0f - opacity)) + (blend_parameter * b.z * opacity);
      break;

    case DEVELOP_BLEND_DIVIDE:
      o = a * (1.0f - opacity) + a / max(b * blend_parameter, 1e-6f) * opacity;
      break;

    case DEVELOP_BLEND_DIVIDE_INVERSE:
      o = a * (1.0f - opacity) + b / max(a * blend_parameter, 1e-6f) * opacity;
      break;

    case DEVELOP_BLEND_GEOMETRIC_MEAN:
      o = a * (1.0f - opacity) + dtcl_sqrt(max(a * b, 0.0f)) * opacity;
      break;

    case DEVELOP_BLEND_HARMONIC_MEAN:
      o = a * (1.0f - opacity) + 2.0f * a * b / (max(a, 5e-7f) + max(b, 5e-7f)) * opacity;
      break;

    // fallback to normal blend
    case DEVELOP_BLEND_BOUNDED:
    case DEVELOP_BLEND_NORMAL2:
    default:
      o = (a * (1.0f - opacity)) + (b * opacity);
      break;
  }

  // we transfer alpha channel of input if mask_display is set, else we save opacity into alpha channel
  o.w = mask_display != 0 ? a.w : opacity;

  out.write(int2(x, y), o);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void blendop_mask_tone_curve(uint3 gid : SV_DispatchThreadID,
                             uniform ImageIn mask_in, uniform ImageOut mask_out,
                             uniform int width, uniform int height,
                             uniform float e, uniform float brightness, uniform float gopacity)
{
  const float mask_epsilon = 16.0f * 1.1920928955078125e-07f; // empirical mask threshold for fully transparent masks
  const int x = gid.x;
  const int y = gid.y;

  if(x >= width || y >= height) return;

  float opacity = mask_in.Areadsingle(x, y);
  float scaled_opacity = 2.f * opacity / gopacity - 1.f;
  if(1.f - brightness <= 0.f)
    scaled_opacity = opacity <= mask_epsilon ? -1.f : 1.f;
  else if(1.f + brightness <= 0.f)
    scaled_opacity = opacity >= 1.f - mask_epsilon ? 1.f : -1.f;
  else if(brightness > 0.f)
  {
    scaled_opacity = (scaled_opacity + brightness) / (1.f - brightness);
    scaled_opacity = min(scaled_opacity, 1.f);
  }
  else
  {
    scaled_opacity = (scaled_opacity + brightness) / (1.f + brightness);
    scaled_opacity = max(scaled_opacity, -1.f);
  }
  opacity = 0.5f * (scaled_opacity * e / (1.f + (e - 1.f) * abs(scaled_opacity))) + 0.5f;
  opacity = clipf(opacity > 1e-6f ? opacity : 0.0f) * gopacity;
  mask_out.write(int2(x, y), float4(opacity));
}

// fills a mask with one value. There is no OpenCL counterpart: OpenCL uses
// clEnqueueFillImage, which neither Vulkan storage images nor Metal textures
// have.
[shader("compute")]
[numthreads(16, 16, 1)]
void blendop_fill_mask(uint3 gid : SV_DispatchThreadID,
                       uniform ImageOut mask, uniform int width, uniform int height, uniform float value)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;
  mask.write(int2(x, y), float4(value));
}

// the luminance, scharr and detail blend kernels of the detail mask and of
// dual demosaic
[shader("compute")]
[numthreads(16, 16, 1)]
void calc_Y0_mask(uint3 gid : SV_DispatchThreadID,
                  uniform RWStructuredBuffer<float> mask, uniform ImageIn in,
                  uniform int w, uniform int height, uniform float4 wb)
{
  const int col = gid.x;
  const int row = gid.y;
  if(col >= w || row >= height) return;
  const int idx = row * w + col;

  const float4 pt = wb * max(0.0f, in.Areadpixel(col, row));
  mask[idx] = dtcl_sqrt((pt.x + pt.y + pt.z) / 3.0f);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void calc_scharr_mask(uint3 gid : SV_DispatchThreadID,
                      uniform StructuredBuffer<float> in, uniform RWStructuredBuffer<float> out,
                      uniform int w, uniform int height)
{
  const int col = gid.x;
  const int row = gid.y;
  if(col >= w || row >= height) return;

  const int oidx = row * w + col;
  const int incol = clamp(col, 1, w - 2);
  const int inrow = clamp(row, 1, height - 2);
  const int idx = inrow * w + incol;
  const float gradient_magnitude = scharr_gradient(in, idx, w);
  out[oidx] = clipf(gradient_magnitude / 16.0f);
}

float _calc_blend_factor(float val, float threshold)
{
  // sigmoid with its inflexion point at (threshold, 0.5)
  return 1.0f / (1.0f + dt_fast_expf(16.0f - (16.0f / threshold) * val));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void calc_detail_blend(uint3 gid : SV_DispatchThreadID,
                       uniform StructuredBuffer<float> in, uniform RWStructuredBuffer<float> out,
                       uniform int w, uniform int height, uniform float threshold, uniform int detail)
{
  const int col = gid.x;
  const int row = gid.y;
  if(col >= w || row >= height) return;

  const int idx = row * w + col;
  const float blend = clipf(_calc_blend_factor(in[idx], threshold));
  out[idx] = detail != 0 ? blend : 1.0f - blend;
}
