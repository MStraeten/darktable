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

// Slang ports of the filmic rgb kernels in ../filmic.cl. The filmic v3
// kernel at the top of that file belongs to a module that no longer exists
// and is not ported.

import colorspace;
import color_conversion;
import noise_generator;

static const float FLT_MAX = 3.402823466e+38f;
static const float INVERSE_SQRT_3 = 0.5773502691896258f;

// dt_iop_filmicrgb_methods_type_t
static const int DT_FILMIC_METHOD_NONE = 0;
static const int DT_FILMIC_METHOD_MAX_RGB = 1;
static const int DT_FILMIC_METHOD_LUMINANCE = 2;
static const int DT_FILMIC_METHOD_POWER_NORM = 3;
static const int DT_FILMIC_METHOD_EUCLIDEAN_NORM_V1 = 4;
static const int DT_FILMIC_METHOD_EUCLIDEAN_NORM_V2 = 5;

// dt_iop_filmicrgb_colorscience_type_t
static const int DT_FILMIC_COLORSCIENCE_V1 = 0;
static const int DT_FILMIC_COLORSCIENCE_V2 = 1;
static const int DT_FILMIC_COLORSCIENCE_V3 = 2;
static const int DT_FILMIC_COLORSCIENCE_V4 = 3;
static const int DT_FILMIC_COLORSCIENCE_V5 = 4;

// dt_iop_filmicrgb_reconstruction_type_t
static const int DT_FILMIC_RECONSTRUCT_RGB = 0;
static const int DT_FILMIC_RECONSTRUCT_RATIOS = 1;

// dt_iop_filmicrgb_curve_type_t
static const int DT_FILMIC_CURVE_POLY_4 = 0;
static const int DT_FILMIC_CURVE_POLY_3 = 1;
static const int DT_FILMIC_CURVE_RATIONAL = 2;

// the spline and its parameters, as filmic_spline() takes them
struct Spline
{
  float4 M1, M2, M3, M4, M5;
  float latitude_min, latitude_max;
  int type[2];
};

// the matrices of the v4 and v5 gamut mapping
struct GamutMatrices
{
  StructuredBuffer<float> matrix_in;
  StructuredBuffer<float> matrix_out;
  StructuredBuffer<float> export_matrix_in;
  StructuredBuffer<float> export_matrix_out;
};

/* norms */
float pixel_rgb_norm_power(float4 pixel)
{
  // weird norm sort of perceptual. This is black magic really, but it looks good.
  const float4 RGB = abs(pixel);
  const float4 RGB_square = RGB * RGB;
  const float4 RGB_cubic = RGB_square * RGB;
  return (RGB_cubic.x + RGB_cubic.y + RGB_cubic.z) / max(RGB_square.x + RGB_square.y + RGB_square.z, 1e-12f);
}

float pixel_rgb_norm_euclidean(float4 pixel)
{
  const float4 RGB_square = pixel * pixel;
  return dtcl_sqrt(RGB_square.x + RGB_square.y + RGB_square.z);
}

float get_pixel_norm(float4 pixel, int variant, ProfileInfo profile_info, ImageIn lut, int use_work_profile)
{
  switch(variant)
  {
    case DT_FILMIC_METHOD_MAX_RGB:
      return max(max(pixel.x, pixel.y), pixel.z);

    case DT_FILMIC_METHOD_POWER_NORM:
      return pixel_rgb_norm_power(pixel);

    case DT_FILMIC_METHOD_EUCLIDEAN_NORM_V1:
      return pixel_rgb_norm_euclidean(pixel);

    case DT_FILMIC_METHOD_EUCLIDEAN_NORM_V2:
      return pixel_rgb_norm_euclidean(pixel) * INVERSE_SQRT_3;

    case DT_FILMIC_METHOD_LUMINANCE:
    case DT_FILMIC_METHOD_NONE:
    default:
      return (use_work_profile != 0) ? get_rgb_matrix_luminance(pixel, profile_info, profile_info.matrix_in, lut)
                                     : dt_camera_rgb_luminance(pixel);
  }
}

float get_luminance(float4 pixel, ProfileInfo profile_info, ImageIn lut, int use_work_profile)
{
  return (use_work_profile != 0) ? get_rgb_matrix_luminance(pixel, profile_info, profile_info.matrix_in, lut)
                                 : dt_camera_rgb_luminance(pixel);
}

/* saturation */
float filmic_desaturate_v1(float x, float sigma_toe, float sigma_shoulder, float saturation)
{
  const float radius_toe = x;
  const float radius_shoulder = 1.0f - x;

  const float key_toe = dtcl_exp(-0.5f * radius_toe * radius_toe / sigma_toe);
  const float key_shoulder = dtcl_exp(-0.5f * radius_shoulder * radius_shoulder / sigma_shoulder);

  return 1.0f - clipf((key_toe + key_shoulder) / saturation);
}

float filmic_desaturate_v2(float x, float sigma_toe, float sigma_shoulder, float saturation)
{
  const float radius_toe = x;
  const float radius_shoulder = 1.0f - x;
  const float sat2 = 0.5f / dtcl_sqrt(saturation);
  const float key_toe = dtcl_exp(-radius_toe * radius_toe / sigma_toe * sat2);
  const float key_shoulder = dtcl_exp(-radius_shoulder * radius_shoulder / sigma_shoulder * sat2);

  return (saturation - (key_toe + key_shoulder) * (saturation));
}

float4 linear_saturation(float4 x, float luminance, float saturation)
{
  return float4(luminance) + float4(saturation) * (x - float4(luminance));
}

float filmic_spline(float x, Spline sp)
{
  // if type polynomial:
  // y = M5 * x⁴ + M4 * x³ + M3 * x² + M2 * x¹ + M1 * x⁰
  // but we rewrite it using Horner factorisation, to spare ops and enable FMA in available
  // else if type rational:
  // y = M1 * (M2 * (x - x_0)² + (x - x_0)) / (M2 * (x - x_0)² + (x - x_0) + M3)
  float result;

  if(x < sp.latitude_min)
  {
    // toe
    if(sp.type[0] == DT_FILMIC_CURVE_POLY_4)
      result = sp.M1.x + x * (sp.M2.x + x * (sp.M3.x + x * (sp.M4.x + x * sp.M5.x)));
    else if(sp.type[0] == DT_FILMIC_CURVE_POLY_3)
      result = sp.M1.x + x * (sp.M2.x + x * (sp.M3.x + x * sp.M4.x));
    else
    {
      const float xi = sp.latitude_min - x;
      const float rat = xi * (xi * sp.M2.x + 1.f);
      result = sp.M4.x - sp.M1.x * rat / (rat + sp.M3.x);
    }
  }
  else if(x > sp.latitude_max)
  {
    // shoulder
    if(sp.type[1] == DT_FILMIC_CURVE_POLY_4)
      result = sp.M1.y + x * (sp.M2.y + x * (sp.M3.y + x * (sp.M4.y + x * sp.M5.y)));
    else if(sp.type[1] == DT_FILMIC_CURVE_POLY_3)
      result = sp.M1.y + x * (sp.M2.y + x * (sp.M3.y + x * sp.M4.y));
    else
    {
      const float xi = x - sp.latitude_max;
      const float rat = xi * (xi * sp.M2.y + 1.f);
      result = sp.M4.y + sp.M1.y * rat / (rat + sp.M3.y);
    }
  }
  else
  {
    // latitude
    result = sp.M1.z + x * sp.M2.z;
  }

  return result;
}

float log_tonemapping_v1(float x, float grey, float black, float dynamic_range)
{
  const float temp = (dtcl_log2(x / grey) - black) / dynamic_range;
  return clamp(temp, NORM_MIN, 1.f);
}

float log_tonemapping_v2(float x, float grey, float black, float dynamic_range)
{
  return clipf((dtcl_log2(x / grey) - black) / dynamic_range);
}

float4 pipe_RGB_to_Ych(float4 rgb, StructuredBuffer<float> matrix)
{
  // go from pipeline RGB to CIE 2006 LMS D65, then Kirk/Filmlight Yrg in polar coordinates
  const float4 LMS = matrix_product_float4(rgb, matrix);
  const float4 Yrg = LMS_to_Yrg(LMS);
  return Yrg_to_Ych(Yrg);
}

float4 Ych_to_pipe_RGB(float4 Ych, StructuredBuffer<float> matrix)
{
  const float4 Yrg = Ych_to_Yrg(Ych);
  const float4 LMS = Yrg_to_LMS(Yrg);
  return matrix_product_float4(LMS, matrix);
}

float4 filmic_desaturate_v4(float4 Ych_original, float4 Ych_final, float saturation)
{
  // Ych is normalized through the LMS conversion, so c is actually a
  // saturation. See filmic.cl for the model.
  const float chroma_original = Ych_original.y * Ych_original.x; // c2
  float chroma_final = Ych_final.y * Ych_final.x;                // c1

  const float delta_chroma = saturation * (chroma_original - chroma_final);

  const bool filmic_brightens = (Ych_final.x > Ych_original.x);
  const bool filmic_resat = (chroma_original < chroma_final);
  const bool filmic_desat = (chroma_original > chroma_final);
  const bool user_resat = (saturation > 0.f);
  const bool user_desat = (saturation < 0.f);

  chroma_final = (filmic_brightens && filmic_resat)
                     ? (chroma_original + chroma_final) / 2.f // force original lower sat if brightening
                 : ((user_resat && filmic_desat) || user_desat)
                     ? chroma_final + delta_chroma // allow resaturation only if filmic desaturated, allow desat anytime
                     : chroma_final;

  Ych_final.y = max(chroma_final / Ych_final.x, 0.f);
  return Ych_final;
}

// pipeline and ICC luminance is CIE Y 1931, Kirk Ych/Yrg uses CIE Y 2006:
// see filmic.cl. Only applies to achromatic pixels.
float CIE_Y_1931_to_CIE_Y_2006(float x) { return 1.05785528f * x; }

// coeffs is the row of the LMS -> RGB matrix that starts at matrix[row]
float clip_chroma_white_raw(StructuredBuffer<float> matrix, int row, float target_white, float Y,
                            float cos_h, float sin_h)
{
  const float c0 = matrix[row], c1 = matrix[row + 1], c2 = matrix[row + 2];
  const float denominator_Y_coeff = c0 * (0.979381443298969f * cos_h + 0.391752577319588f * sin_h)
                                    + c1 * (0.0206185567010309f * cos_h + 0.608247422680412f * sin_h)
                                    - c2 * (cos_h + sin_h);
  const float denominator_target_term = target_white * (0.68285981628866f * cos_h + 0.482137060515464f * sin_h);

  // this channel won't limit the chroma
  if(denominator_Y_coeff == 0.f) return FLT_MAX;

  // the equation for max chroma has an asymptote at this point (zero of
  // denominator). Any Y below that value won't give us sensible results for
  // the upper bound and we should consider the lower bound instead.
  const float Y_asymptote = denominator_target_term / denominator_Y_coeff;
  if(Y <= Y_asymptote) return FLT_MAX;

  // see tools/derive_filmic_v6_gamut_mapping.py for derivation of these equations
  const float denominator = Y * denominator_Y_coeff - denominator_target_term;
  const float numerator = -0.427506877216495f
                          * (Y * (c0 + 0.856492345150334f * c1 + 0.554995960637719f * c2)
                             - 0.988237752433297f * target_white);

  return numerator / denominator;
}

float clip_chroma_white(StructuredBuffer<float> matrix, int row, float target_white, float Y,
                        float cos_h, float sin_h)
{
  // the chroma clipping curves of the RGB channels may disagree slightly at
  // the max luminance, because of numerical inaccuracies in color matrices:
  // interpolate each clipping line to zero chroma near max luminance
  const float eps = 1e-3f;
  const float max_Y = CIE_Y_1931_to_CIE_Y_2006(target_white);
  const float delta_Y = max(max_Y - Y, 0.f);
  float max_chroma;
  if(delta_Y < eps)
  {
    max_chroma = delta_Y / (eps * max_Y)
                 * clip_chroma_white_raw(matrix, row, target_white, (1.f - eps) * max_Y, cos_h, sin_h);
  }
  else
  {
    max_chroma = clip_chroma_white_raw(matrix, row, target_white, Y, cos_h, sin_h);
  }
  return max_chroma >= 0.f ? max_chroma : FLT_MAX;
}

float clip_chroma_black(StructuredBuffer<float> matrix, int row, float cos_h, float sin_h)
{
  // clip_chroma_white_raw() with a target value of 0
  const float c0 = matrix[row], c1 = matrix[row + 1], c2 = matrix[row + 2];
  const float denominator = c0 * (0.979381443298969f * cos_h + 0.391752577319588f * sin_h)
                            + c1 * (0.0206185567010309f * cos_h + 0.608247422680412f * sin_h)
                            - c2 * (cos_h + sin_h);

  // this channel won't limit the chroma
  if(denominator == 0.f) return FLT_MAX;

  const float numerator = -0.427506877216495f * (c0 + 0.856492345150334f * c1 + 0.554995960637719f * c2);
  const float max_chroma = numerator / denominator;
  return max_chroma >= 0.f ? max_chroma : FLT_MAX;
}

float clip_chroma(StructuredBuffer<float> matrix_out, float target_white, float Y,
                  float cos_h, float sin_h, float chroma)
{
  const float chroma_R_white = clip_chroma_white(matrix_out, 0, target_white, Y, cos_h, sin_h);
  const float chroma_G_white = clip_chroma_white(matrix_out, 4, target_white, Y, cos_h, sin_h);
  const float chroma_B_white = clip_chroma_white(matrix_out, 8, target_white, Y, cos_h, sin_h);
  const float max_chroma_white = min(min(chroma_R_white, chroma_G_white), chroma_B_white);

  const float chroma_R_black = clip_chroma_black(matrix_out, 0, cos_h, sin_h);
  const float chroma_G_black = clip_chroma_black(matrix_out, 4, cos_h, sin_h);
  const float chroma_B_black = clip_chroma_black(matrix_out, 8, cos_h, sin_h);
  const float max_chroma_black = min(min(chroma_R_black, chroma_G_black), chroma_B_black);

  return min(min(chroma, max_chroma_black), max_chroma_white);
}

float4 gamut_check_RGB(StructuredBuffer<float> matrix_in, StructuredBuffer<float> matrix_out,
                       float display_black, float display_white, float4 Ych_in)
{
  // heuristic: if there are negatives, calculate the amount (luminance) of
  // white light that would need to be mixed in to bring the pixel back in gamut
  float4 RGB_brightened = Ych_to_pipe_RGB(Ych_in, matrix_out);
  const float min_pix = min(min(RGB_brightened.x, RGB_brightened.y), RGB_brightened.z);
  const float black_offset = max(-min_pix, 0.f);
  RGB_brightened += black_offset;
  const float4 Ych_brightened = pipe_RGB_to_Ych(RGB_brightened, matrix_in);

  // increase the input luminance a little by the value we calculated above.
  // This doesn't actually desaturate the color like mixing white would do:
  // the chroma change comes next.
  const float Y = clamp((Ych_in.x + Ych_brightened.x) / 2.f, CIE_Y_1931_to_CIE_Y_2006(display_black),
                        CIE_Y_1931_to_CIE_Y_2006(display_white));

  const float cos_h = Ych_in.z;
  const float sin_h = Ych_in.w;
  const float new_chroma = clip_chroma(matrix_out, display_white, Y, cos_h, sin_h, Ych_in.y);

  // go to RGB, using existing luminance and hue and the new chroma
  const float4 Ych = float4(Y, new_chroma, cos_h, sin_h);
  const float4 RGB_out = Ych_to_pipe_RGB(Ych, matrix_out);

  // clamp in target RGB as a final catch-all
  return clamp(RGB_out, float4(0.f), float4(display_white));
}

float4 gamut_mapping(float4 Ych_final, float4 Ych_original, GamutMatrices m,
                     float display_black, float display_white, float saturation, int use_output_profile)
{
  // force final hue to original
  Ych_final.z = Ych_original.z;
  Ych_final.w = Ych_original.w;

  // clip luminance
  Ych_final.x = clamp(Ych_final.x, CIE_Y_1931_to_CIE_Y_2006(display_black),
                      CIE_Y_1931_to_CIE_Y_2006(display_white));

  // massage chroma
  Ych_final = filmic_desaturate_v4(Ych_original, Ych_final, saturation);
  Ych_final = gamut_check_Yrg(Ych_final);

  // a channel can still be over display white or under display black because
  // of saturation. Y is clipped already, so any problem now is caused by c.
  if(use_output_profile == 0)
    return gamut_check_RGB(m.matrix_in, m.matrix_out, display_black, display_white, Ych_final);

  const float4 export_RGB = gamut_check_RGB(m.export_matrix_in, m.export_matrix_out, display_black,
                                            display_white, Ych_final);

  // go from export RGB to CIE LMS 2006 D65, then to pipeline RGB D50
  const float4 LMS = matrix_product_float4(export_RGB, m.export_matrix_in);
  return matrix_product_float4(LMS, m.matrix_out);
}

float4 tonemap_channels(float4 i, float dynamic_range, float black_exposure, float grey_value,
                        Spline sp, float display_white, float output_power)
{
  // log tonemapping, then the S curve on individual channels
  float4 o;
  o.x = filmic_spline(log_tonemapping_v2(i.x, grey_value, black_exposure, dynamic_range), sp);
  o.y = filmic_spline(log_tonemapping_v2(i.y, grey_value, black_exposure, dynamic_range), sp);
  o.z = filmic_spline(log_tonemapping_v2(i.z, grey_value, black_exposure, dynamic_range), sp);
  o.w = 0.f;

  // clamp to [0, display_white]: clamping individual channels to
  // display_black would limit the max available saturation. Luminance is
  // clipped to display_black later. Apply the output power afterwards.
  return dtcl_pow(clamp(o, float4(0.f), float4(display_white)), output_power);
}

float tonemap_norm(float norm, float dynamic_range, float black_exposure, float grey_value,
                   Spline sp, float display_black, float display_white, float output_power)
{
  // log tonemapping, the S curve and the transfer function of the display
  norm = log_tonemapping_v2(norm, grey_value, black_exposure, dynamic_range);
  return dtcl_pow(clamp(filmic_spline(norm, sp), display_black, display_white), output_power);
}

float4 filmic_split_v1(float4 i, float dynamic_range, float black_exposure, float grey_value,
                       ProfileInfo profile_info, ImageIn lut, int use_work_profile,
                       float sigma_toe, float sigma_shoulder, float saturation, Spline sp, float output_power)
{
  float4 o = float4(0.f);

  o.x = log_tonemapping_v1(max(i.x, NORM_MIN), grey_value, black_exposure, dynamic_range);
  o.y = log_tonemapping_v1(max(i.y, NORM_MIN), grey_value, black_exposure, dynamic_range);
  o.z = log_tonemapping_v1(max(i.z, NORM_MIN), grey_value, black_exposure, dynamic_range);

  // selective desaturation of extreme luminances
  const float luminance = get_luminance(o, profile_info, lut, use_work_profile);
  const float desaturation = filmic_desaturate_v1(luminance, sigma_toe, sigma_shoulder, saturation);
  o = linear_saturation(o, luminance, desaturation);

  o.x = filmic_spline(o.x, sp);
  o.y = filmic_spline(o.y, sp);
  o.z = filmic_spline(o.z, sp);

  return dtcl_pow(clip4(o), output_power);
}

float4 filmic_split_v2_v3(float4 i, float dynamic_range, float black_exposure, float grey_value,
                          ProfileInfo profile_info, ImageIn lut, int use_work_profile,
                          float sigma_toe, float sigma_shoulder, float saturation, Spline sp, float output_power)
{
  float4 o = float4(0.f);

  o.x = log_tonemapping_v2(max(i.x, NORM_MIN), grey_value, black_exposure, dynamic_range);
  o.y = log_tonemapping_v2(max(i.y, NORM_MIN), grey_value, black_exposure, dynamic_range);
  o.z = log_tonemapping_v2(max(i.z, NORM_MIN), grey_value, black_exposure, dynamic_range);

  // selective desaturation of extreme luminances
  const float luminance = get_luminance(o, profile_info, lut, use_work_profile);
  const float desaturation = filmic_desaturate_v2(luminance, sigma_toe, sigma_shoulder, saturation);
  o = linear_saturation(o, luminance, desaturation);

  o.x = filmic_spline(o.x, sp);
  o.y = filmic_spline(o.y, sp);
  o.z = filmic_spline(o.z, sp);

  return dtcl_pow(clip4(o), output_power);
}

float4 filmic_split_v4(float4 i, float dynamic_range, float black_exposure, float grey_value,
                       float saturation, Spline sp, float output_power, GamutMatrices m,
                       float display_black, float display_white, int use_output_profile)
{
  const float4 o = tonemap_channels(i, dynamic_range, black_exposure, grey_value, sp, display_white, output_power);

  // save Ych in Kirk/Filmlight Yrg, and get the final one
  const float4 Ych_original = pipe_RGB_to_Ych(i, m.matrix_in);
  float4 Ych_final = pipe_RGB_to_Ych(o, m.matrix_in);

  Ych_final.y = min(Ych_original.y, Ych_final.y);

  return gamut_mapping(Ych_final, Ych_original, m, display_black, display_white, saturation, use_output_profile);
}

float4 filmic_chroma_v1(float4 i, float dynamic_range, float black_exposure, float grey_value,
                        ProfileInfo profile_info, ImageIn lut, int use_work_profile,
                        float sigma_toe, float sigma_shoulder, float saturation, Spline sp, float output_power,
                        int variant)
{
  float norm = max(get_pixel_norm(i, variant, profile_info, lut, use_work_profile), NORM_MIN);

  // save the ratios
  float4 o = i / float4(norm);

  // sanitize the ratios
  const float min_ratios = min(min(o.x, o.y), o.z);
  if(min_ratios < 0.0f) o -= float4(min_ratios);

  norm = log_tonemapping_v1(norm, grey_value, black_exposure, dynamic_range);

  // selective desaturation of extreme luminances
  o *= float4(norm);
  const float luminance = get_luminance(o, profile_info, lut, use_work_profile);
  const float desaturation = filmic_desaturate_v1(norm, sigma_toe, sigma_shoulder, saturation);
  o = linear_saturation(o, luminance, desaturation);
  o /= float4(norm);

  // filmic S curve on the max RGB, then the transfer function of the display
  norm = dtcl_pow(clipf(filmic_spline(norm, sp)), output_power);

  return o * norm;
}

float4 filmic_chroma_v2_v3(float4 i, float dynamic_range, float black_exposure, float grey_value,
                           ProfileInfo profile_info, ImageIn lut, int use_work_profile,
                           float sigma_toe, float sigma_shoulder, float saturation, Spline sp, float output_power,
                           int variant, int colorscience_version)
{
  float norm = max(get_pixel_norm(i, variant, profile_info, lut, use_work_profile), NORM_MIN);

  // save the ratios
  float4 ratios = i / float4(norm);

  // sanitize the ratios
  const float min_ratios = min(min(ratios.x, ratios.y), ratios.z);
  if(min_ratios < 0.0f) ratios -= float4(min_ratios);

  norm = log_tonemapping_v2(norm, grey_value, black_exposure, dynamic_range);

  // get the desaturation value based on the log value
  const float4 desaturation = float4(filmic_desaturate_v2(norm, sigma_toe, sigma_shoulder, saturation));

  // filmic S curve on the max RGB, then the transfer function of the display
  norm = dtcl_pow(clipf(filmic_spline(norm, sp)), output_power);

  // re-apply ratios with saturation change
  ratios = max(ratios + (float4(1.0f) - ratios) * (float4(1.0f) - desaturation), float4(0.f));

  if(colorscience_version == DT_FILMIC_COLORSCIENCE_V3)
    norm /= max(get_pixel_norm(ratios, variant, profile_info, lut, use_work_profile), NORM_MIN);

  float4 o = float4(norm) * ratios;

  // penalize the ratios by the amount of clipping
  const float max_pix = fmax3(o);
  if(max_pix > 1.0f)
  {
    ratios = max(ratios + (float4(1.0f) - float4(max_pix)), float4(0.0f));
    o = clip4(float4(norm) * ratios);
  }

  return o;
}

float4 filmic_chroma_v4(float4 i, float dynamic_range, float black_exposure, float grey_value,
                        ProfileInfo profile_info, ImageIn lut, int use_work_profile,
                        float saturation, Spline sp, float output_power, int variant, GamutMatrices m,
                        float display_black, float display_white, int use_output_profile,
                        float norm_min, float norm_max)
{
  // clamp the norm to the valid input range early, otherwise
  // log_tonemapping_v2 clamps it later and the ratios are wrong: colorful
  // patches end up darker than their surrounding where the raw data clips
  float norm = clamp(get_pixel_norm(i, variant, profile_info, lut, use_work_profile), norm_min, norm_max);

  // save the ratios
  const float4 ratios = i / float4(norm);

  norm = tonemap_norm(norm, dynamic_range, black_exposure, grey_value, sp, display_black, display_white,
                      output_power);

  // restore RGB
  const float4 o = norm * ratios;

  // save Ych in Kirk/Filmlight Yrg, and get the final one
  const float4 Ych_original = pipe_RGB_to_Ych(i, m.matrix_in);
  const float4 Ych_final = pipe_RGB_to_Ych(o, m.matrix_in);

  return gamut_mapping(Ych_final, Ych_original, m, display_black, display_white, saturation, use_output_profile);
}

float4 filmic_chroma_v5(float4 i, float dynamic_range, float black_exposure, float grey_value,
                        float saturation, Spline sp, float output_power, GamutMatrices m,
                        float display_black, float display_white, int use_output_profile,
                        float norm_min, float norm_max)
{
  // clamp the norm early, as in filmic_chroma_v4()
  float norm = clamp(max(max(i.x, i.y), i.z), norm_min, norm_max);

  // save the ratios
  const float4 ratios = i / float4(norm);

  norm = tonemap_norm(norm, dynamic_range, black_exposure, grey_value, sp, display_black, display_white,
                      output_power);

  // restore RGB
  const float4 max_rgb = norm * ratios;

  const float4 naive_rgb = tonemap_channels(i, dynamic_range, black_exposure, grey_value, sp, display_white,
                                            output_power);

  // mix max RGB with naive RGB
  const float4 o = (0.5f - saturation) * naive_rgb + (0.5f + saturation) * max_rgb;

  // save Ych in Kirk/Filmlight Yrg, and get the final one
  const float4 Ych_original = pipe_RGB_to_Ych(i, m.matrix_in);
  float4 Ych_final = pipe_RGB_to_Ych(o, m.matrix_in);

  Ych_final.y = min(Ych_original.y, Ych_final.y);

  return gamut_mapping(Ych_final, Ych_original, m, display_black, display_white, 0.f, use_output_profile);
}

// the spline coefficients are float[4] rather than float4 so that the
// parameters stay in 256 bytes of push constants: see extended.slang
Spline make_spline(float M1[4], float M2[4], float M3[4], float M4[4], float M5[4],
                   float latitude_min, float latitude_max, int type_1, int type_2)
{
  Spline sp;
  sp.M1 = float4(M1[0], M1[1], M1[2], M1[3]);
  sp.M2 = float4(M2[0], M2[1], M2[2], M2[3]);
  sp.M3 = float4(M3[0], M3[1], M3[2], M3[3]);
  sp.M4 = float4(M4[0], M4[1], M4[2], M4[3]);
  sp.M5 = float4(M5[0], M5[1], M5[2], M5[3]);
  sp.latitude_min = latitude_min;
  sp.latitude_max = latitude_max;
  sp.type[0] = type_1;
  sp.type[1] = type_2;
  return sp;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void filmicrgb_split(uint3 gid : SV_DispatchThreadID,
                     uniform ImageIn in, uniform ImageOut out,
                     uniform int width, uniform int height,
                     uniform float dynamic_range, uniform float black_exposure, uniform float grey_value,
                     uniform StructuredBuffer<ProfileInfo> profile_info,
                     uniform ImageIn lut, uniform int use_work_profile,
                     uniform float sigma_toe, uniform float sigma_shoulder, uniform float saturation,
                     uniform float M1[4], uniform float M2[4], uniform float M3[4], uniform float M4[4],
                     uniform float M5[4],
                     uniform float latitude_min, uniform float latitude_max, uniform float output_power,
                     uniform int color_science,
                     uniform int type_1, uniform int type_2,
                     uniform StructuredBuffer<float> matrix_in, uniform StructuredBuffer<float> matrix_out,
                     uniform float display_black, uniform float display_white,
                     uniform int use_output_profile,
                     uniform StructuredBuffer<float> export_matrix_in, uniform StructuredBuffer<float> export_matrix_out)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float4 i = in.readpixel(x, y);
  const Spline sp = make_spline(M1, M2, M3, M4, M5, latitude_min, latitude_max, type_1, type_2);
  GamutMatrices m;
  m.matrix_in = matrix_in;
  m.matrix_out = matrix_out;
  m.export_matrix_in = export_matrix_in;
  m.export_matrix_out = export_matrix_out;

  float4 o;
  switch(color_science)
  {
    case DT_FILMIC_COLORSCIENCE_V1:
      o = filmic_split_v1(i, dynamic_range, black_exposure, grey_value, profile_info[0], lut, use_work_profile,
                          sigma_toe, sigma_shoulder, saturation, sp, output_power);
      break;
    case DT_FILMIC_COLORSCIENCE_V2:
    case DT_FILMIC_COLORSCIENCE_V3:
      o = filmic_split_v2_v3(i, dynamic_range, black_exposure, grey_value, profile_info[0], lut,
                             use_work_profile, sigma_toe, sigma_shoulder, saturation, sp, output_power);
      break;
    case DT_FILMIC_COLORSCIENCE_V4:
      o = filmic_split_v4(i, dynamic_range, black_exposure, grey_value, saturation, sp, output_power, m,
                          display_black, display_white, use_output_profile);
      break;
    default:
      // v5 is handled as a chroma variant, it should not end up here
      o = float4(1.f, 0.f, 0.f, 1.f);
      break;
  }

  // copy alpha layer and save
  o.w = i.w;
  out.write(int2(x, y), o);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void filmicrgb_chroma(uint3 gid : SV_DispatchThreadID,
                      uniform ImageIn in, uniform ImageOut out,
                      uniform int width, uniform int height,
                      uniform float dynamic_range, uniform float black_exposure, uniform float grey_value,
                      uniform StructuredBuffer<ProfileInfo> profile_info,
                      uniform ImageIn lut, uniform int use_work_profile,
                      uniform float sigma_toe, uniform float sigma_shoulder, uniform float saturation,
                      uniform float M1[4], uniform float M2[4], uniform float M3[4], uniform float M4[4],
                      uniform float M5[4],
                      uniform float latitude_min, uniform float latitude_max, uniform float output_power,
                      uniform int variant, uniform int color_science,
                      uniform int type_1, uniform int type_2,
                      uniform StructuredBuffer<float> matrix_in, uniform StructuredBuffer<float> matrix_out,
                      uniform float display_black, uniform float display_white,
                      uniform int use_output_profile,
                      uniform StructuredBuffer<float> export_matrix_in, uniform StructuredBuffer<float> export_matrix_out,
                      uniform float norm_min, uniform float norm_max)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float4 i = in.readpixel(x, y);
  const Spline sp = make_spline(M1, M2, M3, M4, M5, latitude_min, latitude_max, type_1, type_2);
  GamutMatrices m;
  m.matrix_in = matrix_in;
  m.matrix_out = matrix_out;
  m.export_matrix_in = export_matrix_in;
  m.export_matrix_out = export_matrix_out;

  float4 o;
  switch(color_science)
  {
    case DT_FILMIC_COLORSCIENCE_V1:
      o = filmic_chroma_v1(i, dynamic_range, black_exposure, grey_value, profile_info[0], lut, use_work_profile,
                           sigma_toe, sigma_shoulder, saturation, sp, output_power, variant);
      break;
    case DT_FILMIC_COLORSCIENCE_V2:
    case DT_FILMIC_COLORSCIENCE_V3:
      o = filmic_chroma_v2_v3(i, dynamic_range, black_exposure, grey_value, profile_info[0], lut,
                              use_work_profile, sigma_toe, sigma_shoulder, saturation, sp, output_power, variant,
                              color_science);
      break;
    case DT_FILMIC_COLORSCIENCE_V4:
      o = filmic_chroma_v4(i, dynamic_range, black_exposure, grey_value, profile_info[0], lut, use_work_profile,
                           saturation, sp, output_power, variant, m, display_black, display_white,
                           use_output_profile, norm_min, norm_max);
      break;
    default:
      o = filmic_chroma_v5(i, dynamic_range, black_exposure, grey_value, saturation, sp, output_power, m,
                           display_black, display_white, use_output_profile, norm_min, norm_max);
      break;
  }

  o.w = i.w;
  out.write(int2(x, y), o);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void filmic_mask_clipped_pixels(uint3 gid : SV_DispatchThreadID,
                                uniform ImageIn in, uniform ImageOut out,
                                uniform int width, uniform int height,
                                uniform float normalize, uniform float feathering,
                                uniform RWStructuredBuffer<int> clipped)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float4 i = in.Areadpixel(x, y);
  const float4 i2 = i * i;

  const float pix_max = dtcl_sqrt(i2.x + i2.y + i2.z);
  const float argument = -pix_max * normalize + feathering;
  const float weight = clipf(1.0f / (1.0f + dtcl_exp2(argument)));

  if(4.f > argument) InterlockedAdd(clipped[0], 1);
  out.write(int2(x, y), float4(weight));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void filmic_show_mask(uint3 gid : SV_DispatchThreadID,
                      uniform ImageIn in, uniform ImageOut out,
                      uniform int width, uniform int height)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float i = in.readsingle(x, y);
  out.write(int2(x, y), float4(i, i, i, 1.0f));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void filmic_inpaint_noise(uint3 gid : SV_DispatchThreadID,
                          uniform ImageIn in, uniform ImageIn mask, uniform ImageOut out,
                          uniform int width, uniform int height, uniform float noise_level,
                          uniform float threshold, uniform int noise_distribution)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  // init random number generator, seeded as in filmic.cl
  uint state[4] = { splitmix32(uint64_t(uint(x + 1))), splitmix32(uint64_t(uint((x + 1) * (y + 3)))),
                    splitmix32(1337ULL), splitmix32(666ULL) };
  xoshiro128plus(state);
  xoshiro128plus(state);
  xoshiro128plus(state);
  xoshiro128plus(state);

  // create noise
  const float4 i = in.readpixel(x, y);
  const float4 sigma = i * noise_level / threshold;
  const float4 noise = dt_noise_generator_simd(noise_distribution, i, sigma, state);
  const float weight = mask.readsingle(x, y);
  const float4 o = max(i * (1.0f - weight) + weight * noise, float4(0.f));
  out.write(int2(x, y), o);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void init_reconstruct(uint3 gid : SV_DispatchThreadID,
                      uniform ImageIn in, uniform ImageIn mask, uniform ImageOut out,
                      uniform int width, uniform int height)
{
  // init the reconstructed buffer with non-clipped and partially clipped
  // pixels: a multiplied alpha blending where mask = alpha weight
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float4 i = in.readpixel(x, y);
  const float4 weight = float4(1.f - mask.readsingle(x, y));
  float4 o = max(i * weight, float4(0.f));

  // copy masks and alpha
  o.w = i.w;
  out.write(int2(x, y), o);
}

float fmaxabsf(float a, float b)
{
  // the max in absolute value, with its sign
  return (abs(a) > abs(b) && !isnan(a)) ? a : (isnan(b)) ? 0.f : b;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void wavelets_reconstruct(uint3 gid : SV_DispatchThreadID,
                          uniform ImageIn HF, uniform ImageIn LF, uniform ImageIn texture,
                          uniform ImageIn mask,
                          uniform ImageIn reconstructed_read, uniform ImageOut reconstructed_write,
                          uniform int width, uniform int height,
                          uniform float gamma, uniform float gamma_comp, uniform float beta, uniform float beta_comp,
                          uniform float delta,
                          uniform int s, uniform int scales, uniform int variant)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float alpha = mask.readsingle(x, y);
  const float4 HF_c = HF.readpixel(x, y);
  const float4 LF_c = LF.readpixel(x, y);
  const float4 TT_c = texture.readpixel(x, y);

  const float grey_texture = fmaxabsf(fmaxabsf(TT_c.x, TT_c.y), TT_c.z);
  const float grey_details = (HF_c.x + HF_c.y + HF_c.z) / 3.f;

  float4 details;
  float4 residual;

  if(variant == DT_FILMIC_RECONSTRUCT_RGB)
  {
    const float grey_HF = beta_comp * (gamma_comp * grey_details + gamma * grey_texture);
    const float grey_residual = beta_comp * (LF_c.x + LF_c.y + LF_c.z) / 3.f;

    details = (gamma_comp * HF_c + gamma * TT_c) * beta + grey_HF;
    residual = (s == scales - 1) ? grey_residual + LF_c * beta : float4(0.f);
  }
  else
  {
    const float grey_HF = (gamma_comp * grey_details + gamma * grey_texture);

    details = 0.5f * ((gamma_comp * HF_c + gamma * TT_c) + grey_HF);
    residual = (s == scales - 1) ? LF_c : float4(0.f);
  }

  const float4 i = reconstructed_read.readpixel(x, y);
  const float4 o = i + alpha * (delta * details + residual);
  reconstructed_write.write(int2(x, y), o);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void compute_ratios(uint3 gid : SV_DispatchThreadID,
                    uniform ImageIn in, uniform ImageOut norms, uniform ImageOut ratios,
                    uniform int variant, uniform int width, uniform int height)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  // variant is unused, as in OpenCL, but Metal drops parameters nothing reads
  if(width < 0) ratios.write(int2(0), float4(float(variant)));

  const float4 i = in.readpixel(x, y);
  const float norm = max(pixel_rgb_norm_euclidean(i), NORM_MIN);
  const float4 ratio = i / norm;
  norms.write(int2(x, y), float4(norm));
  ratios.write(int2(x, y), ratio);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void restore_ratios(uint3 gid : SV_DispatchThreadID,
                    uniform ImageIn ratios, uniform ImageIn norms, uniform ImageOut out,
                    uniform int width, uniform int height)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float4 ratio = ratios.readpixel(x, y);
  const float norm = norms.readsingle(x, y);
  const float4 o = clip4(ratio) * norm;
  out.write(int2(x, y), o);
}

// the legacy filmic module, the first kernel of ../filmic.cl. It takes no
// contrast argument, which the OpenCL kernel has but does not read
[shader("compute")]
[numthreads(16, 16, 1)]
void filmic(uint3 gid : SV_DispatchThreadID,
            uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
            uniform float dynamic_range, uniform float shadows_range, uniform float grey,
            uniform ImageIn table, uniform ImageIn diff,
            uniform float power, uniform int preserve_color, uniform float saturation)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  float4 i = in.readpixel(x, y);
  const float4 xyz = Lab_to_XYZ(i);
  float4 o = XYZ_to_prophotorgb(xyz);

  const float noise = 1.0f / 65536.0f;
  float derivative, luma;

  // global desaturation
  if(saturation != 1.0f)
    o = float4(xyz.y) + saturation * (o - float4(xyz.y));

  if(preserve_color != 0)
  {
    // save the ratios
    float maxRGB = fmax3(o);
    const float4 ratios = o / maxRGB;

    // log profile
    maxRGB = maxRGB / grey;
    maxRGB = (maxRGB < noise) ? noise : maxRGB;
    maxRGB = (dtcl_log2(maxRGB) - shadows_range) / dynamic_range;
    maxRGB = clipf(maxRGB);

    const float index = maxRGB;

    // curve S LUT
    maxRGB = lookup(table, maxRGB);

    // re-apply the ratios
    o = maxRGB * ratios;

    derivative = lookup(diff, index);
    luma = maxRGB;
  }
  else
  {
    // log profile
    o = o / grey;
    o = select(o < float4(noise), float4(noise), o);
    o = (dtcl_log2(o) - shadows_range) / dynamic_range;
    o = clip4(o);

    const float index = prophotorgb_to_XYZ(o).y;

    // curve S LUT
    o.x = lookup(table, o.x);
    o.y = lookup(table, o.y);
    o.z = lookup(table, o.z);

    derivative = lookup(diff, index);
    luma = prophotorgb_to_XYZ(o).y;
  }

  // desaturate selectively
  o = float4(luma) + derivative * (o - float4(luma));
  o = clip4(o);

  // apply the transfer function of the display
  o = dtcl_pow(o, power);

  i.xyz = prophotorgb_to_Lab(o).xyz;
  out.write(int2(x, y), i);
}
