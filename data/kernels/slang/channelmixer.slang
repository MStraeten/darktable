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

// Slang port of ../channelmixer.cl

import colorspace;

// dt_iop_channelmixer_rgb_version_t
static const int CHANNELMIXERRGB_V_1 = 0;
static const int CHANNELMIXERRGB_V_2 = 1;
static const int CHANNELMIXERRGB_V_3 = 2;

// dt_adaptation_t
static const int DT_ADAPTATION_LINEAR_BRADFORD = 0;
static const int DT_ADAPTATION_CAT16           = 1;
static const int DT_ADAPTATION_FULL_BRADFORD   = 2;
static const int DT_ADAPTATION_XYZ             = 3;
static const int DT_ADAPTATION_RGB             = 4;

static const float INVERSE_SQRT_3 = 0.5773502691896258f;

float sqf(float x)
{
  return x * x;
}

float euclidean_norm(float4 input)
{
  return max(dtcl_sqrt(sqf(input.x) + sqf(input.y) + sqf(input.z)), NORM_MIN);
}

float4 gamut_mapping(float4 input, float compression, int clip)
{
  // Get the sum XYZ
  const float sum = input.x + input.y + input.z;
  const float Y = input.y;

  float4 output;
  float4 xyY = {  sum > 0.0f ? input.x / sum : 0.34567f,
                  sum > 0.0f ? input.y / sum : 0.35850f,
                  Y,
                  0.0f };

  // Convert to uvY
  float4 uvY = dt_xyY_to_uvY(xyY);

  // Get the chromaticity difference with white point uv
  const float2 D50 = { 0.20915914598542354f, 0.488075320769787f };
  const float2 delta = D50 - uvY.xy;
  const float Delta = Y * (sqf(delta.x) + sqf(delta.y));

  // Compress chromaticity (move toward white point)
  const float correction = (compression == 0.0f) ? 0.f : dtcl_pow(Delta, compression);

  // Ensure the correction does not bring our uyY vector the other side of D50
  // that would switch to the opposite color, so we clip at D50
  const float2 tmp = correction * delta + uvY.xy;
  uvY.xy = select(uvY.xy > D50, max(tmp, D50), min(tmp, D50));

  // Convert back to xyY
  xyY = dt_uvY_to_xyY(uvY);

  // Clip upon request
  if(clip != 0) xyY.xy = max(xyY.xy, float2(0.0f));

  // Check sanity of y
  // since we later divide by y, it can't be zero
  xyY.y = max(xyY.y, NORM_MIN);

  // Check sanity of x and y :
  // since Z = Y (1 - x - y) / y, if x + y >= 1, Z will be negative
  const float scale = xyY.x + xyY.y;
  const bool sanitize = (scale >= 1.f);
  xyY.xy = sanitize ? xyY.xy / scale : xyY.xy;

  // Convert back to XYZ
  output = dt_xyY_to_XYZ(xyY);
  return output;
}

float4 luma_chroma(float4 input, float4 saturation, float4 lightness, int version)
{
  float4 output;

  // Compute euclidean norm
  float norm = euclidean_norm(input);
  const float avg = max((input.x + input.y + input.z) / 3.0f, NORM_MIN);

  if(norm > 0.f && avg > 0.f)
  {
    // Compute flat lightness adjustment
    const float mix = dot(input, lightness);

    // Compensate the norm to get color ratios (R, G, B) = (1, 1, 1) for grey (colorless) pixels.
    if(version == CHANNELMIXERRGB_V_3) norm *= INVERSE_SQRT_3;

    // Ratios
    // WARNING : dot product below uses all 4 channels, you need to make sure
    // input.w != NaN since saturation.w = 0.f
    output = input / norm;

    // Compute ratios and a flat colorfulness adjustment for the whole pixel
    float coeff_ratio = 0.f;

    if(version == CHANNELMIXERRGB_V_1)
      coeff_ratio = dot((1.f - output) * (1.f - output), saturation);
    else
      coeff_ratio = dot(output, saturation) / 3.f;

    // Adjust the RGB ratios with the pixel correction

    // if the ratio was already invalid (negative), we accept the result to be invalid too
    // otherwise bright saturated blues end up solid black
    const float4 min_ratio = select(output < 0.0f, output, float4(0.0f));
    const float4 output_inverse = 1.0f - output;
    output = max(output_inverse * coeff_ratio + output, min_ratio);

    // The above interpolation between original pixel ratios and (1, 1, 1) might change the norm of the
    // ratios. Compensate for that.
    if(version == CHANNELMIXERRGB_V_3) norm /= euclidean_norm(output) * INVERSE_SQRT_3;

    // Apply colorfulness adjustment channel-wise and repack with lightness to get LMS back
    norm *= max(1.f + mix / avg, 0.f);
    output *= norm;
  }
  else
  {
    // we have black, 0 stays 0, no luminance = no color
    output = input;
  }

  return output;
}

float4 convert_any_LMS_to_XYZ(float4 LMS, int kind, StructuredBuffer<float> RGB_to_XYZ)
{
  switch(kind)
  {
    case DT_ADAPTATION_FULL_BRADFORD:
    case DT_ADAPTATION_LINEAR_BRADFORD:
      return convert_bradford_LMS_to_XYZ(LMS);
    case DT_ADAPTATION_CAT16:
      return convert_CAT16_LMS_to_XYZ(LMS);
    case DT_ADAPTATION_XYZ:
      return LMS;
    default:
      return matrix_product_float4(LMS, RGB_to_XYZ);
  }
}

float4 convert_XYZ_to_any_LMS(float4 XYZ, int kind, StructuredBuffer<float> XYZ_to_RGB)
{
  switch(kind)
  {
    case DT_ADAPTATION_FULL_BRADFORD:
    case DT_ADAPTATION_LINEAR_BRADFORD:
      return convert_XYZ_to_bradford_LMS(XYZ);
    case DT_ADAPTATION_CAT16:
      return convert_XYZ_to_CAT16_LMS(XYZ);
    case DT_ADAPTATION_XYZ:
      return XYZ;
    default:
      return matrix_product_float4(XYZ, XYZ_to_RGB);
  }
}

void downscale_vector(inout float4 vector, float scaling)
{
  const bool valid = (scaling > NORM_MIN) && !isnan(scaling);
  vector /= valid ? (scaling + NORM_MIN) : NORM_MIN;
}

void upscale_vector(inout float4 vector, float scaling)
{
  const bool valid = (scaling > NORM_MIN) && !isnan(scaling);
  vector *= valid ? (scaling + NORM_MIN) : NORM_MIN;
}

float4 chroma_adapt_bradford(float4 RGB, StructuredBuffer<float> RGB_to_XYZ, StructuredBuffer<float> MIX,
                             float4 illuminant, float p, int full)
{
  // Convert from RGB to XYZ
  const float4 XYZ = matrix_product_float4(RGB, RGB_to_XYZ);
  const float Y = XYZ.y;

  // Convert to LMS
  float4 LMS = convert_XYZ_to_bradford_LMS(XYZ);

  // Do white balance
  downscale_vector(LMS, Y);
  bradford_adapt_D50(LMS, illuminant, p, full);
  upscale_vector(LMS, Y);

  // Compute the 3D mix - this is a rotation + homothety of the vector base
  const float4 LMS_mixed = matrix_product_float4(LMS, MIX);

  return convert_bradford_LMS_to_XYZ(LMS_mixed);
}

float4 chroma_adapt_CAT16(float4 RGB, StructuredBuffer<float> RGB_to_XYZ, StructuredBuffer<float> MIX,
                          float4 illuminant, float p, int full)
{
  // Convert from RGB to XYZ
  const float4 XYZ = matrix_product_float4(RGB, RGB_to_XYZ);
  const float Y = XYZ.y;

  // Convert to LMS
  float4 LMS = convert_XYZ_to_CAT16_LMS(XYZ);

  // Do white balance
  downscale_vector(LMS, Y);
  CAT16_adapt_D50(LMS, illuminant, p, full); // force full-adaptation
  upscale_vector(LMS, Y);

  // Compute the 3D mix - this is a rotation + homothety of the vector base
  const float4 LMS_mixed = matrix_product_float4(LMS, MIX);

  return convert_CAT16_LMS_to_XYZ(LMS_mixed);
}

float4 chroma_adapt_XYZ(float4 RGB, StructuredBuffer<float> RGB_to_XYZ, StructuredBuffer<float> MIX,
                        float4 illuminant)
{
  // Convert from RGB to XYZ
  float4 XYZ_mixed = matrix_product_float4(RGB, RGB_to_XYZ);
  const float Y = XYZ_mixed.y;

  // Do white balance in XYZ
  downscale_vector(XYZ_mixed, Y);
  XYZ_adapt_D50(XYZ_mixed, illuminant);
  upscale_vector(XYZ_mixed, Y);

  // Compute the 3D mix in XYZ - this is a rotation + homothety of the vector base
  return matrix_product_float4(XYZ_mixed, MIX);
}

float4 chroma_adapt_RGB(float4 RGB, StructuredBuffer<float> RGB_to_XYZ, StructuredBuffer<float> MIX)
{
  // No white balance.

  // Compute the 3D mix in RGB - this is a rotation + homothety of the vector base
  float4 RGB_mixed = matrix_product_float4(RGB, MIX);

  // Convert from RGB to XYZ
  return matrix_product_float4(RGB_mixed, RGB_to_XYZ);
}

float4 chroma_adapt(float4 RGB, int kind, StructuredBuffer<float> RGB_to_XYZ, StructuredBuffer<float> MIX,
                    float4 illuminant, float p)
{
  switch(kind)
  {
    case DT_ADAPTATION_FULL_BRADFORD:
      return chroma_adapt_bradford(RGB, RGB_to_XYZ, MIX, illuminant, p, 1);
    case DT_ADAPTATION_LINEAR_BRADFORD:
      return chroma_adapt_bradford(RGB, RGB_to_XYZ, MIX, illuminant, p, 0);
    case DT_ADAPTATION_CAT16:
      return chroma_adapt_CAT16(RGB, RGB_to_XYZ, MIX, illuminant, 1.f, 0);
    case DT_ADAPTATION_XYZ:
      return chroma_adapt_XYZ(RGB, RGB_to_XYZ, MIX, illuminant);
    default:
      return chroma_adapt_RGB(RGB, RGB_to_XYZ, MIX);
  }
}

// The OpenCL source duplicates one kernel per adaptation, with switch macros,
// to take the switches out of the pixel loop. Here each entry point passes
// a constant kind, and the compiler drops the other branches.
float4 _channelmixerrgb(int kind, float4 pix_in, StructuredBuffer<float> RGB_to_XYZ,
                        StructuredBuffer<float> XYZ_to_RGB, StructuredBuffer<float> MIX,
                        float4 illuminant, float4 saturation, float4 lightness, float4 grey,
                        float p, float gamut, int clip, int apply_grey, int version)
{
  float4 XYZ, LMS;
  float4 RGB = pix_in;
  RGB.w = 0.f;

  if(clip != 0) RGB = max(RGB, float4(0.f));

  /* WE START IN PIPELINE RGB */
  XYZ = chroma_adapt(RGB, kind, RGB_to_XYZ, MIX, illuminant, p);

  /* FROM HERE WE ARE MANDATORILY IN XYZ - DATA IS IN temp_one */

  // Gamut mapping happens in XYZ space no matter what
  if(clip != 0) XYZ = max(XYZ, float4(0.0f));
  XYZ = gamut_mapping(XYZ, gamut, clip);

  // convert to LMS, XYZ or pipeline RGB
  LMS = convert_XYZ_to_any_LMS(XYZ, kind, XYZ_to_RGB);

  /* FROM HERE WE ARE IN LMS, XYZ OR PIPELINE RGB depending on user param */

  // Clip in LMS
  if(clip != 0) LMS = max(LMS, float4(0.0f));

  // Apply lightness / saturation adjustment
  LMS = luma_chroma(LMS, saturation, lightness, version);

  // Clip in LMS
  if(clip != 0) LMS = max(LMS, float4(0.0f));

  // Save
  if(apply_grey != 0)
  {
    // Turn LMS, XYZ or pipeline RGB into monochrome
    const float grey_mix = max(dot(LMS, grey), 0.0f);

    RGB.xyz = grey_mix;
    RGB.w = pix_in.w; // alpha mask
  }
  else
  {
    // Convert back to XYZ
    XYZ = convert_any_LMS_to_XYZ(LMS, kind, RGB_to_XYZ);

    /* FROM HERE WE ARE MANDATORILY IN XYZ */

    // Clip in XYZ
    if(clip != 0) XYZ = max(XYZ, float4(0.0f));

    // Convert back to RGB
    RGB = matrix_product_float4(XYZ, XYZ_to_RGB);

    if(clip != 0) RGB = max(RGB, float4(0.f));
    RGB.w = pix_in.w;
  }

  return RGB;
}

#define CHANNELMIXERRGB_KERNEL(name, kind)                                                             \
  [shader("compute")]                                                                                  \
  [numthreads(16, 16, 1)]                                                                              \
  void name(uint3 gid : SV_DispatchThreadID,                                                           \
            uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,           \
            uniform StructuredBuffer<float> RGB_to_XYZ, uniform StructuredBuffer<float> XYZ_to_RGB,    \
            uniform StructuredBuffer<float> MIX, uniform float4 illuminant, uniform float4 saturation, \
            uniform float4 lightness, uniform float4 grey, uniform float p, uniform float gamut,       \
            uniform int clip, uniform int apply_grey, uniform int version)                             \
  {                                                                                                    \
    const int x = gid.x;                                                                               \
    const int y = gid.y;                                                                               \
    if(x >= width || y >= height) return;                                                              \
    out.write(int2(x, y), _channelmixerrgb(kind, in.read(int2(x, y)), RGB_to_XYZ, XYZ_to_RGB, MIX,     \
                                           illuminant, saturation, lightness, grey, p, gamut, clip,    \
                                           apply_grey, version));                                      \
  }

CHANNELMIXERRGB_KERNEL(channelmixerrgb_CAT16, DT_ADAPTATION_CAT16)
CHANNELMIXERRGB_KERNEL(channelmixerrgb_bradford_linear, DT_ADAPTATION_LINEAR_BRADFORD)
CHANNELMIXERRGB_KERNEL(channelmixerrgb_bradford_full, DT_ADAPTATION_FULL_BRADFORD)
CHANNELMIXERRGB_KERNEL(channelmixerrgb_XYZ, DT_ADAPTATION_XYZ)
CHANNELMIXERRGB_KERNEL(channelmixerrgb_RGB, DT_ADAPTATION_RGB)
