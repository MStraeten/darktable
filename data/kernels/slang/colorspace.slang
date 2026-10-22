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

// Slang module form of ../colorspace.h. Only the conversions used by kernels
// that have been ported so far are here; the rest follow with their kernels.

module colorspace;

__exported import common;

public float4 matrix_dot(float4 vector, float4 matrix[3])
{
  float4 output;
  const float4 vector_copy = float4(vector.xyz, 0.f);
  output.x = dot(vector_copy, matrix[0]);
  output.y = dot(vector_copy, matrix[1]);
  output.z = dot(vector_copy, matrix[2]);
  output.w = vector.w;
  return output;
}

public float4 matrix_product(float4 xyz, float matrix[9])
{
  const float R = matrix[0] * xyz.x + matrix[1] * xyz.y + matrix[2] * xyz.z;
  const float G = matrix[3] * xyz.x + matrix[4] * xyz.y + matrix[5] * xyz.z;
  const float B = matrix[6] * xyz.x + matrix[7] * xyz.y + matrix[8] * xyz.z;
  const float a = xyz.w;
  return float4(R, G, B, a);
}

// same as above but with 4×float padded matrix
public float4 matrix_product_float4(float4 xyz, float matrix[12])
{
  const float R = matrix[0] * xyz.x + matrix[1] * xyz.y + matrix[2]  * xyz.z;
  const float G = matrix[4] * xyz.x + matrix[5] * xyz.y + matrix[6]  * xyz.z;
  const float B = matrix[8] * xyz.x + matrix[9] * xyz.y + matrix[10] * xyz.z;
  const float a = xyz.w;
  return float4(R, G, B, a);
}

public float4 Lab_2_LCH(float4 Lab)
{
  float H = atan2(Lab.z, Lab.y);

  H = (H > 0.0f) ? H / DT_2PI_F : 1.0f - abs(H) / DT_2PI_F;

  const float L = Lab.x;
  const float C = dt_fast_hypot(Lab.y, Lab.z);

  return float4(L, C, H, Lab.w);
}

public float4 LCH_2_Lab(float4 LCH)
{
  const float L = LCH.x;
  const float a = dtcl_cos(DT_2PI_F*LCH.z) * LCH.y;
  const float b = dtcl_sin(DT_2PI_F*LCH.z) * LCH.y;

  return float4(L, a, b, LCH.w);
}

public float4 lab_f(float4 x)
{
  const float4 epsilon = 216.0f / 24389.0f;
  const float4 kappa = 24389.0f / 27.0f;
  return select(x > epsilon, dt_cbrt(x), (kappa * x + float4(16.0f)) / float4(116.0f));
}

public float4 XYZ_to_Lab(float4 xyz)
{
  float4 lab = float4(0.0f);
  const float4 d50i = float4(1.0f / 0.9642f, 1.0f, 1.0f / 0.8249f, 1.0f);
  xyz = lab_f(xyz * d50i);
  lab.x = 116.0f * xyz.y - 16.0f;
  lab.y = 500.0f * (xyz.x - xyz.y);
  lab.z = 200.0f * (xyz.y - xyz.z);

  return lab;
}

public float4 lab_f_inv(float4 x)
{
  const float4 epsilon = 0.20689655172413796f; // cbrtf(216.0f/24389.0f)
  const float4 kappa   = 24389.0f / 27.0f;
  return select(x > epsilon, x*x*x, (float4(116.0f) * x - float4(16.0f))/kappa);
}

public float4 Lab_to_XYZ(float4 Lab)
{
  const float4 d50 = float4(0.9642f, 1.0f, 0.8249f, 0.0f);
  float4 f = float4(0.0f);
  f.y = (Lab.x + 16.0f)/116.0f;
  f.x = Lab.y/500.0f + f.y;
  f.z = f.y - Lab.z/200.0f;
  return d50 * lab_f_inv(f);
}

// same as matrix_product_float4, for a matrix the host passes as a buffer
public float4 matrix_product_float4(float4 xyz, StructuredBuffer<float> matrix)
{
  const float R = matrix[0] * xyz.x + matrix[1] * xyz.y + matrix[2]  * xyz.z;
  const float G = matrix[4] * xyz.x + matrix[5] * xyz.y + matrix[6]  * xyz.z;
  const float B = matrix[8] * xyz.x + matrix[9] * xyz.y + matrix[10] * xyz.z;
  const float a = xyz.w;
  return float4(R, G, B, a);
}

public float4 prophotorgb_to_XYZ(float4 rgb)
{
  const float rgb_to_xyz[3][3] = { // prophoto rgb
    {0.7976749f, 0.1351917f, 0.0313534f},
    {0.2880402f, 0.7118741f, 0.0000857f},
    {0.0000000f, 0.0000000f, 0.8252100f},
  };
  float4 XYZ = float4(0.0f, 0.0f, 0.0f, 0.0f);
  XYZ.x += rgb_to_xyz[0][0] * rgb.x;
  XYZ.x += rgb_to_xyz[0][1] * rgb.y;
  XYZ.x += rgb_to_xyz[0][2] * rgb.z;
  XYZ.y += rgb_to_xyz[1][0] * rgb.x;
  XYZ.y += rgb_to_xyz[1][1] * rgb.y;
  XYZ.y += rgb_to_xyz[1][2] * rgb.z;
  XYZ.z += rgb_to_xyz[2][0] * rgb.x;
  XYZ.z += rgb_to_xyz[2][1] * rgb.y;
  XYZ.z += rgb_to_xyz[2][2] * rgb.z;
  return XYZ;
}

public float4 XYZ_to_prophotorgb(float4 XYZ)
{
  const float xyz_to_rgb[3][3] = { // prophoto rgb d50
    { 1.3459433f, -0.2556075f, -0.0511118f},
    {-0.5445989f,  1.5081673f,  0.0205351f},
    { 0.0000000f,  0.0000000f,  1.2118128f},
  };
  float4 rgb = float4(0.0f, 0.0f, 0.0f, 0.0f);
  rgb.x += xyz_to_rgb[0][0] * XYZ.x;
  rgb.x += xyz_to_rgb[0][1] * XYZ.y;
  rgb.x += xyz_to_rgb[0][2] * XYZ.z;
  rgb.y += xyz_to_rgb[1][0] * XYZ.x;
  rgb.y += xyz_to_rgb[1][1] * XYZ.y;
  rgb.y += xyz_to_rgb[1][2] * XYZ.z;
  rgb.z += xyz_to_rgb[2][0] * XYZ.x;
  rgb.z += xyz_to_rgb[2][1] * XYZ.y;
  rgb.z += xyz_to_rgb[2][2] * XYZ.z;
  return rgb;
}

public float4 Lab_to_prophotorgb(float4 Lab)
{
  const float4 XYZ = Lab_to_XYZ(Lab);
  return XYZ_to_prophotorgb(XYZ);
}

public float4 prophotorgb_to_Lab(float4 rgb)
{
  const float4 XYZ = prophotorgb_to_XYZ(rgb);
  return XYZ_to_Lab(XYZ);
}

public float4 RGB_2_HSV(float4 RGB)
{
  float4 HSV = float4(0.0f);

  const float minv = min(RGB.x, min(RGB.y, RGB.z));
  const float maxv = max(RGB.x, max(RGB.y, RGB.z));
  const float delta = maxv - minv;

  HSV.z = maxv;
  HSV.w = RGB.w;

  if(abs(maxv) > 1e-6f && abs(delta) > 1e-6f)
  {
    HSV.y = delta / maxv;
  }
  else
  {
    HSV.x = 0.0f;
    HSV.y = 0.0f;
    return HSV;
  }

  if (RGB.x == maxv)
   HSV.x = (RGB.y - RGB.z) / delta;
  else if (RGB.y == maxv)
   HSV.x = 2.0f + (RGB.z - RGB.x) / delta;
  else
   HSV.x = 4.0f + (RGB.x - RGB.y) / delta;

  HSV.x /= 6.0f;

  HSV.x -= floor(HSV.x);

  return HSV;
}

public float4 HSV_2_RGB(float4 HSV)
{
  float4 RGB;

  if (abs(HSV.y) < 1e-6f)
  {
    RGB.x = RGB.y = RGB.z = HSV.z;
    RGB.w = HSV.w;
    return RGB;
  }

  const int i = int(floor(6.0f*HSV.x));
  const float v = HSV.z;
  const float w = HSV.w;
  const float p = v * (1.0f - HSV.y);
  const float q = v * (1.0f - HSV.y * (6.0f*HSV.x - i));
  const float t = v * (1.0f - HSV.y * (1.0f - (6.0f*HSV.x - i)));

  switch (i)
  {
    case 0:
      RGB = float4(v, t, p, w);
      break;
    case 1:
      RGB = float4(q, v, p, w);
      break;
    case 2:
      RGB = float4(p, v, t, w);
      break;
    case 3:
      RGB = float4(p, q, v, w);
      break;
    case 4:
      RGB = float4(t, p, v, w);
      break;
    case 5:
    default:
      RGB = float4(v, p, q, w);
      break;
  }
  return RGB;
}

public float4 dt_xyY_to_uvY(float4 xyY)
{
  // This is the linear part of the chromaticity transform from CIE L*u*v* e.g. u'v'.
  // See https://en.wikipedia.org/wiki/CIELUV
  // It rescales the chromaticity diagram xyY in a more perceptual way,
  // but it is still not hue-linear and not perfectly perceptual.
  // As such, it is the only radiometricly-accurate representation of hue non-linearity in human vision system.
  // Use it for "hue preserving" (as much as possible) gamut mapping in scene-referred space
  const float denominator = -2.f * xyY.x + 12.f * xyY.y + 3.f;
  float4 uvY;
  uvY.x = 4.f * xyY.x / denominator; // u'
  uvY.y = 9.f * xyY.y / denominator; // v'
  uvY.z = xyY.z;                     // Y
  uvY.w = xyY.w;
  return uvY;
}

public float4 dt_uvY_to_xyY(float4 uvY)
{
  // This is the linear part of chromaticity transform from CIE L*u*v* e.g. u'v'.
  // See https://en.wikipedia.org/wiki/CIELUV
  // It rescales the chromaticity diagram xyY in a more perceptual way,
  // but it is still not hue-linear and not perfectly perceptual.
  // As such, it is the only radiometricly-accurate representation of hue non-linearity in human vision system.
  // Use it for "hue preserving" (as much as possible) gamut mapping in scene-referred space
  const float denominator = 6.0f * uvY.x - 16.f * uvY.y + 12.0f;
  float4 xyY;
  xyY.x = 9.f * uvY.x / denominator; // x
  xyY.y = 4.f * uvY.y / denominator; // y
  xyY.z = uvY.z;                     // Y
  xyY.w = uvY.w;
  return xyY;
}

public float4 dt_xyY_to_XYZ(float4 xyY)
{
  // see cpu implementation for details
  float4 XYZ = float4(0.0f);
  if(xyY.y != 0.0f)
  {
    XYZ.x = xyY.z * xyY.x / xyY.y;
    XYZ.y = xyY.z;
    XYZ.z = xyY.z * (1.f - xyY.x - xyY.y) / xyY.y;
  }
  XYZ.w = xyY.w;
  return XYZ;
}

// port src/common/chromatic_adaptation.h

public float4 convert_XYZ_to_bradford_LMS(float4 XYZ)
{
  // Warning : needs XYZ normalized with Y - you need to downscale before
  const float4 XYZ_to_Bradford_LMS[3] = { {  0.8951f,  0.2664f, -0.1614f, 0.f },
                                          { -0.7502f,  1.7135f,  0.0367f, 0.f },
                                          {  0.0389f, -0.0685f,  1.0296f, 0.f } };

  return matrix_dot(XYZ, XYZ_to_Bradford_LMS);
}

public float4 convert_bradford_LMS_to_XYZ(float4 LMS)
{
  // Warning : output XYZ normalized with Y - you need to upscale later
  const float4 Bradford_LMS_to_XYZ[3] = { {  0.9870f, -0.1471f,  0.1600f, 0.f },
                                          {  0.4323f,  0.5184f,  0.0493f, 0.f },
                                          { -0.0085f,  0.0400f,  0.9685f, 0.f } };

  return matrix_dot(LMS, Bradford_LMS_to_XYZ);
}

public float4 convert_XYZ_to_CAT16_LMS(float4 XYZ)
{
  // Warning : needs XYZ normalized with Y - you need to downscale before
  const float4 XYZ_to_CAT16_LMS[3] = { {  0.401288f, 0.650173f, -0.051461f, 0.f },
                                       { -0.250268f, 1.204414f,  0.045854f, 0.f },
                                       { -0.002079f, 0.048952f,  0.953127f, 0.f } };

  return matrix_dot(XYZ, XYZ_to_CAT16_LMS);
}

public float4 convert_CAT16_LMS_to_XYZ(float4 LMS)
{
  // Warning : output XYZ normalized with Y - you need to upscale later
  const float4 CAT16_LMS_to_XYZ[3] = { {  1.862068f, -1.011255f,  0.149187f, 0.f },
                                       {  0.38752f ,  0.621447f, -0.008974f, 0.f },
                                       { -0.015841f, -0.034123f,  1.049964f, 0.f } };

  return matrix_dot(LMS, CAT16_LMS_to_XYZ);
}

public void bradford_adapt_D50(inout float4 lms_in,
                               float4 origin_illuminant,
                               float p, int full)
{
  // Bradford chromatic adaptation from origin to target D50 illuminant in LMS space
  // p = powf(origin_illuminant[2] / D50[2], 0.0834f) needs to be precomputed for performance,
  // since it is independent from current pixel values
  // origin illuminant need also to be precomputed to LMS

  // Precomputed D50 primaries in Bradford LMS for ICC transforms
  const float4 D50 = { 0.996078f, 1.020646f, 0.818155f, 0.f };

  if(full != 0)
  {
    float4 temp = lms_in / origin_illuminant;

    // use linear Bradford if B is negative
    temp.z = (temp.z > 0.f) ? dtcl_pow(temp.z, p) : temp.z;

    lms_in = D50 * temp;
  }
  else
    lms_in *= D50 / origin_illuminant;
}

public void CAT16_adapt_D50(inout float4 lms_in,
                            float4 origin_illuminant,
                            float D, int full)
{
  // CAT16 chromatic adaptation from origin to target D50 illuminant in LMS space
  // D is the coefficient of adaptation, depending of the surround lighting
  // origin illuminant need also to be precomputed to LMS

  // Precomputed D50 primaries in CAT16 LMS for ICC transforms
  const float4 D50 = { 0.994535f, 1.000997f, 0.833036f, 0.f };

  if(full != 0) lms_in *= D50 / origin_illuminant;
  else lms_in *= (D * D50 / origin_illuminant + 1.f - D);
}

public void XYZ_adapt_D50(inout float4 lms_in,
                          float4 origin_illuminant)
{
  // XYZ chromatic adaptation from origin to target D65 illuminant in XYZ space
  // origin illuminant need also to be precomputed to XYZ

  // Precomputed D50 primaries in XYZ for camera WB adjustment
  const float4 D50 = { 0.9642119944211994f, 1.0f, 0.8251882845188288f, 0.f };
  lms_in *= D50 / origin_illuminant;
}
