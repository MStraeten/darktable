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

public float4 XYZ_to_JzAzBz(float4 XYZ_D65)
{
  const float4 M[3] = { float4(0.41478972f, 0.579999f, 0.0146480f, 0.0f),
                        float4(-0.2015100f, 1.120649f, 0.0531008f, 0.0f),
                        float4(-0.0166008f, 0.264800f, 0.6684799f, 0.0f) };

  const float4 A[3] = { float4(0.5f, 0.5f, 0.0f, 0.0f),
                        float4(3.524000f, -4.066708f, 0.542708f, 0.0f),
                        float4(0.199076f, 1.096799f, -1.295875f, 0.0f) };

  const float n = 0.159301758f;
  const float p = 134.034375f;
  const float c1 = 0.8359375f;
  const float c2 = 18.8515625f;
  const float c3 = 18.6875f;

  float4 temp1, temp2;
  // XYZ -> X'Y'Z
  temp1.x = 1.15f * XYZ_D65.x - 0.15f * XYZ_D65.z;
  temp1.y = 0.66f * XYZ_D65.y + 0.34f * XYZ_D65.x;
  temp1.z = XYZ_D65.z;
  temp1.w = 0.f;
  // X'Y'Z -> LMS
  temp2.x = dot(M[0], temp1);
  temp2.y = dot(M[1], temp1);
  temp2.z = dot(M[2], temp1);
  temp2.w = 0.f;

  const float l = temp2.x * 1e-4f;
  const float m = temp2.y * 1e-4f;
  const float s = temp2.z * 1e-4f;

  float4 res;
  // Az and Bz are differences of near-equal numbers, computed as such: see
  // XYZ_to_JzAzBz() in ../colorspace.h
  if(l > 0.0f && m > 0.0f && s > 0.0f)
  {
    const float d_lm = dot(M[0] - M[1], temp1) * 1e-4f;
    const float d_sm = dot(M[2] - M[1], temp1) * 1e-4f;
    const float d_ls = dot(M[0] - M[2], temp1) * 1e-4f;

    const float t_l = dtcl_pow(l, n), t_m = dtcl_pow(m, n), t_s = dtcl_pow(s, n);
    const float dt_lm = t_m * dt_expm1(n * dt_log1p(d_lm / m));
    const float dt_sm = t_m * dt_expm1(n * dt_log1p(d_sm / m));
    const float dt_ls = t_s * dt_expm1(n * dt_log1p(d_ls / s));

    const float k = c2 - c1 * c3;
    const float q_l = 1.0f + c3 * t_l, q_m = 1.0f + c3 * t_m, q_s = 1.0f + c3 * t_s;
    const float y_l = (c1 + c2 * t_l) / q_l;
    const float y_m = (c1 + c2 * t_m) / q_m;
    const float y_s = (c1 + c2 * t_s) / q_s;
    const float dy_lm = k * dt_lm / (q_l * q_m);
    const float dy_sm = k * dt_sm / (q_s * q_m);
    const float dy_ls = k * dt_ls / (q_l * q_s);

    const float Mp = dtcl_pow(y_m, p);
    const float Sp = dtcl_pow(y_s, p);
    const float d_LM = Mp * dt_expm1(p * dt_log1p(dy_lm / y_m));
    const float d_SM = Mp * dt_expm1(p * dt_log1p(dy_sm / y_m));
    const float d_LS = Sp * dt_expm1(p * dt_log1p(dy_ls / y_s));

    res.x = 0.5f * dtcl_pow(y_l, p) + 0.5f * Mp;
    res.y = 3.524000f * d_LM + 0.542708f * d_SM;
    res.z = 0.199076f * d_LS - 1.096799f * d_SM;
    res.w = 0.f;
  }
  else
  {
    // LMS -> L'M'S'
    temp2 = dtcl_pow(max(temp2 / 10000.f, float4(0.0f)), n);
    temp2 = dtcl_pow((c1 + c2 * temp2) / (1.0f + c3 * temp2), p);
    // L'M'S' -> Izazbz
    res.x = dot(A[0], temp2);
    res.y = dot(A[1], temp2);
    res.z = dot(A[2], temp2);
    res.w = 0.f;
  }
  // Iz -> Jz
  res.x = max(0.44f * res.x / (1.0f - 0.56f * res.x) - 1.6295499532821566e-11f, 0.f);
  return res;
}

public float4 JzAzBz_2_XYZ(float4 JzAzBz)
{
  const float b = 1.15f;
  const float g = 0.66f;
  const float c1 = 0.8359375f; // 3424 / 2^12
  const float c2 = 18.8515625f; // 2413 / 2^7
  const float c3 = 18.6875f; // 2392 / 2^7
  const float n_inv = 1.0f / 0.159301758f; // 2610 / 2^14
  const float p_inv = 1.0f / 134.034375f; // 1.7 x 2523 / 2^5
  const float d = -0.56f;
  const float d0 = 1.6295499532821566e-11f;
  const float4 MI[3] = { float4(1.9242264357876067f, -1.0047923125953657f, 0.0376514040306180f, 0.0f),
                         float4(0.3503167620949991f, 0.7264811939316552f, -0.0653844229480850f, 0.0f),
                         float4(-0.0909828109828475f, -0.3127282905230739f, 1.5227665613052603f, 0.0f) };
  const float4 AI[3] = { float4(1.0f, 0.1386050432715393f, 0.0580473161561189f, 0.0f),
                         float4(1.0f, -0.1386050432715393f, -0.0580473161561189f, 0.0f),
                         float4(1.0f, -0.0960192420263190f, -0.8118918960560390f, 0.0f) };

  float4 XYZ, LMS, IzAzBz;
  // Jz -> Iz
  IzAzBz = JzAzBz;
  IzAzBz.x += d0;
  IzAzBz.x = max(IzAzBz.x / (1.0f + d - d * IzAzBz.x), 0.f);
  // IzAzBz -> L'M'S'
  LMS.x = dot(AI[0], IzAzBz);
  LMS.y = dot(AI[1], IzAzBz);
  LMS.z = dot(AI[2], IzAzBz);
  LMS.w = 0.f;
  // L'M'S' -> LMS
  LMS = dtcl_pow(max(LMS, float4(0.0f)), p_inv);
  LMS = 10000.f * dtcl_pow(max((c1 - LMS) / (c3 * LMS - c2), float4(0.0f)), n_inv);
  // LMS -> X'Y'Z
  XYZ.x = dot(MI[0], LMS);
  XYZ.y = dot(MI[1], LMS);
  XYZ.z = dot(MI[2], LMS);
  XYZ.w = 0.f;
  // X'Y'Z -> XYZ_D65
  float4 XYZ_D65;
  XYZ_D65.x = (XYZ.x + (b - 1.0f) * XYZ.z) / b;
  XYZ_D65.y = (XYZ.y + (g - 1.0f) * XYZ_D65.x) / g;
  XYZ_D65.z = XYZ.z;
  XYZ_D65.w = JzAzBz.w;
  return XYZ_D65;
}

// CIE 1931 2° XYZ D65 to CIE 2006 LMS D65 (cone space), see ../colorspace.h
public float4 XYZ_to_LMS(float4 XYZ)
{
  const float4 XYZ_D65_to_LMS_2006_D65[3]
    = { float4(0.257085f, 0.859943f, -0.031061f, 0.f),
        float4(-0.394427f, 1.175800f, 0.106423f, 0.f),
        float4(0.064856f, -0.076250f, 0.559067f, 0.f) };

  return matrix_dot(XYZ, XYZ_D65_to_LMS_2006_D65);
}

public float4 LMS_to_XYZ(float4 LMS)
{
  const float4 LMS_2006_D65_to_XYZ_D65[3]
    = { float4(1.80794659f, -1.29971660f, 0.34785879f, 0.f),
        float4(0.61783960f, 0.39595453f, -0.04104687f, 0.f),
        float4(-0.12546960f, 0.20478038f, 1.74274183f, 0.f) };

  return matrix_dot(LMS, LMS_2006_D65_to_XYZ_D65);
}

// CIE 2006 LMS D65 and Filmlight RGB, see ../colorspace.h
public float4 gradingRGB_to_LMS(float4 RGB)
{
  const float4 filmlightRGB_D65_to_LMS_D65[3]
    = { float4(0.95f, 0.38f, 0.00f, 0.f),
        float4(0.05f, 0.62f, 0.03f, 0.f),
        float4(0.00f, 0.00f, 0.97f, 0.f) };

  return matrix_dot(RGB, filmlightRGB_D65_to_LMS_D65);
}

public float4 LMS_to_gradingRGB(float4 LMS)
{
  const float4 LMS_D65_to_filmlightRGB_D65[3]
    = { float4(1.0877193f, -0.66666667f, 0.02061856f, 0.f),
        float4(-0.0877193f, 1.66666667f, -0.05154639f, 0.f),
        float4(0.f, 0.f, 1.03092784f, 0.f) };

  return matrix_dot(LMS, LMS_D65_to_filmlightRGB_D65);
}

// the Filmlight RGB triplet as Yrg luminance/chromaticity coordinates
public float4 LMS_to_Yrg(float4 LMS)
{
  // compute luminance
  const float Y = 0.68990272f * LMS.x + 0.34832189f * LMS.y;

  // normalize LMS
  const float a = LMS.x + LMS.y + LMS.z;
  const float4 lms = (a == 0.f) ? float4(0.f) : LMS / a;

  // convert to Filmlight rgb (normalized)
  const float4 rgb = LMS_to_gradingRGB(lms);

  return float4(Y, rgb.x, rgb.y, LMS.w);
}

public float4 Yrg_to_LMS(float4 Yrg)
{
  const float Y = Yrg.x;

  // reform rgb (normalized) from chroma
  const float r = Yrg.y;
  const float g = Yrg.z;
  const float b = 1.f - r - g;
  const float4 rgb = float4(r, g, b, 0.f);

  // convert to lms (normalized)
  const float4 lms = gradingRGB_to_LMS(rgb);

  // denormalize to LMS
  const float denom = (0.68990272f * lms.x + 0.34832189f * lms.y);
  const float a = (denom == 0.f) ? 0.f : Y / denom;
  return lms * a;
}

// Filmlight Yrg in polar coordinates Ych, with the cosine and sine of the
// hue angle instead of the angle
public float4 Yrg_to_Ych(float4 Yrg)
{
  const float Y = Yrg.x;
  // subtract the white point, see ../colorspace.h
  const float r = Yrg.y - 0.21902143f;
  const float g = Yrg.z - 0.54371398f;
  const float c = dt_fast_hypot(g, r);
  const float cos_h = c != 0.f ? r / c : 1.f;
  const float sin_h = c != 0.f ? g / c : 0.f;
  return float4(Y, c, cos_h, sin_h);
}

public float4 Ych_to_Yrg(float4 Ych)
{
  const float Y = Ych.x;
  const float c = Ych.y;
  const float cos_h = Ych.z;
  const float sin_h = Ych.w;
  const float r = c * cos_h + 0.21902143f;
  const float g = c * sin_h + 0.54371398f;
  return float4(Y, r, g, 0.f);
}

public float4 dt_D65_XYZ_to_xyY(float4 sXYZ)
{
  // see cpu implementation for details, use D65_xy as fallback
  const float4 XYZ = max(float4(0.0f), sXYZ);
  float4 xyY;
  const float sum = XYZ.x + XYZ.y + XYZ.z;
  if(sum > 0.0f)
  {
    xyY.xy = XYZ.xy / sum;
  }
  else
  {
    xyY.x = 0.31271f;
    xyY.y = 0.32902f;
  }

  xyY.z = XYZ.y;
  xyY.w = XYZ.w;

  return xyY;
}

public float4 gamut_check_Yrg(float4 Ych)
{
  // do a test conversion to Yrg
  const float4 Yrg = Ych_to_Yrg(Ych);

  // gamut-clip in Yrg at constant hue and luminance, e.g. find the max
  // chroma value that fits in gamut at the current hue
  const float D65_r = 0.21902143f;
  const float D65_g = 0.54371398f;
  float max_c = Ych.y;
  const float cos_h = Ych.z;
  const float sin_h = Ych.w;

  if(Yrg.y < 0.f)
  {
    max_c = min(-D65_r / cos_h, max_c);
  }
  if(Yrg.z < 0.f)
  {
    max_c = min(-D65_g / sin_h, max_c);
  }
  if(Yrg.y + Yrg.z > 1.f)
  {
    max_c = min((1.f - D65_r - D65_g) / (cos_h + sin_h), max_c);
  }

  // overwrite chroma with the sanitized value
  Ych.y = max_c;

  return Ych;
}

// darktable Uniform Color Space 2022, see ../colorspace.h
public static const float DT_UCS_L_STAR_RANGE = 2.098883786377f;
public static const float DT_UCS_L_STAR_UPPER_LIMIT = 2.09885f;
public static const float DT_UCS_Y_UPPER_LIMIT = 1e8f;
static const float DT_FLT_MIN = 1.175494350822287507969e-38f;

public float Y_to_dt_UCS_L_star(float Y)
{
  const float Y_hat = dtcl_pow(Y, 0.631651345306265f);
  return DT_UCS_L_STAR_RANGE * Y_hat / (Y_hat + 1.12426773749357f);
}

public float dt_UCS_L_star_to_Y(float L_star)
{
  return dtcl_pow((1.12426773749357f * L_star / (DT_UCS_L_STAR_RANGE - L_star)), 1.5831518565279648f);
}

public float2 xyY_to_dt_UCS_UV(float4 xyY)
{
  const float4 x_factors = float4(-0.783941002840055f, 0.745273540913283f, 0.318707282433486f, 0.f);
  const float4 y_factors = float4(0.277512987809202f, -0.205375866083878f, 2.16743692732158f, 0.f);
  const float4 offsets = float4(0.153836578598858f, -0.165478376301988f, 0.291320554395942f, 0.f);

  float4 UVD = x_factors * xyY.x + y_factors * xyY.y + offsets;
  const float div = (UVD.z >= 0.0f) ? max(DT_FLT_MIN, UVD.z) : min(-DT_FLT_MIN, UVD.z);
  UVD.xy /= div;

  const float2 factors = float2(1.39656225667f, 1.4513954287f);
  const float2 half_values = float2(1.49217352929f, 1.52488637914f);
  const float2 UV_star = float2(factors.x * UVD.x / (abs(UVD.x) + half_values.x),
                                factors.y * UVD.y / (abs(UVD.y) + half_values.y));
  // the following is equivalent to a 2D matrix product
  return float2(-1.124983854323892f * UV_star.x - 0.980483721769325f * UV_star.y,
                1.86323315098672f * UV_star.x + 1.971853092390862f * UV_star.y);
}

public float4 xyY_to_dt_UCS_JCH(float4 xyY, float L_white)
{
  const float2 UV_star_prime = xyY_to_dt_UCS_UV(xyY);

  // Y upper limit is calculated from the L star upper limit
  const float L_star = Y_to_dt_UCS_L_star(clamp(xyY.z, 0.f, DT_UCS_Y_UPPER_LIMIT));
  const float M2 = UV_star_prime.x * UV_star_prime.x + UV_star_prime.y * UV_star_prime.y; // square of colorfulness M

  return float4(L_star / L_white,
                15.932993652962535f * dtcl_pow(L_star, 0.6523997524738018f) * dtcl_pow(M2, 0.6007557017508491f) / L_white,
                atan2(UV_star_prime.y, UV_star_prime.x),
                0.0f);
}

public float4 dt_UCS_JCH_to_xyY(float4 JCH, float L_white)
{
  // clipping L_star below its upper limit keeps dt_UCS_L_star_to_Y() away
  // from dividing by values close to zero, see ../colorspace.h
  const float L_star = clamp(JCH.x * L_white, 0.f, DT_UCS_L_STAR_UPPER_LIMIT);
  const float M = L_star != 0.f
    ? dtcl_pow(JCH.y * L_white / (15.932993652962535f * dtcl_pow(L_star, 0.6523997524738018f)), 0.8322850678616855f)
    : 0.f;

  const float U_star_prime = M * dtcl_cos(JCH.z);
  const float V_star_prime = M * dtcl_sin(JCH.z);

  // the following is equivalent to a 2D matrix product
  const float2 UV_star = float2(-5.037522385190711f * U_star_prime - 2.504856328185843f * V_star_prime,
                                4.760029407436461f * U_star_prime + 2.874012963239247f * V_star_prime);

  const float2 factors = float2(1.39656225667f, 1.4513954287f);
  const float2 half_values = float2(1.49217352929f, 1.52488637914f);
  const float2 UV = float2(-half_values.x * UV_star.x / (abs(UV_star.x) - factors.x),
                           -half_values.y * UV_star.y / (abs(UV_star.y) - factors.y));

  const float4 U_factors = float4(0.167171472114775f, -0.150959086409163f, 0.940254742367256f, 0.f);
  const float4 V_factors = float4(0.141299802443708f, -0.155185060382272f, 1.000000000000000f, 0.f);
  const float4 offsets = float4(-0.00801531300850582f, -0.00843312433578007f, -0.0256325967652889f, 0.f);

  const float4 xyD = U_factors * UV.x + V_factors * UV.y + offsets;

  const float div = (xyD.z >= 0.0f) ? max(DT_FLT_MIN, xyD.z) : min(-DT_FLT_MIN, xyD.z);
  return float4(xyD.x / div, xyD.y / div, dt_UCS_L_star_to_Y(L_star), 0.0f);
}

public float4 dt_UCS_JCH_to_HSB(float4 JCH)
{
  float4 HSB;
  HSB.z = JCH.x * (dtcl_pow(JCH.y, 1.33654221029386f) + 1.f);
  HSB.y = (HSB.z > 0.f) ? JCH.y / HSB.z : 0.f;
  HSB.x = JCH.z;
  HSB.w = 0.f;
  return HSB;
}

public float4 dt_UCS_HSB_to_JCH(float4 HSB)
{
  float4 JCH;
  JCH.z = HSB.x;
  JCH.y = HSB.y * HSB.z;
  JCH.x = HSB.z / (dtcl_pow(JCH.y, 1.33654221029386f) + 1.f);
  JCH.w = 0.f;
  return JCH;
}

public float4 dt_UCS_JCH_to_HCB(float4 JCH)
{
  float4 HCB;
  HCB.z = JCH.x * (dtcl_pow(JCH.y, 1.33654221029386f) + 1.f);
  HCB.y = JCH.y;
  HCB.x = JCH.z;
  HCB.w = 0.f;
  return HCB;
}

public float4 dt_UCS_HCB_to_JCH(float4 HCB)
{
  float4 JCH;
  JCH.z = HCB.x;
  JCH.y = HCB.y;
  JCH.x = HCB.z / (dtcl_pow(HCB.y, 1.33654221029386f) + 1.f);
  JCH.w = 0.f;
  return JCH;
}

public float soft_clip(float x, float soft_threshold, float hard_threshold)
{
  // use an exponential soft clipping above soft_threshold, hard threshold
  // must be > soft threshold
  const float norm = hard_threshold - soft_threshold;
  return (x > soft_threshold) ? soft_threshold + (1.f - dtcl_exp(-(x - soft_threshold) / norm)) * norm : x;
}

public float lookup_gamut(StructuredBuffer<float> gamut_lut, float x)
{
  // linearly interpolate the gamut LUT at the hue angle in radians, in LUT
  // coordinates
  const float x_test = (float)LUT_ELEM * (x + M_PI_F) / DT_2PI_F;

  // the 2 closest integer coordinates, cycling on the hue ring
  const float x_prev = floor(x_test);
  const float x_next = ceil(x_test);
  const int xi = ((int)x_prev) & (LUT_ELEM - 1);
  const int xii = ((int)x_next) & (LUT_ELEM - 1);

  const float y_prev = gamut_lut[xi];

  // y_prev on an integer LUT element, a linear interpolation otherwise
  return y_prev + ((xi != xii) ? (x_test - x_prev) * (gamut_lut[xii] - y_prev) : 0.0f);
}
