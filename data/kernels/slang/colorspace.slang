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
