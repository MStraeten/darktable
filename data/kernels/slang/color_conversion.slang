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

// Slang module form of ../color_conversion.h

module color_conversion;

__exported import common;
__exported import colorspace;
__exported import hdr_transfer;

// must be in synch with dt_iop_colorspace_type_t in colorspaces.h
public enum dt_iop_colorspace_type_t
{
  IOP_CS_NONE = -1,
  IOP_CS_RAW = 0,
  IOP_CS_LAB = 1,
  IOP_CS_RGB = 2,
  IOP_CS_LCH = 3,
  IOP_CS_HSL = 4,
  IOP_CS_JZCZHZ = 5,
};

// must be in synch with dt_colorspaces_iccprofile_info_cl_t. Bound as a
// StructuredBuffer, whose std430 layout packs these scalar arrays like C does.
public struct dt_colorspaces_iccprofile_info_cl_t
{
  public float matrix_in[9];
  public float matrix_out[9];
  public int lutsize;
  public float unbounded_coeffs_in[3][3];
  public float unbounded_coeffs_out[3][3];
  public int nonlinearlut;
  public float grey;
  public int hdr_transfer;
};

public typealias ProfileInfo = dt_colorspaces_iccprofile_info_cl_t;

public float lerp_lookup_unbounded(float x, ImageIn lut, float unbounded_coeffs[3],
                                   int n_lut, int lutsize)
{
  // in case the tone curve is marked as linear, return the fast
  // path to linear unbounded (does not clip x at 1)
  if(unbounded_coeffs[0] >= 0.0f)
  {
    if(x < 1.0f)
    {
      const float ft = clamp(x * float(lutsize - 1), 0.0f, float(lutsize - 1));
      const int t = ft < lutsize - 2 ? int(ft) : lutsize - 2;
      const float f = ft - t;
      const int2 p1 = int2((t & 0xff), (t >> 8) + n_lut * 256);
      const int2 p2 = int2(((t + 1) & 0xff), ((t + 1) >> 8) + n_lut * 256);
      const float l1 = lut.read(p1).x;
      const float l2 = lut.read(p2).x;
      return l1 * (1.0f - f) + l2 * f;
    }
    else return unbounded_coeffs[1] * dtcl_pow(x*unbounded_coeffs[0], unbounded_coeffs[2]);
  }
  else return x;
}

public float lookup(ImageIn lut, float x)
{
  const int xi = clamp(int(x * 65536.0f), 0, 0xffff);
  const int2 p = int2((xi & 0xff), (xi >> 8));
  return lut.read(p).x;
}

public float lookup_unbounded(ImageIn lut, float x, float a[3])
{
  // in case the tone curve is marked as linear, return the fast
  // path to linear unbounded (does not clip x at 1)
  if(a[0] >= 0.0f)
  {
    if(x < 1.0f)
    {
      const int xi = clamp(int(x * 65536.0f), 0, 0xffff);
      const int2 p = int2((xi & 0xff), (xi >> 8));
      return lut.read(p).x;
    }
    else return a[1] * dtcl_pow(x*a[0], a[2]);
  }
  else return x;
}

public float4 apply_trc_in(float4 rgb_in, ProfileInfo profile_info, ImageIn lut)
{
  float3 rgb = rgb_in.xyz;
  if(profile_info.hdr_transfer != 0)
  {
    dt_hdr_decode(rgb, profile_info.hdr_transfer);
  }
  else
  {
    rgb[0] = lerp_lookup_unbounded(rgb_in.x, lut, profile_info.unbounded_coeffs_in[0], 0, profile_info.lutsize);
    rgb[1] = lerp_lookup_unbounded(rgb_in.y, lut, profile_info.unbounded_coeffs_in[1], 1, profile_info.lutsize);
    rgb[2] = lerp_lookup_unbounded(rgb_in.z, lut, profile_info.unbounded_coeffs_in[2], 2, profile_info.lutsize);
  }
  return float4(rgb, rgb_in.w);
}

public float4 apply_trc_out(float4 rgb_in, ProfileInfo profile_info, ImageIn lut)
{
  float3 rgb = rgb_in.xyz;
  if(profile_info.hdr_transfer != 0)
  {
    dt_hdr_encode(rgb, profile_info.hdr_transfer);
  }
  else
  {
    rgb[0] = lerp_lookup_unbounded(rgb_in.x, lut, profile_info.unbounded_coeffs_out[0], 3, profile_info.lutsize);
    rgb[1] = lerp_lookup_unbounded(rgb_in.y, lut, profile_info.unbounded_coeffs_out[1], 4, profile_info.lutsize);
    rgb[2] = lerp_lookup_unbounded(rgb_in.z, lut, profile_info.unbounded_coeffs_out[2], 5, profile_info.lutsize);
  }
  return float4(rgb, rgb_in.w);
}

public float get_rgb_matrix_luminance(float4 rgb, ProfileInfo profile_info,
                                      float matrix[9], ImageIn lut)
{
  float luminance = 0.f;

  if(profile_info.nonlinearlut != 0)
  {
    const float4 linear_rgb = apply_trc_in(rgb, profile_info, lut);
    luminance = matrix[3] * linear_rgb.x + matrix[4] * linear_rgb.y + matrix[5] * linear_rgb.z;
  }
  else
    luminance = matrix[3] * rgb.x + matrix[4] * rgb.y + matrix[5] * rgb.z;

  return luminance;
}

public float4 rgb_matrix_to_xyz(float4 rgb, ProfileInfo profile_info,
                                float matrix[9], ImageIn lut)
{
  float4 out;
  if(profile_info.nonlinearlut != 0)
  {
    const float4 linear_rgb = apply_trc_in(rgb, profile_info, lut);
    out = matrix_product(linear_rgb, matrix);
  }
  else
  {
    out = matrix_product(rgb, matrix);
  }
  return out;
}

public float dt_camera_rgb_luminance(float4 rgb)
{
  const float4 coeffs = float4(0.2225045f, 0.7168786f, 0.0606169f, 0.0f);
  return dot(rgb, coeffs);
}
