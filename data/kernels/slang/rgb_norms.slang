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

// Slang module form of ../rgb_norms.h

module rgb_norms;

__exported import color_conversion;

public static const int DT_RGB_NORM_NONE = 0;
public static const int DT_RGB_NORM_LUMINANCE = 1;
public static const int DT_RGB_NORM_MAX = 2;
public static const int DT_RGB_NORM_AVERAGE = 3;
public static const int DT_RGB_NORM_SUM = 4;
public static const int DT_RGB_NORM_NORM = 5;
public static const int DT_RGB_NORM_POWER = 6;

public float dt_rgb_norm(float4 in, int norm, int work_profile, ProfileInfo profile_info, ImageIn lut)
{
  if(norm == DT_RGB_NORM_LUMINANCE)
  {
    return (work_profile == 0) ? dt_camera_rgb_luminance(in)
                               : get_rgb_matrix_luminance(in, profile_info, profile_info.matrix_in, lut);
  }
  else if(norm == DT_RGB_NORM_MAX)
  {
    return max(in.x, max(in.y, in.z));
  }
  else if(norm == DT_RGB_NORM_AVERAGE)
  {
    return (in.x + in.y + in.z) / 3.0f;
  }
  else if(norm == DT_RGB_NORM_SUM)
  {
    return in.x + in.y + in.z;
  }
  else if(norm == DT_RGB_NORM_NORM)
  {
    return dtcl_pow(in.x * in.x + in.y * in.y + in.z * in.z, 0.5f);
  }
  else if(norm == DT_RGB_NORM_POWER)
  {
    const float R = in.x * in.x;
    const float G = in.y * in.y;
    const float B = in.z * in.z;
    return (in.x * R + in.y * G + in.z * B) / (R + G + B);
  }
  else return (in.x + in.y + in.z) / 3.0f;
}
