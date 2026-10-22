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

// Slang port of ../rgbcurve.cl

import rgb_norms;

[shader("compute")]
[numthreads(16, 16, 1)]
void rgbcurve(uint3 gid : SV_DispatchThreadID,
              uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
              uniform ImageIn table_r, uniform ImageIn table_g, uniform ImageIn table_b,
              uniform StructuredBuffer<float> coeffs_r, uniform StructuredBuffer<float> coeffs_g,
              uniform StructuredBuffer<float> coeffs_b,
              uniform int autoscale, uniform int preserve_colors,
              uniform StructuredBuffer<ProfileInfo> profile_info, uniform ImageIn lut,
              uniform int use_work_profile)
{
  const int x = gid.x;
  const int y = gid.y;

  if(x >= width || y >= height) return;

  float4 pixel = in.read(int2(x, y));

  const float a_r[3] = { coeffs_r[0], coeffs_r[1], coeffs_r[2] };
  const float a_g[3] = { coeffs_g[0], coeffs_g[1], coeffs_g[2] };
  const float a_b[3] = { coeffs_b[0], coeffs_b[1], coeffs_b[2] };

  if(autoscale == 1) // DT_S_SCALE_MANUAL_RGB
  {
    pixel.x = lookup_unbounded(table_r, pixel.x, a_r);
    pixel.y = lookup_unbounded(table_g, pixel.y, a_g);
    pixel.z = lookup_unbounded(table_b, pixel.z, a_b);
  }
  else if(autoscale == 0) // DT_S_SCALE_AUTOMATIC_RGB
  {
    if(preserve_colors == DT_RGB_NORM_NONE)
    {
      pixel.x = lookup_unbounded(table_r, pixel.x, a_r);
      pixel.y = lookup_unbounded(table_r, pixel.y, a_r);
      pixel.z = lookup_unbounded(table_r, pixel.z, a_r);
    }
    else
    {
      float ratio = 1.f;
      const float lum = dt_rgb_norm(pixel, preserve_colors, use_work_profile, profile_info[0], lut);
      if(lum > 0.f)
      {
        const float curve_lum = lookup_unbounded(table_r, lum, a_r);
        ratio = curve_lum / lum;
      }
      pixel.xyz *= ratio;
    }
  }

  out.write(int2(x, y), pixel);
}
