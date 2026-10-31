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

// Slang port of ../rgblevels.cl

import rgb_norms;

float rgblevels_1c(float pixel, float levels[3], float inv_gamma, ImageIn lut)
{
  float level;

  if(pixel <= levels[0])
  {
    // Anything below the lower threshold just clips to zero
    level = 0.0f;
  }
  else if(pixel >= levels[2])
  {
    const float percentage = (pixel - levels[0]) / (levels[2] - levels[0]);
    level = dtcl_pow(percentage, inv_gamma);
  }
  else
  {
    // Within the expected input range we can use the lookup table
    const float percentage = (pixel - levels[0]) / (levels[2] - levels[0]);
    level = lookup(lut, percentage);
  }

  return level;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void rgblevels(uint3 gid : SV_DispatchThreadID,
               uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
               uniform int autoscale, uniform int preserve_colors,
               uniform ImageIn lutr, uniform ImageIn lutg, uniform ImageIn lutb,
               uniform StructuredBuffer<float> levels, uniform StructuredBuffer<float> inv_gamma,
               uniform StructuredBuffer<ProfileInfo> profile_info, uniform ImageIn lut,
               uniform int use_work_profile)
{
  const int x = gid.x;
  const int y = gid.y;

  if(x >= width || y >= height) return;

  float4 pixel = in.read(int2(x, y));

  // levels is float[3][3] on the host
  const float levels_r[3] = { levels[0], levels[1], levels[2] };
  const float levels_g[3] = { levels[3], levels[4], levels[5] };
  const float levels_b[3] = { levels[6], levels[7], levels[8] };

  if(autoscale == 1 || preserve_colors == DT_RGB_NORM_NONE) // autoscale == DT_IOP_RGBLEVELS_INDEPENDENT_CHANNELS
  {
    pixel.x = rgblevels_1c(pixel.x, levels_r, inv_gamma[0], lutr);
    pixel.y = rgblevels_1c(pixel.y, levels_g, inv_gamma[1], lutg);
    pixel.z = rgblevels_1c(pixel.z, levels_b, inv_gamma[2], lutb);
  }
  else
  {
    float ratio = 0.f;
    const float lum = dt_rgb_norm(pixel, preserve_colors, use_work_profile, profile_info[0], lut);
    if(lum > levels_r[0])
    {
      const float curve_lum = rgblevels_1c(lum, levels_r, inv_gamma[0], lutr);
      ratio = curve_lum / lum;
    }
    pixel.xyz *= ratio;
  }

  out.write(int2(x, y), pixel);
}
