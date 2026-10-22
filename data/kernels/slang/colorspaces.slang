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

// Slang port of ../colorspaces.cl

import color_conversion;

[shader("compute")]
[numthreads(16, 16, 1)]
void colorspaces_transform_lab_to_rgb_matrix(uint3 gid : SV_DispatchThreadID,
                                             uniform ImageIn in, uniform ImageOut out,
                                             uniform int width, uniform int height,
                                             uniform StructuredBuffer<ProfileInfo> profile_info,
                                             uniform ImageIn lut)
{
  const int x = gid.x;
  const int y = gid.y;

  if(x >= width || y >= height) return;

  const ProfileInfo info = profile_info[0];
  float4 pixel = in.Areadpixel(x, y);
  pixel.xyz = Lab_to_XYZ(pixel).xyz;
  pixel = matrix_product(pixel, info.matrix_out);

  if(info.nonlinearlut != 0)
    pixel = apply_trc_out(pixel, info, lut);

  out.write(int2(x, y), pixel);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void colorspaces_transform_rgb_matrix_to_lab(uint3 gid : SV_DispatchThreadID,
                                             uniform ImageIn in, uniform ImageOut out,
                                             uniform int width, uniform int height,
                                             uniform StructuredBuffer<ProfileInfo> profile_info,
                                             uniform ImageIn lut)
{
  const int x = gid.x;
  const int y = gid.y;

  if(x >= width || y >= height) return;

  const ProfileInfo info = profile_info[0];
  float4 pixel = in.Areadpixel(x, y);

  if(info.nonlinearlut != 0)
    pixel = apply_trc_in(pixel, info, lut);

  pixel = matrix_product(pixel, info.matrix_in);
  pixel.xyz = XYZ_to_Lab(pixel).xyz;

  out.write(int2(x, y), pixel);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void colorspaces_transform_rgb_matrix_to_rgb(uint3 gid : SV_DispatchThreadID,
                                             uniform ImageIn in, uniform ImageOut out,
                                             uniform int width, uniform int height,
                                             uniform StructuredBuffer<ProfileInfo> profile_info_from,
                                             uniform ImageIn lut_from,
                                             uniform StructuredBuffer<ProfileInfo> profile_info_to,
                                             uniform ImageIn lut_to,
                                             uniform StructuredBuffer<float> matrix)
{
  const int x = gid.x;
  const int y = gid.y;

  if(x >= width || y >= height) return;

  const ProfileInfo from = profile_info_from[0];
  const ProfileInfo to = profile_info_to[0];
  float m[9];
  for(int k = 0; k < 9; k++) m[k] = matrix[k];

  float4 pixel = in.Areadpixel(x, y);

  if(from.nonlinearlut != 0)
    pixel = apply_trc_in(pixel, from, lut_from);

  pixel = matrix_product(pixel, m);

  if(to.nonlinearlut != 0)
    pixel = apply_trc_out(pixel, to, lut_to);

  out.write(int2(x, y), pixel);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void colorspaces_transform_gamma(uint3 gid : SV_DispatchThreadID,
                                 uniform ImageIn in, uniform ImageOut out,
                                 uniform int width, uniform int height,
                                 uniform float gamma)
{
  const int x = gid.x;
  const int y = gid.y;

  if(x >= width || y >= height) return;

  float4 pixel = in.Areadpixel(x, y);
  // sanitize and fix NaN as we do for CPU
  pixel = select(isnan(pixel), float4(0.0f), max(pixel, float4(-1e6f)));
  pixel = copysign(dtcl_pow(abs(pixel), gamma), pixel);
  out.write(int2(x, y), pixel);
}
