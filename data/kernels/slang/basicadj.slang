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

// Slang port of ../basicadj.cl

import rgb_norms;

float get_lut_gamma(float x, float gamma, ImageIn lut)
{
  if(x > 1.0f) return dtcl_pow(x, gamma);

  const int xi = clamp(int(x * 65536.0f), 0, 0xffff);
  return lut.read(int2((xi & 0xff), (xi >> 8))).x;
}

float get_lut_contrast(float x, float contrast, float middle_grey, float inv_middle_grey, ImageIn lut)
{
  if(x > 1.0f) return dtcl_pow(x * inv_middle_grey, contrast) * middle_grey;

  const int xi = clamp(int(x * 65536.0f), 0, 0xffff);
  return lut.read(int2((xi & 0xff), (xi >> 8))).x;
}

float hlcurve(float level, float hlcomp, float hlrange)
{
  if(hlcomp <= 0.0f) return 1.f;

  float val = level + (hlrange - 1.f);
  // to avoid division by zero
  if(val == 0.0f) val = 0.000001f;

  float Y = val / hlrange;
  Y *= hlcomp;
  // to avoid log(<=0)
  if(Y <= -1.0f) Y = -.999999f;

  const float R = hlrange / (val * hlcomp);
  return dt_log1p(Y) * R;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void basicadj(uint3 gid : SV_DispatchThreadID,
              uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
              uniform ImageIn lut_gamma, uniform ImageIn lut_contrast,
              uniform float black_point, uniform float scale,
              uniform int process_gamma, uniform float gamma,
              uniform int plain_contrast, uniform int preserve_colors, uniform float contrast,
              uniform int process_saturation_vibrance, uniform float saturation, uniform float vibrance,
              uniform int process_hlcompr, uniform float hlcomp, uniform float hlrange,
              uniform float middle_grey, uniform float inv_middle_grey,
              uniform StructuredBuffer<ProfileInfo> profile_info, uniform ImageIn lut,
              uniform int use_work_profile)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const ProfileInfo info = profile_info[0];
  float4 pixel = in.read(int2(x, y));
  const float w = pixel.w;

  // exposure
  pixel = (pixel - black_point) * scale;

  // highlight compression
  if(process_hlcompr != 0)
  {
    const float lum = (use_work_profile == 0) ? dt_camera_rgb_luminance(pixel)
                                              : get_rgb_matrix_luminance(pixel, info, info.matrix_in, lut);
    if(lum > 0.f) pixel *= hlcurve(lum, hlcomp, hlrange);
  }

  // gamma
  if(process_gamma != 0)
  {
    if(pixel.x > 0.f) pixel.x = get_lut_gamma(pixel.x, gamma, lut_gamma);
    if(pixel.y > 0.f) pixel.y = get_lut_gamma(pixel.y, gamma, lut_gamma);
    if(pixel.z > 0.f) pixel.z = get_lut_gamma(pixel.z, gamma, lut_gamma);
  }

  // contrast
  if(plain_contrast != 0)
  {
    if(pixel.x > 0.f) pixel.x = get_lut_contrast(pixel.x, contrast, middle_grey, inv_middle_grey, lut_contrast);
    if(pixel.y > 0.f) pixel.y = get_lut_contrast(pixel.y, contrast, middle_grey, inv_middle_grey, lut_contrast);
    if(pixel.z > 0.f) pixel.z = get_lut_contrast(pixel.z, contrast, middle_grey, inv_middle_grey, lut_contrast);
  }

  // contrast (with preserve colors)
  if(preserve_colors != DT_RGB_NORM_NONE)
  {
    float ratio = 1.f;
    const float lum = dt_rgb_norm(pixel, preserve_colors, use_work_profile, info, lut);
    if(lum > 0.f)
    {
      const float contrast_lum = dtcl_pow(lum / middle_grey, contrast) * middle_grey;
      ratio = contrast_lum / lum;
    }
    pixel *= ratio;
  }

  // saturation
  if(process_saturation_vibrance != 0)
  {
    const float average = (pixel.x + pixel.y + pixel.z) / 3;
    // rgb only, as process() does: the OpenCL kernel also counts the alpha channel
    const float delta = length(pixel.xyz - average);
    const float P = vibrance * (1 - dtcl_pow(delta, abs(vibrance)));
    pixel = average + (saturation + P) * (pixel - average);
  }

  pixel.w = w;
  out.write(int2(x, y), pixel);
}
