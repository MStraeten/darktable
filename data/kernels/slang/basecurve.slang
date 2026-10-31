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


// Slang port of ../basecurve.cl, which does what process() does on the CPU

import rgb_norms;

// the curve applied to a luminance norm, so that the channels keep their
// ratios. mul is the exposure multiplier of a fusion image, 1 otherwise.
[shader("compute")]
[numthreads(16, 16, 1)]
void basecurve_lut(uint3 gid : SV_DispatchThreadID,
                   uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
                   uniform float mul, uniform ImageIn table, uniform StructuredBuffer<float> coeffs,
                   uniform int preserve_colors, uniform StructuredBuffer<ProfileInfo> profile_info,
                   uniform ImageIn lut, uniform int use_work_profile)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float a[3] = { coeffs[0], coeffs[1], coeffs[2] };
  float4 pixel = in.read(int2(x, y));
  float ratio = 1.0f;
  const float lum = mul * dt_rgb_norm(pixel, preserve_colors, use_work_profile, profile_info[0], lut);
  if(lum > 0.0f)
    ratio = mul * lookup_unbounded(table, lum, a) / lum;
  pixel.xyz *= ratio;
  out.write(int2(x, y), max(pixel, 0.0f));
}

// the curve applied to each channel, which can shift colors
[shader("compute")]
[numthreads(16, 16, 1)]
void basecurve_legacy_lut(uint3 gid : SV_DispatchThreadID,
                          uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
                          uniform float mul, uniform ImageIn table, uniform StructuredBuffer<float> coeffs)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float a[3] = { coeffs[0], coeffs[1], coeffs[2] };
  float4 pixel = in.read(int2(x, y));
  pixel.x = lookup_unbounded(table, mul * pixel.x, a);
  pixel.y = lookup_unbounded(table, mul * pixel.y, a);
  pixel.z = lookup_unbounded(table, mul * pixel.z, a);
  out.write(int2(x, y), max(pixel, 0.0f));
}

// the fusion weight of well exposedness and saturation, in alpha
[shader("compute")]
[numthreads(16, 16, 1)]
void basecurve_compute_features(uint3 gid : SV_DispatchThreadID,
                                uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  float4 value = in.read(int2(x, y));
  const float ma = max(value.x, max(value.y, value.z));
  const float mi = min(value.x, min(value.y, value.z));
  const float sat = 0.1f + 0.1f * (ma - mi) / max(1.0e-4f, ma);

  const float c = 0.54f;
  float v = abs(value.x - c);
  v = max(abs(value.y - c), v);
  v = max(abs(value.z - c), v);
  const float var = 0.5f;
  value.w = sat * (0.2f + dt_fast_expf(-v * v / (var * var)));
  out.write(int2(x, y), value);
}

// the 5 tap blur along dir (1, 0) or (0, 1), mirrored at the borders
[shader("compute")]
[numthreads(16, 16, 1)]
void basecurve_blur(uint3 gid : SV_DispatchThreadID,
                    uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
                    uniform int2 dir)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float w[5] = { 1.0f / 16.0f, 4.0f / 16.0f, 6.0f / 16.0f, 4.0f / 16.0f, 1.0f / 16.0f };
  const int p = dir.x != 0 ? x : y;
  const int n = dir.x != 0 ? width : height;
  float4 sum = float4(0.0f);
  for(int i = -2; i <= 2; i++)
  {
    const int pp = min(max(-p - i, p + i), n - (p + i - n + 1));
    sum += in.read(dir.x != 0 ? int2(pp, y) : int2(x, pp)) * w[i + 2];
  }
  out.write(int2(x, y), sum);
}

// coarse values in the even pixels, zero in the odd ones
[shader("compute")]
[numthreads(16, 16, 1)]
void basecurve_expand(uint3 gid : SV_DispatchThreadID,
                      uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float4 pixel = (x % 2 == 0 && y % 2 == 0) ? 4.0f * in.read(int2(x / 2, y / 2)) : float4(0.0f);
  out.write(int2(x, y), pixel);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void basecurve_reduce(uint3 gid : SV_DispatchThreadID,
                      uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  out.write(int2(x, y), in.read(int2(2 * x, 2 * y)));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void basecurve_detail(uint3 gid : SV_DispatchThreadID,
                      uniform ImageIn in, uniform ImageIn det, uniform ImageOut out,
                      uniform int width, uniform int height)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  out.write(int2(x, y), in.read(int2(x, y)) - det.read(int2(x, y)));
}

// the weight in alpha gains the local contrast of the detail level
[shader("compute")]
[numthreads(16, 16, 1)]
void basecurve_adjust_features(uint3 gid : SV_DispatchThreadID,
                               uniform ImageIn in, uniform ImageIn det, uniform ImageOut out,
                               uniform int width, uniform int height)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  float4 input = in.read(int2(x, y));
  const float4 detail = det.read(int2(x, y));
  input.w *= 0.1f + sqrt(detail.x * detail.x + detail.y * detail.y + detail.z * detail.z);
  out.write(int2(x, y), input);
}

// adds a fusion image to the output pyramid: its gaussian base at the
// coarsest level, its laplacian (col - tmp) at the others. The first image
// starts from zero, as device images aren't cleared.
[shader("compute")]
[numthreads(16, 16, 1)]
void basecurve_blend(uint3 gid : SV_DispatchThreadID,
                     uniform ImageIn in, uniform ImageIn col, uniform ImageIn tmp, uniform ImageOut out,
                     uniform int width, uniform int height, uniform int laplacian, uniform int first)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  float4 comb = first != 0 ? float4(0.0f) : in.read(int2(x, y));
  const float4 collect = col.read(int2(x, y));
  const float4 temp = laplacian != 0 ? tmp.read(int2(x, y)) : float4(0.0f);
  comb.xyz += (collect.xyz - temp.xyz) * collect.w;
  comb.w += collect.w;
  out.write(int2(x, y), comb);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void basecurve_normalize(uint3 gid : SV_DispatchThreadID,
                         uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  float4 comb = in.read(int2(x, y));
  comb.xyz /= (comb.w > 1.0e-8f) ? comb.w : 1.0f;
  out.write(int2(x, y), comb);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void basecurve_reconstruct(uint3 gid : SV_DispatchThreadID,
                           uniform ImageIn in, uniform ImageIn tmp, uniform ImageOut out,
                           uniform int width, uniform int height)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  out.write(int2(x, y), in.read(int2(x, y)) + tmp.read(int2(x, y)));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void basecurve_finalize(uint3 gid : SV_DispatchThreadID,
                        uniform ImageIn in, uniform ImageIn comb, uniform ImageOut out,
                        uniform int width, uniform int height)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  float4 pixel = max(comb.read(int2(x, y)), 0.0f);
  pixel.w = in.read(int2(x, y)).w;
  out.write(int2(x, y), pixel);
}
