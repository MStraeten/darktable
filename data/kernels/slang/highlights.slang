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

// Slang port of the guided laplacians kernels in ../basic.cl. They follow
// the CPU code in src/iop/hlreconstruct/laplacian.c where the two differ.

import noise_generator;

// enum wavelets_scale_t
static const uint FIRST_SCALE = 1 << 1;
static const uint LAST_SCALE = 1 << 2;

static const float B_SPLINE_TO_LAPLACIAN = 3.182727439285017f;

[shader("compute")]
[numthreads(16, 16, 1)]
void hl_interpolate_and_mask(uint3 gid : SV_DispatchThreadID,
                             uniform ImageIn input, uniform ImageOut interpolated, uniform ImageOut clipping_mask,
                             uniform float4 clips, uniform float4 wb, uniform int filters,
                             uniform int width, uniform int height)
{
  // bilinear interpolation
  const int j = gid.x;
  const int i = gid.y;
  if(j >= width || i >= height) return;

  const uint f = uint(filters);
  const int c = FC(i, j, f);
  const float center = input.Areadsingle(j, i);

  float R = 0.f;
  float G = 0.f;
  float B = 0.f;
  bool R_clipped = false;
  bool G_clipped = false;
  bool B_clipped = false;

  if(i == 0 || j == 0 || i == height - 1 || j == width - 1)
  {
    // on the edges, R = G = B = center: the image is remosaiced at the end
    R = G = B = center;
    R_clipped = G_clipped = B_clipped = (center > clips[c]);
  }
  else
  {
    const float north = input.Areadsingle(j, i - 1);
    const float south = input.Areadsingle(j, i + 1);
    const float west = input.Areadsingle(j - 1, i);
    const float east = input.Areadsingle(j + 1, i);

    const float north_east = input.Areadsingle(j + 1, i - 1);
    const float north_west = input.Areadsingle(j - 1, i - 1);
    const float south_east = input.Areadsingle(j + 1, i + 1);
    const float south_west = input.Areadsingle(j - 1, i + 1);

    if(c == GREEN)
    {
      G = center;
      G_clipped = (center > clips[GREEN]);
    }
    else
    {
      // interpolate inside an X/Y cross
      G = (north + south + east + west) / 4.f;
      G_clipped = (north > clips[GREEN] || south > clips[GREEN] || east > clips[GREEN] || west > clips[GREEN]);
    }

    if(c == RED)
    {
      R = center;
      R_clipped = (center > clips[RED]);
    }
    else if(FC(i - 1, j, f) == RED && FC(i + 1, j, f) == RED)
    {
      R = (north + south) / 2.f;
      R_clipped = (north > clips[RED] || south > clips[RED]);
    }
    else if(FC(i, j - 1, f) == RED && FC(i, j + 1, f) == RED)
    {
      R = (west + east) / 2.f;
      R_clipped = (west > clips[RED] || east > clips[RED]);
    }
    else
    {
      R = (north_west + north_east + south_east + south_west) / 4.f;
      R_clipped = (north_west > clips[RED] || north_east > clips[RED] || south_west > clips[RED]
                   || south_east > clips[RED]);
    }

    if(c == BLUE)
    {
      B = center;
      B_clipped = (center > clips[BLUE]);
    }
    else if(FC(i - 1, j, f) == BLUE && FC(i + 1, j, f) == BLUE)
    {
      B = (north + south) / 2.f;
      B_clipped = (north > clips[BLUE] || south > clips[BLUE]);
    }
    else if(FC(i, j - 1, f) == BLUE && FC(i, j + 1, f) == BLUE)
    {
      B = (west + east) / 2.f;
      B_clipped = (west > clips[BLUE] || east > clips[BLUE]);
    }
    else
    {
      B = (north_west + north_east + south_east + south_west) / 4.f;
      B_clipped = (north_west > clips[BLUE] || north_east > clips[BLUE] || south_west > clips[BLUE]
                   || south_east > clips[BLUE]);
    }
  }

  const float4 RGB = float4(R, G, B, sqrt(R * R + G * G + B * B));
  const float4 clipped = float4(R_clipped ? 1.f : 0.f, G_clipped ? 1.f : 0.f, B_clipped ? 1.f : 0.f,
                                (R_clipped || G_clipped || B_clipped) ? 1.f : 0.f);
  interpolated.write(int2(j, i), max(RGB / float4(wb.xyz, 1.f), float4(0.f)));
  clipping_mask.write(int2(j, i), clipped);
}

// interpolate_bilinear() of common/fast_guided_filter.h on 4-channel images
[shader("compute")]
[numthreads(16, 16, 1)]
void hl_bilinear(uint3 gid : SV_DispatchThreadID,
                 uniform ImageIn in, uniform int width_in, uniform int height_in,
                 uniform ImageOut out, uniform int width_out, uniform int height_out)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width_out || y >= height_out) return;

  const float x_in = (float(x) / float(width_out)) * float(width_in);
  const float y_in = (float(y) / float(height_out)) * float(height_in);

  int x_prev = int(floor(x_in));
  int x_next = x_prev + 1;
  int y_prev = int(floor(y_in));
  int y_next = y_prev + 1;
  x_prev = min(x_prev, width_in - 1);
  x_next = min(x_next, width_in - 1);
  y_prev = min(y_prev, height_in - 1);
  y_next = min(y_next, height_in - 1);

  const float4 Q_NW = in.Areadpixel(x_prev, y_prev);
  const float4 Q_NE = in.Areadpixel(x_next, y_prev);
  const float4 Q_SE = in.Areadpixel(x_next, y_next);
  const float4 Q_SW = in.Areadpixel(x_prev, y_next);

  // spatial differences between nodes, next - prev = 1
  const float Dy_next = float(y_next) - y_in;
  const float Dy_prev = 1.f - Dy_next;
  const float Dx_next = float(x_next) - x_in;
  const float Dx_prev = 1.f - Dx_next;

  out.write(int2(x, y), Dy_prev * (Q_SW * Dx_next + Q_SE * Dx_prev)
                        + Dy_next * (Q_NW * Dx_next + Q_NE * Dx_prev));
}

void _fetch_neighbours(ImageIn HF, int x, int y, int width, int height, int mult, out float4 n[9])
{
  const int j_neighbours[3] = { max(x - mult, 0), x, min(x + mult, width - 1) };
  const int i_neighbours[3] = { max(y - mult, 0), y, min(y + mult, height - 1) };
  [unroll]
  for(int ii = 0; ii < 3; ii++)
    [unroll]
    for(int jj = 0; jj < 3; jj++)
      n[3 * ii + jj] = HF.Areadpixel(j_neighbours[jj], i_neighbours[ii]);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void hl_guide_laplacians(uint3 gid : SV_DispatchThreadID,
                         uniform ImageIn HF, uniform ImageIn LF, uniform ImageIn mask,
                         uniform ImageIn output_r, uniform ImageOut output_w,
                         uniform int width, uniform int height, uniform int mult,
                         uniform float noise_level, uniform int salt, uniform uint scale,
                         uniform float radius_sq)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float4 alpha_ch = mask.Areadpixel(x, y);
  const float alpha = alpha_ch.w;
  const float alpha_comp = 1.f - alpha;

  float4 high_frequency = HF.Areadpixel(x, y);

  if(alpha > 0.f) // reconstruct
  {
    float4 n[9];
    _fetch_neighbours(HF, x, y, width, height, mult, n);

    // linear fit of the laplacian of chromaticity against the laplacian of the norm
    float4 means_HF = float4(0.f);
    for(int k = 0; k < 9; k++) means_HF += n[k] / 9.f;

    float4 variance_HF = float4(0.f);
    for(int k = 0; k < 9; k++) variance_HF += (n[k] - means_HF) * (n[k] - means_HF) / 9.f;

    // the channel most likely to contain details = max(variance(HF)). As on
    // the CPU, the norm guides when no RGB channel has any variance
    int guide = ALPHA;
    float guiding_value_HF = 0.f;
    for(int c = 0; c < 3; c++)
    {
      if(variance_HF[c] > guiding_value_HF)
      {
        guiding_value_HF = variance_HF[c];
        guide = c;
      }
    }

    float4 covariance_HF = float4(0.f);
    for(int k = 0; k < 9; k++)
      covariance_HF += (n[k] - means_HF) * (n[k][guide] - means_HF[guide]) / 9.f;

    const float scale_multiplier = 1.f / radius_sq;
    const float4 a_HF = max(covariance_HF / variance_HF[guide], float4(0.f));
    const float4 b_HF = means_HF - a_HF * means_HF[guide];

    high_frequency = alpha_ch * scale_multiplier * (a_HF * high_frequency[guide] + b_HF)
                     + (1.f - alpha_ch * scale_multiplier) * high_frequency;
  }

  float4 out;
  if((scale & FIRST_SCALE) != 0)
    out = high_frequency;
  else
    out = output_r.Areadpixel(x, y) + high_frequency;

  if((scale & LAST_SCALE) != 0)
    out = max(out + LF.Areadpixel(x, y), float4(0.f));

  // last step of RGB reconstruct: add noise
  if((scale & LAST_SCALE) != 0 && salt != 0 && alpha > 0.f)
  {
    uint state[4] = { splitmix32(uint64_t(uint(x + 1))), splitmix32(uint64_t(uint((x + 1) * (y + 3)))),
                      splitmix32(1337ULL), splitmix32(666ULL) };
    xoshiro128plus(state);
    xoshiro128plus(state);
    xoshiro128plus(state);
    xoshiro128plus(state);

    const float4 sigma = out * noise_level;
    float4 noise = dt_noise_generator_simd(DT_NOISE_POISSONIAN, out, sigma, state);

    // the noise only brightens the image, since it's clipped
    noise = out + abs(noise - out);
    out = max(alpha * noise + alpha_comp * out, float4(0.f));
  }

  if((scale & LAST_SCALE) != 0)
  {
    // break the RGB channels into ratios/norm for the next step of reconstruction
    const float norm = max(sqrt(out.x * out.x + out.y * out.y + out.z * out.z), 1e-6f);
    out /= norm;
    out.w = norm;
  }

  output_w.write(int2(x, y), out);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void hl_diffuse_color(uint3 gid : SV_DispatchThreadID,
                      uniform ImageIn HF, uniform ImageIn LF, uniform ImageIn mask,
                      uniform ImageIn output_r, uniform ImageOut output_w,
                      uniform int width, uniform int height, uniform int mult,
                      uniform uint scale, uniform float first_order_factor)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float4 alpha = mask.Areadpixel(x, y);
  float4 high_frequency = HF.Areadpixel(x, y);
  const float norm_backup = high_frequency.w;

  if(alpha.w > 0.f) // reconstruct
  {
    float4 n[9];
    _fetch_neighbours(HF, x, y, width, height, mult, n);

    // laplacian in the direction parallel to the steepest gradient on the norm
    const float anisotropic_kernel_isophote[9] = { 0.25f, 0.5f, 0.25f, 0.5f, -3.f, 0.5f, 0.25f, 0.5f, 0.25f };
    float4 laplacian_HF = float4(0.f);
    for(int k = 0; k < 9; k++) laplacian_HF += n[k] * anisotropic_kernel_isophote[k];

    const float4 multipliers_HF = float4(float3(1.f / B_SPLINE_TO_LAPLACIAN), 0.f);
    high_frequency += alpha * multipliers_HF * (laplacian_HF - first_order_factor * high_frequency);

    // the norm is not diffused
    high_frequency.w = norm_backup;
  }

  float4 out;
  if((scale & FIRST_SCALE) != 0)
    out = high_frequency;
  else
    out = output_r.Areadpixel(x, y) + high_frequency;

  if((scale & LAST_SCALE) != 0)
  {
    out = max(out + LF.Areadpixel(x, y), float4(0.f));

    // renormalize ratios
    if(alpha.w > 0.f)
    {
      const float norm = sqrt(out.x * out.x + out.y * out.y + out.z * out.z);
      if(norm > 1e-4f) out.xyz /= norm;
    }

    // reconstruct RGB from ratios and norm, the norm stays in the 4th channel
    out.xyz *= out.w;
  }

  output_w.write(int2(x, y), out);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void hl_remosaic_and_replace(uint3 gid : SV_DispatchThreadID,
                             uniform ImageIn input, uniform ImageIn interpolated, uniform ImageIn clipping_mask,
                             uniform ImageOut output, uniform float4 wb, uniform int filters,
                             uniform int width, uniform int height)
{
  // take RGB ratios and norm, reconstruct RGB and remosaic the image
  const int j = gid.x;
  const int i = gid.y;
  if(j >= width || i >= height) return;

  const int c = FC(i, j, uint(filters));
  const float opacity = clipf(clipping_mask.Areadpixel(j, i).w);
  const float pix_in = input.Areadsingle(j, i);
  const float correction = max(interpolated.Areadpixel(j, i)[c] * wb[c], 0.f);
  output.write(int2(j, i), float4(opacity * correction + (1.f - opacity) * pix_in, 0.f, 0.f, 0.f));
}
