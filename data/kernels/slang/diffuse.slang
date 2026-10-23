/*
    This file is part of darktable,
    Copyright (C) 2021-2026 darktable developers.

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

// Slang port of ../diffuse.cl. The OpenCL mask is an 8 bit image; the Slang
// images are float, so here the mask holds 0 or 1 in a single channel float
// image.

import common;
import noise_generator;

// discretization parameters for the partial derivative equation solver
static const int H_STEP = 1;    // spatial step

static const int DT_ISOTROPY_ISOTROPE = 0; // diffuse in all directions with same intensity
static const int DT_ISOTROPY_ISOPHOTE = 1; // diffuse more in the isophote direction (orthogonal to gradient)
static const int DT_ISOTROPY_GRADIENT = 2; // diffuse more in the gradient direction

void find_gradient(const float4 pixels[9], out float4 xy[2])
{
  // compute the gradient with centered finite differences in a 3x3 stencil
  // warning : x is vertical, y is horizontal
  xy[0] = (pixels[7] - pixels[1]) / 2.f;
  xy[1] = (pixels[5] - pixels[3]) / 2.f;
}

float4 sqf(const float4 x)
{
  return x * x;
}

void rotation_matrix_isophote(const float4 c2, const float4 cos_theta_sin_theta,
                              const float4 cos_theta2, const float4 sin_theta2,
                              out float4 a[2][2])
{
  // the coefficients of a square symmetrical matrix of rotation of the gradient:
  // [[ a11, a12 ],
  //  [ a12, a22 ]]
  // taken from https://www.researchgate.net/publication/220663968
  // c dampens the gradient direction
  a[0][0] = cos_theta2 + c2 * sin_theta2;
  a[1][1] = c2 * cos_theta2 + sin_theta2;
  a[0][1] = (c2 - 1.0f) * cos_theta_sin_theta;
  a[1][0] = a[0][1];
}

void rotation_matrix_gradient(const float4 c2, const float4 cos_theta_sin_theta,
                              const float4 cos_theta2, const float4 sin_theta2,
                              out float4 a[2][2])
{
  // based on https://www.researchgate.net/publication/220663968 and inverted
  // c dampens the isophote direction
  a[0][0] = c2 * cos_theta2 + sin_theta2;
  a[1][1] = cos_theta2 + c2 * sin_theta2;
  a[0][1] = (1.0f - c2) * cos_theta_sin_theta;
  a[1][0] = a[0][1];
}

void build_matrix(const float4 a[2][2], out float4 kern[9])
{
  const float4 b11 = a[0][1] / 2.0f;
  const float4 b13 = -b11;
  const float4 b22 = -2.0f * (a[0][0] + a[1][1]);

  // the kernel of the rotated anisotropic laplacian, see ../diffuse.cl
  kern[0] = b11;
  kern[1] = a[1][1];
  kern[2] = b13;
  kern[3] = a[0][0];
  kern[4] = b22;
  kern[5] = a[0][0];
  kern[6] = b13;
  kern[7] = a[1][1];
  kern[8] = b11;
}

void isotrope_laplacian(out float4 kern[9])
{
  // see https://eng.aurelienpierre.com/2021/03/rotation-invariant-laplacian-for-2d-grids/#Second-order-isotropic-finite-differences
  // for references (Oono & Puri)
  kern[0] = float4(0.25f);
  kern[1] = float4(0.5f);
  kern[2] = float4(0.25f);
  kern[3] = float4(0.5f);
  kern[4] = float4(-3.f);
  kern[5] = float4(0.5f);
  kern[6] = float4(0.25f);
  kern[7] = float4(0.5f);
  kern[8] = float4(0.25f);
}

void compute_kern(const float4 c2, const float4 cos_theta_sin_theta,
                  const float4 cos_theta2, const float4 sin_theta2,
                  const int isotropy_type, out float4 kern[9])
{
  float4 a[2][2];
  if(isotropy_type == DT_ISOTROPY_ISOPHOTE)
  {
    rotation_matrix_isophote(c2, cos_theta_sin_theta, cos_theta2, sin_theta2, a);
    build_matrix(a, kern);
  }
  else if(isotropy_type == DT_ISOTROPY_GRADIENT)
  {
    rotation_matrix_gradient(c2, cos_theta_sin_theta, cos_theta2, sin_theta2, a);
    build_matrix(a, kern);
  }
  else
    isotrope_laplacian(kern);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void diffuse_pde(uint3 gid : SV_DispatchThreadID,
                 uniform ImageIn HF, uniform ImageIn LF,
                 uniform ImageIn mask, uniform int has_mask,
                 uniform ImageOut output,
                 uniform int width, uniform int height,
                 uniform float4 anisotropy, uniform int4 isotropy_type,
                 uniform float regularization, uniform float variance_threshold,
                 uniform float current_radius_square, uniform int mult,
                 uniform float4 ABCD, uniform float strength)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const bool opacity = has_mask ? mask.Areadsingle(x, y) != 0.0f : true;

  const float4 regularization_factor = float4(regularization * current_radius_square / 9.f);

  float4 result;

  if(opacity)
  {
    // non-local neighbours coordinates
    const int j_neighbours[3] = { clamp((x - mult * H_STEP), 0, width - 1),
                                  x,
                                  clamp((x + mult * H_STEP), 0, width - 1) };
    const int i_neighbours[3] = { clamp((y - mult * H_STEP), 0, height - 1),
                                  y,
                                  clamp((y + mult * H_STEP), 0, height - 1) };

    // fetch non-local pixels and store them locally and contiguously
    float4 neighbour_pixel_HF[9];
    float4 neighbour_pixel_LF[9];

    for(int ii = 0; ii < 3; ii++)
      for(int jj = 0; jj < 3; jj++)
      {
        neighbour_pixel_HF[3 * ii + jj] = HF.Areadpixel(j_neighbours[ii], i_neighbours[jj]);
        neighbour_pixel_LF[3 * ii + jj] = LF.Areadpixel(j_neighbours[ii], i_neighbours[jj]);
      }

    // build the local anisotropic convolution filters for gradients and laplacians
    float4 gradient[2], laplacian[2];
    find_gradient(neighbour_pixel_LF, gradient);
    find_gradient(neighbour_pixel_HF, laplacian);

    // cos and sin of arg(grad) and arg(lapl), with arg forced to 0 where the
    // magnitude is 0
    const float4 magnitude_grad = dtcl_sqrt(sqf(gradient[0]) + sqf(gradient[1]));
    gradient[0] = select(magnitude_grad != 0.f, gradient[0] / magnitude_grad, float4(1.f));
    gradient[1] = select(magnitude_grad != 0.f, gradient[1] / magnitude_grad, float4(0.f));

    const float4 magnitude_lapl = dtcl_sqrt(sqf(laplacian[0]) + sqf(laplacian[1]));
    laplacian[0] = select(magnitude_lapl != 0.f, laplacian[0] / magnitude_lapl, float4(1.f));
    laplacian[1] = select(magnitude_lapl != 0.f, laplacian[1] / magnitude_lapl, float4(0.f));

    const float4 cos_theta_grad_sq = sqf(gradient[0]);
    const float4 sin_theta_grad_sq = sqf(gradient[1]);
    const float4 cos_theta_sin_theta_grad = gradient[0] * gradient[1];
    const float4 cos_theta_lapl_sq = sqf(laplacian[0]);
    const float4 sin_theta_lapl_sq = sqf(laplacian[1]);
    const float4 cos_theta_sin_theta_lapl = laplacian[0] * laplacian[1];

    // c² in https://www.researchgate.net/publication/220663968
    // warning : in c2[s], s is the order of the derivative
    const float4 c2[4] = { dtcl_exp(-magnitude_grad * anisotropy.x),
                           dtcl_exp(-magnitude_lapl * anisotropy.y),
                           dtcl_exp(-magnitude_grad * anisotropy.z),
                           dtcl_exp(-magnitude_lapl * anisotropy.w) };

    float4 kern_first[9], kern_second[9], kern_third[9], kern_fourth[9];
    compute_kern(c2[0], cos_theta_sin_theta_grad, cos_theta_grad_sq, sin_theta_grad_sq, isotropy_type.x, kern_first);
    compute_kern(c2[1], cos_theta_sin_theta_lapl, cos_theta_lapl_sq, sin_theta_lapl_sq, isotropy_type.y, kern_second);
    compute_kern(c2[2], cos_theta_sin_theta_grad, cos_theta_grad_sq, sin_theta_grad_sq, isotropy_type.z, kern_third);
    compute_kern(c2[3], cos_theta_sin_theta_lapl, cos_theta_lapl_sq, sin_theta_lapl_sq, isotropy_type.w, kern_fourth);

    // convolve filters and compute the variance and the regularization term
    float4 derivatives[4] = { float4(0.f), float4(0.f), float4(0.f), float4(0.f) };
    float4 variance = float4(0.f);

    [unroll]
    for(int k = 0; k < 9; k++)
    {
      derivatives[0] += kern_first[k] * neighbour_pixel_LF[k];
      derivatives[1] += kern_second[k] * neighbour_pixel_LF[k];
      derivatives[2] += kern_third[k] * neighbour_pixel_HF[k];
      derivatives[3] += kern_fourth[k] * neighbour_pixel_HF[k];
      variance += sqf(neighbour_pixel_HF[k]);
    }

    // regularize the variance taking into account the blurring scale, so that
    // the scene-referred variance stays roughly constant whatever the wavelet
    // scale, which prevents large scale halos when deblurring
    variance = variance_threshold + variance * regularization_factor;

    // compute the update
    float4 acc = float4(0.f);
    for(int k = 0; k < 4; k++) acc += derivatives[k] * ABCD[k];
    const float4 hf = HF.Areadpixel(x, y);
    acc = (hf * strength + acc / variance);

    // update the solution
    const float4 lf = LF.Areadpixel(x, y);
    result = max(acc + lf, float4(0.f));
  }
  else
  {
    const float4 hf = HF.Areadpixel(x, y);
    const float4 lf = LF.Areadpixel(x, y);
    result = hf + lf;
  }

  output.write(int2(x, y), result);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void build_mask(uint3 gid : SV_DispatchThreadID,
                uniform ImageIn in, uniform ImageOut mask,
                uniform float threshold, uniform int width, uniform int height)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float4 pix_in = in.Areadpixel(x, y);
  const bool m = pix_in.x > threshold || pix_in.y > threshold || pix_in.z > threshold;
  mask.write(int2(x, y), float4(m ? 1.0f : 0.0f));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void inpaint_mask(uint3 gid : SV_DispatchThreadID,
                  uniform ImageOut inpainted, uniform ImageIn original,
                  uniform ImageIn mask, uniform int width, uniform int height)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float4 pix_in = original.Areadpixel(x, y);
  float4 pix_out = pix_in;

  if(mask.Areadsingle(x, y) != 0.0f)
  {
    uint state[4] = { splitmix32(x + 1), splitmix32((x + 1) * (y + 3)), splitmix32(1337), splitmix32(666) };
    xoshiro128plus(state);
    xoshiro128plus(state);
    xoshiro128plus(state);
    xoshiro128plus(state);
    pix_out = abs(dt_noise_generator_simd(DT_NOISE_GAUSSIAN, pix_in, pix_in, state));
  }

  inpainted.write(int2(x, y), pix_out);
}
