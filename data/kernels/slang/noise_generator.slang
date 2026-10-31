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

// Slang module form of ../noise_generator.h

module noise_generator;

__exported import common;

// dt_noise_distribution_t
public static const int DT_NOISE_UNIFORM = 0;
public static const int DT_NOISE_GAUSSIAN = 1;
public static const int DT_NOISE_POISSONIAN = 2;

public uint splitmix32(uint64_t seed)
{
  // fast random number generator
  // reference: https://gist.github.com/imneme/6179748664e88ef3c34860f44309fc71
  uint64_t result = (seed ^ (seed >> 33)) * 0x62a9d9ed799705f5ULL;
  result = (result ^ (result >> 28)) * 0xcb24d0a5c88c35b3ULL;
  return uint(result >> 32);
}

uint rol32(uint x, int k)
{
  return (x << k) | (x >> (32 - k));
}

public float xoshiro128plus(inout uint state[4])
{
  // fast random number generator
  // reference: http://prng.di.unimi.it/
  const uint result = state[0] + state[3];
  const uint t = state[1] << 9;

  state[2] ^= state[0];
  state[3] ^= state[1];
  state[1] ^= state[2];
  state[0] ^= state[3];

  state[2] ^= t;
  state[3] = rol32(state[3], 11);

  return float(result >> 8) * 5.9604644775390625e-8f; // 2^-24: the first 24 bits in the mantissa
}

float4 uniform_noise_simd(float4 mu, float4 sigma, inout uint state[4])
{
  const float n0 = xoshiro128plus(state);
  const float n1 = xoshiro128plus(state);
  const float n2 = xoshiro128plus(state);
  const float4 noise = float4(n0, n1, n2, 0.f);
  return mu + 2.0f * (noise - 0.5f) * sigma;
}

float4 gaussian_noise_simd(float4 mu, float4 sigma, inout uint state[4])
{
  // Box-Muller transform: https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
  float4 u1 = float4(0.f);
  float4 u2 = float4(0.f);

  u1.x = xoshiro128plus(state);
  u1.y = xoshiro128plus(state);
  u1.z = xoshiro128plus(state);

  u2.x = xoshiro128plus(state);
  u2.y = xoshiro128plus(state);
  u2.z = xoshiro128plus(state);

  u1 = max(u1, float4(1.175494350822287507969e-38f));

  const float4 flip = float4(1.0f, 0.0f, 1.0f, 0.0f);
  const float4 flip_comp = float4(0.0f, 1.0f, 0.0f, 0.0f);

  const float4 noise = flip * dtcl_sqrt(-2.0f * dtcl_log(u1)) * dtcl_cos(DT_2PI_F * u2)
                       + flip_comp * dtcl_sqrt(-2.0f * dtcl_log(u1)) * dtcl_sin(DT_2PI_F * u2);
  return noise * sigma + mu;
}

float4 poisson_noise_simd(float4 mu, float4 sigma, inout uint state[4])
{
  // gaussian noise with the Anscombe transform applied
  float4 u1 = float4(0.f);
  float4 u2 = float4(0.f);

  // the CPU path draws the numbers in this order
  u1.x = xoshiro128plus(state);
  u2.x = xoshiro128plus(state);
  u1.y = xoshiro128plus(state);
  u2.y = xoshiro128plus(state);
  u1.z = xoshiro128plus(state);
  u2.z = xoshiro128plus(state);

  u1 = max(u1, float4(1.175494350822287507969e-38f));

  const float4 flip = float4(1.0f, 0.0f, 1.0f, 0.0f);
  const float4 flip_comp = float4(0.0f, 1.0f, 0.0f, 0.0f);

  const float4 noise = flip * dtcl_sqrt(-2.0f * dtcl_log(u1)) * dtcl_cos(DT_2PI_F * u2)
                       + flip_comp * dtcl_sqrt(-2.0f * dtcl_log(u1)) * dtcl_sin(DT_2PI_F * u2);

  const float4 r = noise * sigma + 2.0f * dtcl_sqrt(max(mu + (3.f / 8.f), float4(0.0f)));
  return ((r * r - sigma * sigma) / (4.f)) - (3.f / 8.f);
}

public float4 dt_noise_generator_simd(int distribution, float4 mu, float4 param, inout uint state[4])
{
  if(distribution == DT_NOISE_GAUSSIAN) return gaussian_noise_simd(mu, param, state);
  if(distribution == DT_NOISE_POISSONIAN) return poisson_noise_simd(mu, param, state);
  return uniform_noise_simd(mu, param, state);
}
