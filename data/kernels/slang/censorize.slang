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

// Slang port of the pixelation and noise steps of process() in
// src/iop/censorize.c, which has no OpenCL code. The blurs around them are
// the gaussian of gaussian.slang.

import common;
import noise_generator;

// each work item paints its pixel with the average of the 5 points of its
// big pixel that the CPU code takes. The CPU code paints each big pixel up
// to, but not including, its bottom right corner, so the last column and row
// of the image are never written there and usually come out black; they are
// written as zeroes here, to keep both paths alike.
[shader("compute")]
[numthreads(16, 16, 1)]
void censorize_pixelate(uint3 gid : SV_DispatchThreadID,
                        uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
                        uniform int radius)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const int2 size = int2(width, height);
  const int2 tl = clamp(2 * radius * (int2(x, y) / (2 * radius)), int2(0), size - 1);
  const int2 cc = clamp(tl + radius, int2(0), size - 1);
  const int2 br = clamp(cc + radius, int2(0), size - 1);
  if(x >= br.x || y >= br.y)
  {
    out.write(int2(x, y), float4(0.0f));
    return;
  }

  const float4 RGB = (in.readA(tl) + in.readA(int2(br.x, tl.y)) + in.readA(cc)
                      + in.readA(int2(tl.x, br.y)) + in.readA(br)) / 5.0f;
  out.write(int2(x, y), RGB);
}

// multiplicative gaussian noise, scaled by the green channel, with the
// random state of make_noise()
[shader("compute")]
[numthreads(16, 16, 1)]
void censorize_noise(uint3 gid : SV_DispatchThreadID,
                     uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
                     uniform float noise)
{
  const int j = gid.x;
  const int i = gid.y;
  if(j >= width || i >= height) return;

  uint state[4] = { splitmix32(uint64_t(j + 1)), splitmix32(uint64_t(j + 1) * uint64_t(i + 3)),
                    splitmix32(1337), splitmix32(666) };
  xoshiro128plus(state);
  xoshiro128plus(state);
  xoshiro128plus(state);
  xoshiro128plus(state);

  float4 pixel = in.Areadpixel(j, i);
  const float norm = pixel.y;

  // gaussian_noise() of src/develop/noise_generator.h
  const float u1 = max(xoshiro128plus(state), 1.175494350822287507969e-38f);
  const float u2 = xoshiro128plus(state);
  const bool flip = (i % 2) != 0 || (j % 2) != 0;
  const float n = flip ? sqrt(-2.0f * log(u1)) * cos(DT_2PI_F * u2) : sqrt(-2.0f * log(u1)) * sin(DT_2PI_F * u2);
  const float epsilon = (n * noise * norm + norm) / norm;

  pixel.xyz = max(pixel.xyz * epsilon, 0.0f);
  out.write(int2(j, i), pixel);
}
