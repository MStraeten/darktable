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

// Slang ports of the kernels in ../sharpen.cl. The OpenCL blurs stage a row
// or column in local memory; these read the image directly, in the same
// order, so the sums are the same.

import common;

[shader("compute")]
[numthreads(16, 16, 1)]
void sharpen_hblur(uint3 gid : SV_DispatchThreadID,
                   uniform ImageIn in, uniform ImageOut out,
                   uniform StructuredBuffer<float> m,
                   uniform int rad, uniform int width, uniform int height)
{
  const int x = int(gid.x);
  const int y = int(gid.y);
  if(x >= width || y >= height) return;

  float4 pixel = in.readpixel(x, y);
  if(x >= rad && x < width - rad)
  {
    float sum = 0.0f;
    for(int i = -rad; i <= rad; i++)
      sum += in.readsingle(x + i, y) * m[i + rad];
    pixel.x = sum;
  }
  out.write(int2(x, y), pixel);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void sharpen_vblur(uint3 gid : SV_DispatchThreadID,
                   uniform ImageIn in, uniform ImageOut out,
                   uniform StructuredBuffer<float> m,
                   uniform int rad, uniform int width, uniform int height)
{
  const int x = int(gid.x);
  const int y = int(gid.y);
  if(x >= width || y >= height) return;

  float4 pixel = in.readpixel(x, y);
  if(y >= rad && y < height - rad)
  {
    float sum = 0.0f;
    for(int i = -rad; i <= rad; i++)
      sum += in.readsingle(x, y + i) * m[i + rad];
    pixel.x = sum;
  }
  out.write(int2(x, y), pixel);
}

// final mixing step: in_a is the original image, in_b the blurred one
[shader("compute")]
[numthreads(16, 16, 1)]
void sharpen_mix(uint3 gid : SV_DispatchThreadID,
                 uniform ImageIn in_a, uniform ImageIn in_b, uniform ImageOut out,
                 uniform int width, uniform int height,
                 uniform float sharpen, uniform float thrs, uniform int rad)
{
  const int x = int(gid.x);
  const int y = int(gid.y);
  if(x >= width || y >= height) return;

  float4 pixel = in_a.readpixel(x, y);
  if(x >= rad && y >= rad && x < width - rad && y < height - rad)
  {
    const float blurredx = in_b.readsingle(x, y);
    const float delta = pixel.x - blurredx;
    const float amount = sharpen * copysign(max(0.0f, abs(delta) - thrs), delta);
    pixel.x = pixel.x + amount;
  }
  out.write(int2(x, y), pixel);
}
