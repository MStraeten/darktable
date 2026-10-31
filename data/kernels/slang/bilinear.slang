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

// Slang ports of the buffer kernels in ../bilinear.cl, with the sampling of
// ../bilinear.h: a GPU counterpart of interpolate_bilinear() in
// common/fast_guided_filter.h, which resamples the working buffers of the
// guided filters

import common;

[shader("compute")]
[numthreads(16, 16, 1)]
void bilinear1(uint3 gid : SV_DispatchThreadID,
               uniform StructuredBuffer<float> in, uniform int width_in, uniform int height_in,
               uniform RWStructuredBuffer<float> out, uniform int width_out, uniform int height_out)
{
  const int x = int(gid.x);
  const int y = int(gid.y);
  if(x >= width_out || y >= height_out) return;

  // relative coordinates of the pixel in output space, then absolute ones in input space
  const float x_in = (float(x) / float(width_out)) * float(width_in);
  const float y_in = (float(y) / float(height_out)) * float(height_in);

  // nearest neighbours in input space
  int x_prev = int(floor(x_in));
  int x_next = x_prev + 1;
  int y_prev = int(floor(y_in));
  int y_next = y_prev + 1;
  x_prev = (x_prev < width_in) ? x_prev : width_in - 1;
  x_next = (x_next < width_in) ? x_next : width_in - 1;
  y_prev = (y_prev < height_in) ? y_prev : height_in - 1;
  y_next = (y_next < height_in) ? y_next : height_in - 1;

  const float Q_NW = in[y_prev * width_in + x_prev];
  const float Q_NE = in[y_prev * width_in + x_next];
  const float Q_SE = in[y_next * width_in + x_next];
  const float Q_SW = in[y_next * width_in + x_prev];

  // spatial differences between nodes, next - prev = 1
  const float Dy_next = float(y_next) - y_in;
  const float Dy_prev = 1.0f - Dy_next;
  const float Dx_next = float(x_next) - x_in;
  const float Dx_prev = 1.0f - Dx_next;

  out[y * width_out + x] = Dy_prev * (Q_SW * Dx_next + Q_SE * Dx_prev)
                         + Dy_next * (Q_NW * Dx_next + Q_NE * Dx_prev);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void bilinear2(uint3 gid : SV_DispatchThreadID,
               uniform StructuredBuffer<float2> in, uniform int width_in, uniform int height_in,
               uniform RWStructuredBuffer<float2> out, uniform int width_out, uniform int height_out)
{
  const int x = int(gid.x);
  const int y = int(gid.y);
  if(x >= width_out || y >= height_out) return;

  // relative coordinates of the pixel in output space, then absolute ones in input space
  const float x_in = (float(x) / float(width_out)) * float(width_in);
  const float y_in = (float(y) / float(height_out)) * float(height_in);

  // nearest neighbours in input space
  int x_prev = int(floor(x_in));
  int x_next = x_prev + 1;
  int y_prev = int(floor(y_in));
  int y_next = y_prev + 1;
  x_prev = (x_prev < width_in) ? x_prev : width_in - 1;
  x_next = (x_next < width_in) ? x_next : width_in - 1;
  y_prev = (y_prev < height_in) ? y_prev : height_in - 1;
  y_next = (y_next < height_in) ? y_next : height_in - 1;

  const float2 Q_NW = in[y_prev * width_in + x_prev];
  const float2 Q_NE = in[y_prev * width_in + x_next];
  const float2 Q_SE = in[y_next * width_in + x_next];
  const float2 Q_SW = in[y_next * width_in + x_prev];

  // spatial differences between nodes, next - prev = 1
  const float Dy_next = float(y_next) - y_in;
  const float Dy_prev = 1.0f - Dy_next;
  const float Dx_next = float(x_next) - x_in;
  const float Dx_prev = 1.0f - Dx_next;

  out[y * width_out + x] = Dy_prev * (Q_SW * Dx_next + Q_SE * Dx_prev)
                         + Dy_next * (Q_NW * Dx_next + Q_NE * Dx_prev);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void bilinear4(uint3 gid : SV_DispatchThreadID,
               uniform StructuredBuffer<float4> in, uniform int width_in, uniform int height_in,
               uniform RWStructuredBuffer<float4> out, uniform int width_out, uniform int height_out)
{
  const int x = int(gid.x);
  const int y = int(gid.y);
  if(x >= width_out || y >= height_out) return;

  // relative coordinates of the pixel in output space, then absolute ones in input space
  const float x_in = (float(x) / float(width_out)) * float(width_in);
  const float y_in = (float(y) / float(height_out)) * float(height_in);

  // nearest neighbours in input space
  int x_prev = int(floor(x_in));
  int x_next = x_prev + 1;
  int y_prev = int(floor(y_in));
  int y_next = y_prev + 1;
  x_prev = (x_prev < width_in) ? x_prev : width_in - 1;
  x_next = (x_next < width_in) ? x_next : width_in - 1;
  y_prev = (y_prev < height_in) ? y_prev : height_in - 1;
  y_next = (y_next < height_in) ? y_next : height_in - 1;

  const float4 Q_NW = in[y_prev * width_in + x_prev];
  const float4 Q_NE = in[y_prev * width_in + x_next];
  const float4 Q_SE = in[y_next * width_in + x_next];
  const float4 Q_SW = in[y_next * width_in + x_prev];

  // spatial differences between nodes, next - prev = 1
  const float Dy_next = float(y_next) - y_in;
  const float Dy_prev = 1.0f - Dy_next;
  const float Dx_next = float(x_next) - x_in;
  const float Dx_prev = 1.0f - Dx_next;

  out[y * width_out + x] = Dy_prev * (Q_SW * Dx_next + Q_SE * Dx_prev)
                         + Dy_next * (Q_NW * Dx_next + Q_NE * Dx_prev);
}
