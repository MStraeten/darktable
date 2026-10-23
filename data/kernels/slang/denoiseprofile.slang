/*
    This file is part of darktable,
    copyright (c) 2011-2026 darktable developers.

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

// Slang port of ../denoiseprofile.cl. The OpenCL box filters and the first
// reduction stage stage their data in local memory; here each work item reads
// what it needs directly. The second reduction stage has no port: the host
// adds up the first stage's block sums.

import common;

/*
    To speed up processing we use an algorithm proposed by B. Goossens, H.Q. Luong, J. Aelterman, A. Pizurica,  and W. Philips,
    "A GPU-Accelerated Real-Time NLMeans Algorithm for Denoising Color Video Sequences", in Proc. ACIVS (2), 2010, pp.46-57.
*/

float ddirac(const int2 q)
{
  return ((q.x != 0 || q.y != 0) ? 1.0f : 0.0f);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void denoiseprofile_precondition(uint3 gid : SV_DispatchThreadID,
                                 uniform ImageIn in, uniform ImageOut out,
                                 uniform int width, uniform int height,
                                 uniform float4 a, uniform float4 sigma2)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float4 pixel = in.readpixel(x, y);
  const float alpha = pixel.w;

  const float4 t = max(pixel / a, 0.f);
  const float4 d = max(float4(0.0f), t + float4(0.375f) + sigma2);
  float4 s = 2.0f * dtcl_sqrt(d);

  s.w = alpha;
  out.write(int2(x, y), s);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void denoiseprofile_precondition_v2(uint3 gid : SV_DispatchThreadID,
                                    uniform ImageIn in, uniform ImageOut out,
                                    uniform int width, uniform int height,
                                    uniform float4 a, uniform float4 p, uniform float4 b, uniform float4 wb)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float4 pixel = in.readpixel(x, y);
  const float alpha = pixel.w;

  float4 t = max(2.0f * dtcl_pow(max(float4(0.0f), pixel / wb + b), 1.0f - p / 2.0f) / ((-p + 2.0f) * dtcl_sqrt(a)), 0.f);

  t.w = alpha;
  out.write(int2(x, y), t);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void denoiseprofile_precondition_Y0U0V0(uint3 gid : SV_DispatchThreadID,
                                        uniform ImageIn in, uniform ImageOut out,
                                        uniform int width, uniform int height,
                                        uniform float4 a, uniform float4 p, uniform float4 b,
                                        uniform StructuredBuffer<float> toY0U0V0)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float4 pixel = in.readpixel(x, y);
  const float alpha = pixel.w;

  const float4 t = max(2.0f * dtcl_pow(max(float4(0.0f), pixel + b), 1.0f - p / 2.0f) / ((-p + 2.0f) * dtcl_sqrt(a)), 0.f);

  float4 outpx = float4(0.0f);
  outpx.x += toY0U0V0[0] * t.x;
  outpx.x += toY0U0V0[1] * t.y;
  outpx.x += toY0U0V0[2] * t.z;
  outpx.y += toY0U0V0[3] * t.x;
  outpx.y += toY0U0V0[4] * t.y;
  outpx.y += toY0U0V0[5] * t.z;
  outpx.z += toY0U0V0[6] * t.x;
  outpx.z += toY0U0V0[7] * t.y;
  outpx.z += toY0U0V0[8] * t.z;

  outpx.w = alpha;
  out.write(int2(x, y), outpx);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void denoiseprofile_dist(uint3 gid : SV_DispatchThreadID,
                         uniform ImageIn in, uniform RWStructuredBuffer<float> U4,
                         uniform int width, uniform int height, uniform int2 q)
{
  const int x = gid.x;
  const int y = gid.y;
  const int gidx = y * width + x;
  if(x >= width || y >= height) return;

  int xpq = x + q.x;
  int ypq = y + q.y;
  // out of bounds indexes become 0. q.x and q.y can be negative
  xpq *= (x + q.x < width && x + q.x >= 0) ? 1 : 0;
  ypq *= (y + q.y < height && y + q.y >= 0) ? 1 : 0;

  const float4 p1 = in.readpixel(x, y);
  const float4 p2 = in.readpixel(xpq, ypq);
  const float4 tmp = (p1 - p2) * (p1 - p2);
  float dist = tmp.x + tmp.y + tmp.z;

  // dist is 0 where xpq or ypq is out of bounds
  dist *= (x + q.x < width && x + q.x >= 0 && y + q.y < height && y + q.y >= 0) ? 1.0f : 0.0f;

  U4[gidx] = dist;
}

// the OpenCL kernels take q but don't use it, and Metal drops parameters
// nothing reads, which would shift every later one
void _keep_q(RWStructuredBuffer<float> U4_out, const int2 q, const int width)
{
  if(width < 0) U4_out[0] = float(q.x + q.y);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void denoiseprofile_horiz(uint3 gid : SV_DispatchThreadID,
                          uniform StructuredBuffer<float> U4_in, uniform RWStructuredBuffer<float> U4_out,
                          uniform int width, uniform int height, uniform int2 q, uniform int P)
{
  const int x = gid.x;
  const int y = gid.y;
  _keep_q(U4_out, q, width);
  if(x >= width || y >= height) return;

  // the same sum as the OpenCL kernel takes from its local buffer
  float distacc = 0.0f;
  for(int pi = -P; pi <= P; pi++)
    distacc += U4_in[y * width + clamp(x + pi, 0, width - 1)];

  U4_out[y * width + x] = distacc;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void denoiseprofile_vert(uint3 gid : SV_DispatchThreadID,
                         uniform StructuredBuffer<float> U4_in, uniform RWStructuredBuffer<float> U4_out,
                         uniform int width, uniform int height, uniform int2 q, uniform int P,
                         uniform float norm, uniform float central_pixel_weight,
                         uniform StructuredBuffer<float> U4_single_pixel)
{
  const int x = gid.x;
  const int y = gid.y;
  _keep_q(U4_out, q, width);
  if(x >= width || y >= height) return;

  const int gidx = y * width + x;

  float distacc = 0.0f;
  for(int pj = -P; pj <= P; pj++)
    distacc += U4_in[clamp(y + pj, 0, height - 1) * width + x];

  distacc += U4_single_pixel[gidx] * (2 * P + 1) * (2 * P + 1) * central_pixel_weight;
  distacc /= (1.0f + central_pixel_weight);

  distacc = fast_mexp2f(max(0.0f, distacc * norm - 2.0f));

  U4_out[gidx] = distacc;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void denoiseprofile_accu(uint3 gid : SV_DispatchThreadID,
                         uniform ImageIn in, uniform RWStructuredBuffer<float4> U2,
                         uniform StructuredBuffer<float> U4,
                         uniform int width, uniform int height, uniform int2 q)
{
  const int x = gid.x;
  const int y = gid.y;
  const int gidx = y * width + x;
  if(x >= width || y >= height) return;

  // wpq and wmq weigh the reads at (x, y) + q and (x, y) - q, and are 1 only
  // where their position is in bounds
  int wpq = 1;
  int wmq = 1;

  wpq *= (x + q.x < width) ? 1 : 0;
  wmq *= (x - q.x < width) ? 1 : 0;
  wpq *= (x + q.x >= 0) ? 1 : 0;
  wmq *= (x - q.x >= 0) ? 1 : 0;

  wpq *= (y + q.y >= 0) ? 1 : 0;
  wmq *= (y - q.y >= 0) ? 1 : 0;
  wpq *= (y + q.y < height) ? 1 : 0;
  wmq *= (y - q.y < height) ? 1 : 0;

  const float4 u1_pq = wpq != 0 ? in.readpixel(x + q.x, y + q.y) : float4(0.0f);
  const float4 u1_mq = wmq != 0 ? in.readpixel(x - q.x, y - q.y) : float4(0.0f);

  const float u4 = U4[gidx];
  const float u4_mq = U4[clamp(y - q.y, 0, height - 1) * width + clamp(x - q.x, 0, width - 1)];

  const float u4_mq_dd = u4_mq * ddirac(q);

  float4 accu = (u4 * u1_pq) + (u4_mq_dd * u1_mq);
  accu.w = (wpq * u4 + wmq * u4_mq_dd);

  U2[gidx] += accu;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void denoiseprofile_finish(uint3 gid : SV_DispatchThreadID,
                           uniform ImageIn in, uniform StructuredBuffer<float4> U2, uniform ImageOut out,
                           uniform int width, uniform int height, uniform float4 a, uniform float4 sigma2)
{
  const int x = gid.x;
  const int y = gid.y;
  const int gidx = y * width + x;
  if(x >= width || y >= height) return;

  const float4 u2 = U2[gidx];
  const float alpha = in.readalpha(x, y);

  float4 px = u2.w > 0.0f ? u2 / u2.w : float4(0.0f);

  px = select(px < float4(0.5f), float4(0.0f),
              0.25f * px * px + 0.25f * sqrt(1.5f) / px - 1.375f / (px * px) + 0.625f * dtcl_sqrt(1.5f) / (px * px * px) - 0.125f - sigma2);

  px *= a;
  px.w = alpha;
  out.write(int2(x, y), px);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void denoiseprofile_finish_v2(uint3 gid : SV_DispatchThreadID,
                              uniform ImageIn in, uniform StructuredBuffer<float4> U2, uniform ImageOut out,
                              uniform int width, uniform int height, uniform float4 a, uniform float4 p,
                              uniform float4 b, uniform float bias, uniform float4 wb)
{
  const int x = gid.x;
  const int y = gid.y;
  const int gidx = y * width + x;
  if(x >= width || y >= height) return;

  const float4 u2 = U2[gidx];
  const float alpha = in.readalpha(x, y);

  float4 px = u2.w > 0.0f ? u2 / u2.w : float4(0.0f);

  const float4 delta = px * px + float4(bias);
  const float4 denominator = 4.0f / (sqrt(a) * (2.0f - p));
  const float4 z1 = (px + dtcl_sqrt(max(float4(0.0f), delta))) / denominator;
  px = max(dtcl_pow(z1, 1.0f / (1.0f - p / 2.0f)) - b, 0.f);
  px = px * wb;
  px.w = alpha;
  out.write(int2(x, y), px);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void denoiseprofile_backtransform(uint3 gid : SV_DispatchThreadID,
                                  uniform ImageIn in, uniform ImageOut out,
                                  uniform int width, uniform int height, uniform float4 a, uniform float4 sigma2)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  float4 px = in.readpixel(x, y);
  const float alpha = px.w;

  px = select(px < float4(0.5f), float4(0.0f),
              0.25f * px * px + 0.25f * sqrt(1.5f) / px - 1.375f / (px * px) + 0.625f * sqrt(1.5f) / (px * px * px) - 0.125f - sigma2);

  px *= a;
  px.w = alpha;
  out.write(int2(x, y), px);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void denoiseprofile_backtransform_v2(uint3 gid : SV_DispatchThreadID,
                                     uniform ImageIn in, uniform ImageOut out,
                                     uniform int width, uniform int height, uniform float4 a, uniform float4 p,
                                     uniform float4 b, uniform float bias, uniform float4 wb)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  float4 px = in.readpixel(x, y);
  const float alpha = px.w;

  px = max(float4(0.0f), px);
  const float4 delta = px * px + float4(bias);
  const float4 denominator = 4.0f / (sqrt(a) * (2.0f - p));
  const float4 z1 = (px + sqrt(max(float4(0.0f), delta))) / denominator;
  px = max(dtcl_pow(z1, 1.0f / (1.0f - p / 2.0f)) - b, 0.f);
  px = px * wb;
  px.w = alpha;
  out.write(int2(x, y), px);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void denoiseprofile_backtransform_Y0U0V0(uint3 gid : SV_DispatchThreadID,
                                         uniform ImageIn in, uniform ImageOut out,
                                         uniform int width, uniform int height, uniform float4 a, uniform float4 p,
                                         uniform float4 b, uniform float bias, uniform float4 wb,
                                         uniform StructuredBuffer<float> toRGB)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float4 t = in.readpixel(x, y);
  const float alpha = t.w;

  float4 px = float4(0.0f);
  px.x += toRGB[0] * t.x;
  px.x += toRGB[1] * t.y;
  px.x += toRGB[2] * t.z;
  px.y += toRGB[3] * t.x;
  px.y += toRGB[4] * t.y;
  px.y += toRGB[5] * t.z;
  px.z += toRGB[6] * t.x;
  px.z += toRGB[7] * t.y;
  px.z += toRGB[8] * t.z;

  px = max(float4(0.0f), px);
  const float4 delta = px * px + float4(bias) * wb;
  const float4 denominator = 4.0f / (sqrt(a) * (2.0f - p));
  const float4 z1 = (px + sqrt(max(float4(0.0f), delta))) / denominator;
  px = max(dtcl_pow(z1, 1.0f / (1.0f - p / 2.0f)) - b, 0.f);
  px.w = alpha;
  out.write(int2(x, y), px);
}

float4 weight(const float4 c1, const float4 c2, const float inv_sigma2)
{
  const float4 sqr = (c1 - c2) * (c1 - c2);
  const float dt = (sqr.x + sqr.y + sqr.z) * inv_sigma2;
  const float var = 0.02f;
  const float off2 = 9.0f;
  const float r = fast_mexp2f(max(0.0f, dt * var - off2));

  return float4(r);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void denoiseprofile_decompose(uint3 gid : SV_DispatchThreadID,
                              uniform ImageIn in, uniform ImageOut coarse, uniform ImageOut detail,
                              uniform int width, uniform int height, uniform uint scale, uniform float inv_sigma2,
                              uniform StructuredBuffer<float> filter)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const int mult = 1 << scale;

  const float4 pixel = in.readpixel(x, y);
  float4 sum = float4(0.0f);
  float4 wgt = float4(0.0f);

  for(int j = 0; j < 5; j++)
    for(int i = 0; i < 5; i++)
    {
      const int xx = mult * (i - 2) + x;
      const int yy = mult * (j - 2) + y;
      const int k = j * 5 + i;

      const float4 px = in.readpixel(xx, yy);
      const float4 w = filter[k] * weight(pixel, px, inv_sigma2);

      sum += w * px;
      wgt += w;
    }

  sum /= wgt;
  sum.w = pixel.w;

  detail.write(int2(x, y), pixel - sum);
  coarse.write(int2(x, y), sum);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void denoiseprofile_synthesize(uint3 gid : SV_DispatchThreadID,
                               uniform ImageIn coarse, uniform ImageIn detail, uniform ImageOut out,
                               uniform int width, uniform int height,
                               uniform float4 threshold, uniform float4 boost)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float4 c = coarse.readpixel(x, y);
  const float4 d = detail.readpixel(x, y);
  const float4 amount = copysign(max(float4(0.0f), abs(d) - threshold), d);
  float4 sum = c + boost * amount;
  sum.w = c.w;
  out.write(int2(x, y), sum);
}

// the sum of the squared pixels in each 16x16 block, one work item per
// block, where the OpenCL kernel runs a work group per block
static const int REDUCE_BLOCK = 16;

[shader("compute")]
[numthreads(16, 16, 1)]
void denoiseprofile_reduce_first(uint3 gid : SV_DispatchThreadID,
                                 uniform ImageIn in, uniform int width, uniform int height,
                                 uniform RWStructuredBuffer<float4> accu)
{
  const int bx = gid.x;
  const int by = gid.y;
  const int nbx = (width + REDUCE_BLOCK - 1) / REDUCE_BLOCK;
  const int nby = (height + REDUCE_BLOCK - 1) / REDUCE_BLOCK;
  if(bx >= nbx || by >= nby) return;

  float4 sum = float4(0.0f);
  for(int j = 0; j < REDUCE_BLOCK; j++)
  {
    const int y = by * REDUCE_BLOCK + j;
    if(y >= height) break;
    for(int i = 0; i < REDUCE_BLOCK; i++)
    {
      const int x = bx * REDUCE_BLOCK + i;
      if(x >= width) break;
      const float4 pixel = in.readpixel(x, y);
      sum += pixel * pixel;
    }
  }
  accu[by * nbx + bx] = sum;
}
