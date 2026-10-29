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

// Slang port of ../spektrafilm.cl, which documents every kernel; the
// comments here only say where the two differ. The working buffers are
// float4 device buffers, as there. The grain helpers are ../grain.h in Slang,
// with the same explicit fma() nesting. Vulkan lets an fma round twice
// (lavapipe does), so the grain draws there can differ from the CPU on
// isolated pixels, as the comments in ../grain.h describe.

import common;

static const int SF_NLE = 256;
static const float SF_LOG_EPS = 1e-10f;

// offsets (in floats) into the packed matrix/constant buffer
static const int SF_M_IN = 0;
static const int SF_M_OUT = 9;
static const int SF_M_COUPLERS = 18;
static const int SF_M_RGB2XYZ = 27;
static const int SF_M_XYZ2RGB = 36;
static const int SF_M_OK1 = 45;
static const int SF_M_OK2 = 54;
static const int SF_M_OK1I = 63;
static const int SF_M_OK2I = 72;
static const int SF_M_LM_DONOR = 81;
static const int SF_M_LM_RECV = 87;

// ---- ../grain.h ----------------------------------------------------------

float grain_clampf(float x, float lo, float hi)
{
  return min(max(x, lo), hi);
}

uint grain_hash(uint x)
{
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

float grain_uniform(uint s)
{
  return float(grain_hash(s) & 0xffffffu) / float(0x1000000);
}

float grain_normal(uint s)
{
  const float u = grain_uniform(s) + grain_uniform(s * 2654435761u + 1u) + grain_uniform(s * 2246822519u + 2u)
                  + grain_uniform(s * 3266489917u + 3u);
  return (u - 2.0f) * 1.7320508f;
}

uint grain_pixel_seed(uint xi, uint yi, uint chan, uint salt)
{
  return (xi * 73856093u ^ yi * 19349663u ^ chan * 83492791u) ^ salt;
}

static const float GRAIN_POISSON_EXACT_MAX = 12.0f;

float grain_exp2i(int k)
{
  return asfloat(uint(k + 127) << 23);
}

float grain_exp_neg(float lam)
{
  const float t = -lam;
  const int k = int(floor(fma(t, 1.4426950216293335f, 0.5f)));
  const float r = fma(-float(k), 0.6931471824645996f, t);
  float p = 0.00138888892f;
  p = fma(p, r, 0.00833333377f);
  p = fma(p, r, 0.0416666679f);
  p = fma(p, r, 0.166666672f);
  p = fma(p, r, 0.5f);
  p = fma(p, r, 1.0f);
  p = fma(p, r, 1.0f);
  return p * grain_exp2i(k);
}

float grain_exp2f(float x)
{
  const int k = int(floor(x + 0.5f));
  const float t = (x - float(k)) * 0.6931471824645996f;
  float p = 0.000198412700f;
  p = fma(p, t, 0.00138888892f);
  p = fma(p, t, 0.00833333377f);
  p = fma(p, t, 0.0416666679f);
  p = fma(p, t, 0.166666672f);
  p = fma(p, t, 0.5f);
  p = fma(p, t, 1.0f);
  p = fma(p, t, 1.0f);
  return p * grain_exp2i(k);
}

float grain_log2f(float x)
{
  const uint xu = asuint(x);
  int e = int((xu >> 23) & 0xffu) - 127;
  float m = asfloat((xu & 0x007fffffu) | 0x3f800000u);
  if(m > 1.41421356f)
  {
    m *= 0.5f;
    e += 1;
  }
  const float s = (m - 1.0f) / (m + 1.0f);
  const float s2 = s * s;
  float p = 0.222222224f;
  p = fma(p, s2, 0.285714298f);
  p = fma(p, s2, 0.400000006f);
  p = fma(p, s2, 0.666666687f);
  p = fma(p, s2, 2.0f);
  return fma(p * s, 1.4426950216293335f, float(e));
}

float grain_poisson(float lam, uint seed)
{
  if(lam <= 0.0f) return 0.0f;
  if(lam < GRAIN_POISSON_EXACT_MAX)
  {
    const float limit = grain_exp_neg(lam);
    float prod = 1.0f;
    int k = 0;
    do
    {
      prod *= grain_uniform(seed + uint(k) * 0x9e3779b9u);
      k++;
    } while(prod > limit && k < 64);
    return float(k - 1);
  }
  return fma(sqrt(lam), grain_normal(seed), lam);
}

float grain_layer_particle(float density, float dmax, float npart, float unif, uint seed)
{
  const float p = grain_clampf(density / dmax, 1e-6f, 1.0f - 1e-6f);
  const float od = dmax / max(npart, 1e-9f);
  const float sat = fma(-(p * unif), 1.0f - 1e-6f, 1.0f);
  return grain_poisson(npart * p / sat, seed * 0x9e3779b9u + 1u) * od * sat;
}

// ---- helpers ---------------------------------------------------------------

float3 sf_mat3(StructuredBuffer<float> mats, int o, float3 v)
{
  return float3(fma(mats[o + 0], v.x, fma(mats[o + 1], v.y, mats[o + 2] * v.z)),
                fma(mats[o + 3], v.x, fma(mats[o + 4], v.y, mats[o + 5] * v.z)),
                fma(mats[o + 6], v.x, fma(mats[o + 7], v.y, mats[o + 8] * v.z)));
}

float sf_mitchell(float t)
{
  const float B = 1.0f / 3.0f, C = 1.0f / 3.0f;
  const float x = abs(t);
  if(x < 1.0f)
    return (1.0f / 6.0f)
           * fma((12.0f - 9.0f * B - 6.0f * C) * x * x, x,
                 fma((-18.0f + 12.0f * B + 6.0f * C) * x, x, (6.0f - 2.0f * B)));
  else if(x < 2.0f)
    return (1.0f / 6.0f)
           * fma((-B - 6.0f * C) * x * x, x,
                 fma((6.0f * B + 30.0f * C) * x, x,
                     fma((-12.0f * B - 48.0f * C), x, (8.0f * B + 24.0f * C))));
  return 0.0f;
}

int sf_reflect(int idx, int L)
{
  if(idx < 0) return -idx;
  if(idx >= L) return 2 * (L - 1) - idx;
  return idx;
}

void sf_base_frac(float coord, int L, out int base, out float frac)
{
  coord = grain_clampf(coord, 0.0f, float(L - 1));
  if(coord >= float(L - 1))
  {
    base = L - 2;
    frac = 1.0f;
    return;
  }
  base = int(floor(coord));
  frac = coord - float(base);
}

float3 sf_cubic2d(StructuredBuffer<float> lut, int L, float x, float y)
{
  int xb, yb;
  float xf, yf;
  sf_base_frac(x, L, xb, xf);
  sf_base_frac(y, L, yb, yf);
  float wx[4], wy[4];
  for(int i = 0; i < 4; i++)
  {
    wx[i] = sf_mitchell(xf + 1.0f - float(i));
    wy[i] = sf_mitchell(yf + 1.0f - float(i));
  }
  float3 acc = float3(0.0f);
  float wsum = 0.0f;
  for(int i = 0; i < 4; i++)
  {
    const int xi = sf_reflect(xb - 1 + i, L);
    for(int j = 0; j < 4; j++)
    {
      const int yj = sf_reflect(yb - 1 + j, L);
      const float w = wx[i] * wy[j];
      wsum += w;
      const int o = (xi * L + yj) * 3;
      acc = float3(fma(w, lut[o], acc.x), fma(w, lut[o + 1], acc.y), fma(w, lut[o + 2], acc.z));
    }
  }
  return (wsum != 0.0f) ? acc / wsum : acc;
}

float sf_curve(StructuredBuffer<float> curves, float x, float le0, float inv_le_step, int c)
{
  const float t = (x - le0) * inv_le_step;
  if(t <= 0.0f) return curves[c];
  if(t >= float(SF_NLE - 1)) return curves[(SF_NLE - 1) * 3 + c];
  const int i = int(t);
  const float f = t - float(i);
  return fma(f, curves[(i + 1) * 3 + c] - curves[i * 3 + c], curves[i * 3 + c]);
}

float sf_hermite(float y0, float y1, float m0, float m1, float t)
{
  const float t2 = t * t, t3 = t2 * t;
  const float h00 = fma(2.0f, t3, fma(-3.0f, t2, 1.0f));
  const float h10 = fma(-2.0f, t2, t3) + t;
  const float h01 = fma(-2.0f, t3, 3.0f * t2);
  const float h11 = t3 - t2;
  return fma(h00, y0, fma(h10, m0, fma(h01, y1, h11 * m1)));
}

float sf_lerp(float v0, float v1, float t)
{
  return fma(t, v1 - v0, v0);
}

float3 sf_pchip3d(StructuredBuffer<float> lut, StructuredBuffer<float> sx, StructuredBuffer<float> sy,
                  StructuredBuffer<float> sz, StructuredBuffer<float> cmin, StructuredBuffer<float> cmax,
                  int n, float r, float g, float b)
{
  const int m = n - 1;
  int i, j, k;
  float tr, tg, tb;
  sf_base_frac(r, n, i, tr);
  sf_base_frac(g, n, j, tg);
  sf_base_frac(b, n, k, tb);
  float o[3];
  for(int c = 0; c < 3; c++)
  {
    // the corners of the cell, as indices into the n^3 x 3 tables
    const int a000 = (((i) * n + (j)) * n + (k)) * 3 + c;
    const int a100 = (((i + 1) * n + (j)) * n + (k)) * 3 + c;
    const int a010 = (((i) * n + (j + 1)) * n + (k)) * 3 + c;
    const int a110 = (((i + 1) * n + (j + 1)) * n + (k)) * 3 + c;
    const int a001 = (((i) * n + (j)) * n + (k + 1)) * 3 + c;
    const int a101 = (((i + 1) * n + (j)) * n + (k + 1)) * 3 + c;
    const int a011 = (((i) * n + (j + 1)) * n + (k + 1)) * 3 + c;
    const int a111 = (((i + 1) * n + (j + 1)) * n + (k + 1)) * 3 + c;

    const float v000 = sf_hermite(lut[a000], lut[a100], sx[a000], sx[a100], tr);
    const float v010 = sf_hermite(lut[a010], lut[a110], sx[a010], sx[a110], tr);
    const float v001 = sf_hermite(lut[a001], lut[a101], sx[a001], sx[a101], tr);
    const float v011 = sf_hermite(lut[a011], lut[a111], sx[a011], sx[a111], tr);
    const float sy00 = sf_lerp(sy[a000], sy[a100], tr);
    const float sy10 = sf_lerp(sy[a010], sy[a110], tr);
    const float sy01 = sf_lerp(sy[a001], sy[a101], tr);
    const float sy11 = sf_lerp(sy[a011], sy[a111], tr);
    const float vz0 = sf_hermite(v000, v010, sy00, sy10, tg);
    const float vz1 = sf_hermite(v001, v011, sy01, sy11, tg);
    const float sz0 = sf_lerp(sf_lerp(sz[a000], sz[a100], tr), sf_lerp(sz[a010], sz[a110], tr), tg);
    const float sz1 = sf_lerp(sf_lerp(sz[a001], sz[a101], tr), sf_lerp(sz[a011], sz[a111], tr), tg);
    float v = sf_hermite(vz0, vz1, sz0, sz1, tb);
    const int ci = ((i * m + j) * m + k) * 3 + c;
    o[c] = grain_clampf(v, cmin[ci], cmax[ci]);
  }
  return float3(o[0], o[1], o[2]);
}

float sf_log10f(float x)
{
  return grain_log2f(x) * 0.3010299956639812f;
}

float sf_pow10f(float x)
{
  return grain_exp2f(x * 3.321928094887362f);
}

float sf_knee(float d, float threshold, float limit, float power)
{
  if(d <= threshold) return d;
  const float scale = limit - threshold;
  const float x = (d - threshold) / scale;
  const float y = x / pow(1.0f + pow(x, power), 1.0f / power);
  return fma(scale, y, threshold);
}

float3 sf_xyz_to_oklab(StructuredBuffer<float> mats, float3 xyz)
{
  float3 lms = sf_mat3(mats, SF_M_OK1, xyz);
  lms = float3(dt_cbrt(lms.x), dt_cbrt(lms.y), dt_cbrt(lms.z));
  return sf_mat3(mats, SF_M_OK2, lms);
}

float3 sf_oklab_to_xyz(StructuredBuffer<float> mats, float3 lab)
{
  float3 lms = sf_mat3(mats, SF_M_OK2I, lab);
  lms = lms * lms * lms;
  return sf_mat3(mats, SF_M_OK1I, lms);
}

float sf_cmax_lookup(StructuredBuffer<float> table, int nl, int nh, float L, float h)
{
  const float L_lo_v = 0.02f, L_hi_v = 1.0f;
  L = grain_clampf(L, L_lo_v, L_hi_v);
  const float h_step = 2.0f * M_PI_F / float(nh);
  const float h_idx = (h + M_PI_F) / h_step;
  const float h_floor = floor(h_idx);
  int h_lo = int(h_floor) % nh;
  if(h_lo < 0) h_lo += nh;
  const int h_hi = (h_lo + 1) % nh;
  const float h_frac = h_idx - h_floor;
  const float L_idx = (L - L_lo_v) / (L_hi_v - L_lo_v) * float(nl - 1);
  int L_lo = int(floor(L_idx));
  L_lo = clamp(L_lo, 0, nl - 2);
  const float L_frac = L_idx - float(L_lo);
  const float v00 = table[L_lo * nh + h_lo];
  const float v01 = table[L_lo * nh + h_hi];
  const float v10 = table[(L_lo + 1) * nh + h_lo];
  const float v11 = table[(L_lo + 1) * nh + h_hi];
  const float l0 = 1.0f - L_frac, hh0 = 1.0f - h_frac;
  return fma(v00 * l0, hh0, fma(v01 * l0, h_frac, fma(v10 * L_frac, hh0, (v11 * L_frac) * h_frac)));
}

// ---- per-pixel stage kernels ---------------------------------------------------

[shader("compute")]
[numthreads(16, 16, 1)]
void spektrafilm_expose(uint3 gid : SV_DispatchThreadID,
                        uniform ImageIn in, uniform RWStructuredBuffer<float4> plane,
                        uniform int w, uniform int h, uniform StructuredBuffer<float> mats,
                        uniform StructuredBuffer<float> tc_lut, uniform int tc_n, uniform float ev_scale)
{
  const int x = gid.x, y = gid.y;
  if(x >= w || y >= h) return;
  const float4 px = in.readpixel(x, y);
  const float3 xyz = sf_mat3(mats, SF_M_IN, px.xyz);
  const float b = xyz.x + xyz.y + xyz.z;
  const float inv = 1.0f / max(b, 1e-10f);
  const float xx = xyz.x * inv, yy = xyz.y * inv;
  const float tcx = grain_clampf((1.0f - xx) * (1.0f - xx), 0.0f, 1.0f);
  const float tcy = grain_clampf(yy / max(1.0f - xx, 1e-10f), 0.0f, 1.0f);
  const float scale = float(tc_n - 1);
  float3 raw = sf_cubic2d(tc_lut, tc_n, tcx * scale, tcy * scale);
  const float bb = isfinite(b) ? b : 0.0f;
  raw = (raw * bb) * ev_scale;
  raw = max(raw, float3(0.0f));
  plane[y * w + x] = float4(raw, px.w);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void spektrafilm_lograw(uint3 gid : SV_DispatchThreadID,
                        uniform RWStructuredBuffer<float4> plane, uniform int w, uniform int h,
                        uniform float raw_scale)
{
  const int x = gid.x, y = gid.y;
  if(x >= w || y >= h) return;
  const int k = y * w + x;
  float4 p = plane[k];
  p.x = sf_log10f(max(p.x * raw_scale, 0.0f) + SF_LOG_EPS);
  p.y = sf_log10f(max(p.y * raw_scale, 0.0f) + SF_LOG_EPS);
  p.z = sf_log10f(max(p.z * raw_scale, 0.0f) + SF_LOG_EPS);
  plane[k] = p;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void spektrafilm_develop_corr(uint3 gid : SV_DispatchThreadID,
                              uniform StructuredBuffer<float4> lograw, uniform RWStructuredBuffer<float4> corr,
                              uniform int w, uniform int h, uniform StructuredBuffer<float> curves_norm,
                              uniform StructuredBuffer<float> mats, uniform float le0, uniform float inv_le_step,
                              uniform float dmax0, uniform float dmax1, uniform float dmax2,
                              uniform int positive, uniform int donor_lm)
{
  const int x = gid.x, y = gid.y;
  if(x >= w || y >= h) return;
  const int k = y * w + x;
  const float4 lg = lograw[k];
  const float dmx[3] = { dmax0, dmax1, dmax2 };
  const float lgv[3] = { lg.x, lg.y, lg.z };
  float silver[3];
  for(int c = 0; c < 3; c++)
  {
    const float d = sf_curve(curves_norm, lgv[c], le0, inv_le_step, c);
    silver[c] = positive != 0 ? dmx[c] - d : d;
    if(donor_lm != 0)
    {
      const float K = mats[SF_M_LM_DONOR + c], Dref = mats[SF_M_LM_DONOR + 3 + c];
      silver[c] = silver[c] * (K + Dref) / (K + silver[c]);
    }
  }
  float o[3];
  for(int m = 0; m < 3; m++)
    o[m] = fma(silver[0], mats[SF_M_COUPLERS + 0 * 3 + m],
               fma(silver[1], mats[SF_M_COUPLERS + 1 * 3 + m], silver[2] * mats[SF_M_COUPLERS + 2 * 3 + m]));
  corr[k] = float4(o[0], o[1], o[2], 0.0f);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void spektrafilm_develop(uint3 gid : SV_DispatchThreadID,
                         uniform StructuredBuffer<float4> lograw, uniform StructuredBuffer<float4> corr,
                         uniform int use_corr, uniform RWStructuredBuffer<float4> cmy, uniform int w,
                         uniform int h, uniform StructuredBuffer<float> curves,
                         uniform StructuredBuffer<float> mats, uniform float le0, uniform float inv_le_step,
                         uniform int recv_lm)
{
  const int x = gid.x, y = gid.y;
  if(x >= w || y >= h) return;
  const int k = y * w + x;
  const float4 lg = lograw[k];
  // read the correction always, so that Metal keeps the argument
  const float4 cr_in = corr[k];
  const float4 cr = use_corr != 0 ? cr_in : float4(0.0f);
  float crv[3] = { cr.x, cr.y, cr.z };
  if(use_corr != 0 && recv_lm != 0)
    for(int c = 0; c < 3; c++)
    {
      const float Kr = mats[SF_M_LM_RECV + c], cref = mats[SF_M_LM_RECV + 3 + c];
      crv[c] = crv[c] * (Kr + cref) / (Kr + crv[c]);
    }
  const float lgv[3] = { lg.x - crv[0], lg.y - crv[1], lg.z - crv[2] };
  float o[3];
  for(int c = 0; c < 3; c++) o[c] = sf_curve(curves, lgv[c], le0, inv_le_step, c);
  cmy[k] = float4(o[0], o[1], o[2], lg.w);
}

float sf_grain_curve_inverse(StructuredBuffer<float> arr, int base, int n, int stride, float target)
{
  const bool increasing = arr[base + (n - 1) * stride] >= arr[base];
  int lo = 0, hi = n - 1;
  while(hi - lo > 1)
  {
    const int mid = (lo + hi) / 2;
    const float v = arr[base + mid * stride];
    if((increasing && v <= target) || (!increasing && v >= target)) lo = mid;
    else hi = mid;
  }
  const float v0 = arr[base + lo * stride], v1 = arr[base + hi * stride];
  const float denom = v1 - v0;
  float frac = (abs(denom) > 1e-9f) ? (target - v0) / denom : 0.0f;
  frac = clamp(frac, 0.0f, 1.0f);
  return float(lo) + frac;
}

float sf_grain_curve_sample(StructuredBuffer<float> arr, int base, int n, int stride, float pos)
{
  int i0 = int(pos);
  if(i0 < 0) i0 = 0;
  if(i0 > n - 2) i0 = (n - 2 < 0) ? 0 : n - 2;
  const float frac = pos - float(i0);
  return fma(arr[base + (i0 + 1) * stride], frac, arr[base + i0 * stride] * (1.0f - frac));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void spektrafilm_grain_gen_raw_sl(uint3 gid : SV_DispatchThreadID,
                                  uniform StructuredBuffer<float4> dens, uniform RWStructuredBuffer<float> raw_out,
                                  uniform int w, uniform int h, uniform int roi_x, uniform int roi_y,
                                  uniform int mono, uniform int channel_idx, uniform int seed_ch,
                                  uniform int sl_idx, uniform int nle, uniform int max_sub,
                                  uniform float unif_ch, uniform float npart_scale, uniform uint salt,
                                  uniform StructuredBuffer<float> layer_dmax,
                                  uniform StructuredBuffer<float> layer_npart,
                                  uniform StructuredBuffer<float> layer_dmin,
                                  uniform StructuredBuffer<float> layer_curve_total,
                                  uniform StructuredBuffer<float> layer_curve)
{
  const int x = gid.x, y = gid.y;
  if(x >= w || y >= h) return;
  const int k = y * w + x;
  const float4 d4 = dens[k];
  const float density = mono != 0 ? (d4.x + d4.y + d4.z) / 3.0f
                                  : (channel_idx == 0 ? d4.x : (channel_idx == 1 ? d4.y : d4.z));
  const int lstride = max_sub * 3;
  const int idx = sl_idx * 3 + channel_idx;
  const float pos = sf_grain_curve_inverse(layer_curve_total, channel_idx, nle, 3, density);
  const float raw = sf_grain_curve_sample(layer_curve, idx, nle, lstride, pos);
  const float d_abs = raw + layer_dmin[idx];
  const uint seed = grain_pixel_seed(uint(x + roi_x), uint(y + roi_y), uint(seed_ch + sl_idx * 10), salt);
  raw_out[k] = grain_layer_particle(d_abs, layer_dmax[idx], layer_npart[idx] * npart_scale, unif_ch, seed);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void spektrafilm_grain_accumulate_1c(uint3 gid : SV_DispatchThreadID,
                                     uniform RWStructuredBuffer<float> acc, uniform StructuredBuffer<float> src,
                                     uniform int w, uniform int h, uniform int reset)
{
  const int x = gid.x, y = gid.y;
  if(x >= w || y >= h) return;
  const int k = y * w + x;
  if(reset != 0) acc[k] = src[k];
  else acc[k] += src[k];
}

[shader("compute")]
[numthreads(16, 16, 1)]
void spektrafilm_grain_finalize_channel(uint3 gid : SV_DispatchThreadID,
                                        uniform RWStructuredBuffer<float4> grain_buf,
                                        uniform StructuredBuffer<float> acc, uniform StructuredBuffer<float4> dens,
                                        uniform int w, uniform int h, uniform int mono,
                                        uniform int channel_idx, uniform int out_ch,
                                        uniform float dmin_ch, uniform float amount)
{
  const int x = gid.x, y = gid.y;
  if(x >= w || y >= h) return;
  const int k = y * w + x;
  const float4 d4 = dens[k];
  const float density = mono != 0 ? (d4.x + d4.y + d4.z) / 3.0f
                                  : (channel_idx == 0 ? d4.x : (channel_idx == 1 ? d4.y : d4.z));
  const float g = acc[k] - dmin_ch;
  const float delta = (g - density) * amount;
  float4 gv = (mono != 0 || out_ch == 0) ? float4(0.f) : grain_buf[k];
  if(mono != 0) gv = float4(delta, delta, delta, 0.f);
  else if(out_ch == 0) gv.x = delta;
  else if(out_ch == 1) gv.y = delta;
  else gv.z = delta;
  grain_buf[k] = gv;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void spektrafilm_grain_add(uint3 gid : SV_DispatchThreadID,
                           uniform RWStructuredBuffer<float4> dens_buf, uniform StructuredBuffer<float4> grain_buf,
                           uniform int w, uniform int h)
{
  const int x = gid.x, y = gid.y;
  if(x >= w || y >= h) return;
  const int k = y * w + x;
  const float4 d = dens_buf[k];
  const float4 g = grain_buf[k];
  dens_buf[k] = float4(d.xyz + g.xyz, d.w);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void spektrafilm_grain_usm(uint3 gid : SV_DispatchThreadID,
                           uniform RWStructuredBuffer<float4> cmy, uniform StructuredBuffer<float4> orig,
                           uniform int w, uniform int h, uniform float amount,
                           uniform float dmin0, uniform float dmin1, uniform float dmin2)
{
  const int x = gid.x, y = gid.y;
  if(x >= w || y >= h) return;
  const int k = y * w + x;
  const float4 dmin = float4(dmin0, dmin1, dmin2, 0.0f);
  const float4 D = max(orig[k] + dmin, float4(0.0f));
  const float4 blur = cmy[k] + dmin;
  const float eps = 1e-6f;
  const float ratmax = 4.0f, ratmin = 1.0f / ratmax;
  float4 o;
  o.x = fma(D.x, pow(max(min(D.x / max(blur.x, eps), ratmax), ratmin), amount), -dmin.x);
  o.y = fma(D.y, pow(max(min(D.y / max(blur.y, eps), ratmax), ratmin), amount), -dmin.y);
  o.z = fma(D.z, pow(max(min(D.z / max(blur.z, eps), ratmax), ratmin), amount), -dmin.z);
  o.w = 0.0f;
  cmy[k] = o;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void spektrafilm_print_expose(uint3 gid : SV_DispatchThreadID,
                              uniform StructuredBuffer<float4> cmy, uniform RWStructuredBuffer<float4> loge,
                              uniform int w, uniform int h,
                              uniform StructuredBuffer<float> lut, uniform StructuredBuffer<float> sx,
                              uniform StructuredBuffer<float> sy, uniform StructuredBuffer<float> sz,
                              uniform StructuredBuffer<float> cmn, uniform StructuredBuffer<float> cmx,
                              uniform int steps, uniform float lo0, uniform float lo1, uniform float lo2,
                              uniform float inv0, uniform float inv1, uniform float inv2,
                              uniform float log10_print_exposure)
{
  const int x = gid.x, y = gid.y;
  if(x >= w || y >= h) return;
  const int k = y * w + x;
  const float4 v = cmy[k];
  const float scale = float(steps - 1);
  const float r = (v.x - lo0) * inv0 * scale;
  const float g = (v.y - lo1) * inv1 * scale;
  const float b = (v.z - lo2) * inv2 * scale;
  const float3 l1 = sf_pchip3d(lut, sx, sy, sz, cmn, cmx, steps, r, g, b);
  loge[k] = float4(max(l1 + log10_print_exposure, float3(-30.0f)), v.w);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void spektrafilm_pow10(uint3 gid : SV_DispatchThreadID,
                       uniform RWStructuredBuffer<float4> plane, uniform int w, uniform int h)
{
  const int x = gid.x, y = gid.y;
  if(x >= w || y >= h) return;
  const int k = y * w + x;
  const float4 p = plane[k];
  plane[k] = float4(sf_pow10f(p.x), sf_pow10f(p.y), sf_pow10f(p.z), p.w);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void spektrafilm_log10(uint3 gid : SV_DispatchThreadID,
                       uniform RWStructuredBuffer<float4> plane, uniform int w, uniform int h)
{
  const int x = gid.x, y = gid.y;
  if(x >= w || y >= h) return;
  const int k = y * w + x;
  const float4 p = plane[k];
  plane[k] = float4(sf_log10f(max(p.x, 0.0f) + SF_LOG_EPS), sf_log10f(max(p.y, 0.0f) + SF_LOG_EPS),
                    sf_log10f(max(p.z, 0.0f) + SF_LOG_EPS), p.w);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void spektrafilm_print_develop(uint3 gid : SV_DispatchThreadID,
                               uniform StructuredBuffer<float4> loge, uniform RWStructuredBuffer<float4> cmy,
                               uniform int w, uniform int h, uniform StructuredBuffer<float> print_curves,
                               uniform float le0, uniform float inv_le_step)
{
  const int x = gid.x, y = gid.y;
  if(x >= w || y >= h) return;
  const int k = y * w + x;
  const float4 v = loge[k];
  cmy[k] = float4(sf_curve(print_curves, v.x, le0, inv_le_step, 0),
                  sf_curve(print_curves, v.y, le0, inv_le_step, 1),
                  sf_curve(print_curves, v.z, le0, inv_le_step, 2), v.w);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void spektrafilm_scan(uint3 gid : SV_DispatchThreadID,
                      uniform StructuredBuffer<float4> cmy, uniform RWStructuredBuffer<float4> rgb_out,
                      uniform int w, uniform int h,
                      uniform StructuredBuffer<float> lut, uniform StructuredBuffer<float> sx,
                      uniform StructuredBuffer<float> sy, uniform StructuredBuffer<float> sz,
                      uniform StructuredBuffer<float> cmn, uniform StructuredBuffer<float> cmx,
                      uniform int steps, uniform float lo0, uniform float lo1, uniform float lo2,
                      uniform float inv0, uniform float inv1, uniform float inv2,
                      uniform StructuredBuffer<float> mats, uniform StructuredBuffer<float> cmax_table,
                      uniform int cmax_nl, uniform int cmax_nh, uniform int compress_mode,
                      uniform float out_luminance_boost, uniform float out_scale,
                      uniform int bw_on, uniform float bw_m, uniform float bw_q,
                      uniform StructuredBuffer<float4> glare, uniform int glare_on,
                      uniform float glare_x, uniform float glare_y, uniform float glare_z)
{
  const int x = gid.x, y = gid.y;
  if(x >= w || y >= h) return;
  const int k = y * w + x;
  const float4 c4 = cmy[k];
  const float scale = float(steps - 1);
  const float r = (c4.x - lo0) * inv0 * scale;
  const float g = (c4.y - lo1) * inv1 * scale;
  const float b = (c4.z - lo2) * inv2 * scale;
  const float3 lx = sf_pchip3d(lut, sx, sy, sz, cmn, cmx, steps, r, g, b);
  float3 xyz = float3(sf_pow10f(lx.x), sf_pow10f(lx.y), sf_pow10f(lx.z));
  if(out_luminance_boost != 1.0f) xyz *= out_luminance_boost;
  if(bw_on != 0)
  {
    const float yc = grain_clampf(fma(bw_m, xyz.y, bw_q), 0.0f, 1.0f);
    xyz *= yc / (xyz.y + 1e-10f);
  }
  // read the glare field always, so that Metal keeps the argument
  const float gv = glare[k].x;
  if(glare_on != 0)
    xyz = float3(fma(gv, glare_x, xyz.x), fma(gv, glare_y, xyz.y), fma(gv, glare_z, xyz.z));
  float3 rgb = sf_mat3(mats, SF_M_OUT, xyz);

  if(compress_mode == 1)
  {
    const float3 lab = sf_xyz_to_oklab(mats, sf_mat3(mats, SF_M_RGB2XYZ, rgb));
    const float L = sf_knee(lab.x, 0.7f, 1.0f, 2.2f);
    const float C = dt_hypot(lab.y, lab.z);
    const float hh = atan2(lab.z, lab.y);
    const float C_max = max(sf_cmax_lookup(cmax_table, cmax_nl, cmax_nh, L, hh), 1e-9f);
    const float d = sf_knee(C / C_max, 0.0f, 1.0f, 6.0f);
    const float C_new = d * C_max;
    const float3 lab_new = float3(L, C_new * cos(hh), C_new * sin(hh));
    rgb = sf_mat3(mats, SF_M_XYZ2RGB, sf_oklab_to_xyz(mats, lab_new));
  }
  else if(compress_mode == 2)
  {
    const float ach = max(rgb.x, max(rgb.y, rgb.z));
    if(ach > 1e-12f)
    {
      float v[3] = { rgb.x, rgb.y, rgb.z };
      for(int c = 0; c < 3; c++)
      {
        const float d = (ach - v[c]) / ach;
        const float dc = sf_knee(d, 0.0f, 1.0f, 6.0f);
        v[c] = ach * (1.0f - dc);
      }
      rgb = float3(v[0], v[1], v[2]);
    }
  }
  else
  {
    // keep cmax_table used, so that Metal keeps the argument
    if(cmax_nl < 0) rgb.x += cmax_table[0];
  }

  if(out_scale != 1.0f) rgb *= out_scale;

  rgb_out[k] = float4(rgb, 0.0f);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void spektrafilm_crop_out(uint3 gid : SV_DispatchThreadID,
                          uniform StructuredBuffer<float4> rgb, uniform ImageIn in, uniform ImageOut out,
                          uniform int w, uniform int ow, uniform int oh, uniform int ox, uniform int oy)
{
  const int x = gid.x, y = gid.y;
  if(x >= ow || y >= oh) return;
  const float4 v = rgb[(y + oy) * w + (x + ox)];
  const float4 px = in.readpixel(x + ox, y + oy);
  out.write(int2(x, y), float4(v.xyz, px.w));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void spektrafilm_scan_usm(uint3 gid : SV_DispatchThreadID,
                          uniform RWStructuredBuffer<float4> rgb, uniform StructuredBuffer<float4> orig,
                          uniform int w, uniform int h, uniform float amount)
{
  const int x = gid.x, y = gid.y;
  if(x >= w || y >= h) return;
  const int k = y * w + x;
  const float4 D = orig[k], blur = rgb[k];
  rgb[k] = float4(fma(amount, D.x - blur.x, D.x), fma(amount, D.y - blur.y, D.y),
                  fma(amount, D.z - blur.z, D.z), D.w);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void spektrafilm_glare_gen(uint3 gid : SV_DispatchThreadID,
                           uniform RWStructuredBuffer<float4> field, uniform int w, uniform int h,
                           uniform int roi_x, uniform int roi_y, uniform float mean,
                           uniform float s, uniform float bias, uniform uint salt)
{
  const int x = gid.x, y = gid.y;
  if(x >= w || y >= h) return;
  const uint seed = grain_pixel_seed(uint(x + roi_x), uint(y + roi_y), 0x5eedu, salt);
  const float g = mean * exp(fma(s, grain_normal(seed), bias));
  field[y * w + x] = float4(g, g, g, 0.0f);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void spektrafilm_passthrough(uint3 gid : SV_DispatchThreadID,
                             uniform ImageIn in, uniform ImageOut out,
                             uniform int ow, uniform int oh, uniform int ox, uniform int oy)
{
  const int x = gid.x, y = gid.y;
  if(x >= ow || y >= oh) return;
  out.write(int2(x, y), in.readpixel(x + ox, y + oy));
}

// ---- spatial-effect kernels ----------------------------------------------------

// Young-van Vliet recursion, one work item per line, dispatched with
// dt_slang_gpu_run_1d_args()
[shader("compute")]
[numthreads(16, 16, 1)]
void spektrafilm_yvv_row_4c(uint3 gid : SV_DispatchThreadID,
                            uniform StructuredBuffer<float4> src, uniform RWStructuredBuffer<float4> dst,
                            uniform int w, uniform int h, uniform float B, uniform float B2, uniform float B3)
{
  const int row = gid.y * 16 + gid.x;
  if(row >= h) return;
  const int o = row * w;
  float4 w1 = src[o], w2 = src[o], w3 = src[o];
  for(int j = 0; j < w; j++)
  {
    const float4 v = w1 + fma(B, src[o + j] - w1, fma(B2, w2 - w1, B3 * (w3 - w1)));
    dst[o + j] = v;
    w3 = w2;
    w2 = w1;
    w1 = v;
  }
  float4 v1 = dst[o + w - 1], v2 = v1, v3 = v1;
  for(int j = w - 1; j >= 0; j--)
  {
    const float4 v = v1 + fma(B, dst[o + j] - v1, fma(B2, v2 - v1, B3 * (v3 - v1)));
    dst[o + j] = v;
    v3 = v2;
    v2 = v1;
    v1 = v;
  }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void spektrafilm_yvv_col_4c(uint3 gid : SV_DispatchThreadID,
                            uniform StructuredBuffer<float4> src, uniform RWStructuredBuffer<float4> dst,
                            uniform int w, uniform int h, uniform float B, uniform float B2, uniform float B3)
{
  const int col = gid.y * 16 + gid.x;
  if(col >= w) return;
  float4 w1 = src[col], w2 = w1, w3 = w1;
  for(int i = 0; i < h; i++)
  {
    const int k = i * w + col;
    const float4 v = w1 + fma(B, src[k] - w1, fma(B2, w2 - w1, B3 * (w3 - w1)));
    dst[k] = v;
    w3 = w2;
    w2 = w1;
    w1 = v;
  }
  float4 v1 = dst[(h - 1) * w + col], v2 = v1, v3 = v1;
  for(int i = h - 1; i >= 0; i--)
  {
    const int k = i * w + col;
    const float4 v = v1 + fma(B, dst[k] - v1, fma(B2, v2 - v1, B3 * (v3 - v1)));
    dst[k] = v;
    v3 = v2;
    v2 = v1;
    v1 = v;
  }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void spektrafilm_yvv_row_1c(uint3 gid : SV_DispatchThreadID,
                            uniform StructuredBuffer<float> src, uniform RWStructuredBuffer<float> dst,
                            uniform int w, uniform int h, uniform float B, uniform float B2, uniform float B3)
{
  const int row = gid.y * 16 + gid.x;
  if(row >= h) return;
  const int o = row * w;
  float w1 = src[o], w2 = src[o], w3 = src[o];
  for(int j = 0; j < w; j++)
  {
    const float v = w1 + fma(B, src[o + j] - w1, fma(B2, w2 - w1, B3 * (w3 - w1)));
    dst[o + j] = v;
    w3 = w2;
    w2 = w1;
    w1 = v;
  }
  float v1 = dst[o + w - 1], v2 = v1, v3 = v1;
  for(int j = w - 1; j >= 0; j--)
  {
    const float v = v1 + fma(B, dst[o + j] - v1, fma(B2, v2 - v1, B3 * (v3 - v1)));
    dst[o + j] = v;
    v3 = v2;
    v2 = v1;
    v1 = v;
  }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void spektrafilm_yvv_col_1c(uint3 gid : SV_DispatchThreadID,
                            uniform StructuredBuffer<float> src, uniform RWStructuredBuffer<float> dst,
                            uniform int w, uniform int h, uniform float B, uniform float B2, uniform float B3)
{
  const int col = gid.y * 16 + gid.x;
  if(col >= w) return;
  float w1 = src[col], w2 = w1, w3 = w1;
  for(int i = 0; i < h; i++)
  {
    const int k = i * w + col;
    const float v = w1 + fma(B, src[k] - w1, fma(B2, w2 - w1, B3 * (w3 - w1)));
    dst[k] = v;
    w3 = w2;
    w2 = w1;
    w1 = v;
  }
  float v1 = dst[(h - 1) * w + col], v2 = v1, v3 = v1;
  for(int i = h - 1; i >= 0; i--)
  {
    const int k = i * w + col;
    const float v = v1 + fma(B, dst[k] - v1, fma(B2, v2 - v1, B3 * (v3 - v1)));
    dst[k] = v;
    v3 = v2;
    v2 = v1;
    v1 = v;
  }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void spektrafilm_wide_rows(uint3 gid : SV_DispatchThreadID,
                           uniform StructuredBuffer<float4> in, uniform RWStructuredBuffer<float4> rows,
                           uniform int w, uniform int h, uniform int k, uniform int lw, uniform int ox)
{
  const int X = gid.x, y = gid.y;
  if(X >= lw || y >= h) return;
  const int x0 = max(X * k - ox, 0), x1 = min((X + 1) * k - ox, w);
  float4 rs = float4(0.0f);
  for(int x = x0; x < x1; x++) rs += in[y * w + x];
  rows[y * lw + X] = rs;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void spektrafilm_wide_down(uint3 gid : SV_DispatchThreadID,
                           uniform StructuredBuffer<float4> rows, uniform RWStructuredBuffer<float4> low,
                           uniform int w, uniform int h, uniform int k, uniform int lw, uniform int lh,
                           uniform int ox, uniform int oy, uniform StructuredBuffer<float> inv)
{
  const int X = gid.x, Y = gid.y;
  if(X >= lw || Y >= lh) return;
  const int x0 = max(X * k - ox, 0), x1 = min((X + 1) * k - ox, w);
  const int y0 = max(Y * k - oy, 0), y1 = min((Y + 1) * k - oy, h);
  const float ix = inv[x1 - x0];
  float4 lo = float4(0.0f);
  for(int y = y0; y < y1; y++) lo = fma(rows[y * lw + X], float4(ix), lo);
  low[Y * lw + X] = lo * inv[y1 - y0];
}

[shader("compute")]
[numthreads(16, 16, 1)]
void spektrafilm_wide_up(uint3 gid : SV_DispatchThreadID,
                         uniform StructuredBuffer<float4> low, uniform RWStructuredBuffer<float4> out,
                         uniform int w, uniform int h, uniform int k, uniform int lw, uniform int lh,
                         uniform int ox, uniform int oy, uniform float inv2k)
{
  const int x = gid.x, y = gid.y;
  if(x >= w || y >= h) return;
  const float v = float(2 * (y + oy) + 1 - k) * inv2k;
  int j0 = int(floor(v)), j1;
  float fy;
  if(j0 < 0) { j0 = j1 = 0; fy = 0.0f; }
  else if(j0 >= lh - 1) { j0 = j1 = lh - 1; fy = 0.0f; }
  else { j1 = j0 + 1; fy = v - float(j0); }
  const float u = float(2 * (x + ox) + 1 - k) * inv2k;
  int i0 = int(floor(u)), i1;
  float fx;
  if(i0 < 0) { i0 = i1 = 0; fx = 0.0f; }
  else if(i0 >= lw - 1) { i0 = i1 = lw - 1; fx = 0.0f; }
  else { i1 = i0 + 1; fx = u - float(i0); }
  const float4 p00 = low[j0 * lw + i0], p01 = low[j0 * lw + i1];
  const float4 p10 = low[j1 * lw + i0], p11 = low[j1 * lw + i1];
  const float4 top = fma(float4(fx), p01 - p00, p00);
  const float4 bot = fma(float4(fx), p11 - p10, p10);
  out[y * w + x] = fma(float4(fy), bot - top, top);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void spektrafilm_gauss_row_4c(uint3 gid : SV_DispatchThreadID,
                              uniform StructuredBuffer<float4> src, uniform RWStructuredBuffer<float4> dst,
                              uniform int w, uniform int h, uniform StructuredBuffer<float> weights,
                              uniform int radius)
{
  const int x = gid.x, y = gid.y;
  if(x >= w || y >= h) return;
  float4 acc = float4(0.0f);
  for(int k = -radius; k <= radius; k++)
  {
    const int xx = clamp(x + k, 0, w - 1);
    acc = fma(float4(weights[k + radius]), src[y * w + xx], acc);
  }
  dst[y * w + x] = acc;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void spektrafilm_gauss_col_4c(uint3 gid : SV_DispatchThreadID,
                              uniform StructuredBuffer<float4> src, uniform RWStructuredBuffer<float4> dst,
                              uniform int w, uniform int h, uniform StructuredBuffer<float> weights,
                              uniform int radius)
{
  const int x = gid.x, y = gid.y;
  if(x >= w || y >= h) return;
  float4 acc = float4(0.0f);
  for(int k = -radius; k <= radius; k++)
  {
    const int yy = clamp(y + k, 0, h - 1);
    acc = fma(float4(weights[k + radius]), src[yy * w + x], acc);
  }
  dst[y * w + x] = acc;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void spektrafilm_gauss_row_1c(uint3 gid : SV_DispatchThreadID,
                              uniform StructuredBuffer<float> src, uniform RWStructuredBuffer<float> dst,
                              uniform int w, uniform int h, uniform StructuredBuffer<float> weights,
                              uniform int radius)
{
  const int x = gid.x, y = gid.y;
  if(x >= w || y >= h) return;
  float acc = 0.0f;
  for(int k = -radius; k <= radius; k++)
  {
    const int xx = clamp(x + k, 0, w - 1);
    acc = fma(weights[k + radius], src[y * w + xx], acc);
  }
  dst[y * w + x] = acc;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void spektrafilm_gauss_col_1c(uint3 gid : SV_DispatchThreadID,
                              uniform StructuredBuffer<float> src, uniform RWStructuredBuffer<float> dst,
                              uniform int w, uniform int h, uniform StructuredBuffer<float> weights,
                              uniform int radius)
{
  const int x = gid.x, y = gid.y;
  if(x >= w || y >= h) return;
  float acc = 0.0f;
  for(int k = -radius; k <= radius; k++)
  {
    const int yy = clamp(y + k, 0, h - 1);
    acc = fma(weights[k + radius], src[yy * w + x], acc);
  }
  dst[y * w + x] = acc;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void spektrafilm_scatter_combine(uint3 gid : SV_DispatchThreadID,
                                 uniform StructuredBuffer<float4> raw, uniform StructuredBuffer<float4> core,
                                 uniform StructuredBuffer<float4> tail, uniform RWStructuredBuffer<float4> out,
                                 uniform int w, uniform int h, uniform float s_amount,
                                 uniform float ws_r, uniform float ws_g, uniform float ws_b)
{
  const int x = gid.x, y = gid.y;
  if(x >= w || y >= h) return;
  const int k = y * w + x;
  const float4 r = raw[k], c = core[k], t = tail[k];
  float4 o;
  o.x = fma(s_amount, fma(ws_r, t.x, (1.f - ws_r) * c.x) - r.x, r.x);
  o.y = fma(s_amount, fma(ws_g, t.y, (1.f - ws_g) * c.y) - r.y, r.y);
  o.z = fma(s_amount, fma(ws_b, t.z, (1.f - ws_b) * c.z) - r.z, r.z);
  o.w = r.w;
  out[k] = o;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void spektrafilm_accum(uint3 gid : SV_DispatchThreadID,
                       uniform StructuredBuffer<float4> blurred, uniform RWStructuredBuffer<float4> acc,
                       uniform int w, uniform int h, uniform float wk, uniform int reset)
{
  const int x = gid.x, y = gid.y;
  if(x >= w || y >= h) return;
  const int k = y * w + x;
  const float4 b = blurred[k];
  float4 a = reset != 0 ? float4(0.f) : acc[k];
  a.x = fma(wk, b.x, a.x);
  a.y = fma(wk, b.y, a.y);
  a.z = fma(wk, b.z, a.z);
  acc[k] = a;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void spektrafilm_channel_extract(uint3 gid : SV_DispatchThreadID,
                                 uniform StructuredBuffer<float4> src, uniform RWStructuredBuffer<float> dst,
                                 uniform int w, uniform int h, uniform int channel)
{
  const int x = gid.x, y = gid.y;
  if(x >= w || y >= h) return;
  const int k = y * w + x;
  const float4 s = src[k];
  dst[k] = (channel == 0) ? s.x : (channel == 1) ? s.y : s.z;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void spektrafilm_channel_accum(uint3 gid : SV_DispatchThreadID,
                               uniform StructuredBuffer<float> blurred, uniform RWStructuredBuffer<float4> acc,
                               uniform int w, uniform int h, uniform float weight, uniform int channel,
                               uniform int reset)
{
  const int x = gid.x, y = gid.y;
  if(x >= w || y >= h) return;
  const int k = y * w + x;
  const float bv = blurred[k];
  float4 a = reset != 0 ? float4(0.f) : acc[k];
  const float av = (channel == 0) ? a.x : (channel == 1) ? a.y : a.z;
  const float nv = fma(weight, bv, av);
  if(channel == 0) a.x = nv;
  else if(channel == 1) a.y = nv;
  else a.z = nv;
  acc[k] = a;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void spektrafilm_halation_apply(uint3 gid : SV_DispatchThreadID,
                                uniform RWStructuredBuffer<float4> raw, uniform StructuredBuffer<float4> blur,
                                uniform int w, uniform int h, uniform float a_r, uniform float a_g,
                                uniform float a_b)
{
  const int x = gid.x, y = gid.y;
  if(x >= w || y >= h) return;
  const int k = y * w + x;
  float4 r = raw[k];
  const float4 b = blur[k];
  r.x = fma(a_r, b.x, r.x) / (1.f + a_r);
  r.y = fma(a_g, b.y, r.y) / (1.f + a_g);
  r.z = fma(a_b, b.z, r.z) / (1.f + a_b);
  raw[k] = r;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void spektrafilm_boost(uint3 gid : SV_DispatchThreadID,
                       uniform RWStructuredBuffer<float4> plane, uniform int w, uniform int h,
                       uniform float raw_x0, uniform float xmax, uniform float inv_max,
                       uniform float a, uniform float a_log2e, uniform float scale,
                       uniform float ext, uniform float slope)
{
  const int x = gid.x, y = gid.y;
  if(x >= w || y >= h) return;
  const int k = y * w + x;
  const float4 p = plane[k];
  float v[3] = { p.x, p.y, p.z };
  for(int c = 0; c < 3; c++)
  {
    const float xv = v[c];
    if(xv <= raw_x0) continue;
    if(xv <= xmax)
    {
      const float dx = (xv - raw_x0) * inv_max;
      const float t = a * dx;
      v[c] = fma(scale, (grain_exp2f(a_log2e * dx) - t) - 1.0f, xv);
    }
    else
      v[c] = fma(slope, xv - xmax, xv + ext);
  }
  plane[k] = float4(v[0], v[1], v[2], p.w);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void spektrafilm_diffusion_accum(uint3 gid : SV_DispatchThreadID,
                                 uniform StructuredBuffer<float4> blurred, uniform RWStructuredBuffer<float4> acc,
                                 uniform int w, uniform int h, uniform float wr, uniform float wg,
                                 uniform float wb, uniform int reset)
{
  const int x = gid.x, y = gid.y;
  if(x >= w || y >= h) return;
  const int k = y * w + x;
  const float4 b = blurred[k];
  const float4 a = reset != 0 ? float4(0.f) : acc[k];
  acc[k] = float4(fma(wr, b.x, a.x), fma(wg, b.y, a.y), fma(wb, b.z, a.z), b.w);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void spektrafilm_diffusion_mix(uint3 gid : SV_DispatchThreadID,
                               uniform RWStructuredBuffer<float4> plane, uniform StructuredBuffer<float4> acc,
                               uniform int w, uniform int h, uniform float p_s)
{
  const int x = gid.x, y = gid.y;
  if(x >= w || y >= h) return;
  const int k = y * w + x;
  const float4 e = plane[k], s = acc[k];
  plane[k] = float4(fma(p_s, s.x, (1.f - p_s) * e.x), fma(p_s, s.y, (1.f - p_s) * e.y),
                    fma(p_s, s.z, (1.f - p_s) * e.z), e.w);
}

// buffer to buffer copy, for clEnqueueCopyBuffer
[shader("compute")]
[numthreads(16, 16, 1)]
void spektrafilm_copy(uint3 gid : SV_DispatchThreadID,
                      uniform StructuredBuffer<float4> src, uniform RWStructuredBuffer<float4> dst,
                      uniform int w, uniform int h)
{
  const int x = gid.x, y = gid.y;
  if(x >= w || y >= h) return;
  dst[y * w + x] = src[y * w + x];
}
