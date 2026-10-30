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

// Runs each ported kernel twice on the same input, once from its .cl source
// through OpenCL and once from the Slang-built SPIR-V through Vulkan, or the
// .metallib files through Metal on macOS (parity_metal.m), and compares the
// output images. It needs no GPU: Mesa's lavapipe and PoCL are enough, which
// is how it runs in CI. Without a device for either API it exits with SKIP.
//
// usage: darktable-test-slang-parity <dir with .cl files> <dir with the Slang kernels> [program]

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#ifndef PARITY_METAL
#include <vulkan/vulkan.h>
#endif

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "parity.h"

#define W 517
#define H 263

// -------------------------------------------------------------------------------------------------
// test cases

// must match dt_colorspaces_iccprofile_info_cl_t in src/common/iop_profile.h
typedef struct profile_info_t
{
  float matrix_in[9];
  float matrix_out[9];
  int lutsize;
  float unbounded_coeffs_in[3][3];
  float unbounded_coeffs_out[3][3];
  int nonlinearlut;
  float grey;
  int hdr_transfer;
} profile_info_t;

static float *image_in, *image_lab, *image_raw, *lut_curve, *lut_profile;
static float *rcd_cfa, *rcd_lpf, *rcd_dir, *rcd_rgb[3];
static uint8_t *mask_bytes;
static profile_info_t profile_linear, profile_nonlinear, profile_pq;

static float frand(void)
{
  return (float)rand() / (float)RAND_MAX;
}

static void make_inputs(void)
{
  srand(42);
  image_in = malloc(sizeof(float) * 4 * W * H);
  image_lab = malloc(sizeof(float) * 4 * W * H);
  for(size_t k = 0; k < (size_t)4 * W * H; k++) image_in[k] = frand() * 1.2f - 0.05f; // a few negatives, a few > 1
  // a raw image: smooth gradients with noise and a few clipped areas, as a
  // demosaic of pure noise would test little
  image_raw = malloc(sizeof(float) * W * H);
  for(int y = 0; y < H; y++)
    for(int x = 0; x < W; x++)
    {
      const float base = 0.5f + 0.4f * sinf(x * 0.05f) * cosf(y * 0.07f) + 0.05f * frand();
      const int bayer = ((y & 1) << 1) | (x & 1);
      const float gain = bayer == 0 ? 0.6f : bayer == 3 ? 0.8f : 1.0f;
      image_raw[(size_t)y * W + x] = ((x / 40 + y / 30) % 7 == 0) ? 1.05f : base * gain - 0.01f;
    }

  // intermediate RCD data: the normalized raw image, low pass values,
  // directional weights in [0, 1] and color planes near the raw values
  rcd_cfa = malloc(sizeof(float) * W * H);
  rcd_lpf = malloc(sizeof(float) * W * H);
  rcd_dir = malloc(sizeof(float) * W * H);
  for(int c = 0; c < 3; c++) rcd_rgb[c] = malloc(sizeof(float) * W * H);
  for(size_t k = 0; k < (size_t)W * H; k++)
  {
    rcd_cfa[k] = fmaxf(0.0f, image_raw[k]);
    rcd_lpf[k] = 4.0f * rcd_cfa[k] + 0.1f * frand();
    rcd_dir[k] = frand();
    for(int c = 0; c < 3; c++) rcd_rgb[c][k] = rcd_cfa[k] * (0.9f + 0.2f * frand());
  }

  // a sparse mask of 3x3 cells for the opposed highlights, for three colors
  const int msize = ((W / 3 + 3) / 4 * 4) * ((H / 3 + 3) / 4 * 4);
  mask_bytes = calloc(1, 3 * msize);
  for(int k = 0; k < 3 * msize; k++) mask_bytes[k] = frand() < 0.05f;
  for(size_t k = 0; k < (size_t)W * H; k++)
  {
    image_lab[4 * k + 0] = frand() * 100.0f;
    image_lab[4 * k + 1] = frand() * 160.0f - 80.0f;
    image_lab[4 * k + 2] = frand() * 160.0f - 80.0f;
    image_lab[4 * k + 3] = frand();
  }

  // 256x256 tone curve, like the tables rgbcurve and rgblevels upload
  lut_curve = malloc(sizeof(float) * 256 * 256);
  for(int k = 0; k < 256 * 256; k++) lut_curve[k] = powf(k / 65535.0f, 0.6f);

  // 256x(256*6) profile LUT: three input and three output curves
  lut_profile = malloc(sizeof(float) * 256 * 256 * 6);
  for(int n = 0; n < 6; n++)
    for(int k = 0; k < 256 * 256; k++)
      lut_profile[n * 65536 + k] = n < 3 ? powf(k / 65535.0f, 2.2f) : powf(k / 65535.0f, 1.0f / 2.2f);

  const float m_in[9] = { 0.4361f, 0.3851f, 0.1431f, 0.2225f, 0.7169f, 0.0606f, 0.0139f, 0.0971f, 0.7141f };
  const float m_out[9] = { 3.1339f, -1.6169f, -0.4906f, -0.9788f, 1.9161f, 0.0335f, 0.0719f, -0.2290f, 1.4052f };
  profile_info_t *all[3] = { &profile_linear, &profile_nonlinear, &profile_pq };
  for(int p = 0; p < 3; p++)
  {
    memset(all[p], 0, sizeof(profile_info_t));
    memcpy(all[p]->matrix_in, m_in, sizeof(m_in));
    memcpy(all[p]->matrix_out, m_out, sizeof(m_out));
    all[p]->lutsize = 65536;
    all[p]->grey = 0.1842f;
    for(int c = 0; c < 3; c++)
    {
      // extrapolation above 1: a * (x * b)^g, as fitted by dt_iop_estimate_exp()
      all[p]->unbounded_coeffs_in[c][0] = 1.0f;
      all[p]->unbounded_coeffs_in[c][1] = 1.0f;
      all[p]->unbounded_coeffs_in[c][2] = 2.2f;
      all[p]->unbounded_coeffs_out[c][0] = 1.0f;
      all[p]->unbounded_coeffs_out[c][1] = 1.0f;
      all[p]->unbounded_coeffs_out[c][2] = 1.0f / 2.2f;
    }
  }
  profile_nonlinear.nonlinearlut = 1;
  profile_pq.nonlinearlut = 1;
  profile_pq.hdr_transfer = 1; // DT_HDR_PQ
}

// the VNG tables as _vng_gpu_tables() in src/iop/demosaicing/vng.c builds
// them, for one Bayer pattern or the X-Trans one
static const signed char vng_terms[]
      = { -2, -2, +0, -1, 1, 0x01, -2, -2, +0, +0, 2, 0x01, -2, -1, -1, +0, 1, 0x01, -2, -1, +0, -1, 1, 0x02,
          -2, -1, +0, +0, 1, 0x03, -2, -1, +0, +1, 2, 0x01, -2, +0, +0, -1, 1, 0x06, -2, +0, +0, +0, 2, 0x02,
          -2, +0, +0, +1, 1, 0x03, -2, +1, -1, +0, 1, 0x04, -2, +1, +0, -1, 2, 0x04, -2, +1, +0, +0, 1, 0x06,
          -2, +1, +0, +1, 1, 0x02, -2, +2, +0, +0, 2, 0x04, -2, +2, +0, +1, 1, 0x04, -1, -2, -1, +0, 1, 0x80,
          -1, -2, +0, -1, 1, 0x01, -1, -2, +1, -1, 1, 0x01, -1, -2, +1, +0, 2, 0x01, -1, -1, -1, +1, 1, 0x88,
          -1, -1, +1, -2, 1, 0x40, -1, -1, +1, -1, 1, 0x22, -1, -1, +1, +0, 1, 0x33, -1, -1, +1, +1, 2, 0x11,
          -1, +0, -1, +2, 1, 0x08, -1, +0, +0, -1, 1, 0x44, -1, +0, +0, +1, 1, 0x11, -1, +0, +1, -2, 2, 0x40,
          -1, +0, +1, -1, 1, 0x66, -1, +0, +1, +0, 2, 0x22, -1, +0, +1, +1, 1, 0x33, -1, +0, +1, +2, 2, 0x10,
          -1, +1, +1, -1, 2, 0x44, -1, +1, +1, +0, 1, 0x66, -1, +1, +1, +1, 1, 0x22, -1, +1, +1, +2, 1, 0x10,
          -1, +2, +0, +1, 1, 0x04, -1, +2, +1, +0, 2, 0x04, -1, +2, +1, +1, 1, 0x04, +0, -2, +0, +0, 2, 0x80,
          +0, -1, +0, +1, 2, 0x88, +0, -1, +1, -2, 1, 0x40, +0, -1, +1, +0, 1, 0x11, +0, -1, +2, -2, 1, 0x40,
          +0, -1, +2, -1, 1, 0x20, +0, -1, +2, +0, 1, 0x30, +0, -1, +2, +1, 2, 0x10, +0, +0, +0, +2, 2, 0x08,
          +0, +0, +2, -2, 2, 0x40, +0, +0, +2, -1, 1, 0x60, +0, +0, +2, +0, 2, 0x20, +0, +0, +2, +1, 1, 0x30,
          +0, +0, +2, +2, 2, 0x10, +0, +1, +1, +0, 1, 0x44, +0, +1, +1, +2, 1, 0x10, +0, +1, +2, -1, 2, 0x40,
          +0, +1, +2, +0, 1, 0x60, +0, +1, +2, +1, 1, 0x20, +0, +1, +2, +2, 1, 0x10, +1, -2, +1, +0, 1, 0x80,
          +1, -1, +1, +1, 1, 0x88, +1, +0, +1, +2, 1, 0x08, +1, +0, +2, -1, 1, 0x40, +1, +0, +2, +1, 1, 0x10 };
static const signed char vng_chood[] = { -1, -1, -1, 0, -1, +1, 0, +1, +1, +1, +1, 0, +1, -1, 0, -1 };

static int fcol(const int row, const int col, const uint32_t filters, const uint8_t (*const xtrans)[6])
{
  if(filters == 9) return xtrans[(row + 600) % 6][(col + 600) % 6];
  return filters >> ((((row << 1) & 14) + (col & 1)) << 1) & 3;
}

#define VNG_IPS_SIZE (6 * 6 * 352 * sizeof(int))
typedef struct vng_tables_t
{
  int32_t lookup[16][16][32];
  int ips[6 * 6 * 352];
  int code[16][16];
} vng_tables_t;

static void vng_tables(vng_tables_t *tb, const uint32_t filters4, const uint8_t (*const xtrans)[6])
{
  const int is_xtrans = filters4 == 9;
  const int lsize = is_xtrans ? 6 : 16;
  const int colors = is_xtrans ? 3 : 4;
  const int prow = is_xtrans ? 6 : 8;
  const int pcol = is_xtrans ? 6 : 2;
  memset(tb, 0, sizeof(*tb));
  for(int row = 0; row < lsize; row++)
    for(int col = 0; col < lsize; col++)
    {
      int32_t *ip = &tb->lookup[row][col][1];
      int sum[4] = { 0 };
      const int f = fcol(row, col, filters4, xtrans);
      for(int y = -1; y <= 1; y++)
        for(int x = -1; x <= 1; x++)
        {
          const int weight = 1 << ((y == 0) + (x == 0));
          const int color = fcol(row + y, col + x, filters4, xtrans);
          if(color == f) continue;
          *ip++ = (y << 16) | (x & 0xffffu);
          *ip++ = weight;
          *ip++ = color;
          sum[color] += weight;
        }
      tb->lookup[row][col][0] = (ip - &tb->lookup[row][col][0]) / 3;
      for(int c = 0; c < colors; c++)
        if(c != f)
        {
          *ip++ = c;
          *ip++ = sum[c];
        }
      *ip = f;
    }

  int *ip = tb->ips;
  for(int row = 0; row < prow; row++)
    for(int col = 0; col < pcol; col++)
    {
      tb->code[row][col] = ip - tb->ips;
      const signed char *cp = vng_terms;
      for(int t = 0; t < 64; t++)
      {
        const int y1 = *cp++, x1 = *cp++;
        const int y2 = *cp++, x2 = *cp++;
        const int weight = *cp++;
        const int grads = *cp++;
        const int color = fcol(row + y1, col + x1, filters4, xtrans);
        if(fcol(row + y2, col + x2, filters4, xtrans) != color) continue;
        const int diag = (fcol(row, col + 1, filters4, xtrans) == color
                          && fcol(row + 1, col, filters4, xtrans) == color) ? 2 : 1;
        if(abs(y1 - y2) == diag && abs(x1 - x2) == diag) continue;
        *ip++ = (y1 << 16) | (x1 & 0xffffu);
        *ip++ = (y2 << 16) | (x2 & 0xffffu);
        *ip++ = (color << 16) | (weight & 0xffffu);
        for(int g = 0; g < 8; g++)
          if(grads & 1 << g) *ip++ = g;
        *ip++ = -1;
      }
      *ip++ = 0x7fffffff;
      cp = vng_chood;
      for(int g = 0; g < 8; g++)
      {
        const int y = *cp++, x = *cp++;
        *ip++ = (y << 16) | (x & 0xffffu);
        const int color = fcol(row, col, filters4, xtrans);
        if(fcol(row + y, col + x, filters4, xtrans) != color
           && fcol(row + y * 2, col + x * 2, filters4, xtrans) == color)
        {
          *ip++ = (2 * y << 16) | (2 * x & 0xffffu);
          *ip++ = color;
        }
        else
        {
          *ip++ = 0;
          *ip++ = 0;
        }
      }
    }
}

// allhex and sgreen as process_markesteijn_cl() builds them
static void markesteijn_hex(const uint8_t (*const xtrans)[6], signed char allhex[3][3][8][2], int sgreen[2])
{
  static const short orth[12] = { 1, 0, 0, 1, -1, 0, 0, -1, 1, 0, 0, 1 },
                     patt[2][16] = { { 0, 1, 0, -1, 2, 0, -1, 0, 1, 1, 1, -1, 0, 0, 0, 0 },
                                     { 0, 1, 0, -2, 1, 0, -2, 0, 1, 1, -2, -2, 1, -1, -1, 1 } };
  memset(allhex, 0, sizeof(signed char) * 3 * 3 * 8 * 2);
  sgreen[0] = sgreen[1] = 0;
  for(int row = 0; row < 3; row++)
    for(int col = 0; col < 3; col++)
      for(int ng = 0, d = 0; d < 10; d += 2)
      {
        const int g = fcol(row, col, 9, xtrans) == 1;
        if(fcol(row + orth[d] + 6, col + orth[d + 2] + 6, 9, xtrans) == 1)
          ng = 0;
        else
          ng++;
        if(ng == 4)
        {
          sgreen[0] = col;
          sgreen[1] = row;
        }
        if(ng == g + 1)
          for(int c = 0; c < 8; c++)
          {
            const int v = orth[d] * patt[g][c * 2] + orth[d + 1] * patt[g][c * 2 + 1];
            const int h = orth[d + 2] * patt[g][c * 2] + orth[d + 3] * patt[g][c * 2 + 1];
            allhex[row][col][c ^ (g * 2 & d)][0] = h;
            allhex[row][col][c ^ (g * 2 & d)][1] = v;
          }
      }
}

#define IMG_IN(n, px) { .kind = ARG_IMAGE_IN, .name = n, .w = W, .h = H, .channels = 4, .pixels = px }
#define LUT(n, px, lw, lh) { .kind = ARG_IMAGE_IN, .name = n, .w = lw, .h = lh, .channels = 1, .pixels = px }
#define IMG_OUT(n) { .kind = ARG_IMAGE_OUT, .name = n, .w = W, .h = H, .channels = 4 }
#define RAW_IN(n, px) { .kind = ARG_IMAGE_IN, .name = n, .w = W, .h = H, .channels = 1, .pixels = px }
#define RAW_OUT(n) { .kind = ARG_IMAGE_OUT, .name = n, .w = W, .h = H, .channels = 1 }
#define BUFFER_OUT(n, ptr, sz) { .kind = ARG_BUFFER_OUT, .name = n, .data = ptr, .size = sz }
#define BYTES_OUT(n, sz) { .kind = ARG_BUFFER_OUT, .name = n, .size = sz, .bytes = 1 }
#define INTS_OUT(n, sz) { .kind = ARG_BUFFER_OUT, .name = n, .size = sz, .ints = 1 }
#define LOCAL(n, sz) { .kind = ARG_LOCAL, .name = n, .size = sz }
#define INT(n, val) { .kind = ARG_INT, .name = n, .i = val }
#define FLOAT(n, val) { .kind = ARG_FLOAT, .name = n, .f = val }
#define FLOAT4(n, a, b, c, d) { .kind = ARG_FLOAT4, .name = n, .v = { a, b, c, d } }
#define INT2(n, a, b) { .kind = ARG_INT2, .name = n, .iv = { a, b } }
#define INT4(n, a, b, c, d) { .kind = ARG_INT4, .name = n, .iv = { a, b, c, d } }
#define STRUCT(n, ptr) { .kind = ARG_STRUCT, .name = n, .data = &(ptr), .size = sizeof(ptr) }
#define FLOAT2(n, a, b) { .kind = ARG_FLOAT2, .name = n, .v = { a, b } }
#define BUFFER(n, ptr, sz) { .kind = ARG_BUFFER, .name = n, .data = ptr, .size = sz }
#define FLOAT4_BUF(n, a, b, c, d) { .kind = ARG_FLOAT4_BUF, .name = n, .v = { a, b, c, d } }
#define SLANG_INT(n, val) { .kind = ARG_INT, .name = n, .i = val, .slang_only = 1 }
#define SLANG_BUFFER(n, ptr, sz) { .kind = ARG_BUFFER, .name = n, .data = ptr, .size = sz, .slang_only = 1 }
#define SLANG_RAW_IN(n, px) { .kind = ARG_IMAGE_IN, .name = n, .w = W, .h = H, .channels = 1, .pixels = px, \
                              .slang_only = 1 }
#define CHAR2(n, a, b) { .kind = ARG_CHAR2, .name = n, .iv = { a, b } }
#define RAW_U16(n, ptr, rw, rh) { .kind = ARG_RAW_U16, .name = n, .data = ptr, .w = rw, .h = rh, \
                                  .size = (size_t)(rw) * (rh) * sizeof(uint16_t) }

static const float levels[9] = { 0.05f, 0.45f, 0.9f, 0.0f, 0.5f, 1.0f, 0.1f, 0.4f, 0.95f };
static const float inv_gamma[3] = { 1.1f, 0.9f, 1.3f };
static const float coeffs_linear[3] = { 1.0f, 1.0f, 0.6f };
static const float coeffs_off[3] = { -1.0f, 0.0f, 0.0f };
static const float matrix3x3[9] = { 0.9f, 0.05f, 0.05f, 0.1f, 0.8f, 0.1f, 0.0f, 0.2f, 0.8f };

static int make_tests(test_t *t)
{
  int n = 0;
  // an X-Trans pattern as the host passes it: 6x6 bytes
  static const uint8_t xtrans6[6][6] = { { 1, 1, 0, 1, 1, 2 }, { 1, 1, 2, 1, 1, 0 }, { 2, 0, 1, 0, 2, 1 },
                                         { 1, 1, 2, 1, 1, 0 }, { 1, 1, 0, 1, 1, 2 }, { 0, 2, 1, 2, 0, 1 } };
#define TEST_GRID(lbl, prog, kern, tol, w, h, ...)                                        \
  {                                                                                      \
    const arg_t a[] = { __VA_ARGS__ };                                                   \
    t[n] = (test_t){ .label = lbl, .program = prog, .kernel = kern, .tolerance = tol,    \
                     .gw = w, .gh = h, .nargs = sizeof(a) / sizeof(a[0]) };             \
    memcpy(t[n].args, a, sizeof(a));                                                     \
    n++;                                                                                 \
  }
// a line kernel runs one work item per row or column, laid out 16 wide on
// the Slang side (see dt_slang_gpu_run_1d_args) and as len x 1 in OpenCL
#define TEST_LINE(lbl, prog, kern, tol, len, ...)                                         \
  {                                                                                      \
    TEST_GRID(lbl, prog, kern, tol, 16, ((len) + 15) / 16, __VA_ARGS__);                 \
    t[n - 1].clgw = (len);                                                               \
    t[n - 1].clgh = 1;                                                                   \
    t[n - 1].clly = 1;                                                                   \
  }
#define TEST_TOL(lbl, prog, kern, tol, ...) TEST_GRID(lbl, prog, kern, tol, W, H, __VA_ARGS__)
#define TEST(lbl, prog, kern, ...) TEST_TOL(lbl, prog, kern, 0.0, __VA_ARGS__)

  TEST("negadoctor", "negadoctor", "negadoctor",
       IMG_IN("in", image_in), IMG_OUT("out"), INT("width", W), INT("height", H),
       FLOAT4("Dmin", 1.13f, 0.49f, 0.27f, 1.0f), FLOAT4("wb_high", 1.6f, 1.7f, 1.8f, 1.0f),
       FLOAT4("offset", -0.05f, -0.02f, -0.01f, 0.0f), FLOAT("exposure", -0.9f), FLOAT("black", 0.0755f),
       FLOAT("gamma", 4.0f), FLOAT("soft_clip", 0.75f), FLOAT("soft_clip_comp", 0.25f));

  for(int mode = 0; mode < 3; mode++)
  {
    // independent channels; preserve colors by luminance (work profile); by power norm
    const int autoscale = mode == 0 ? 1 : 0;
    const int preserve = mode == 0 ? 0 : mode == 1 ? 1 : 6;
    static const char *labels[3] = { "rgblevels independent", "rgblevels luminance", "rgblevels power" };
    // with preserved colors the curve is looked up, without interpolation, at
    // a computed norm: float noise can move that to the next of the 65536 LUT
    // entries, which changes the output by up to about 2e-4
    TEST_TOL(labels[mode], "rgblevels", "rgblevels", mode == 0 ? 0.0 : 1e-3,
         IMG_IN("in", image_in), IMG_OUT("out"), INT("width", W), INT("height", H),
         INT("autoscale", autoscale), INT("preserve_colors", preserve),
         LUT("lutr", lut_curve, 256, 256), LUT("lutg", lut_curve, 256, 256), LUT("lutb", lut_curve, 256, 256),
         BUFFER("levels", levels, sizeof(levels)), BUFFER("inv_gamma", inv_gamma, sizeof(inv_gamma)),
         BUFFER("profile_info", &profile_nonlinear, sizeof(profile_info_t)),
         LUT("lut", lut_profile, 256, 256 * 6), INT("use_work_profile", 1));
  }

  for(int mode = 0; mode < 3; mode++)
  {
    // manual RGB; automatic without norm; automatic with luminance norm and camera profile
    const int autoscale = mode == 0 ? 1 : 0;
    const int preserve = mode == 2 ? 1 : 0;
    static const char *labels[3] = { "rgbcurve manual", "rgbcurve automatic", "rgbcurve luminance" };
    // same LUT step as for rgblevels above
    TEST_TOL(labels[mode], "rgbcurve", "rgbcurve", mode == 2 ? 1e-3 : 0.0,
         IMG_IN("in", image_in), IMG_OUT("out"), INT("width", W), INT("height", H),
         LUT("table_r", lut_curve, 256, 256), LUT("table_g", lut_curve, 256, 256),
         LUT("table_b", lut_curve, 256, 256),
         BUFFER("coeffs_r", coeffs_linear, sizeof(coeffs_linear)), BUFFER("coeffs_g", coeffs_off, sizeof(coeffs_off)),
         BUFFER("coeffs_b", coeffs_linear, sizeof(coeffs_linear)),
         INT("autoscale", autoscale), INT("preserve_colors", preserve),
         BUFFER("profile_info", &profile_linear, sizeof(profile_info_t)),
         LUT("lut", lut_profile, 256, 256 * 6), INT("use_work_profile", 0));
  }

  const profile_info_t *profiles[3] = { &profile_linear, &profile_nonlinear, &profile_pq };
  static const char *suffix[3] = { "linear", "lut", "pq" };
  static char labels[3][4][96];
  for(int p = 0; p < 3; p++)
  {
    snprintf(labels[p][0], sizeof(labels[p][0]), "colorspaces lab_to_rgb_matrix %s", suffix[p]);
    // Lab to linear RGB cancels large terms for saturated colors, and the
    // output LUT is steep near zero: OpenCL and Vulkan both land about 1e-4
    // from a double precision reference there, on opposite sides
    TEST_TOL(labels[p][0], "colorspaces", "colorspaces_transform_lab_to_rgb_matrix", 5e-4,
         IMG_IN("in", image_lab), IMG_OUT("out"), INT("width", W), INT("height", H),
         BUFFER("profile_info", profiles[p], sizeof(profile_info_t)), LUT("lut", lut_profile, 256, 256 * 6));
    snprintf(labels[p][1], sizeof(labels[p][1]), "colorspaces rgb_matrix_to_lab %s", suffix[p]);
    // a and b are 500 and 200 times a difference of cube roots. For PQ input
    // L reaches 300, where Vulkan's pow() leaves a off by 0.01 (OpenCL 0.003)
    // against a double precision reference, a dE of about 0.03
    TEST_TOL(labels[p][1], "colorspaces", "colorspaces_transform_rgb_matrix_to_lab", 0.05,
         IMG_IN("in", image_in), IMG_OUT("out"), INT("width", W), INT("height", H),
         BUFFER("profile_info", profiles[p], sizeof(profile_info_t)), LUT("lut", lut_profile, 256, 256 * 6));
    snprintf(labels[p][2], sizeof(labels[p][2]), "colorspaces rgb_matrix_to_rgb %s", suffix[p]);
    TEST(labels[p][2], "colorspaces", "colorspaces_transform_rgb_matrix_to_rgb",
         IMG_IN("in", image_in), IMG_OUT("out"), INT("width", W), INT("height", H),
         BUFFER("profile_info_from", profiles[p], sizeof(profile_info_t)),
         LUT("lut_from", lut_profile, 256, 256 * 6),
         BUFFER("profile_info_to", profiles[p], sizeof(profile_info_t)),
         LUT("lut_to", lut_profile, 256, 256 * 6), BUFFER("matrix", matrix3x3, sizeof(matrix3x3)));
  }

  TEST("colorspaces gamma", "colorspaces", "colorspaces_transform_gamma",
       IMG_IN("in", image_in), IMG_OUT("out"), INT("width", W), INT("height", H), FLOAT("gamma", 0.45f));

  // the clipping indicators. tmp stands for the input in the histogram
  // profile, with values on both sides of the thresholds
  static float *ox_tmp = NULL;
  if(!ox_tmp)
  {
    ox_tmp = malloc(sizeof(float) * 4 * W * H);
    for(size_t k = 0; k < (size_t)4 * W * H; k++) ox_tmp[k] = image_in[(k * 7) % ((size_t)4 * W * H)] * 1.05f;
  }
  static const char *ox_labels[5] = { "overexposed gamut", "overexposed anyrgb", "overexposed luminance",
                                      "overexposed saturation", "overexposed no profile" };
  for(int m = 0; m < 5; m++)
    TEST(ox_labels[m], "basic", "overexposed",
         IMG_IN("in", image_in), IMG_OUT("out"), IMG_IN("tmp", ox_tmp), INT("width", W), INT("height", H),
         FLOAT("lower", 0.0625f), FLOAT("upper", 0.95f),
         FLOAT4("lower_color", 0.0f, 0.0f, 1.0f, 1.0f), FLOAT4("upper_color", 1.0f, 0.0f, 0.0f, 1.0f),
         BUFFER("profile_info", &profile_nonlinear, sizeof(profile_info_t)), LUT("lut", lut_profile, 256, 256 * 6),
         INT("use_work_profile", m < 4), INT("mode", m < 4 ? m : 0));

  // raw overexposed: output pixel coordinates mapped back into a raw of a
  // different size, partly outside of it, with fractional positions
  enum { RW = 300, RH = 200 };
  static uint16_t *ro_raw = NULL;
  static float *ro_coord = NULL;
  if(!ro_raw)
  {
    ro_raw = malloc(sizeof(uint16_t) * RW * RH);
    for(size_t k = 0; k < (size_t)RW * RH; k++) ro_raw[k] = (uint16_t)(frand() * 65535.0f);
    ro_coord = malloc(sizeof(float) * 2 * W * H);
    for(int y = 0; y < H; y++)
      for(int x = 0; x < W; x++)
      {
        ro_coord[2 * ((size_t)y * W + x)] = x * 0.61f - 10.3f;
        ro_coord[2 * ((size_t)y * W + x) + 1] = y * 0.83f - 7.6f;
      }
  }
  static const unsigned int ro_threshold[4] = { 40000, 45000, 50000, 45000 };
  static const float ro_colors[4][4] = { { 1.0f, 0.0f, 0.0f, 1.0f }, { 0.0f, 1.0f, 0.0f, 1.0f },
                                         { 0.0f, 0.0f, 1.0f, 1.0f }, { 0.0f, 0.0f, 0.0f, 1.0f } };
  static const unsigned int ro_filters[2] = { 0x94949494u, 9u };
  static char ro_labels[2][3][48];
  for(int f = 0; f < 2; f++)
  {
    const char *cfa = f ? "xtrans" : "bayer";
    snprintf(ro_labels[f][0], sizeof(ro_labels[f][0]), "rawoverexposed_mark_cfa %s", cfa);
    snprintf(ro_labels[f][1], sizeof(ro_labels[f][1]), "rawoverexposed_mark_solid %s", cfa);
    snprintf(ro_labels[f][2], sizeof(ro_labels[f][2]), "rawoverexposed_falsecolor %s", cfa);
    TEST(ro_labels[f][0], "basic", "rawoverexposed_mark_cfa",
         IMG_IN("in", image_in), IMG_OUT("out"), BUFFER("pi", ro_coord, sizeof(float) * 2 * W * H),
         INT("width", W), INT("height", H), RAW_U16("raw", ro_raw, RW, RH), INT("raw_width", RW),
         INT("raw_height", RH), INT("filters", ro_filters[f]), BUFFER("xtrans", xtrans6, sizeof(xtrans6)),
         BUFFER("threshold", ro_threshold, sizeof(ro_threshold)), BUFFER("colors", ro_colors, sizeof(ro_colors)));
    TEST(ro_labels[f][1], "basic", "rawoverexposed_mark_solid",
         IMG_IN("in", image_in), IMG_OUT("out"), BUFFER("pi", ro_coord, sizeof(float) * 2 * W * H),
         INT("width", W), INT("height", H), RAW_U16("raw", ro_raw, RW, RH), INT("raw_width", RW),
         INT("raw_height", RH), INT("filters", ro_filters[f]), BUFFER("xtrans", xtrans6, sizeof(xtrans6)),
         BUFFER("threshold", ro_threshold, sizeof(ro_threshold)), FLOAT4("solid_color", 0.0f, 0.0f, 0.0f, 1.0f));
    TEST(ro_labels[f][2], "basic", "rawoverexposed_falsecolor",
         IMG_IN("in", image_in), IMG_OUT("out"), BUFFER("pi", ro_coord, sizeof(float) * 2 * W * H),
         INT("width", W), INT("height", H), RAW_U16("raw", ro_raw, RW, RH), INT("raw_width", RW),
         INT("raw_height", RH), INT("filters", ro_filters[f]), BUFFER("xtrans", xtrans6, sizeof(xtrans6)),
         BUFFER("threshold", ro_threshold, sizeof(ro_threshold)));
  }

  // 4x4 matrices as the host passes them, of which the kernel reads 3 rows
  static const float to_base[16] = { 0.95f, 0.03f, 0.02f, 0.0f, 0.04f, 0.92f, 0.04f, 0.0f, 0.01f, 0.05f, 0.94f, 0.0f };
  static const float to_rendering[16] = { 1.1f, -0.06f, -0.04f, 0.0f, -0.05f, 1.08f, -0.03f, 0.0f, -0.02f, -0.07f, 1.09f, 0.0f };
  static const float to_pipe[16] = { 0.91f, 0.05f, 0.04f, 0.0f, 0.06f, 0.9f, 0.04f, 0.0f, 0.02f, 0.04f, 0.94f, 0.0f };
  // the hue preservation divides by the sum of the two smaller channels,
  // which goes through zero for out of gamut colors: there it multiplies the
  // float noise of pow() by a few hundred. Against a double precision
  // reference, PoCL and lavapipe land about 1e-4 away, Apple's OpenCL 2e-3
  // and Metal 6e-3
  TEST_TOL("sigmoid per channel", "sigmoid", "sigmoid_loglogistic_per_channel", 1e-2,
       IMG_IN("in", image_in), IMG_OUT("out"), INT("width", W), INT("height", H),
       FLOAT("white_target", 1.0f), FLOAT("paper_exp", 0.18f), FLOAT("film_fog", 0.0f),
       FLOAT("contrast_power", 1.5f), FLOAT("skew_power", 1.2f), FLOAT("hue_preservation", 0.7f),
       BUFFER("pipe_to_base", to_base, sizeof(to_base)), BUFFER("base_to_rendering", to_rendering, sizeof(to_rendering)),
       BUFFER("rendering_to_pipe", to_pipe, sizeof(to_pipe)));
  TEST("sigmoid rgb ratio", "sigmoid", "sigmoid_loglogistic_rgb_ratio",
       IMG_IN("in", image_in), IMG_OUT("out"), INT("width", W), INT("height", H),
       FLOAT("white_target", 1.0f), FLOAT("black_target", 0.0002f), FLOAT("paper_exp", 0.18f),
       FLOAT("film_fog", 0.0f), FLOAT("contrast_power", 1.5f), FLOAT("skew_power", 1.2f));

  TEST("exposure", "basic", "exposure",
       IMG_IN("in", image_in), IMG_OUT("out"), INT("width", W), INT("height", H),
       FLOAT("black", 0.01f), FLOAT("scale", 1.7f));

  // camera to XYZ, camera to linear RGB and linear RGB to XYZ
  static const float cam_to_xyz[9] = { 0.62f, 0.25f, 0.09f, 0.28f, 0.82f, -0.1f, 0.03f, -0.12f, 0.92f };
  static const float cam_to_rgb[9] = { 1.6f, -0.45f, -0.15f, -0.2f, 1.5f, -0.3f, 0.05f, -0.5f, 1.45f };
  static const float rgb_to_xyz[9] = { 0.4361f, 0.3851f, 0.1431f, 0.2225f, 0.7169f, 0.0606f, 0.0139f, 0.0971f, 0.7141f };
  static const float xyz_to_rgb[9] = { 3.1339f, -1.6169f, -0.4906f, -0.9788f, 1.9161f, 0.0335f, 0.0719f, -0.2290f, 1.4052f };
  // per channel extrapolation above 1: identity for red, curves for green and blue
  static const float unbounded[9] = { -1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.6f, 1.0f, 1.0f, 0.6f };
  static const float corr[4] = { 1.1f, 1.0f, 0.9f, 1.0f };
  // a DNG hue/saturation map of 6 hues, 3 saturations and 4 values
  enum { HUE_DIV = 6, SAT_DIV = 3, VAL_DIV = 4 };
  static float hsm[HUE_DIV * SAT_DIV * VAL_DIV * 3];
  for(int k = 0; k < HUE_DIV * SAT_DIV * VAL_DIV; k++)
  {
    hsm[3 * k + 0] = 10.0f * sinf(k);        // hue shift in degrees
    hsm[3 * k + 1] = 1.0f + 0.1f * cosf(k);  // saturation scale
    hsm[3 * k + 2] = 1.0f + 0.05f * sinf(2 * k); // value scale
  }

  // tone curve and blue mapping, with and without the hue/saturation map
  // (sRGB encoded), and PQ input
  static const char *colorin_labels[2][3] = {
    { "colorin unbound", "colorin unbound hsm", "colorin unbound pq" },
    { "colorin clipping", "colorin clipping hsm", "colorin clipping pq" } };
  for(int clipping = 0; clipping < 2; clipping++)
    for(int mode = 0; mode < 3; mode++)
    {
      // XYZ to Lab cancels large terms for saturated colors, as in the
      // colorspaces tests above, and the hue/saturation map goes through Lab
      // to ProPhoto and back once more. PQ decodes to values up to 49, which
      // the camera matrices above would cancel down to near zero luminance,
      // so a positive matrix stands in for them; L then reaches 375, where a
      // and b land about 0.01 from a double precision reference, on opposite
      // sides for OpenCL and Vulkan, as in colorspaces rgb_matrix_to_lab pq
      TEST_TOL(colorin_labels[clipping][mode], "basic", clipping ? "colorin_clipping" : "colorin_unbound",
           mode == 0 ? 5e-4 : mode == 1 ? 5e-3 : 0.05,
           IMG_IN("in", image_in), IMG_OUT("out"), INT("width", W), INT("height", H),
           BUFFER("cmat", mode == 2 ? rgb_to_xyz : clipping ? cam_to_rgb : cam_to_xyz, sizeof(cam_to_xyz)),
           BUFFER("lmat", rgb_to_xyz, sizeof(rgb_to_xyz)),
           LUT("lutr", lut_curve, 256, 256), LUT("lutg", lut_curve, 256, 256), LUT("lutb", lut_curve, 256, 256),
           INT("blue_mapping", mode != 2), BUFFER("a", unbounded, sizeof(unbounded)),
           BUFFER("corr", corr, sizeof(corr)), FLOAT("scale", 1.2f),
           INT("hue_div", HUE_DIV), INT("sat_div", SAT_DIV), INT("val_div", VAL_DIV),
           BUFFER("dev_hsm", hsm, sizeof(hsm)), INT("encoding", 1), INT("use_hsm", mode == 1),
           INT("hdr_transfer", mode == 2 ? 1 : 0));
    }

  TEST("colorout", "basic", "colorout",
       IMG_IN("in", image_lab), IMG_OUT("out"), INT("width", W), INT("height", H),
       BUFFER("mat", xyz_to_rgb, sizeof(xyz_to_rgb)),
       LUT("lutr", lut_curve, 256, 256), LUT("lutg", lut_curve, 256, 256), LUT("lutb", lut_curve, 256, 256),
       BUFFER("a", unbounded, sizeof(unbounded)), INT("hdr_transfer", 0));
  // the steep part of the PQ curve, as for colorspaces lab_to_rgb_matrix pq
  TEST_TOL("colorout pq", "basic", "colorout", 5e-4,
       IMG_IN("in", image_lab), IMG_OUT("out"), INT("width", W), INT("height", H),
       BUFFER("mat", xyz_to_rgb, sizeof(xyz_to_rgb)),
       LUT("lutr", lut_curve, 256, 256), LUT("lutg", lut_curve, 256, 256), LUT("lutb", lut_curve, 256, 256),
       BUFFER("a", unbounded, sizeof(unbounded)), INT("hdr_transfer", 1));

  // 4x4 matrices as the host passes them, of which the kernels read 3 rows
  static const float cm_rgb_to_xyz[12] = { 0.4361f, 0.3851f, 0.1431f, 0.0f, 0.2225f, 0.7169f, 0.0606f, 0.0f,
                                           0.0139f, 0.0971f, 0.7141f, 0.0f };
  static const float cm_xyz_to_rgb[12] = { 3.1339f, -1.6169f, -0.4906f, 0.0f, -0.9788f, 1.9161f, 0.0335f, 0.0f,
                                           0.0719f, -0.2290f, 1.4052f, 0.0f };
  static const float cm_mix[12] = { 1.05f, -0.03f, -0.02f, 0.0f, -0.04f, 1.06f, -0.02f, 0.0f,
                                    0.01f, -0.06f, 1.05f, 0.0f };
  static const char *cm_kernels[5] = { "channelmixerrgb_CAT16", "channelmixerrgb_bradford_linear",
                                       "channelmixerrgb_bradford_full", "channelmixerrgb_XYZ",
                                       "channelmixerrgb_RGB" };
  static char cm_labels[5][3][96];
  for(int k = 0; k < 5; k++)
    for(int mode = 0; mode < 3; mode++)
    {
      // version 3 with gamut compression and clipping; version 1 unclipped;
      // version 2 to monochrome
      static const char *cm_suffix[3] = { "v3 clip", "v1", "v2 grey" };
      snprintf(cm_labels[k][mode], sizeof(cm_labels[k][mode]), "%s %s", cm_kernels[k], cm_suffix[mode]);
      // unclipped, out of gamut inputs come out between about -1e3 and 1e6,
      // where Metal lands up to 2.5e-4 (relative) from OpenCL. The squared
      // ratios of the v1 saturation take a few of them further: on an M4,
      // one pixel of bradford_linear lands 5.5e-3 off
      TEST_TOL(cm_labels[k][mode], "channelmixer", cm_kernels[k], mode == 1 ? 1e-3 : 0.0,
           IMG_IN("in", image_in), IMG_OUT("out"), INT("width", W), INT("height", H),
           BUFFER("RGB_to_XYZ", cm_rgb_to_xyz, sizeof(cm_rgb_to_xyz)),
           BUFFER("XYZ_to_RGB", cm_xyz_to_rgb, sizeof(cm_xyz_to_rgb)), BUFFER("MIX", cm_mix, sizeof(cm_mix)),
           FLOAT4("illuminant", 0.97f, 1.01f, 0.78f, 0.0f), FLOAT4("saturation", 0.1f, -0.05f, 0.08f, 0.0f),
           FLOAT4("lightness", 0.02f, 0.05f, -0.03f, 0.0f), FLOAT4("grey", 0.3f, 0.6f, 0.1f, 0.0f),
           FLOAT("p", 1.02f), FLOAT("gamut", mode == 0 ? 0.5f : 0.0f), INT("clip", mode == 0),
           INT("apply_grey", mode == 2), INT("version", mode == 0 ? 2 : mode == 1 ? 0 : 1));
      t[n - 1].outliers = mode == 1 ? 8 : 0;
    }

  // the raw stage. Two Bayer layouts, RGGB and BGGR, and an X-Trans pattern
  // as the host passes it: 6x6 bytes
  static const unsigned int bayer[2] = { 0x94949494u, 0x16161616u };
  static const uint8_t xtrans[6][6] = { { 1, 1, 0, 1, 1, 2 }, { 1, 1, 2, 1, 1, 0 }, { 2, 0, 1, 0, 2, 1 },
                                        { 1, 1, 2, 1, 1, 0 }, { 1, 1, 0, 1, 1, 2 }, { 0, 2, 1, 2, 0, 1 } };
  static const float wb_coeffs[4] = { 2.1f, 1.0f, 1.6f, 1.0f };
  static const float hl_clips[4] = { 0.95f, 0.98f, 0.9f, 1.0f };
  static const float hl_correction[4] = { 1.1f, 1.0f, 0.9f, 1.0f };
  static const float hl_chroma[4] = { 0.05f, -0.02f, 0.03f, 0.0f };
  static const char *bayer_labels[2][5] = {
    { "whitebalance_1f rggb", "highlights_1f_clip rggb", "highlights_initmask rggb",
      "highlights_opposed rggb", "border_interpolate rggb" },
    { "whitebalance_1f bggr", "highlights_1f_clip bggr", "highlights_initmask bggr",
      "highlights_opposed bggr", "border_interpolate bggr" } };

  // the opposed method's 3x3 downscaled masks, one byte per cell and color
  const int mwidth = W / 3, mheight = H / 3;
  const int msize = ((mwidth + 3) / 4 * 4) * ((mheight + 3) / 4 * 4);

  for(int b = 0; b < 2; b++)
  {
    TEST(bayer_labels[b][0], "basic", "whitebalance_1f",
         RAW_IN("in", image_raw), RAW_OUT("out"), INT("width", W), INT("height", H),
         BUFFER("coeffs", wb_coeffs, sizeof(wb_coeffs)), INT("filters", bayer[b]),
         BUFFER("xtrans", xtrans, sizeof(xtrans)));
    // a crop of 3 columns and 2 rows
    TEST(bayer_labels[b][1], "basic", "highlights_1f_clip",
         RAW_IN("in", image_raw), RAW_OUT("out"), INT("iwidth", W), INT("iheight", H),
         INT("owidth", W - 5), INT("oheight", H - 4), FLOAT4("clip", 0.95f, 0.98f, 0.9f, 0.98f),
         INT("dx", 3), INT("dy", 2), INT("filters", bayer[b]), BUFFER("xtrans", xtrans, sizeof(xtrans)));
    TEST_GRID(bayer_labels[b][2], "basic", "highlights_initmask", 0.0, mwidth, mheight,
         RAW_IN("in", image_raw), BYTES_OUT("inmask", 3 * msize), INT("msize", msize), INT("mwidth", mwidth),
         INT("mheight", mheight), INT("iwidth", W), INT("iheight", H), INT("filters", bayer[b]),
         BUFFER("xtrans", xtrans, sizeof(xtrans)), BUFFER("clips", hl_clips, sizeof(hl_clips)));
    TEST(bayer_labels[b][3], "basic", "highlights_opposed",
         RAW_IN("in", image_raw), RAW_OUT("out"), INT("owidth", W), INT("oheight", H),
         INT("iwidth", W), INT("iheight", H), INT("dx", 0), INT("dy", 0), INT("filters", bayer[b]),
         BUFFER("xtrans", xtrans, sizeof(xtrans)), BUFFER("clips", hl_clips, sizeof(hl_clips)),
         BUFFER("chroma", hl_chroma, sizeof(hl_chroma)), BUFFER("correction", hl_correction, sizeof(hl_correction)));
    TEST(bayer_labels[b][4], "demosaic_ppg", "border_interpolate",
         RAW_IN("in", image_raw), IMG_OUT("out"), INT("width", W), INT("height", H), INT("filters", bayer[b]));
  }
  TEST("whitebalance_1f_xtrans", "basic", "whitebalance_1f_xtrans",
       RAW_IN("in", image_raw), RAW_OUT("out"), INT("width", W), INT("height", H),
       BUFFER("coeffs", wb_coeffs, sizeof(wb_coeffs)), INT("filters", 9),
       BUFFER("xtrans", xtrans, sizeof(xtrans)));
  TEST("whitebalance_4f", "basic", "whitebalance_4f",
       IMG_IN("in", image_in), IMG_OUT("out"), INT("width", W), INT("height", H),
       BUFFER("coeffs", wb_coeffs, sizeof(wb_coeffs)), INT("filters", 0),
       BUFFER("xtrans", xtrans, sizeof(xtrans)));
  TEST_GRID("highlights_initmask xtrans", "basic", "highlights_initmask", 0.0, mwidth, mheight,
       RAW_IN("in", image_raw), BYTES_OUT("inmask", 3 * msize), INT("msize", msize), INT("mwidth", mwidth),
       INT("mheight", mheight), INT("iwidth", W), INT("iheight", H), INT("filters", 9),
       BUFFER("xtrans", xtrans, sizeof(xtrans)), BUFFER("clips", hl_clips, sizeof(hl_clips)));
  TEST_GRID("highlights_dilatemask", "basic", "highlights_dilatemask", 0.0, mwidth, mheight,
       BUFFER("in", mask_bytes, 3 * msize), BYTES_OUT("out", 3 * msize),
       INT("msize", msize), INT("mwidth", mwidth), INT("mheight", mheight));

  // the PPG kernels that demosaic the borders for RCD, with the local memory
  // the OpenCL kernels need for 16x16 work groups
  TEST("ppg_demosaic_green", "demosaic_ppg", "ppg_demosaic_green",
       RAW_IN("in", image_raw), IMG_OUT("out"), INT("width", W), INT("height", H), INT("filters", bayer[0]),
       LOCAL("buffer", sizeof(float) * (16 + 6) * (16 + 6)), INT("border", 32));
  TEST("ppg_demosaic_redblue", "demosaic_ppg", "ppg_demosaic_redblue",
       IMG_IN("in", image_in), IMG_OUT("out"), INT("width", W), INT("height", H), INT("filters", bayer[0]),
       LOCAL("buffer", 4 * sizeof(float) * (16 + 2) * (16 + 2)), INT("border", 16));
  // the passthroughs, and the downscaling demosaic of low quality thumbnails
  // at a quarter and a fifth of the size
  TEST("passthrough_monochrome", "demosaic_other", "passthrough_monochrome",
       RAW_IN("in", image_raw), IMG_OUT("out"), INT("width", W), INT("height", H));
  TEST("passthrough_color rggb", "demosaic_other", "passthrough_color",
       RAW_IN("in", image_raw), IMG_OUT("out"), INT("width", W), INT("height", H), INT("filters", bayer[0]),
       BUFFER("xtrans", xtrans, sizeof(xtrans)));
  TEST("passthrough_color xtrans", "demosaic_other", "passthrough_color",
       RAW_IN("in", image_raw), IMG_OUT("out"), INT("width", W), INT("height", H), INT("filters", 9),
       BUFFER("xtrans", xtrans, sizeof(xtrans)));
  {
    static const int zw[2] = { W / 4, W / 5 }, zh[2] = { H / 4, H / 5 };
    static const float zs[2] = { 0.25f, 0.2f };
    static const char *zlabels[2][3] = {
      { "zoom_passthrough_monochrome 1/4", "zoom_half_size 1/4", "zoom_third_size_xtrans 1/4" },
      { "zoom_passthrough_monochrome 1/5", "zoom_half_size 1/5", "zoom_third_size_xtrans 1/5" } };
    for(int k = 0; k < 2; k++)
    {
      TEST_GRID(zlabels[k][0], "demosaic_other", "clip_and_zoom_demosaic_passthrough_monochrome", 0.0, zw[k], zh[k],
                RAW_IN("in", image_raw),
                { .kind = ARG_IMAGE_OUT, .name = "out", .w = zw[k], .h = zh[k], .channels = 4 },
                INT("width", zw[k]), INT("height", zh[k]), INT("rin_wd", W), INT("rin_ht", H),
                FLOAT("r_scale", zs[k]));
      TEST_GRID(zlabels[k][1], "demosaic_ppg", "clip_and_zoom_demosaic_half_size", 0.0, zw[k], zh[k],
                RAW_IN("in", image_raw),
                { .kind = ARG_IMAGE_OUT, .name = "out", .w = zw[k], .h = zh[k], .channels = 4 },
                INT("width", zw[k]), INT("height", zh[k]), INT("rin_wd", W), INT("rin_ht", H),
                FLOAT("r_scale", zs[k]), INT("filters", bayer[0]));
      TEST_GRID(zlabels[k][2], "demosaic_vng", "clip_and_zoom_demosaic_third_size_xtrans", 0.0, zw[k], zh[k],
                RAW_IN("in", image_raw),
                { .kind = ARG_IMAGE_OUT, .name = "out", .w = zw[k], .h = zh[k], .channels = 4 },
                INT("width", zw[k]), INT("height", zh[k]), INT("rin_wd", W), INT("rin_ht", H),
                FLOAT("r_scale", zs[k]), BUFFER("xtrans", xtrans, sizeof(xtrans)));
    }
  }
  // the threshold lets about half the neighbors into the median
  TEST("pre_median", "demosaic_ppg", "pre_median",
       RAW_IN("in", image_raw), RAW_OUT("out"), INT("width", W), INT("height", H), INT("filters", bayer[0]),
       FLOAT("threshold", 0.25f), LOCAL("buffer", sizeof(float) * (16 + 4) * (16 + 4)));

  // green equilibration and color smoothing. The full average reduces to one
  // partial sum per 16x16 work group first; the Slang apply step takes the two
  // sums, from which the OpenCL host computes the ratio
  {
    static const float ge_sums[2] = { 1.0f, 1.07f };
    const int ngroups = ((W + 15) / 16) * ((H + 15) / 16);
    TEST("green_equilibration_lavg", "demosaic_ppg", "green_equilibration_lavg",
         RAW_IN("in", image_raw), RAW_OUT("out"), INT("width", W), INT("height", H), INT("filters", bayer[0]),
         FLOAT("thr", 0.05f), LOCAL("buffer", sizeof(float) * (16 + 4) * (16 + 4)));
    TEST_TOL("green_equilibration_favg_reduce_first", "demosaic_ppg", "green_equilibration_favg_reduce_first",
             1e-6, RAW_IN("in", image_raw), INT("width", W), INT("height", H),
             BUFFER_OUT("accu", NULL, sizeof(float) * 2 * ngroups), INT("filters", bayer[0]),
             LOCAL("buffer", sizeof(float) * 2 * 16 * 16));
    TEST("green_equilibration_favg_apply", "demosaic_ppg", "green_equilibration_favg_apply",
         RAW_IN("in", image_raw), RAW_OUT("out"), INT("width", W), INT("height", H), INT("filters", bayer[0]),
         FLOAT("gr_ratio", ge_sums[1] / ge_sums[0]), SLANG_BUFFER("sums", ge_sums, sizeof(ge_sums)));
    TEST("color_smoothing", "demosaic_ppg", "color_smoothing",
         IMG_IN("in", image_in), IMG_OUT("out"), INT("width", W), INT("height", H),
         LOCAL("buffer", 4 * sizeof(float) * (16 + 2) * (16 + 2)));
  }

  // the RCD steps, on plausible intermediate data rather than the output of
  // the step before, so that each step is compared on its own
  const size_t plane = sizeof(float) * W * H;
  for(int b = 0; b < 2; b++)
  {
    static const char *rcd_labels[2][9] = {
      { "rcd_populate rggb", "rcd_step_1 rggb", "rcd_step_2 rggb", "rcd_step_3 rggb", "rcd_step_4_0 rggb",
        "rcd_step_4_1 rggb", "rcd_step_4_2 rggb", "rcd_step_4_3 rggb", "rcd_write_output rggb" },
      { "rcd_populate bggr", "rcd_step_1 bggr", "rcd_step_2 bggr", "rcd_step_3 bggr", "rcd_step_4_0 bggr",
        "rcd_step_4_1 bggr", "rcd_step_4_2 bggr", "rcd_step_4_3 bggr", "rcd_write_output bggr" } };
    const int f = bayer[b];
    TEST(rcd_labels[b][0], "demosaic_rcd", "rcd_populate",
         RAW_IN("in", image_raw), BUFFER_OUT("cfa", NULL, plane), BUFFER_OUT("rgb0", NULL, plane),
         BUFFER_OUT("rgb1", NULL, plane), BUFFER_OUT("rgb2", NULL, plane), INT("w", W), INT("height", H),
         INT("filters", f), FLOAT("scale", 1.0f / 1.05f));
    TEST(rcd_labels[b][1], "demosaic_rcd", "rcd_step_1",
         BUFFER("cfa", rcd_cfa, plane), BUFFER_OUT("VH_dir", NULL, plane), INT("w", W), INT("height", H),
         LOCAL("buffer", sizeof(float) * (16 + 8) * (16 + 8)));
    TEST_GRID(rcd_labels[b][2], "demosaic_rcd", "rcd_step_2", 0.0, W / 2, H,
         BUFFER_OUT("lpf", NULL, plane), BUFFER("cfa", rcd_cfa, plane), INT("w", W), INT("height", H),
         INT("filters", f));
    TEST_GRID(rcd_labels[b][3], "demosaic_rcd", "rcd_step_3", 0.0, W / 2, H,
         BUFFER("lpf", rcd_lpf, plane), BUFFER("cfa", rcd_cfa, plane), BUFFER_OUT("rgb1", rcd_rgb[1], plane),
         BUFFER("VH_Dir", rcd_dir, plane), INT("w", W), INT("height", H), INT("filters", f));
    TEST_GRID(rcd_labels[b][4], "demosaic_rcd", "rcd_step_4_0", 0.0, W / 2, H,
         BUFFER("cfa", rcd_cfa, plane), BUFFER_OUT("p_diff", NULL, plane), BUFFER_OUT("q_diff", NULL, plane),
         INT("w", W), INT("height", H), INT("filters", f));
    TEST_GRID(rcd_labels[b][5], "demosaic_rcd", "rcd_step_4_1", 0.0, W / 2, H,
         BUFFER_OUT("PQ_dir", NULL, plane), BUFFER("p_diff", rcd_lpf, plane), BUFFER("q_diff", rcd_cfa, plane),
         INT("w", W), INT("height", H), INT("filters", f));
    TEST_GRID(rcd_labels[b][6], "demosaic_rcd", "rcd_step_4_2", 0.0, W / 2, H,
         BUFFER("PQ_dir", rcd_dir, plane), BUFFER_OUT("rgb0", rcd_rgb[0], plane),
         BUFFER_OUT("rgb1", rcd_rgb[1], plane), BUFFER_OUT("rgb2", rcd_rgb[2], plane),
         INT("w", W), INT("height", H), INT("filters", f));
    TEST_GRID(rcd_labels[b][7], "demosaic_rcd", "rcd_step_4_3", 0.0, W / 2, H,
         BUFFER("VH_dir", rcd_dir, plane), BUFFER_OUT("rgb0", rcd_rgb[0], plane),
         BUFFER_OUT("rgb1", rcd_rgb[1], plane), BUFFER_OUT("rgb2", rcd_rgb[2], plane),
         INT("w", W), INT("height", H), INT("filters", f));
    TEST(rcd_labels[b][8], "demosaic_rcd", "rcd_write_output",
         IMG_OUT("out"), BUFFER("rgb0", rcd_rgb[0], plane), BUFFER("rgb1", rcd_rgb[1], plane),
         BUFFER("rgb2", rcd_rgb[2], plane), INT("w", W), INT("height", H), FLOAT("scale", 1.05f),
         INT("border", 9));
  }

  // capture sharpening of demosaic, on a 4-channel demosaiced image and the
  // raw it came from. The kernel table picks small and large 9x9 kernels.
  {
    enum { CS_ALIGN = 32, CS_TABLE = (W * H + 3) & ~3 };
    static float cs_ones[W * H], cs_y[W * H], cs_blend[W * H], cs_lum[W * H], cs_tmp[W * H];
    static float cs_kernels[256 * CS_ALIGN], cs_gauss[25];
    static unsigned char cs_table[CS_TABLE];
    for(int k = 0; k < W * H; k++)
    {
      cs_ones[k] = 1.0f;
      cs_y[k] = 0.02f + fabsf(image_in[4 * k + 1]);
      cs_blend[k] = (k % 11 == 0) ? 0.0f : fminf(fmaxf(image_in[4 * k + 2], 0.0f), 1.0f);
      cs_lum[k] = 0.01f + fabsf(image_in[4 * k]);
      cs_tmp[k] = 0.01f + fabsf(image_in[4 * k + 3]);
      cs_table[k] = (unsigned char)((k * 37) % 256);
    }
    for(int k = 0; k < 256 * CS_ALIGN; k++) cs_kernels[k] = 0.01f + 0.04f * frand();
    for(int k = 0; k < 25; k++) cs_gauss[k] = 0.01f + 0.04f * frand();
    TEST("prepare_blend", "capture", "prepare_blend",
         RAW_IN("cfa", image_raw), IMG_IN("dev_out", image_in), INT("filters", bayer[0]),
         BUFFER("xtrans", xtrans, sizeof(xtrans)), BUFFER_OUT("mask", cs_ones, sizeof(cs_ones)),
         BUFFER_OUT("Yold", NULL, sizeof(cs_y)), FLOAT4("wb", 0.9f, 0.95f, 0.85f, 0.0f), INT("w", W),
         INT("height", H));
    TEST("modify_blend", "capture", "modify_blend",
         BUFFER_OUT("blend", cs_blend, sizeof(cs_blend)), BUFFER("Yold", cs_y, sizeof(cs_y)),
         BUFFER_OUT("luminance", NULL, sizeof(cs_lum)), FLOAT("dthresh", 0.4f), INT("width", W),
         INT("height", H));
    TEST_LINE("final_blend", "capture", "final_blend", 0.0, W * H,
              BUFFER_OUT("blendmask", cs_blend, sizeof(cs_blend)), BUFFER("unblurred", cs_tmp, sizeof(cs_tmp)),
              INT("pixels", W * H));
    TEST("gaussian_kernel_9x9", "gaussian", "gaussian_kernel_9x9",
         BUFFER("input", cs_y, sizeof(cs_y)), BUFFER_OUT("output", NULL, sizeof(cs_y)), INT("width", W),
         INT("height", H), INT("ch", 1), BUFFER("kern", cs_gauss, sizeof(cs_gauss)), FLOAT("minval", 0.0f),
         FLOAT("maxval", 1.0f));
    TEST("kernel_9x9_div", "capture", "kernel_9x9_div",
         BUFFER("in", cs_y, sizeof(cs_y)), BUFFER_OUT("out", cs_tmp, sizeof(cs_tmp)),
         BUFFER("luminance", cs_lum, sizeof(cs_lum)), BUFFER("blend", cs_blend, sizeof(cs_blend)),
         BUFFER("kernels", cs_kernels, sizeof(cs_kernels)), BUFFER("table", cs_table, sizeof(cs_table)),
         INT("w1", W), INT("height", H));
    TEST("kernel_9x9_mul", "capture", "kernel_9x9_mul",
         BUFFER("in", cs_tmp, sizeof(cs_tmp)), BUFFER_OUT("out", cs_y, sizeof(cs_y)),
         BUFFER("blend", cs_blend, sizeof(cs_blend)), BUFFER("kernels", cs_kernels, sizeof(cs_kernels)),
         BUFFER("table", cs_table, sizeof(cs_table)), INT("w1", W), INT("height", H));
    TEST("capture_result", "capture", "capture_result",
         IMG_IN("in", image_in), IMG_OUT("out"), BUFFER("blendmask", cs_blend, sizeof(cs_blend)),
         BUFFER("luminance", cs_lum, sizeof(cs_lum)), BUFFER("tmp", cs_tmp, sizeof(cs_tmp)), INT("width", W),
         INT("height", H));

    // the mask of dual demosaic, which the detail mask shares, and its blend
    static float dual_low[4 * W * H];
    for(int k = 0; k < 4 * W * H; k++) dual_low[k] = image_in[(k + 4 * 97) % (4 * W * H)];
    TEST("calc_Y0_mask", "blendop", "calc_Y0_mask",
         BUFFER_OUT("mask", NULL, sizeof(cs_y)), IMG_IN("in", image_in), INT("w", W), INT("height", H),
         FLOAT4("wb", 0.5f, 1.0f, 0.7f, 1.0f));
    TEST("calc_scharr_mask", "blendop", "calc_scharr_mask",
         BUFFER("in", cs_y, sizeof(cs_y)), BUFFER_OUT("out", NULL, sizeof(cs_tmp)), INT("w", W),
         INT("height", H));
    TEST("calc_detail_blend", "blendop", "calc_detail_blend",
         BUFFER("in", cs_blend, sizeof(cs_blend)), BUFFER_OUT("out", NULL, sizeof(cs_tmp)), INT("w", W),
         INT("height", H), FLOAT("threshold", 0.3f), INT("detail", 1));
    TEST("calc_detail_blend inverted", "blendop", "calc_detail_blend",
         BUFFER("in", cs_blend, sizeof(cs_blend)), BUFFER_OUT("out", NULL, sizeof(cs_tmp)), INT("w", W),
         INT("height", H), FLOAT("threshold", 0.3f), INT("detail", 0));
    TEST("write_blended_dual", "demosaic_rcd", "write_blended_dual",
         IMG_IN("high", image_in), IMG_IN("low", dual_low), IMG_OUT("out"), INT("w", W), INT("height", H),
         BUFFER("mask", cs_blend, sizeof(cs_blend)), INT("showmask", 0));
  }

  // VNG for one Bayer pattern and X-Trans: the linear interpolation over the
  // whole image and only on a border, and the gradients on its output
  {
    static vng_tables_t vng_tb[2];
    static const char *vng_labels[2][4] = {
      { "vng_lin_interpolate rggb", "vng_lin_interpolate border rggb", "vng_lin_interpolate linear rggb",
        "vng_interpolate rggb" },
      { "vng_lin_interpolate xtrans", "vng_lin_interpolate border xtrans", "vng_lin_interpolate linear xtrans",
        "vng_interpolate xtrans" } };
    for(int k = 0; k < 2; k++)
    {
      const uint32_t f4 = k ? 9u : bayer[0] | 0x0c0c0c0cu;
      vng_tables(&vng_tb[k], f4, xtrans);
      TEST(vng_labels[k][0], "demosaic_vng", "vng_lin_interpolate",
           RAW_IN("in", image_raw), IMG_OUT("out"), INT("width", W), INT("height", H), INT("border", 0),
           INT("filters", f4), BUFFER("xtrans", xtrans, sizeof(xtrans)),
           BUFFER("lookup", vng_tb[k].lookup, sizeof(vng_tb[k].lookup)),
           LOCAL("buffer", sizeof(float) * 18 * 18), INT("only_linear", 0));
      TEST(vng_labels[k][1], "demosaic_vng", "vng_lin_interpolate",
           RAW_IN("in", image_raw), IMG_OUT("out"), INT("width", W), INT("height", H), INT("border", 19),
           INT("filters", f4), BUFFER("xtrans", xtrans, sizeof(xtrans)),
           BUFFER("lookup", vng_tb[k].lookup, sizeof(vng_tb[k].lookup)),
           LOCAL("buffer", sizeof(float) * 18 * 18), INT("only_linear", 1));
      TEST(vng_labels[k][2], "demosaic_vng", "vng_lin_interpolate",
           RAW_IN("in", image_raw), IMG_OUT("out"), INT("width", W), INT("height", H), INT("border", 0),
           INT("filters", f4), BUFFER("xtrans", xtrans, sizeof(xtrans)),
           BUFFER("lookup", vng_tb[k].lookup, sizeof(vng_tb[k].lookup)),
           LOCAL("buffer", sizeof(float) * 18 * 18), INT("only_linear", 1));
      TEST(vng_labels[k][3], "demosaic_vng", "vng_interpolate",
           IMG_IN("in", image_in), IMG_OUT("out"), INT("width", W), INT("height", H), INT("filters", f4),
           BUFFER("xtrans", xtrans, sizeof(xtrans)), BUFFER("ips", vng_tb[k].ips, VNG_IPS_SIZE),
           BUFFER("code", vng_tb[k].code, sizeof(vng_tb[k].code)), LOCAL("buffer", 4 * sizeof(float) * 20 * 20));
    }
  }

  // the Markesteijn steps, each on plausible data of its own: color planes
  // with all channels set, green limits around them and the raw data as
  // OpenCL's markesteijn_initial_copy() spreads it over the channels. The
  // Slang kernels that merge the steps after the green interpolation have no
  // OpenCL counterpart to compare with here
  {
    static signed char allhex[3][3][8][2];
    static int sg[2];
    markesteijn_hex(xtrans, allhex, sg);
    const size_t npix = (size_t)W * H;
    const size_t p4 = sizeof(float) * 4 * npix;
    static float *mk_rgb[4], *mk_cfa, *mk_yuv, *mk_minmax;
    for(int c = 0; c < 4; c++)
    {
      mk_rgb[c] = malloc(p4);
      for(size_t k = 0; k < 4 * npix; k++) mk_rgb[c][k] = fmaxf(0.0f, image_in[k] * (0.9f + 0.2f * frand()));
    }
    mk_cfa = calloc(4 * npix, sizeof(float));
    mk_yuv = calloc(4 * npix, sizeof(float));
    mk_minmax = malloc(sizeof(float) * 2 * npix);
    for(int y = 0; y < H; y++)
      for(int x = 0; x < W; x++)
      {
        const size_t k = (size_t)y * W + x;
        mk_cfa[4 * k + fcol(y, x, 9, xtrans)] = fmaxf(0.0f, image_raw[k]);
        const float *p = mk_rgb[0] + 4 * k;
        const float Y = 0.2627f * p[0] + 0.6780f * p[1] + 0.0593f * p[2];
        mk_yuv[4 * k] = Y;
        mk_yuv[4 * k + 1] = (p[2] - Y) * 0.56433f;
        mk_yuv[4 * k + 2] = (p[0] - Y) * 0.67815f;
        const float g = mk_rgb[0][4 * k + 1];
        mk_minmax[2 * k] = g - 0.1f * frand();
        mk_minmax[2 * k + 1] = g + 0.1f * frand();
      }
#define MK_HEX BUFFER("allhex", allhex, sizeof(allhex))
#define MK_XT BUFFER("xtrans", xtrans, sizeof(xtrans))
    TEST("markesteijn_green_minmax", "demosaic_markesteijn", "markesteijn_green_minmax",
         BUFFER("rgb", mk_cfa, p4), SLANG_RAW_IN("in", image_raw),
         BUFFER_OUT("gminmax", NULL, sizeof(float) * 2 * npix), INT("width", W),
         INT("height", H), INT("border", 3), CHAR2("sgreen", sg[0], sg[1]), MK_XT, MK_HEX,
         LOCAL("buffer", sizeof(float) * 22 * 22));
    // the four buffers start as the OpenCL kernel finds them, and the Slang
    // kernel fills them from the raw data as well
    TEST("markesteijn_initial_green", "demosaic_markesteijn", "markesteijn_interpolate_green",
         SLANG_RAW_IN("in", image_raw),
         BUFFER_OUT("rgb_0", mk_cfa, p4), BUFFER_OUT("rgb_1", mk_cfa, p4), BUFFER_OUT("rgb_2", mk_cfa, p4),
         BUFFER_OUT("rgb_3", mk_cfa, p4), BUFFER("gminmax", mk_minmax, sizeof(float) * 2 * npix),
         INT("width", W), INT("height", H), INT("border", 3), CHAR2("sgreen", sg[0], sg[1]), MK_XT, MK_HEX,
         LOCAL("buffer", 4 * sizeof(float) * 28 * 28));
    t[n - 1].slang_kernel = "markesteijn_initial_green";
    for(int d = 0; d < 6; d += 3)
    {
      static const char *mk_labels[6] = { "markesteijn_solitary_green d0", 0, 0, "markesteijn_solitary_green d3" };
      const int i = d & 1 ? 0 : 1;
      // the Slang work items cover only the solitary green pixels
      TEST_GRID(mk_labels[d], "demosaic_markesteijn", "markesteijn_solitary_green", 0.0,
                (W - sg[0] + 2) / 3, (H - sg[1] + 2) / 3,
                BUFFER_OUT("rgb", mk_rgb[0], p4), BUFFER_OUT("aux", mk_rgb[1], p4), INT("width", W),
                INT("height", H), INT("border", 5), INT("d", d), CHAR2("dir", i, i ^ 1),
                INT("hcomp", d & 1 ? 2 : 0), CHAR2("sgreen", sg[0], sg[1]), MK_XT,
                LOCAL("buffer", 4 * sizeof(float) * 20 * 20));
      t[n - 1].clgw = W;
      t[n - 1].clgh = H;
    }
    TEST("markesteijn_recalculate_green", "demosaic_markesteijn", "markesteijn_recalculate_green",
         BUFFER_OUT("rgb_0", mk_rgb[0], p4), BUFFER_OUT("rgb_1", mk_rgb[1], p4), BUFFER_OUT("rgb_2", mk_rgb[2], p4),
         BUFFER_OUT("rgb_3", mk_rgb[3], p4), BUFFER("gminmax", mk_minmax, sizeof(float) * 2 * npix),
         INT("width", W), INT("height", H), INT("border", 6), CHAR2("sgreen", sg[0], sg[1]), MK_XT, MK_HEX);
    for(int d = 0; d < 4; d += 3)
    {
      static const char *mk_labels[4] = { "markesteijn_red_and_blue d0", 0, 0, "markesteijn_red_and_blue d3" };
      TEST(mk_labels[d], "demosaic_markesteijn", "markesteijn_red_and_blue",
           BUFFER_OUT("rgb", mk_rgb[d], p4), INT("width", W), INT("height", H), INT("border", 6), INT("d", d),
           CHAR2("sgreen", sg[0], sg[1]), MK_XT, LOCAL("buffer", 4 * sizeof(float) * 22 * 22));
    }
    for(int d = 0; d < 8; d += 6)
    {
      static const char *mk_labels[8] = { "markesteijn_interpolate_twoxtwo d0", 0, 0, 0, 0, 0,
                                       "markesteijn_interpolate_twoxtwo d6" };
      TEST(mk_labels[d], "demosaic_markesteijn", "markesteijn_interpolate_twoxtwo",
           BUFFER_OUT("rgb", mk_rgb[d / 2], p4), INT("width", W), INT("height", H), INT("border", 8), INT("d", d),
           CHAR2("sgreen", sg[0], sg[1]), MK_XT, MK_HEX, LOCAL("buffer", 4 * sizeof(float) * 20 * 20));
    }
    // OpenCL differentiates the converted buffer, the Slang kernel converts as it reads
    for(int d = 0; d < 8; d += 5)
    {
      static const char *mk_labels[8] = { "markesteijn_differentiate d0", 0, 0, 0, 0, "markesteijn_differentiate d5" };
      TEST(mk_labels[d], "demosaic_markesteijn", "markesteijn_differentiate",
           BUFFER("yuv", mk_yuv, p4), SLANG_BUFFER("rgb", mk_rgb[0], p4),
           BUFFER_OUT("drv", NULL, sizeof(float) * npix), INT("width", W),
           INT("height", H), INT("border", 8), INT("d", d), LOCAL("buffer", 4 * sizeof(float) * 18 * 18));
    }
#undef MK_HEX
#undef MK_XT
  }

  // tone curve on Lab: the a and b curves bounded and unbounded, a and b
  // scaled with L, the curve on XYZ, and on ProPhoto RGB with and without a norm
  {
    static const float tc_coeffs_ab[12] = { 1.25f, 1.0f, 1.4f, 1.3f, 1.0f, 1.2f,
                                            1.2f, 1.0f, 1.3f, 1.35f, 1.0f, 1.5f };
    static const char *tc_labels[6] = { "tonecurve ab", "tonecurve ab unbounded", "tonecurve scaled ab",
                                        "tonecurve xyz", "tonecurve rgb", "tonecurve rgb luminance" };
    static const int tc_autoscale[6] = { 0, 0, 1, 2, 3, 3 };
    // past the a and b curves the L curve is looked up, without
    // interpolation, at a value converted from Lab: float noise can move that
    // to the next LUT entry, and the conversion back from far out of gamut
    // pixels of the noise image amplifies it, up to about 5e-3
    for(int k = 0; k < 6; k++)
      TEST_TOL(tc_labels[k], "basic", "tonecurve", k >= 3 ? 1e-2 : 0.0,
               IMG_IN("in", image_lab), IMG_OUT("out"), INT("width", W), INT("height", H),
               LUT("table_L", lut_curve, 256, 256), LUT("table_a", lut_curve, 256, 256),
               LUT("table_b", lut_curve, 256, 256), INT("autoscale_ab", tc_autoscale[k]),
               INT("unbound_ab", k == 1), BUFFER("coeffs_L", coeffs_linear, sizeof(coeffs_linear)),
               BUFFER("coeffs_ab", tc_coeffs_ab, sizeof(tc_coeffs_ab)), FLOAT("low_approximation", 0.7f),
               INT("preserve_colors", k == 5 ? 1 : 0),
               BUFFER("profile_info", &profile_linear, sizeof(profile_info_t)),
               LUT("lut", lut_profile, 256, 256 * 6));
  }

  // vignette: darkening and brightening, clipped and unbounded
  for(int k = 0; k < 2; k++)
  {
    static const char *vg_labels[2] = { "vignette darken", "vignette brighten unbounded" };
    TEST(vg_labels[k], "extended", "vignette",
         IMG_IN("in", image_in), IMG_OUT("out"), INT("width", W), INT("height", H),
         FLOAT2("scale", 2.0f / W, 2.4f / H), FLOAT2("roi_center_scaled", 1.1f, 0.9f),
         FLOAT2("expt", 2.0f / 1.3f, 1.3f / 2.0f), FLOAT("dscale", 0.3f), FLOAT("fscale", 0.5f),
         FLOAT("brightness", k ? 0.4f : -0.6f), FLOAT("saturation", k ? -0.3f : 0.5f),
         FLOAT("dither", 1.0f / 256), INT("unbound", k));
  }

  // rotate and perspective: a slight rotation with a keystone, partly
  // outside the input
  {
    static const float as_homo[9] = { 0.995f, -0.06f, 9.0f, 0.07f, 1.01f, -6.0f, 1.2e-5f, -2e-5f, 1.0f };
    static const char *as_kernels[4] = { "ashift_bilinear", "ashift_bicubic", "ashift_lanczos2", "ashift_lanczos3" };
    // the homography lands a few ulp apart on the two devices, which on a
    // noise image moves the interpolated values by up to about 1.6e-4
    for(int k = 0; k < 4; k++)
      TEST_TOL(as_kernels[k], "basic", as_kernels[k], 5e-4,
               IMG_IN("in", image_in), IMG_OUT("out"), INT("width", W), INT("height", H),
               INT("iwidth", W), INT("iheight", H), INT2("roi_in", 3, 2), INT2("roi_out", 1, 4),
               FLOAT("in_scale", 1.0f), FLOAT("out_scale", 1.0f), FLOAT2("clip", 2.5f, 1.5f),
               BUFFER("homograph", as_homo, sizeof(as_homo)));
  }

  // contrast equalizer: the edge-aware wavelet steps at a fine and a coarse scale
  {
    static float eaw_filter[25];
    static const float m[5] = { 0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f };
    for(int j = 0; j < 5; j++)
      for(int i = 0; i < 5; i++) eaw_filter[j * 5 + i] = m[i] * m[j];
    for(int s = 0; s < 4; s += 3)
    {
      static const char *eaw_labels[4] = { "eaw_decompose scale 0", 0, 0, "eaw_decompose scale 3" };
      TEST_TOL(eaw_labels[s], "atrous", "eaw_decompose", 1e-5,
               IMG_IN("in", image_in), IMG_OUT("coarse"), IMG_OUT("detail"), INT("width", W), INT("height", H),
               INT("scale", s), FLOAT("sharpen", 2.5f), BUFFER("filter", eaw_filter, sizeof(eaw_filter)));
    }
    TEST("eaw_synthesize", "atrous", "eaw_synthesize",
         IMG_OUT("out"), IMG_IN("coarse", image_in), IMG_IN("detail", image_lab), INT("width", W),
         INT("height", H), FLOAT4("threshold", 10.0f, 5.0f, 6.0f, 0.0f), FLOAT4("boost", 1.2f, 0.8f, 1.5f, 1.0f));
  }

  // haze removal: the box filters along rows and columns, the transition map
  // and the dehazing
  {
    static const char *hz_kernels[4] = { "hazeremoval_box_min_x", "hazeremoval_box_min_y",
                                         "hazeremoval_box_max_x", "hazeremoval_box_max_y" };
    // one work item per pixel in Slang, per row or column in OpenCL
    for(int k = 0; k < 4; k++)
    {
      TEST(hz_kernels[k], "hazeremoval", hz_kernels[k],
           INT("width", W), INT("height", H), RAW_IN("in", image_raw), RAW_OUT("out"), INT("w", 6));
      t[n - 1].clgw = (k & 1) ? W : H;
      t[n - 1].clgh = 1;
      t[n - 1].clly = 1;
    }
    TEST("hazeremoval_transision_map", "hazeremoval", "hazeremoval_transision_map",
         INT("width", W), INT("height", H), IMG_IN("in", image_in), RAW_OUT("out"), FLOAT("strength", 0.6f),
         FLOAT("A0_r", 0.9f), FLOAT("A0_g", 0.95f), FLOAT("A0_b", 1.05f));
    TEST("hazeremoval_dehaze", "hazeremoval", "hazeremoval_dehaze",
         INT("width", W), INT("height", H), IMG_IN("in", image_in), RAW_IN("trans_map", rcd_dir),
         IMG_OUT("out"), FLOAT("t_min", 0.1f), FLOAT4("A0", 0.9f, 0.95f, 1.05f, 0.0f));
  }

  // resampling, with plans like _prepare_resampling_plan() makes them: per
  // output pixel a length index, a kernel offset and an index offset, here
  // with 2 to 6 taps
  {
    enum { OW = 331, OH = 157, TAPS = 6, VTAPS = 8 };
    static int hmeta[OW * 3], vmeta[OH * 3], hlength[OW], vlength[OH];
    static int hindex[OW * (TAPS + 1)], vindex[OH * (TAPS + 1)];
    static float hkernel[OW * (TAPS + 1)], vkernel[OH * (TAPS + 1)];
    for(int pass = 0; pass < 2; pass++)
    {
      const int out = pass ? OH : OW, in = pass ? H : W;
      int *meta = pass ? vmeta : hmeta, *length = pass ? vlength : hlength, *index = pass ? vindex : hindex;
      float *kernel = pass ? vkernel : hkernel;
      for(int k = 0; k < out; k++)
      {
        const int taps = 2 + k % (TAPS - 1);
        const int center = (int)((k + 0.5f) * in / out);
        meta[k * 3] = k;
        meta[k * 3 + 1] = k * (TAPS + 1);
        meta[k * 3 + 2] = k * (TAPS + 1);
        length[k] = taps;
        float norm = 0.0f;
        for(int tap = 0; tap < taps; tap++)
        {
          const int i = center - taps / 2 + tap;
          index[k * (TAPS + 1) + tap] = i < 0 ? 0 : i >= in ? in - 1 : i;
          kernel[k * (TAPS + 1) + tap] = 0.5f + frand();
          norm += kernel[k * (TAPS + 1) + tap];
        }
        for(int tap = 0; tap < taps; tap++) kernel[k * (TAPS + 1) + tap] /= norm;
      }
    }
    // one work group per column and VTAPS work items per output pixel, as
    // dt_interpolation_resample_cl() runs it
    t[n] = (test_t){ .label = "interpolation_resample", .program = "basic", .kernel = "interpolation_resample",
                     .tolerance = 1e-5, .gw = OW, .gh = OH,
                     .clgw = OW, .clgh = OH * VTAPS, .cllx = 1, .clly = 4 * VTAPS };
    const arg_t ra[] = {
      IMG_IN("in", image_in), { .kind = ARG_IMAGE_OUT, .name = "out", .w = OW, .h = OH, .channels = 4 },
      INT("width", OW), INT("height", OH), BUFFER("hmeta", hmeta, sizeof(hmeta)),
      BUFFER("vmeta", vmeta, sizeof(vmeta)), BUFFER("hlength", hlength, sizeof(hlength)),
      BUFFER("vlength", vlength, sizeof(vlength)), BUFFER("hindex", hindex, sizeof(hindex)),
      BUFFER("vindex", vindex, sizeof(vindex)), BUFFER("hkernel", hkernel, sizeof(hkernel)),
      BUFFER("vkernel", vkernel, sizeof(vkernel)), INT("htaps", TAPS), INT("vtaps", VTAPS),
      LOCAL("lkernel", TAPS * sizeof(float)), LOCAL("lindex", TAPS * sizeof(int)),
      LOCAL("buffer", 4 * VTAPS * 4 * sizeof(float)) };
    t[n].nargs = sizeof(ra) / sizeof(ra[0]);
    memcpy(t[n].args, ra, sizeof(ra));
    n++;

    TEST_GRID("interpolation_copy", "basic", "interpolation_copy", 0.0, OW, OH,
              IMG_IN("dev_in", image_in), { .kind = ARG_IMAGE_OUT, .name = "dev_out", .w = OW, .h = OH, .channels = 4 },
              INT("owidth", OW), INT("oheight", OH), INT("iwidth", W), INT("iheight", H),
              INT("dx", W - OW / 2), INT("dy", 7));
  }

  // flip: mirrored, and rotated by 90 degrees with a swapped output
  {
    static const int fl_orientation[4] = { 1, 2, 5, 6 };
    static char fl_labels[4][32];
    for(int k = 0; k < 4; k++)
    {
      const int o = fl_orientation[k], swap = o & 4;
      snprintf(fl_labels[k], sizeof(fl_labels[k]), "flip %d", o);
      TEST(fl_labels[k], "basic", "flip",
           IMG_IN("in", image_in),
           { .kind = ARG_IMAGE_OUT, .name = "out", .w = swap ? H : W, .h = swap ? W : H, .channels = 4 },
           INT("width", W), INT("height", H), INT("owidth", swap ? H : W), INT("oheight", swap ? W : H),
           INT("orientation", o));
    }
  }

  // rawprepare on integer raw values, cropped, with per-site black and white levels
  {
    static float rp_raw[W * H];
    for(int k = 0; k < W * H; k++) rp_raw[k] = floorf(fmaxf(0.0f, image_raw[k]) * 15000.0f) + 512.0f;
    static const float rp_sub[4] = { 512.0f, 510.0f, 514.0f, 509.0f };
    static const float rp_div[4] = { 15872.0f, 15870.0f, 15000.0f, 15869.0f };
    static float rp_rdiv[4];
    for(int k = 0; k < 4; k++) rp_rdiv[k] = 1.0f / rp_div[k];
    static const float rp_sub4[4] = { 0.02f, 0.01f, 0.03f, 0.0f };
    static const float rp_div4[4] = { 0.9f, 0.95f, 0.85f, 1.0f };
    static float rp_rdiv4[4];
    for(int k = 0; k < 4; k++) rp_rdiv4[k] = 1.0f / rp_div4[k];
    const int rw = W - 8, rh = H - 6;
    TEST_GRID("rawprepare 1f", "basic", "rawprepare_1f_unnormalized", 0.0, rw, rh,
              RAW_IN("in", rp_raw), { .kind = ARG_IMAGE_OUT, .name = "out", .w = rw, .h = rh, .channels = 1 },
              INT("width", rw), INT("height", rh), INT("cx", 3), INT("cy", 2),
              BUFFER("sub", rp_sub, sizeof(rp_sub)), BUFFER("div", rp_div, sizeof(rp_div)),
              BUFFER("rdiv", rp_rdiv, sizeof(rp_rdiv)), INT("rx", 5), INT("ry", 1));
    TEST_GRID("rawprepare 4f", "basic", "rawprepare_4f", 0.0, rw, rh,
              IMG_IN("in", image_in), { .kind = ARG_IMAGE_OUT, .name = "out", .w = rw, .h = rh, .channels = 4 },
              INT("width", rw), INT("height", rh), INT("cx", 3), INT("cy", 2),
              BUFFER("sub", rp_sub4, sizeof(rp_sub4)), BUFFER("div", rp_div4, sizeof(rp_div4)),
              BUFFER("rdiv", rp_rdiv4, sizeof(rp_rdiv4)), INT("rx", 5), INT("ry", 1));
  }

  // blending: a module's input and output with an offset between them, as
  // when a module crops, through each blend mode of each blend colorspace,
  // and the parametric masks
  {
    enum { OW = W - 8, OH = H - 6 };
    static float bl_rgb_b[4 * OW * OH], bl_lab_b[4 * OW * OH], bl_raw_b[OW * OH], bl_form[OW * OH];
    static float bl_params[6 * 16];
    for(int k = 0; k < OW * OH; k++)
    {
      for(int c = 0; c < 4; c++)
      {
        bl_rgb_b[4 * k + c] = image_in[(4 * k + c + 52) % (4 * W * H)] * 0.9f + 0.05f;
        bl_lab_b[4 * k + c] = image_lab[(4 * k + c + 52) % (4 * W * H)] * (c == 0 ? 0.8f : 1.1f);
      }
      bl_raw_b[k] = image_raw[(k + 31) % (W * H)] * 0.95f;
      bl_form[k] = 0.5f + 0.5f * sinf(k * 0.003f);
    }
    for(int ch = 0; ch < 16; ch++)
    {
      // a trapezoid per channel, wide enough that most pixels sit on a slope
      const float p0 = -0.1f + 0.02f * ch, p1 = p0 + 0.25f, p2 = p1 + 0.2f, p3 = p2 + 0.3f;
      const float pr[6] = { p0, p1, p2, p3, 1.0f / (p1 - p0), 1.0f / (p3 - p2) };
      memcpy(bl_params + 6 * ch, pr, sizeof(pr));
    }
#define BL_IMG(n, px, ch) { .kind = ARG_IMAGE_IN, .name = n, .w = OW, .h = OH, .channels = ch, .pixels = px }
#define BL_OUT(n, ch) { .kind = ARG_IMAGE_OUT, .name = n, .w = OW, .h = OH, .channels = ch }

    // Lab: L in, L out and h in, the last inverted; RGB: gray in, red out
    // and H in, red out inverted
    static const struct { const char *kernel; int blendif; int rgb_in; int profile; } bl_masks[] = {
      { "blendop_mask_Lab", 0x211 | (0x200 << 16), 0, 0 },
      { "blendop_mask_RAW", 0, 0, 0 },
      { "blendop_mask_rgb_hsl", 0x121 | (0x20 << 16), 1, 0 },
      { "blendop_mask_rgb_hsl", 0x121 | (0x20 << 16), 1, 1 },
      { "blendop_mask_rgb_jzczhz", 0x121 | (0x100 << 16), 1, 1 },
    };
    static char bl_mask_labels[2 * 5][64];
    for(int m = 0; m < 5; m++)
      for(int comb = 0; comb < 2; comb++)
      {
        // DEVELOP_COMBINE_NORM_EXCL, then DEVELOP_COMBINE_INV_INCL
        const unsigned int combine = comb ? 3 : 0;
        char *lbl = bl_mask_labels[2 * m + comb];
        snprintf(lbl, 64, "%s%s %s", bl_masks[m].kernel + 8, bl_masks[m].profile ? " profile" : "",
                 comb ? "inv incl" : "norm excl");
        TEST_GRID(lbl, "blendop", bl_masks[m].kernel, 0.0, OW, OH,
                  IMG_IN("in_a", bl_masks[m].rgb_in ? image_in : image_lab),
                  BL_IMG("in_b", bl_masks[m].rgb_in ? bl_rgb_b : bl_lab_b, 4), BL_IMG("mask_in", bl_form, 1),
                  BL_OUT("mask", 1), INT("width", OW), INT("height", OH), FLOAT("gopacity", 0.8f),
                  INT("blendif", bl_masks[m].blendif), BUFFER("blendif_parameters", bl_params, sizeof(bl_params)),
                  INT("mask_mode", 5), INT("mask_combine", combine), INT2("offs", 3, 2),
                  BUFFER("profile_info", bl_masks[m].profile ? &profile_nonlinear : &profile_linear,
                         sizeof(profile_info_t)),
                  LUT("profile_lut", lut_profile, 256, 256 * 6), INT("use_work_profile", bl_masks[m].profile));
      }

    TEST_GRID("mask_tone_curve", "blendop", "blendop_mask_tone_curve", 0.0, OW, OH,
              BL_IMG("mask_in", bl_form, 1), BL_OUT("mask_out", 1), INT("width", OW), INT("height", OH),
              FLOAT("e", expf(3.0f * 0.4f)), FLOAT("brightness", -0.3f), FLOAT("gopacity", 0.9f));

    static const int bl_modes[] = { 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D,
                                    0x0E, 0x0F, 0x10, 0x11, 0x12, 0x13, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B,
                                    0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x21, 0x22, 0x23, 0x25, 0x26, 0x27, 0x28,
                                    0x29, (int)(0x80000000u | 0x18), (int)(0x80000000u | 0x0B) };
    enum { NMODES = sizeof(bl_modes) / sizeof(bl_modes[0]) };
    static const struct { const char *kernel; int ch; int kind; } bl_kernels[] = {
      { "blendop_Lab", 4, 0 }, { "blendop_RAW", 1, 1 }, { "blendop_RAW4", 4, 2 },
      { "blendop_rgb_hsl", 4, 2 }, { "blendop_rgb_jzczhz", 4, 2 },
    };
    static char bl_labels[5 * NMODES][64];
    for(int b = 0; b < 5; b++)
      for(int m = 0; m < NMODES; m++)
      {
        const int kind = bl_kernels[b].kind;
        char *lbl = bl_labels[b * NMODES + m];
        snprintf(lbl, 64, "%s 0x%x", bl_kernels[b].kernel + 8, (unsigned)bl_modes[m]);
        // the Lab hue, color and color adjustment modes go through LCh and
        // back, and Vulkan only bounds the absolute error of sin and cos, by
        // 2^-11
        const int lch = kind == 0 && (bl_modes[m] == 0x12 || bl_modes[m] == 0x13 || bl_modes[m] == 0x16);
        TEST_GRID(lbl, "blendop", bl_kernels[b].kernel, lch ? 1e-3 : 0.0, OW, OH,
                  { .kind = ARG_IMAGE_IN, .name = "in_a", .w = W, .h = H, .channels = bl_kernels[b].ch,
                    .pixels = kind == 0 ? image_lab : kind == 1 ? image_raw : image_in },
                  BL_IMG("in_b", kind == 0 ? bl_lab_b : kind == 1 ? bl_raw_b : bl_rgb_b, bl_kernels[b].ch),
                  BL_IMG("mask", bl_form, 1), BL_OUT("out", bl_kernels[b].ch), INT("width", OW), INT("height", OH),
                  INT("blend_mode", bl_modes[m]), FLOAT("blend_parameter", 1.5f), INT2("offs", 3, 2),
                  INT("mask_display", 0));
      }
#undef BL_IMG
#undef BL_OUT
  }

  // color balance rgb, with both saturation formulas and its mask display
  {
    static const float cb_matrix_in[12] = { 0.31f, 0.58f, 0.11f, 0.0f, 0.18f, 0.71f, 0.11f, 0.0f,
                                            0.04f, 0.16f, 0.80f, 0.0f };
    static const float cb_matrix_out[12] = { 1.71f, -0.36f, -0.25f, 0.0f, -0.67f, 1.62f, 0.04f, 0.0f,
                                             0.02f, -0.04f, 0.91f, 0.0f };
    static const float cb_hue_rotation[4] = { 0.9848f, -0.1736f, 0.1736f, 0.9848f };
    static float cb_gamut_lut[512];
    for(int k = 0; k < 512; k++) cb_gamut_lut[k] = 0.6f + 0.3f * sinf(k * 0.0491f) + 0.05f * cosf(k * 0.31f);
    static const char *cb_labels[3] = { "colorbalancergb dtucs", "colorbalancergb jzazbz", "colorbalancergb mask" };
    // pow() on Vulkan is exp2(log2()), and the JzAzBz exponent of 134 scales up
    // its error: PoCL and lavapipe land about 1e-3 apart there
    static const double cb_tol[3] = { 5e-4, 5e-3, 0.0 };
    for(int mode = 0; mode < 3; mode++)
      TEST_TOL(cb_labels[mode], "extended", "colorbalancergb", cb_tol[mode],
           IMG_IN("in", image_in), IMG_OUT("out"), INT("width", W), INT("height", H),
           BUFFER("profile_info", &profile_linear, sizeof(profile_info_t)),
           BUFFER("matrix_in", cb_matrix_in, sizeof(cb_matrix_in)),
           BUFFER("matrix_out", cb_matrix_out, sizeof(cb_matrix_out)),
           BUFFER("gamut_lut", cb_gamut_lut, sizeof(cb_gamut_lut)),
           FLOAT("shadows_weight", 2.2f), FLOAT("highlights_weight", 1.8f), FLOAT("midtones_weight", 1.4f),
           FLOAT("mask_grey_fulcrum", 0.1845f), FLOAT("hue_angle", 0.17f), FLOAT("chroma_global", 0.1f),
           FLOAT4("chroma", 0.05f, -0.1f, 0.15f, 0.0f), FLOAT("vibrance", 0.2f),
           FLOAT4("global_offset", 0.002f, -0.001f, 0.003f, 0.0f),
           FLOAT4("shadows", 1.05f, 0.97f, 1.02f, 0.0f), FLOAT4("highlights", 0.96f, 1.03f, 1.01f, 0.0f),
           FLOAT4("midtones", 1.03f, 0.98f, 1.05f, 0.0f), FLOAT("white_fulcrum", 1.2f),
           FLOAT("midtones_Y", 0.95f), FLOAT("grey_fulcrum", 0.1845f), FLOAT("contrast", 1.15f),
           FLOAT("brilliance_global", 0.05f), FLOAT4("brilliance", 0.02f, -0.03f, 0.04f, 0.0f),
           FLOAT("saturation_global", 0.12f), FLOAT4("saturation", -0.05f, 0.08f, 0.1f, 0.0f),
           INT("mask_display", mode == 2), INT("mask_type", 1), INT("checker_1", 8), INT("checker_2", 16),
           FLOAT4("checker_color_1", 0.2f, 0.2f, 0.2f, 1.0f), FLOAT4("checker_color_2", 0.8f, 0.8f, 0.8f, 1.0f),
           FLOAT("L_white", 1.07f), INT("saturation_formula", mode == 1 ? 0 : 1),
           BUFFER("hue_rotation_matrix", cb_hue_rotation, sizeof(cb_hue_rotation)));
  }

  // filmic rgb: the tone curves of every color science, and the highlight
  // reconstruction kernels. The gamut mapping needs an LMS matrix and its
  // inverse, or it clips everything.
  {
    static const float fm_in[12] = { 0.31f, 0.58f, 0.11f, 0.0f, 0.18f, 0.71f, 0.11f, 0.0f,
                                     0.04f, 0.16f, 0.80f, 0.0f };
    static float fm_out[12];
    {
      const float *m = fm_in;
      const float det = m[0] * (m[5] * m[10] - m[6] * m[9]) - m[1] * (m[4] * m[10] - m[6] * m[8])
                        + m[2] * (m[4] * m[9] - m[5] * m[8]);
      fm_out[0] = (m[5] * m[10] - m[6] * m[9]) / det;
      fm_out[1] = (m[2] * m[9] - m[1] * m[10]) / det;
      fm_out[2] = (m[1] * m[6] - m[2] * m[5]) / det;
      fm_out[4] = (m[6] * m[8] - m[4] * m[10]) / det;
      fm_out[5] = (m[0] * m[10] - m[2] * m[8]) / det;
      fm_out[6] = (m[2] * m[4] - m[0] * m[6]) / det;
      fm_out[8] = (m[4] * m[9] - m[5] * m[8]) / det;
      fm_out[9] = (m[1] * m[8] - m[0] * m[9]) / det;
      fm_out[10] = (m[0] * m[5] - m[1] * m[4]) / det;
    }
    // a different output space for the export gamut mapping
    static const float fm_export_in[12] = { 0.35f, 0.55f, 0.10f, 0.0f, 0.20f, 0.68f, 0.12f, 0.0f,
                                            0.03f, 0.13f, 0.84f, 0.0f };
    static float fm_export_out[12];
    {
      const float *m = fm_export_in;
      const float det = m[0] * (m[5] * m[10] - m[6] * m[9]) - m[1] * (m[4] * m[10] - m[6] * m[8])
                        + m[2] * (m[4] * m[9] - m[5] * m[8]);
      fm_export_out[0] = (m[5] * m[10] - m[6] * m[9]) / det;
      fm_export_out[1] = (m[2] * m[9] - m[1] * m[10]) / det;
      fm_export_out[2] = (m[1] * m[6] - m[2] * m[5]) / det;
      fm_export_out[4] = (m[6] * m[8] - m[4] * m[10]) / det;
      fm_export_out[5] = (m[0] * m[10] - m[2] * m[8]) / det;
      fm_export_out[6] = (m[2] * m[4] - m[0] * m[6]) / det;
      fm_export_out[8] = (m[4] * m[9] - m[5] * m[8]) / det;
      fm_export_out[9] = (m[1] * m[8] - m[0] * m[9]) / det;
      fm_export_out[10] = (m[0] * m[5] - m[1] * m[4]) / det;
    }

    // scene-referred input over 10 EV around middle grey, below display
    // white: at white, the chroma clipping of the v4 and v5 gamut mapping
    // jumps between 0 and FLT_MAX for the smallest change of luminance, and
    // OpenCL and Vulkan round differently there, as do the CPU and OpenCL
    static float *fm_image;
    fm_image = malloc(sizeof(float) * 4 * W * H);
    for(size_t k = 0; k < (size_t)4 * W * H; k++)
      fm_image[k] = (k % 4 == 3) ? 1.0f : 0.1845f * exp2f(9.0f * image_in[k] - 6.5f);

    // split: v1, v2, v3 and v4 with and without an output profile.
    // chroma: v1, v2 and v3 with three norms, v4 with the luminance and the
    // euclidean norms, and v5. Curve types cycle through hard, soft and safe.
    static const int fm_split_science[5] = { 0, 1, 2, 3, 3 };
    static const int fm_chroma_science[7] = { 0, 1, 2, 3, 3, 4, 4 };
    static const int fm_chroma_norm[7] = { 1, 3, 4, 2, 5, 1, 1 };
    static char fm_labels[12][48];
    for(int k = 0; k < 12; k++)
    {
      const int split = k < 5;
      const int science = split ? fm_split_science[k] : fm_chroma_science[k - 5];
      const int norm = split ? 0 : fm_chroma_norm[k - 5];
      const int export = (split && k == 4) || k == 11;
      const int type_1 = k % 3, type_2 = (k + 1) % 3;
      snprintf(fm_labels[k], sizeof(fm_labels[k]), "filmicrgb_%s v%d norm %d%s", split ? "split" : "chroma",
               science + 1, norm, export ? " export" : "");
      const float M1[4] = { 0.02f, -0.35f, 0.18f, 0.0f }, M2[4] = { 0.9f, 2.6f, 1.05f, 0.0f };
      const float M3[4] = { 1.6f, -1.9f, 0.0f, 0.0f }, M4[4] = { -1.2f, 0.45f, 0.0f, 0.0f };
      const float M5[4] = { 0.3f, 0.12f, 0.0f, 0.0f };
      if(split)
        TEST_TOL(fm_labels[k], "filmic", "filmicrgb_split", 1e-3,
                 IMG_IN("in", fm_image), IMG_OUT("out"), INT("width", W), INT("height", H),
                 FLOAT("dynamic_range", 12.0f), FLOAT("black_exposure", -7.5f), FLOAT("grey_value", 0.1845f),
                 BUFFER("profile_info", &profile_linear, sizeof(profile_info_t)),
                 LUT("lut", lut_profile, 256, 256 * 6), INT("use_work_profile", k == 1),
                 FLOAT("sigma_toe", 0.12f), FLOAT("sigma_shoulder", 0.21f), FLOAT("saturation", 0.15f),
                 FLOAT4("M1", M1[0], M1[1], M1[2], M1[3]), FLOAT4("M2", M2[0], M2[1], M2[2], M2[3]),
                 FLOAT4("M3", M3[0], M3[1], M3[2], M3[3]), FLOAT4("M4", M4[0], M4[1], M4[2], M4[3]),
                 FLOAT4("M5", M5[0], M5[1], M5[2], M5[3]),
                 FLOAT("latitude_min", 0.32f), FLOAT("latitude_max", 0.61f), FLOAT("output_power", 2.2f),
                 INT("color_science", science), INT("type_1", type_1), INT("type_2", type_2),
                 BUFFER("matrix_in", fm_in, sizeof(fm_in)), BUFFER("matrix_out", fm_out, sizeof(fm_out)),
                 FLOAT("display_black", 0.0002f), FLOAT("display_white", 1.0f), INT("use_output_profile", export),
                 BUFFER("export_matrix_in", fm_export_in, sizeof(fm_export_in)),
                 BUFFER("export_matrix_out", fm_export_out, sizeof(fm_export_out)))
      else
        TEST_TOL(fm_labels[k], "filmic", "filmicrgb_chroma", 1e-3,
                 IMG_IN("in", fm_image), IMG_OUT("out"), INT("width", W), INT("height", H),
                 FLOAT("dynamic_range", 12.0f), FLOAT("black_exposure", -7.5f), FLOAT("grey_value", 0.1845f),
                 BUFFER("profile_info", &profile_linear, sizeof(profile_info_t)),
                 LUT("lut", lut_profile, 256, 256 * 6), INT("use_work_profile", k == 8),
                 FLOAT("sigma_toe", 0.12f), FLOAT("sigma_shoulder", 0.21f), FLOAT("saturation", 0.15f),
                 FLOAT4("M1", M1[0], M1[1], M1[2], M1[3]), FLOAT4("M2", M2[0], M2[1], M2[2], M2[3]),
                 FLOAT4("M3", M3[0], M3[1], M3[2], M3[3]), FLOAT4("M4", M4[0], M4[1], M4[2], M4[3]),
                 FLOAT4("M5", M5[0], M5[1], M5[2], M5[3]),
                 FLOAT("latitude_min", 0.32f), FLOAT("latitude_max", 0.61f), FLOAT("output_power", 2.2f),
                 INT("variant", norm), INT("color_science", science), INT("type_1", type_1), INT("type_2", type_2),
                 BUFFER("matrix_in", fm_in, sizeof(fm_in)), BUFFER("matrix_out", fm_out, sizeof(fm_out)),
                 FLOAT("display_black", 0.0002f), FLOAT("display_white", 1.0f), INT("use_output_profile", export),
                 BUFFER("export_matrix_in", fm_export_in, sizeof(fm_export_in)),
                 BUFFER("export_matrix_out", fm_export_out, sizeof(fm_export_out)),
                 FLOAT("norm_min", 0.0011f), FLOAT("norm_max", 4.5f))
      // next to the asymptotes of the chroma clipping, the last bit of
      // luminance decides whether a channel limits the chroma at all
      t[n - 1].outliers = science >= 3 ? 8 : 0;
    }

    // the counter of clipped pixels races in OpenCL, so only the mask is compared
    static int fm_clipped;
    TEST("filmic_mask_clipped_pixels", "filmic", "filmic_mask_clipped_pixels",
         IMG_IN("in", fm_image), RAW_OUT("out"), INT("width", W), INT("height", H),
         FLOAT("normalize", 1.0f / 1.7f), FLOAT("feathering", 3.0f),
         BUFFER("clipped", &fm_clipped, sizeof(fm_clipped)));
    TEST("filmic_show_mask", "filmic", "filmic_show_mask",
         RAW_IN("in", rcd_dir), IMG_OUT("out"), INT("width", W), INT("height", H));
    static const char *fm_noise[3] = { "filmic_inpaint_noise uniform", "filmic_inpaint_noise gaussian",
                                       "filmic_inpaint_noise poissonian" };
    for(int k = 0; k < 3; k++)
      TEST(fm_noise[k], "filmic", "filmic_inpaint_noise",
           IMG_IN("in", image_in), RAW_IN("mask", rcd_dir), IMG_OUT("out"), INT("width", W), INT("height", H),
           FLOAT("noise_level", 0.2f), FLOAT("threshold", 3.0f), INT("noise_distribution", k));
    TEST("init_reconstruct", "filmic", "init_reconstruct",
         IMG_IN("in", image_in), RAW_IN("mask", rcd_dir), IMG_OUT("out"), INT("width", W), INT("height", H));
    static const char *fm_wr[4] = { "wavelets_reconstruct rgb", "wavelets_reconstruct rgb last",
                                    "wavelets_reconstruct ratios", "wavelets_reconstruct ratios last" };
    for(int k = 0; k < 4; k++)
      TEST(fm_wr[k], "filmic", "wavelets_reconstruct",
           IMG_IN("HF", image_lab), IMG_IN("LF", image_in), IMG_IN("texture", fm_image), RAW_IN("mask", rcd_dir),
           IMG_IN("reconstructed_read", image_in), IMG_OUT("reconstructed_write"), INT("width", W),
           INT("height", H), FLOAT("gamma", 0.3f), FLOAT("gamma_comp", 0.7f), FLOAT("beta", 0.6f),
           FLOAT("beta_comp", 0.4f), FLOAT("delta", 0.8f), INT("s", k % 2 ? 5 : 2), INT("scales", 6),
           INT("variant", k / 2));
    TEST("compute_ratios", "filmic", "compute_ratios",
         IMG_IN("in", image_in), RAW_OUT("norms"), IMG_OUT("ratios"), INT("variant", 1), INT("width", W),
         INT("height", H));
    TEST("restore_ratios", "filmic", "restore_ratios",
         IMG_IN("ratios", image_in), RAW_IN("norms", rcd_lpf), IMG_OUT("out"), INT("width", W), INT("height", H));

    TEST("blur_2D_Bspline_vertical", "bspline", "blur_2D_Bspline_vertical",
         IMG_IN("in", image_in), IMG_OUT("out"), INT("width", W), INT("height", H), INT("mult", 8));
    TEST("blur_2D_Bspline_horizontal", "bspline", "blur_2D_Bspline_horizontal",
         IMG_IN("in", image_in), IMG_OUT("out"), INT("width", W), INT("height", H), INT("mult", 4));
    TEST("wavelets_detail_level", "bspline", "wavelets_detail_level",
         IMG_IN("detail", image_in), IMG_IN("LF", image_lab), IMG_OUT("HF"), INT("width", W), INT("height", H));
  }

  // local laplacian on 1-channel pyramids: the fine level is the test image
  // size, the coarse one half of it, and the padding 16 pixels
  {
    const int CW = (W - 1) / 2 + 1, CH = (H - 1) / 2 + 1, SUPP = 16;
    static float *ll_fine[6], *ll_coarse[6], *ll_out1;
    for(int k = 0; k < 6; k++)
    {
      ll_fine[k] = malloc(sizeof(float) * W * H);
      ll_coarse[k] = malloc(sizeof(float) * CW * CH);
      for(size_t i = 0; i < (size_t)W * H; i++) ll_fine[k][i] = rcd_cfa[i] * (0.8f + 0.05f * k) + 0.02f * frand();
      for(size_t i = 0; i < (size_t)CW * CH; i++) ll_coarse[k][i] = rcd_cfa[i] * (0.8f + 0.05f * k);
    }
    ll_out1 = malloc(sizeof(float) * CW * CH);
    for(size_t i = 0; i < (size_t)CW * CH; i++) ll_out1[i] = frand();
#define LL_IN(n, px, ww, hh) { .kind = ARG_IMAGE_IN, .name = n, .w = ww, .h = hh, .channels = 1, .pixels = px }
#define LL_OUT(n, ww, hh) { .kind = ARG_IMAGE_OUT, .name = n, .w = ww, .h = hh, .channels = 1 }
    TEST_GRID("pad_input", "locallaplacian", "pad_input", 0.0, W + 2 * SUPP, H + 2 * SUPP,
              IMG_IN("input", image_lab), LL_OUT("padded", W + 2 * SUPP, H + 2 * SUPP), INT("wd", W), INT("ht", H),
              INT("max_supp", SUPP), INT("wd2", W + 2 * SUPP), INT("ht2", H + 2 * SUPP));
    TEST_GRID("gauss_reduce", "locallaplacian", "gauss_reduce", 0.0, CW, CH,
              RAW_IN("input", rcd_cfa), LL_OUT("coarse", CW, CH), INT("wd", CW), INT("ht", CH));
    TEST("laplacian_assemble", "locallaplacian", "laplacian_assemble",
         RAW_IN("input", rcd_dir), LL_IN("output1", ll_out1, CW, CH), RAW_OUT("output0"),
         RAW_IN("buf_g0_l0", ll_fine[0]), LL_IN("buf_g0_l1", ll_coarse[0], CW, CH),
         RAW_IN("buf_g1_l0", ll_fine[1]), LL_IN("buf_g1_l1", ll_coarse[1], CW, CH),
         RAW_IN("buf_g2_l0", ll_fine[2]), LL_IN("buf_g2_l1", ll_coarse[2], CW, CH),
         RAW_IN("buf_g3_l0", ll_fine[3]), LL_IN("buf_g3_l1", ll_coarse[3], CW, CH),
         RAW_IN("buf_g4_l0", ll_fine[4]), LL_IN("buf_g4_l1", ll_coarse[4], CW, CH),
         RAW_IN("buf_g5_l0", ll_fine[5]), LL_IN("buf_g5_l1", ll_coarse[5], CW, CH), INT("pw", W), INT("ph", H));
    TEST("process_curve", "locallaplacian", "process_curve",
         RAW_IN("input", rcd_dir), RAW_OUT("output"), FLOAT("g", 0.41f), FLOAT("sigma", 0.5f),
         FLOAT("shadows", 0.7f), FLOAT("highlights", 0.4f), FLOAT("clarity", 0.25f), INT("wd", W), INT("ht", H));
    TEST_GRID("write_back", "locallaplacian", "write_back", 0.0, W - 2 * SUPP, H - 2 * SUPP,
              IMG_IN("input", image_lab), RAW_IN("processed", rcd_dir),
              { .kind = ARG_IMAGE_OUT, .name = "output", .w = W - 2 * SUPP, .h = H - 2 * SUPP, .channels = 4 },
              INT("max_supp", SUPP), INT("wd", W - 2 * SUPP), INT("ht", H - 2 * SUPP));
#undef LL_IN
#undef LL_OUT
  }

  // the recursive gaussian of the mask blur, with the coefficients
  // dt_gaussian_init() computes for sigma 6 and order 0
  {
    const float alpha = 1.695f / 6.0f;
    const float ema = expf(-alpha), ema2 = expf(-2.0f * alpha);
    const float b1 = -2.0f * ema, b2 = ema2;
    const float k = (1.0f - ema) * (1.0f - ema) / (1.0f + (2.0f * alpha * ema) - ema2);
    const float a0 = k, a1 = k * (alpha - 1.0f) * ema, a2 = k * (alpha + 1.0f) * ema, a3 = -k * ema2;
    const float coefp = (a0 + a1) / (1.0f + b1 + b2), coefn = (a2 + a3) / (1.0f + b1 + b2);
    TEST_LINE("gaussian_column_1c", "gaussian", "gaussian_column_1c", 1e-6, W,
              BUFFER("in", image_raw, plane), BUFFER_OUT("out", NULL, plane), INT("width", W), INT("height", H),
              FLOAT("a0", a0), FLOAT("a1", a1), FLOAT("a2", a2), FLOAT("a3", a3), FLOAT("b1", b1), FLOAT("b2", b2),
              FLOAT("coefp", coefp), FLOAT("coefn", coefn), FLOAT("Labmax", 1.0f), FLOAT("Labmin", 0.0f));
    TEST("gaussian_transpose_1c", "gaussian", "gaussian_transpose_1c",
         BUFFER("in", image_raw, plane), BUFFER_OUT("out", NULL, plane), INT("width", W), INT("height", H),
         INT("blocksize", 16), LOCAL("buffer", sizeof(float) * 16 * 17));
  }

  // the guided filter of mask feathering: an RGB guide, read from row 2 on as
  // when the OpenCL code tiles, and a mask
  {
    enum { GH = H - 2 };
    static float gf_var[6][W * GH], gf_cov[3][W * GH];
    for(int k = 0; k < W * GH; k++)
    {
      // positive definite covariance matrices, and a few singular ones
      const int singular = k % 97 == 0;
      for(int c = 0; c < 3; c++)
      {
        gf_var[c == 0 ? 0 : c == 1 ? 3 : 5][k] = singular ? 0.0f : 0.05f + rcd_dir[k + c * 7];
        gf_cov[c][k] = 0.3f * rcd_lpf[k + c * 11] - 0.6f;
      }
      gf_var[1][k] = singular ? 0.0f : 0.02f * rcd_dir[k + 3];
      gf_var[2][k] = singular ? 0.0f : -0.02f * rcd_dir[k + 5];
      gf_var[4][k] = singular ? 0.0f : 0.01f * rcd_dir[k + 9];
    }
#define GF_IN(n, px) { .kind = ARG_IMAGE_IN, .name = n, .w = W, .h = GH, .channels = 1, .pixels = px }
#define GF_OUT(n) { .kind = ARG_IMAGE_OUT, .name = n, .w = W, .h = GH, .channels = 1 }
    TEST_GRID("guided_filter_split_rgb_image", "guided_filter", "guided_filter_split_rgb_image", 0.0, W, GH,
              INT("width", W), INT("height", GH), INT("first", 2), IMG_IN("guide", image_in),
              GF_OUT("out_r"), GF_OUT("out_g"), GF_OUT("out_b"), FLOAT("guide_weight", 10.0f));
    // a narrow window, and one wider than half the image
    // a work item per chunk of a row or column in Slang, per row or column in OpenCL
#define GF_BOX(lbl, kern, along_x, wnd)                                                      \
    {                                                                                    \
      const int chunk = (2 * (wnd) + 1) < 16 ? 16 : (2 * (wnd) + 1) > 64 ? 64 : 2 * (wnd) + 1; \
      TEST_GRID(lbl, "guided_filter", kern, 1e-6, along_x ? (W + chunk - 1) / chunk : W,   \
                along_x ? GH : (GH + chunk - 1) / chunk, INT("width", W), INT("height", GH),  \
                GF_IN("in", rcd_dir), GF_OUT("out"), INT("w", wnd), SLANG_INT("chunk", chunk)); \
      t[n - 1].clgw = along_x ? GH : W;                                                    \
      t[n - 1].clgh = 1;                                                                   \
      t[n - 1].clly = 1;                                                                   \
    }
    // a narrow window, and one wider than half the image
    GF_BOX("guided_filter_box_mean_x w7", "guided_filter_box_mean_x", 1, 7);
    GF_BOX("guided_filter_box_mean_x w300", "guided_filter_box_mean_x", 1, 300);
    GF_BOX("guided_filter_box_mean_y w7", "guided_filter_box_mean_y", 0, 7);
    GF_BOX("guided_filter_box_mean_y w150", "guided_filter_box_mean_y", 0, 150);
#undef GF_BOX
    // windows up to 2 * 32 + 1 wide, one work item per pixel in Slang
    for(int k = 0; k < 2; k++)
    {
      static const char *gf_labels[2] = { "guided_filter_box_mean_x_tile w7", "guided_filter_box_mean_x_tile w32" };
      TEST_GRID(gf_labels[k], "guided_filter", "guided_filter_box_mean_x", 1e-6, W, GH, INT("width", W),
                INT("height", GH), GF_IN("in", rcd_dir), GF_OUT("out"), INT("w", k ? 32 : 7));
      t[n - 1].slang_kernel = "guided_filter_box_mean_x_tile";
      t[n - 1].clgw = GH;
      t[n - 1].clgh = 1;
      t[n - 1].clly = 1;
    }
    TEST_GRID("guided_filter_covariances", "guided_filter", "guided_filter_covariances", 0.0, W, GH,
              INT("width", W), INT("height", GH), INT("first", 2), IMG_IN("guide", image_in), GF_IN("img", rcd_dir),
              GF_OUT("cov_imgg_img_r"), GF_OUT("cov_imgg_img_g"), GF_OUT("cov_imgg_img_b"),
              FLOAT("guide_weight", 10.0f));
    TEST_GRID("guided_filter_variances", "guided_filter", "guided_filter_variances", 0.0, W, GH,
              INT("width", W), INT("height", GH), INT("first", 2), IMG_IN("guide", image_in),
              GF_OUT("var_imgg_rr"), GF_OUT("var_imgg_rg"), GF_OUT("var_imgg_rb"), GF_OUT("var_imgg_gg"),
              GF_OUT("var_imgg_gb"), GF_OUT("var_imgg_bb"), FLOAT("guide_weight", 10.0f));
    TEST_GRID("guided_filter_update_covariance", "guided_filter", "guided_filter_update_covariance", 0.0, W, GH,
              INT("width", W), INT("height", GH), GF_IN("in", rcd_lpf), GF_OUT("out"), GF_IN("a", rcd_dir),
              GF_IN("b", rcd_cfa), FLOAT("eps", 0.25f));
    TEST_GRID("guided_filter_solve", "guided_filter", "guided_filter_solve", 1e-5, W, GH,
              INT("width", W), INT("height", GH), GF_IN("img_mean", rcd_cfa),
              GF_IN("imgg_mean_r", rcd_rgb[0]), GF_IN("imgg_mean_g", rcd_rgb[1]), GF_IN("imgg_mean_b", rcd_rgb[2]),
              GF_IN("cov_imgg_img_r", gf_cov[0]), GF_IN("cov_imgg_img_g", gf_cov[1]),
              GF_IN("cov_imgg_img_b", gf_cov[2]),
              GF_IN("var_imgg_rr", gf_var[0]), GF_IN("var_imgg_rg", gf_var[1]), GF_IN("var_imgg_rb", gf_var[2]),
              GF_IN("var_imgg_gg", gf_var[3]), GF_IN("var_imgg_gb", gf_var[4]), GF_IN("var_imgg_bb", gf_var[5]),
              GF_OUT("a_r"), GF_OUT("a_g"), GF_OUT("a_b"), GF_OUT("b"));
    TEST_GRID("guided_filter_generate_result", "guided_filter", "guided_filter_generate_result", 1e-6, W, GH,
              INT("width", W), INT("height", GH), INT("first", 2), IMG_IN("guide", image_in),
              GF_IN("a_r", gf_cov[0]), GF_IN("a_g", gf_cov[1]), GF_IN("a_b", gf_cov[2]), GF_IN("b", rcd_lpf),
              GF_OUT("res"), FLOAT("guide_weight", 10.0f), FLOAT("minval", 0.0f), FLOAT("maxval", 1.0f));
#undef GF_IN
#undef GF_OUT
  }

  // one diffusion step of each isotropy type, on plausible wavelet data: a
  // positive low frequency layer and signed details. The mask is an 8 bit
  // image in OpenCL and a float one here, so only the unmasked path compares
  {
    static float df_hf[4 * W * H];
    for(int k = 0; k < 4 * W * H; k++) df_hf[k] = 0.2f * (image_in[(k + 4 * 37) % (4 * W * H)] - 0.5f);
    static const int df_types[3][4] = { { 0, 0, 0, 0 }, { 1, 1, 1, 1 }, { 2, 1, 0, 2 } };
    static const char *df_labels[3] = { "diffuse_pde isotrope", "diffuse_pde isophote", "diffuse_pde mixed" };
    for(int d = 0; d < 3; d++)
      TEST_TOL(df_labels[d], "diffuse", "diffuse_pde", 1e-4,
               IMG_IN("HF", df_hf), IMG_IN("LF", image_in), RAW_IN("mask", rcd_dir), INT("has_mask", 0),
               IMG_OUT("output"), INT("width", W), INT("height", H),
               FLOAT4("anisotropy", 2.5f, 1.0f, 0.5f, 4.0f),
               INT4("isotropy_type", df_types[d][0], df_types[d][1], df_types[d][2], df_types[d][3]),
               FLOAT("regularization", 0.7f), FLOAT("variance_threshold", 0.01f),
               FLOAT("current_radius_square", 4.0f + 3.0f * d), INT("mult", 1 << (d + 1)),
               FLOAT4("ABCD", -0.06f, 0.05f, 0.04f, -0.03f), FLOAT("strength", 1.1f));
  }

  // agx on scene-referred data from about -7 to +5 EV around middle gray. One
  // set of params takes the sigmoid toe and shoulder with the look and hue
  // restoration, the other the fallback curves and the hdr darkening
  {
    // tone_mapping_params_t in agx.c, with its hdr_params_t at the end
    typedef struct
    {
      float min_ev, max_ev, range_in_ev, curve_gamma, pivot_x, pivot_y, target_black, toe_power,
          toe_transition_x, toe_transition_y, toe_scale;
      int need_convex_toe;
      float toe_fallback_coefficient, toe_fallback_power, slope, intercept, target_white, shoulder_power,
          shoulder_transition_x, shoulder_transition_y, shoulder_scale;
      int need_concave_shoulder;
      float shoulder_fallback_coefficient, shoulder_fallback_power, look_offset, look_slope, look_power,
          look_saturation, look_original_hue_mix_ratio;
      int look_tuned, restore_hue;
      int hdr_enabled;
      float hdr_headroom, hdr_purity, hdr_log_min, hdr_pivot_x, hdr_pivot_y, hdr_toe_scale, hdr_shoulder_scale;
    } agx_params_t;
    static float ag_image[4 * W * H];
    for(int k = 0; k < 4 * W * H; k++)
      ag_image[k] = (k % 4 == 3) ? image_in[k] : 0.18f * exp2f(12.0f * image_in[k] - 7.0f);
    static const float to_xyz[16] = { 0.41f, 0.36f, 0.18f, 0.0f, 0.21f, 0.72f, 0.07f, 0.0f, 0.02f, 0.12f, 0.95f, 0.0f };
    static agx_params_t ag_params[2];
    ag_params[0] = (agx_params_t){ .min_ev = -10.0f, .max_ev = 6.5f, .range_in_ev = 16.5f, .curve_gamma = 2.2f,
                                   .pivot_x = 10.0f / 16.5f, .pivot_y = 0.459f, .target_black = 0.0f,
                                   .toe_power = 1.5f, .toe_transition_x = 0.4f, .toe_transition_y = 0.047f,
                                   .toe_scale = -0.1f, .need_convex_toe = 0, .toe_fallback_coefficient = 0.3f,
                                   .toe_fallback_power = 1.6f, .slope = 2.0f, .intercept = -0.753f,
                                   .target_white = 1.0f, .shoulder_power = 1.5f, .shoulder_transition_x = 0.75f,
                                   .shoulder_transition_y = 0.747f, .shoulder_scale = 0.3f,
                                   .need_concave_shoulder = 0, .shoulder_fallback_coefficient = 0.5f,
                                   .shoulder_fallback_power = 1.4f, .look_offset = 0.02f, .look_slope = 1.1f,
                                   .look_power = 1.2f, .look_saturation = 1.3f, .look_original_hue_mix_ratio = 0.4f,
                                   .look_tuned = 1, .restore_hue = 1 };
    ag_params[1] = ag_params[0];
    ag_params[1].need_convex_toe = 1;
    ag_params[1].need_concave_shoulder = 1;
    ag_params[1].look_tuned = 0;
    ag_params[1].restore_hue = 0;
    ag_params[1].hdr_enabled = 1;
    ag_params[1].hdr_headroom = 2.0f;
    ag_params[1].hdr_purity = 0.5f;
    ag_params[1].hdr_log_min = -12.0f;
    ag_params[1].hdr_pivot_x = 0.5f;
    ag_params[1].hdr_pivot_y = 0.5f;
    ag_params[1].hdr_toe_scale = -0.4f;
    ag_params[1].hdr_shoulder_scale = 0.5f;
    static const char *ag_labels[2] = { "agx sigmoid look hue", "agx fallback hdr" };
    for(int d = 0; d < 2; d++)
      TEST(ag_labels[d], "agx", "kernel_agx",
           IMG_IN("input", ag_image), IMG_OUT("output"), INT("width", W), INT("height", H),
           STRUCT("params", ag_params[d]),
           BUFFER("pipe_to_base", to_base, sizeof(to_base)), BUFFER("base_to_rendering", to_rendering, sizeof(to_rendering)),
           BUFFER("rendering_to_pipe", to_pipe, sizeof(to_pipe)), BUFFER("rendering_to_xyz", to_xyz, sizeof(to_xyz)),
           INT("base_working_same_profile", d));
  }

  // sharpen on the L channel of Lab, with a radius of 5 as at 100% zoom. The
  // OpenCL blurs run in work groups of 16 along the blur, with a local buffer
  // for the group and its wings, as process_cl() sets them up
  {
    enum { SH_RAD = 5, SH_WD = 2 * SH_RAD + 1, SH_BS = 16 };
    static float sh_m[SH_WD];
    float weight = 0.0f;
    for(int l = -SH_RAD; l <= SH_RAD; l++)
      weight += sh_m[l + SH_RAD] = expf(-l * l / (2.f * 0.64f * SH_RAD * SH_RAD));
    for(int l = 0; l < SH_WD; l++) sh_m[l] /= weight;
    static float sh_blurred[4 * W * H];
    for(int k = 0; k < 4 * W * H; k++) sh_blurred[k] = image_lab[k] + 4.0f * (image_in[k] - 0.5f);

    TEST("sharpen_hblur", "sharpen", "sharpen_hblur",
         IMG_IN("in", image_lab), IMG_OUT("out"), BUFFER("m", sh_m, sizeof(sh_m)), INT("rad", SH_RAD),
         INT("width", W), INT("height", H), INT("blocksize", SH_BS),
         LOCAL("buffer", (SH_BS + 2 * SH_RAD) * sizeof(float)));
    t[n - 1].clgw = (W + SH_BS - 1) / SH_BS * SH_BS;
    t[n - 1].clgh = H;
    t[n - 1].cllx = SH_BS;
    t[n - 1].clly = 1;
    TEST("sharpen_vblur", "sharpen", "sharpen_vblur",
         IMG_IN("in", image_lab), IMG_OUT("out"), BUFFER("m", sh_m, sizeof(sh_m)), INT("rad", SH_RAD),
         INT("width", W), INT("height", H), INT("blocksize", SH_BS),
         LOCAL("buffer", (SH_BS + 2 * SH_RAD) * sizeof(float)));
    t[n - 1].clgw = W;
    t[n - 1].clgh = (H + SH_BS - 1) / SH_BS * SH_BS;
    t[n - 1].cllx = 1;
    t[n - 1].clly = SH_BS;
    TEST("sharpen_mix", "sharpen", "sharpen_mix",
         IMG_IN("in_a", image_lab), IMG_IN("in_b", sh_blurred), IMG_OUT("out"), INT("width", W),
         INT("height", H), FLOAT("sharpen", 0.5f), FLOAT("thrs", 0.5f), INT("rad", SH_RAD));
  }

  // the recursive gaussian on 2 and 4 channel buffers, as the guided filters
  // of color equalizer run it, with the coefficients of the 1 channel case
  // above, and the bilinear resampling of their working buffers
  {
    const float alpha = 1.695f / 6.0f;
    const float ema = expf(-alpha), ema2 = expf(-2.0f * alpha);
    const float b1 = -2.0f * ema, b2 = ema2;
    const float k = (1.0f - ema) * (1.0f - ema) / (1.0f + (2.0f * alpha * ema) - ema2);
    const float a0 = k, a1 = k * (alpha - 1.0f) * ema, a2 = k * (alpha + 1.0f) * ema, a3 = -k * ema2;
    const float coefp = (a0 + a1) / (1.0f + b1 + b2), coefn = (a2 + a3) / (1.0f + b1 + b2);
    const size_t plane2 = 2 * sizeof(float) * W * H, plane4 = 4 * sizeof(float) * W * H;
    TEST_LINE("gaussian_column_2c", "gaussian", "gaussian_column_2c", 1e-6, W,
              BUFFER("in", image_in, plane2), BUFFER_OUT("out", NULL, plane2), INT("width", W), INT("height", H),
              FLOAT("a0", a0), FLOAT("a1", a1), FLOAT("a2", a2), FLOAT("a3", a3), FLOAT("b1", b1), FLOAT("b2", b2),
              FLOAT("coefp", coefp), FLOAT("coefn", coefn), FLOAT2("Labmax", 1.0f, 1.0f),
              FLOAT2("Labmin", 0.0f, 0.0f));
    TEST_LINE("gaussian_column_4c", "gaussian", "gaussian_column_4c", 1e-6, W,
              BUFFER("in", image_in, plane4), BUFFER_OUT("out", NULL, plane4), INT("width", W), INT("height", H),
              FLOAT("a0", a0), FLOAT("a1", a1), FLOAT("a2", a2), FLOAT("a3", a3), FLOAT("b1", b1), FLOAT("b2", b2),
              FLOAT("coefp", coefp), FLOAT("coefn", coefn), FLOAT4("Labmax", 1.0f, 1.0f, 1.0f, 1.0f),
              FLOAT4("Labmin", 0.0f, 0.0f, 0.0f, 0.0f));
    TEST("gaussian_transpose_2c", "gaussian", "gaussian_transpose_2c",
         BUFFER("in", image_in, plane2), BUFFER_OUT("out", NULL, plane2), INT("width", W), INT("height", H),
         INT("blocksize", 16), LOCAL("buffer", 2 * sizeof(float) * 16 * 17));
    TEST("gaussian_transpose_4c", "gaussian", "gaussian_transpose_4c",
         BUFFER("in", image_in, plane4), BUFFER_OUT("out", NULL, plane4), INT("width", W), INT("height", H),
         INT("blocksize", 16), LOCAL("buffer", 4 * sizeof(float) * 16 * 17));

    enum { BW = W / 3, BH = H / 3 };
    static const int bl_ch[3] = { 1, 2, 4 };
    static const char *bl_kernels[3] = { "bilinear1", "bilinear2", "bilinear4" };
    static char bl_labels[3][2][32];
    for(int c = 0; c < 3; c++)
    {
      const size_t px = bl_ch[c] * sizeof(float);
      snprintf(bl_labels[c][0], sizeof(bl_labels[c][0]), "%s down", bl_kernels[c]);
      TEST_GRID(bl_labels[c][0], "bilinear", bl_kernels[c], 1e-6, BW, BH,
                BUFFER("in", image_in, px * W * H), INT("width_in", W), INT("height_in", H),
                BUFFER_OUT("out", NULL, px * BW * BH), INT("width_out", BW), INT("height_out", BH));
      snprintf(bl_labels[c][1], sizeof(bl_labels[c][1]), "%s up", bl_kernels[c]);
      TEST_GRID(bl_labels[c][1], "bilinear", bl_kernels[c], 1e-6, W, H,
                BUFFER("in", image_in, px * BW * BH), INT("width_in", BW), INT("height_in", BH),
                BUFFER_OUT("out", NULL, px * W * H), INT("width_out", W), INT("height_out", H));
    }
  }

  // color equalizer, on plausible working data: chromaticities, saturations,
  // positive definite covariances with a few singular ones, smooth LUTs over
  // hue, and the HSB pixels write_output() maps back to RGB
  {
    enum { CE_SAT = 4096, CE_LUT = 512 };
    static float ce_uv[2 * W * H], ce_sat[W * H], ce_lum[W * H], ce_cov[4 * W * H], ce_corr4[4 * W * H];
    static float ce_a[4 * W * H], ce_b[2 * W * H], ce_corrections[2 * W * H], ce_bcorr[W * H];
    static float ce_hsb[4 * W * H], ce_weights[2 * CE_SAT + 1];
    static float ce_gamut[CE_LUT], ce_lut_hue[CE_LUT], ce_lut_sat[CE_LUT], ce_lut_bright[CE_LUT];
    for(int k = 0; k < W * H; k++)
    {
      const float *p = image_in + 4 * k;
      ce_uv[2 * k] = 0.6f * (p[0] - 0.5f);
      ce_uv[2 * k + 1] = 0.6f * (p[1] - 0.5f);
      ce_sat[k] = fminf(fmaxf(p[2], 0.0f), 1.0f);
      ce_lum[k] = 0.05f + 0.9f * fminf(fmaxf(p[3], 0.0f), 1.0f);
      const int singular = k % 97 == 0;
      ce_cov[4 * k] = singular ? 0.0f : 0.02f + 0.1f * fabsf(p[0]);
      ce_cov[4 * k + 1] = ce_cov[4 * k + 2] = singular ? 0.0f : 0.01f * (p[1] - 0.5f);
      ce_cov[4 * k + 3] = singular ? 0.0f : 0.02f + 0.1f * fabsf(p[2]);
      for(int c = 0; c < 4; c++)
      {
        ce_corr4[4 * k + c] = 0.02f * (p[(c + 1) % 4] - 0.5f);
        ce_a[4 * k + c] = 0.5f * (p[(c + 2) % 4] - 0.5f);
      }
      ce_b[2 * k] = 0.1f * (p[3] - 0.5f);
      ce_b[2 * k + 1] = 0.05f * (p[0] - 0.5f);
      ce_corrections[2 * k] = 0.1f * (p[1] - 0.5f);
      ce_corrections[2 * k + 1] = 1.0f + 0.4f * (p[2] - 0.5f);
      ce_bcorr[k] = 0.1f * (p[3] - 0.5f);
      ce_hsb[4 * k] = 6.2f * (p[0] - 0.5f);
      ce_hsb[4 * k + 1] = 0.3f * fminf(fmaxf(p[1], 0.0f), 1.0f);
      ce_hsb[4 * k + 2] = 0.1f + 0.7f * fminf(fmaxf(p[2], 0.0f), 1.0f);
      ce_hsb[4 * k + 3] = p[3];
    }
    for(int i = 0; i <= 2 * CE_SAT; i++)
    {
      const float x = (float)i / (2 * CE_SAT);
      ce_weights[i] = x * x * (3.0f - 2.0f * x);
    }
    for(int i = 0; i < CE_LUT; i++)
    {
      const float h = 6.2831853f * i / CE_LUT;
      ce_gamut[i] = 0.25f + 0.1f * sinf(h);
      ce_lut_hue[i] = 0.08f * sinf(2.0f * h);
      ce_lut_sat[i] = 1.0f + 0.2f * cosf(h);
      ce_lut_bright[i] = 1.0f + 0.1f * sinf(3.0f * h);
    }
    // linear Rec.709 to XYZ D65 and back, as rows of float4
    static const float ce_to_xyz[12] = { 0.4124f, 0.3576f, 0.1805f, 0.0f, 0.2126f, 0.7152f, 0.0722f, 0.0f,
                                         0.0193f, 0.1192f, 0.9505f, 0.0f };
    static const float ce_from_xyz[12] = { 3.2406f, -1.5372f, -0.4986f, 0.0f, -0.9689f, 1.8758f, 0.0415f, 0.0f,
                                           0.0557f, -0.2040f, 1.0570f, 0.0f };
    const size_t s1 = sizeof(float) * W * H, s2 = 2 * s1, s4 = 4 * s1;
#define CE_BUF(n, p, sz) BUFFER(n, p, sz)
    TEST("init_covariance", "colorequal", "init_covariance",
         BUFFER_OUT("covariance", NULL, s4), CE_BUF("uv", ce_uv, s2), INT("width", W), INT("height", H));
    TEST("finish_covariance", "colorequal", "finish_covariance",
         BUFFER_OUT("covariance", ce_cov, s4), CE_BUF("uv", ce_uv, s2), INT("width", W), INT("height", H));
    TEST("prepare_prefilter", "colorequal", "prepare_prefilter",
         CE_BUF("uv", ce_uv, s2), CE_BUF("covariance", ce_cov, s4), BUFFER_OUT("a", NULL, s4),
         BUFFER_OUT("b", NULL, s2), FLOAT("eps", 0.01f), INT("width", W), INT("height", H));
    TEST("apply_prefilter", "colorequal", "apply_prefilter",
         BUFFER_OUT("uv", ce_uv, s2), CE_BUF("saturation", ce_sat, s1), CE_BUF("a", ce_a, s4),
         CE_BUF("b", ce_b, s2), CE_BUF("weights", ce_weights, sizeof(ce_weights)), FLOAT("sat_shift", 0.1f),
         INT("width", W), INT("height", H));
    TEST("prepare_correlations", "colorequal", "prepare_correlations",
         CE_BUF("corrections", ce_corrections, s2), CE_BUF("b_corrections", ce_bcorr, s1), CE_BUF("uv", ce_uv, s2),
         BUFFER_OUT("correlations", NULL, s4), INT("width", W), INT("height", H));
    TEST("finish_correlations", "colorequal", "finish_correlations",
         CE_BUF("corrections", ce_corrections, s2), CE_BUF("b_corrections", ce_bcorr, s1), CE_BUF("uv", ce_uv, s2),
         BUFFER_OUT("correlations", ce_corr4, s4), BUFFER_OUT("covariance", ce_cov, s4),
         INT("width", W), INT("height", H));
    TEST("final_guide", "colorequal", "final_guide",
         CE_BUF("covariance", ce_cov, s4), CE_BUF("correlations", ce_corr4, s4),
         CE_BUF("corrections", ce_corrections, s2), CE_BUF("b_corrections", ce_bcorr, s1), CE_BUF("uv", ce_uv, s2),
         BUFFER_OUT("a", NULL, s4), BUFFER_OUT("b", NULL, s2), FLOAT("eps", 0.01f), INT("width", W), INT("height", H));
    TEST("apply_guided", "colorequal", "apply_guided",
         CE_BUF("uv", ce_uv, s2), CE_BUF("saturation", ce_sat, s1), CE_BUF("scharr", ce_lum, s1),
         CE_BUF("a", ce_a, s4), CE_BUF("b", ce_b, s2), BUFFER_OUT("corrections", ce_corrections, s2),
         BUFFER_OUT("b_corrections", ce_bcorr, s1), CE_BUF("weights", ce_weights, sizeof(ce_weights)),
         FLOAT("sat_shift", 0.1f), FLOAT("bright_shift", -0.1f), INT("width", W), INT("height", H));
    TEST("sample_input", "colorequal", "sample_input",
         IMG_IN("dev_in", image_in), BUFFER_OUT("saturation", NULL, s1), BUFFER_OUT("lum", NULL, s1),
         BUFFER_OUT("uv", NULL, s2), BUFFER_OUT("pix_out", NULL, s4), CE_BUF("mat", ce_to_xyz, sizeof(ce_to_xyz)),
         INT("width", W), INT("height", H));
    // the gamut mapping and the conversion back go through pow() and atan2()
    TEST_TOL("write_output", "colorequal", "write_output", 1e-3,
             IMG_OUT("dev_out"), BUFFER_OUT("pix_out", ce_hsb, s4), CE_BUF("corrections", ce_corrections, s2),
             CE_BUF("b_corrections", ce_bcorr, s1), CE_BUF("mat", ce_from_xyz, sizeof(ce_from_xyz)),
             CE_BUF("gamut_LUT", ce_gamut, sizeof(ce_gamut)), FLOAT("white", 1.2f), INT("width", W),
             INT("height", H));
    static const char *ce_visual[5] = { "write_visual hue", "write_visual saturation", "write_visual brightness",
                                        "write_visual saturation grad", "write_visual brightness grad" };
    static const int ce_modes[5] = { 0, 1, 2, 5, 6 };
    for(int m = 0; m < 5; m++)
      TEST(ce_visual[m], "colorequal", "write_visual",
           IMG_OUT("dev_out"), CE_BUF("pixout", ce_hsb, s4), CE_BUF("corrections", ce_corrections, s2),
           CE_BUF("b_corrections", ce_bcorr, s1), CE_BUF("saturation", ce_sat, s1), CE_BUF("scharr", ce_lum, s1),
           CE_BUF("weights", ce_weights, sizeof(ce_weights)), FLOAT("bright_shift", -0.1f),
           FLOAT("sat_shift", 0.1f), FLOAT("white", 1.2f), INT("mode", ce_modes[m]), INT("width", W),
           INT("height", H));
    TEST_LINE("draw_weight", "colorequal", "draw_weight", 0.0, W,
              IMG_OUT("dev_out"), CE_BUF("weights", ce_weights, sizeof(ce_weights)), FLOAT("bright_shift", -0.1f),
              FLOAT("sat_shift", 0.1f), INT("mode", 6), INT("width", W), INT("height", H));
    for(int guiding = 0; guiding < 2; guiding++)
      TEST_TOL(guiding ? "process_data guided" : "process_data", "colorequal", "process_data", 1e-3,
               CE_BUF("uv", ce_uv, s2), BUFFER_OUT("Lscharr", ce_lum, s1), CE_BUF("saturation", ce_sat, s1),
               BUFFER_OUT("corrections", NULL, s2), BUFFER_OUT("b_corrections", NULL, s1),
               BUFFER_OUT("pixout", NULL, s4), CE_BUF("LUT_saturation", ce_lut_sat, sizeof(ce_lut_sat)),
               CE_BUF("LUT_hue", ce_lut_hue, sizeof(ce_lut_hue)),
               CE_BUF("LUT_brightness", ce_lut_bright, sizeof(ce_lut_bright)), FLOAT("white", 1.2f),
               FLOAT("gradient_amp", 4.0f), INT("guiding", guiding), INT("width", W), INT("height", H));
#undef CE_BUF
  }

  // saturation curve, with both formulas, and with and without the guided filter masks
  {
    static const float sc_matrix_in[12] = { 0.31f, 0.58f, 0.11f, 0.0f, 0.18f, 0.71f, 0.11f, 0.0f,
                                            0.04f, 0.16f, 0.80f, 0.0f };
    static const float sc_matrix_out[12] = { 1.71f, -0.36f, -0.25f, 0.0f, -0.67f, 1.62f, 0.04f, 0.0f,
                                             0.02f, -0.04f, 0.91f, 0.0f };
    static float sc_gamut_lut[512], sc_sat_lut[256], sc_bri_lut[256];
    for(int k = 0; k < 512; k++) sc_gamut_lut[k] = 0.6f + 0.3f * sinf(k * 0.0491f) + 0.05f * cosf(k * 0.31f);
    for(int k = 0; k < 256; k++)
    {
      sc_sat_lut[k] = 0.5f + 0.35f * sinf(k * 0.021f);
      sc_bri_lut[k] = 0.5f - 0.2f * sinf(k * 0.017f);
    }
    // masks in [0, 1], a filtered mask near the raw one, a noise confidence
    // and a perceptual guide
    static float *sc_mask, *sc_filtered, *sc_conf, *sc_guide;
    sc_mask = malloc(sizeof(float) * W * H);
    sc_filtered = malloc(sizeof(float) * W * H);
    sc_conf = malloc(sizeof(float) * W * H);
    sc_guide = malloc(sizeof(float) * 4 * W * H);
    for(size_t k = 0; k < (size_t)W * H; k++)
    {
      sc_mask[k] = fminf(fmaxf(image_raw[k], 0.0f), 1.0f);
      sc_filtered[k] = fminf(fmaxf(sc_mask[k] + 0.1f * (frand() - 0.5f), 0.0f), 1.0f);
      sc_conf[k] = frand();
    }
    for(size_t k = 0; k < (size_t)4 * W * H; k++) sc_guide[k] = 0.5f * image_in[k] + 0.02f * frand();

    // the JzAzBz error of pow() on Vulkan, as for color balance rgb above
    static const double sc_tol[2] = { 1e-2, 5e-4 };
    static const char *sc_names[2][6] = {
      { "satcurvergb jzazbz", "satcurvergb jzazbz masks", "satcurve_histogram jzazbz", "satcurve_mask jzazbz",
        "satcurve_prepare_scalar_mask jzazbz" },
      { "satcurvergb dtucs", "satcurvergb dtucs masks", "satcurve_histogram dtucs", "satcurve_mask dtucs",
        "satcurve_prepare_scalar_mask dtucs" }
    };
    for(int formula = 0; formula < 2; formula++)
    {
      for(int use_mask = 0; use_mask < 2; use_mask++)
        TEST_TOL(sc_names[formula][use_mask], "satcurve", "satcurvergb", sc_tol[formula],
             IMG_IN("in", image_in), IMG_OUT("out"), RAW_IN("sat_mask_in", sc_mask),
             RAW_IN("bri_mask_in", sc_filtered), RAW_IN("noise_confidence", sc_conf), INT("use_mask", use_mask),
             INT("width", W), INT("height", H), BUFFER("matrix_in", sc_matrix_in, sizeof(sc_matrix_in)),
             BUFFER("matrix_out", sc_matrix_out, sizeof(sc_matrix_out)),
             BUFFER("sat_lut", sc_sat_lut, sizeof(sc_sat_lut)), BUFFER("bri_lut", sc_bri_lut, sizeof(sc_bri_lut)),
             BUFFER("gamut_lut", sc_gamut_lut, sizeof(sc_gamut_lut)), INT("formula", formula),
             FLOAT("L_white", 1.07f), FLOAT("noise_protection", 0.6f));
      // float noise moves a few pixels into the next of the 256 bins, each
      // holding about 530 of them. With JzAzBz that is up to 2 of the 30 in
      // the sparse bins near zero saturation
      TEST_TOL(sc_names[formula][2], "satcurve", "satcurve_histogram", formula == 0 ? 0.1 : 2e-2,
           IMG_IN("in", image_in), INT("width", W), INT("height", H),
           BUFFER("matrix_in", sc_matrix_in, sizeof(sc_matrix_in)),
           BUFFER("gamut_lut", sc_gamut_lut, sizeof(sc_gamut_lut)), INT("formula", formula),
           FLOAT("L_white", 1.07f), INTS_OUT("hist_bins", 256 * sizeof(int32_t)));
      TEST_TOL(sc_names[formula][3], "satcurve", "satcurve_mask", sc_tol[formula],
           IMG_IN("in", image_in), IMG_OUT("out"), INT("width", W), INT("height", H),
           BUFFER("matrix_in", sc_matrix_in, sizeof(sc_matrix_in)),
           BUFFER("gamut_lut", sc_gamut_lut, sizeof(sc_gamut_lut)), INT("formula", formula),
           FLOAT("L_white", 1.07f));
      TEST_TOL(sc_names[formula][4], "satcurve", "satcurve_prepare_scalar_mask", sc_tol[formula],
           IMG_IN("in", image_in), RAW_OUT("out"), INT("width", W), INT("height", H),
           BUFFER("matrix_in", sc_matrix_in, sizeof(sc_matrix_in)),
           BUFFER("gamut_lut", sc_gamut_lut, sizeof(sc_gamut_lut)), INT("formula", formula),
           FLOAT("L_white", 1.07f));
    }
    TEST_TOL("satcurve_prepare_perceptual_guide", "satcurve", "satcurve_prepare_perceptual_guide", 5e-4,
         IMG_IN("in", image_in), IMG_OUT("out"), INT("width", W), INT("height", H),
         BUFFER("matrix_in", sc_matrix_in, sizeof(sc_matrix_in)), FLOAT("L_white", 1.07f));
    TEST("satcurve_prepare_filter_confidence", "satcurve", "satcurve_prepare_filter_confidence",
         RAW_IN("raw", sc_mask), RAW_IN("filtered", sc_filtered), IMG_IN("guide", sc_guide), RAW_OUT("control"),
         RAW_OUT("confidence"), INT("width", W), INT("height", H), FLOAT("protect_from", 0.05f),
         FLOAT("protect_to", 0.3f));
    TEST("satcurve_mask_from_control", "satcurve", "satcurve_mask_from_control",
         RAW_IN("raw", sc_mask), RAW_IN("filtered", sc_filtered), RAW_IN("confidence", sc_conf),
         IMG_IN("in", image_in), IMG_OUT("out"), INT("width", W), INT("height", H), FLOAT("protect_from", 0.05f),
         FLOAT("protect_to", 0.3f), FLOAT("noise_protection", 0.6f));
  }

  // tone equalizer, on positive luminances spread over the [-9; 0] EV range
  // the module works in, and on guided filter terms derived from them
  {
    enum { TE_DW = W / 4, TE_DH = H / 4 };
    static float te_lum[W * H], te_mask[W * H], te_packed[4 * W * H], te_ab[2 * W * H];
    static float te_av4[4 * TE_DW * TE_DH], te_av2[2 * TE_DW * TE_DH];
    for(int k = 0; k < W * H; k++)
    {
      const float *p = image_in + 4 * k;
      te_lum[k] = exp2f(-9.0f * fminf(fmaxf(p[0], 0.0f), 1.0f));
      te_mask[k] = exp2f(-9.0f * fminf(fmaxf(p[1], 0.0f), 1.0f));
      const float g = te_lum[k], m = te_mask[k];
      te_packed[4 * k] = g;
      te_packed[4 * k + 1] = m;
      te_packed[4 * k + 2] = g * g * (1.0f + 0.5f * fabsf(p[2]));
      te_packed[4 * k + 3] = g * m * (1.0f + 0.2f * (p[3] - 0.5f));
      te_ab[2 * k] = 0.5f + p[2];
      te_ab[2 * k + 1] = 0.1f * (p[3] - 0.5f);
    }
    for(int k = 0; k < TE_DW * TE_DH; k++)
    {
      const float g = te_lum[k], m = te_mask[k];
      te_av4[4 * k] = g;
      te_av4[4 * k + 1] = 0.1f * g * g * fabsf(image_in[4 * k + 2]);
      te_av4[4 * k + 2] = m;
      te_av4[4 * k + 3] = 0.05f * g * m * (image_in[4 * k + 3] - 0.5f);
      te_av2[2 * k] = g;
      te_av2[2 * k + 1] = 0.1f * g * g * fabsf(image_in[4 * k + 2]);
    }
    const size_t s1 = sizeof(float) * W * H, s2 = 2 * s1, s4 = 4 * s1;
    const size_t ds2 = 2 * sizeof(float) * TE_DW * TE_DH, ds4 = 2 * ds2;

    static const char *te_methods[7] = { "luminance_mask mean", "luminance_mask lightness", "luminance_mask value",
                                         "luminance_mask norm1", "luminance_mask norm2",
                                         "luminance_mask norm power", "luminance_mask geomean" };
    for(int m = 0; m < 7; m++)
      TEST(te_methods[m], "toneequal", "toneequal_luminance_mask",
           IMG_IN("in", image_in), BUFFER_OUT("luminance", NULL, s1), INT("width", W), INT("height", H),
           INT("method", m), FLOAT("exposure_boost", 1.3f), FLOAT("fulcrum", m & 1 ? 0.1845f : 0.0f),
           FLOAT("contrast_boost", m & 1 ? 1.4f : 1.0f));
    TEST_GRID("toneequal_apply", "toneequal", "toneequal_apply", 0.0, W - 8, H - 6,
              IMG_IN("in", image_in), BUFFER("luminance", te_lum, s1), IMG_OUT("out"), INT("width", W - 8),
              INT("height", H - 6), INT("in_width", W), INT("offset_x", 3), INT("offset_y", 2),
              FLOAT4("factors_low", 1.2f, 0.9f, 1.5f, 0.7f), FLOAT4("factors_high", 1.1f, 1.0f, 0.8f, 1.9f),
              FLOAT("gauss_denom", 0.5f));
    TEST_GRID("toneequal_display_mask", "toneequal", "toneequal_display_mask", 0.0, W - 8, H - 6,
              IMG_IN("in", image_in), BUFFER("luminance", te_lum, s1), IMG_OUT("out"), INT("width", W - 8),
              INT("height", H - 6), INT("in_width", W), INT("offset_x", 3), INT("offset_y", 2));
    static const float te_sampling[3] = { 0.0f, 1.0f, 0.5f };
    static const char *te_quantize[3] = { "toneequal_quantize off", "toneequal_quantize 1 EV",
                                          "toneequal_quantize 0.5 EV" };
    for(int q = 0; q < 3; q++)
      TEST(te_quantize[q], "toneequal", "toneequal_quantize",
           BUFFER("in", te_lum, s1), BUFFER_OUT("out", NULL, s1), INT("width", W), INT("height", H),
           FLOAT("sampling", te_sampling[q]), FLOAT("clip_min", exp2f(-14.0f)), FLOAT("clip_max", 4.0f));

    // a small radius, and one wider than half the image to reach the branch
    // that can neither add nor remove values
    static const int te_radius[2] = { 5, 300 };
    static char te_box[2][4][48];
    static const char *te_box_kernels[4] = { "toneequal_box_mean_x_2c", "toneequal_box_mean_y_2c",
                                             "toneequal_box_mean_x_4c", "toneequal_box_mean_y_4c" };
    for(int r = 0; r < 2; r++)
      for(int b = 0; b < 4; b++)
      {
        snprintf(te_box[r][b], sizeof(te_box[r][b]), "%s r%d", te_box_kernels[b], te_radius[r]);
        const int is_y = b & 1;
        const size_t sz = b < 2 ? s2 : s4;
        TEST_LINE(te_box[r][b], "toneequal", te_box_kernels[b], 1e-5, is_y ? W : H,
                  BUFFER("input", b < 2 ? te_ab : te_packed, sz), BUFFER_OUT("output", NULL, sz),
                  INT("width", W), INT("height", H), INT("radius", te_radius[r]));
      }

    TEST("toneequal_gf_pack", "toneequal", "toneequal_gf_pack",
         BUFFER("guide", te_lum, s1), BUFFER("mask", te_mask, s1), BUFFER_OUT("packed", NULL, s4),
         INT("width", W), INT("height", H));
    TEST("toneequal_gf_ab", "toneequal", "toneequal_gf_ab",
         BUFFER("packed", te_packed, s4), BUFFER_OUT("ab", NULL, s2), INT("width", W), INT("height", H),
         FLOAT("feathering", 0.05f));
    for(int f = 0; f < 2; f++)
    {
      TEST(f ? "toneequal_gf_blend geomean" : "toneequal_gf_blend", "toneequal", "toneequal_gf_blend",
           BUFFER_OUT("image", te_lum, s1), BUFFER("ab", te_ab, s2), INT("width", W), INT("height", H),
           INT("filter", f));
      TEST(f ? "toneequal_gf_blend_upsample geomean" : "toneequal_gf_blend_upsample", "toneequal",
           "toneequal_gf_blend_upsample",
           BUFFER_OUT("image", te_lum, s1), INT("width", W), INT("height", H), BUFFER("ds_ab", te_ab, ds2),
           INT("ds_width", TE_DW), INT("ds_height", TE_DH), INT("filter", f));
    }
    TEST("toneequal_eigf_pack_4c", "toneequal", "toneequal_eigf_pack_4c",
         BUFFER("guide", te_lum, s1), BUFFER("mask", te_mask, s1), BUFFER_OUT("packed", NULL, s4),
         INT("width", W), INT("height", H));
    TEST("toneequal_eigf_finish_4c", "toneequal", "toneequal_eigf_finish_4c",
         BUFFER_OUT("av", te_packed, s4), INT("width", W), INT("height", H));
    TEST("toneequal_eigf_pack_2c", "toneequal", "toneequal_eigf_pack_2c",
         BUFFER("guide", te_lum, s1), BUFFER_OUT("packed", NULL, s2), INT("width", W), INT("height", H));
    TEST("toneequal_eigf_finish_2c", "toneequal", "toneequal_eigf_finish_2c",
         BUFFER_OUT("av", te_packed, s2), INT("width", W), INT("height", H));
    for(int f = 0; f < 2; f++)
    {
      TEST_TOL(f ? "toneequal_eigf_blend geomean" : "toneequal_eigf_blend", "toneequal", "toneequal_eigf_blend",
               1e-4, BUFFER_OUT("image", te_lum, s1), BUFFER("mask", te_mask, s1), INT("width", W),
               INT("height", H), BUFFER("ds_av", te_av4, ds4), INT("ds_width", TE_DW), INT("ds_height", TE_DH),
               INT("filter", f), FLOAT("feathering", 0.05f));
      TEST_TOL(f ? "toneequal_eigf_blend_no_mask geomean" : "toneequal_eigf_blend_no_mask", "toneequal",
               "toneequal_eigf_blend_no_mask", 1e-4,
               BUFFER_OUT("image", te_lum, s1), INT("width", W), INT("height", H), BUFFER("ds_av", te_av2, ds2),
               INT("ds_width", TE_DW), INT("ds_height", TE_DH), INT("filter", f), FLOAT("feathering", 0.05f));
    }
  }

  // lens correction. The lensfun kernels sample at coordinates that lensfun
  // computes on the host, one pair per color: a mild barrel distortion with
  // some lateral chromatic aberration, which runs off the image at the edges,
  // and with the nan checks a few invalid coordinates
  {
    static float ln_pi[2][2 * 3 * W * H];
    for(int v = 0; v < 2; v++)
      for(int y = 0; y < H; y++)
        for(int x = 0; x < W; x++)
        {
          const float cx = x - 0.5f * W, cy = y - 0.5f * H;
          const float r2 = (cx * cx + cy * cy) / (0.25f * W * W);
          for(int c = 0; c < 3; c++)
          {
            const float k = 0.97f + 0.04f * r2 + 0.003f * (c - 1);
            float *p = ln_pi[v] + 6 * (y * W + x) + 2 * c;
            p[0] = 0.5f * W + k * cx + 7.0f;
            p[1] = 0.5f * H + k * cy - 3.0f;
            if(v == 1 && (x * 7 + y * 13) % 101 == 0) p[c & 1] = NAN;
          }
        }
    static const char *ln_kernels[4] = { "lens_distort_bilinear", "lens_distort_bicubic",
                                         "lens_distort_lanczos2", "lens_distort_lanczos3" };
    static const char *ln_labels[2][4] = {
      { "lens_distort_bilinear", "lens_distort_bicubic", "lens_distort_lanczos2", "lens_distort_lanczos3" },
      { "lens_distort_bilinear nan", "lens_distort_bicubic nan", "lens_distort_lanczos2 nan", "lens_distort_lanczos3 nan" } };
    // for blue the bicubic kernel weighs clamped taps, whose weights can
    // nearly cancel: one pixel here lands 2.4e-5 away on lavapipe
    for(int v = 0; v < 2; v++)
      for(int d = 0; d < 4; d++)
        TEST_TOL(ln_labels[v][d], "basic", ln_kernels[d], 1e-4,
                 IMG_IN("in", image_in), IMG_OUT("out"), INT("width", W), INT("height", H),
                 INT("iwidth", W), INT("iheight", H), INT("roi_in_x", 7), INT("roi_in_y", -3),
                 BUFFER("pi", ln_pi[v], sizeof(ln_pi[v])), INT("do_nan_checks", v));

    // lensfun's vignetting correction factors, halved, four to a pixel
    static float ln_vig[4 * W * H];
    for(int k = 0; k < 4 * W * H; k++) ln_vig[k] = 0.5f + 0.3f * image_in[(k + 4 * 11) % (4 * W * H)];
    TEST("lens_vignette", "basic", "lens_vignette",
         IMG_IN("in", image_in), IMG_OUT("out"), INT("width", W), INT("height", H),
         BUFFER("pi", ln_vig, sizeof(ln_vig)));

    // the manual vignette spline, as _init_vignette_spline() makes it
    static float ln_spline[512];
    for(int i = 0; i < 512; i++) ln_spline[i] = 0.1f + 0.4f * tanhf(3.0f * (i / 511.0f - 0.5f));
    for(int m = 0; m < 2; m++)
      TEST(m ? "lens_man_vignette mask" : "lens_man_vignette", "basic", "lens_man_vignette",
           IMG_IN("in", image_in), IMG_OUT("out"), BUFFER("spline", ln_spline, sizeof(ln_spline)),
           INT("width", W), INT("height", H), FLOAT("w2", 0.5f * W + 20.0f), FLOAT("h2", 0.5f * H + 10.0f),
           INT("roix", 20), INT("roiy", 10), FLOAT("inv_maxr", 1.0f / hypotf(0.5f * W + 20.0f, 0.5f * H + 10.0f)),
           FLOAT("intensity", 0.6f), INT("splinesize", 512), INT("vigmask", m));

    // embedded metadata corrections: 9 knots of vignetting and distortion
    static float md_knots[16], md_vig[16], md_cor[3 * 16];
    for(int i = 0; i < 9; i++)
    {
      md_knots[i] = (i + 0.5f) / 8.0f;
      md_vig[i] = 1.0f - 0.04f * i;
      for(int c = 0; c < 3; c++) md_cor[16 * c + i] = 1.0f + 0.006f * i + 0.001f * (c - 1) * i;
    }
    const float md_w2 = 0.5f * W + 4.0f, md_h2 = 0.5f * H + 2.0f;
    const float md_r = 1.0f / hypotf(md_w2, md_h2);
    TEST("md_vignette", "basic", "md_vignette",
         IMG_IN("in", image_in), IMG_OUT("out"), BUFFER("knots_vig", md_knots, sizeof(md_knots)),
         BUFFER("vig", md_vig, sizeof(md_vig)), INT("width", W), INT("height", H),
         FLOAT("w2", md_w2), FLOAT("h2", md_h2), FLOAT("r", md_r), INT("roix", 4), INT("roiy", 2), INT("knots", 9));
    // the interpolator ids and widths of bilinear, bicubic, lanczos2 and lanczos3
    static const int md_itor[4][2] = { { 0, 1 }, { 1, 2 }, { 2, 2 }, { 3, 3 } };
    static const char *md_labels[4] = { "md_lens_correction bilinear", "md_lens_correction bicubic",
                                        "md_lens_correction lanczos2", "md_lens_correction lanczos3 pass" };
    // the sample positions come from hypot() and a division, which may each
    // be an ulp off: at 260 pixels from the center that moves a sample by
    // 3e-5 pixels, and on this noise image moves the result by up to 8e-5.
    // OpenCL computes taps for lanczos3 with the bilinear formula, which at
    // that width has negative taps and a small norm, so it amplifies more.
    for(int d = 0; d < 4; d++)
      TEST_TOL(md_labels[d], "basic", "md_lens_correction", 1e-3,
               IMG_IN("in", image_in), IMG_OUT("out"), BUFFER("knots_dist", md_knots, sizeof(md_knots)),
               BUFFER("cor_rgb", md_cor, sizeof(md_cor)), INT("owidth", W), INT("oheight", H),
               INT("iwidth", W), INT("iheight", H), FLOAT("w2", md_w2), FLOAT("h2", md_h2), FLOAT("r", md_r),
               FLOAT("scale", 1.02f), INT("roix", 4), INT("roiy", 2), INT("roox", 6), INT("rooy", 3),
               INT("knots", 9), INT("itor_mode", md_itor[d][0]), INT("itor_width", md_itor[d][1]),
               INT("pass_mode", d == 3));
  }

  // denoise (profiled): the variance stabilizing transforms and their
  // inverses, one step of non-local means and one wavelet scale
  {
    static const float dp_y0u0v0[9] = { 0.33f, 0.34f, 0.33f, 0.5f, 0.0f, -0.5f, 0.25f, -0.5f, 0.25f };
    static const float dp_rgb[9] = { 1.0f, 1.2f, 0.6f, 1.0f, 0.0f, -0.7f, 1.0f, -1.1f, 0.6f };
#define DP_VST FLOAT4("a", 0.012f, 0.011f, 0.013f, 1.0f), FLOAT4("p", 0.4f, 0.5f, 0.45f, 1.0f), \
               FLOAT4("b", 0.0004f, 0.0003f, 0.0005f, 1.0f)
    TEST("denoiseprofile_precondition", "denoiseprofile", "denoiseprofile_precondition",
         IMG_IN("in", image_in), IMG_OUT("out"), INT("width", W), INT("height", H),
         FLOAT4("a", 0.012f, 0.011f, 0.013f, 1.0f), FLOAT4("sigma2", 0.002f, 0.003f, 0.001f, 0.0f));
    TEST("denoiseprofile_precondition_v2", "denoiseprofile", "denoiseprofile_precondition_v2",
         IMG_IN("in", image_in), IMG_OUT("out"), INT("width", W), INT("height", H), DP_VST,
         FLOAT4("wb", 2.1f, 1.0f, 1.6f, 0.0f));
    TEST("denoiseprofile_precondition_Y0U0V0", "denoiseprofile", "denoiseprofile_precondition_Y0U0V0",
         IMG_IN("in", image_in), IMG_OUT("out"), INT("width", W), INT("height", H), DP_VST,
         BUFFER("toY0U0V0", dp_y0u0v0, sizeof(dp_y0u0v0)));

    // non-local means on preconditioned data, which is around 2 sqrt(x / a)
    static float dp_pre[4 * W * H];
    for(int k = 0; k < 4 * W * H; k++) dp_pre[k] = (k % 4 == 3) ? 1.0f : 2.0f * sqrtf(fmaxf(image_in[k], 0.0f) / 0.012f);
    // DP_B is the mirrored border the distances are computed over
    enum { DP_P = 3, DP_BS = 16, DP_B = 5 };
    static float dp_u4[W * H], dp_u4b[W * H];
    for(int k = 0; k < W * H; k++)
    {
      dp_u4[k] = 4.0f * image_in[4 * k];
      dp_u4b[k] = 30.0f * image_in[4 * k + 1] * image_in[4 * k + 2];
    }
    TEST_GRID("denoiseprofile_dist", "denoiseprofile", "denoiseprofile_dist", 0.0, W + 2 * DP_B, H + 2 * DP_B,
              IMG_IN("in", dp_pre), BUFFER_OUT("U4", NULL, sizeof(float) * (W + 2 * DP_B) * (H + 2 * DP_B)),
              INT("width", W), INT("height", H), INT2("q", 3, -2), INT("border", DP_B));
    TEST("denoiseprofile_horiz", "denoiseprofile", "denoiseprofile_horiz",
         BUFFER("U4_in", dp_u4, sizeof(dp_u4)), BUFFER_OUT("U4_out", NULL, sizeof(float) * W * H),
         INT("width", W), INT("height", H), INT2("q", 3, -2), INT("P", DP_P),
         LOCAL("buffer", sizeof(float) * (DP_BS + 2 * DP_P)));
    t[n - 1].clgw = (W + DP_BS - 1) / DP_BS * DP_BS;
    t[n - 1].cllx = DP_BS;
    t[n - 1].clly = 1;
    TEST("denoiseprofile_vert", "denoiseprofile", "denoiseprofile_vert",
         BUFFER("U4_in", dp_u4b, sizeof(dp_u4b)), BUFFER_OUT("U4_out", NULL, sizeof(float) * W * H),
         INT("width", W), INT("height", H), INT2("q", 3, -2), INT("P", DP_P), FLOAT("norm", 0.04f),
         LOCAL("buffer", sizeof(float) * (DP_BS + 2 * DP_P)), FLOAT("central_pixel_weight", 0.1f),
         BUFFER("U4_single_pixel", dp_u4, sizeof(dp_u4)));
    t[n - 1].clgh = (H + DP_BS - 1) / DP_BS * DP_BS;
    t[n - 1].cllx = 1;
    t[n - 1].clly = DP_BS;
    // accumulated weights so far, then one more offset
    static float dp_u2[4 * W * H];
    for(int k = 0; k < 4 * W * H; k++) dp_u2[k] = (k % 4 == 3) ? 3.0f + image_in[k] : 3.0f * dp_pre[k];
    static float dp_w[(W + 2 * DP_B) * (H + 2 * DP_B)];
    for(int k = 0; k < (W + 2 * DP_B) * (H + 2 * DP_B); k++) dp_w[k] = image_in[4 * (k % (W * H)) + 2];
    TEST("denoiseprofile_accu", "denoiseprofile", "denoiseprofile_accu",
         IMG_IN("in", dp_pre), BUFFER_OUT("U2", dp_u2, sizeof(dp_u2)), BUFFER("U4", dp_w, sizeof(dp_w)),
         INT("width", W), INT("height", H), INT2("q", -2, 3), INT("border", DP_B));
    TEST("denoiseprofile_finish", "denoiseprofile", "denoiseprofile_finish",
         IMG_IN("in", image_in), BUFFER("U2", dp_u2, sizeof(dp_u2)), IMG_OUT("out"), INT("width", W),
         INT("height", H), FLOAT4("a", 0.012f, 0.011f, 0.013f, 1.0f), FLOAT4("sigma2", 0.002f, 0.003f, 0.001f, 0.0f));
    TEST("denoiseprofile_finish_v2", "denoiseprofile", "denoiseprofile_finish_v2",
         IMG_IN("in", image_in), BUFFER("U2", dp_u2, sizeof(dp_u2)), IMG_OUT("out"), INT("width", W),
         INT("height", H), DP_VST, FLOAT("bias", 0.3f), FLOAT4("wb", 2.1f, 1.0f, 1.6f, 0.0f));

    TEST("denoiseprofile_backtransform", "denoiseprofile", "denoiseprofile_backtransform",
         IMG_IN("in", dp_pre), IMG_OUT("out"), INT("width", W), INT("height", H),
         FLOAT4("a", 0.012f, 0.011f, 0.013f, 1.0f), FLOAT4("sigma2", 0.002f, 0.003f, 0.001f, 0.0f));
    TEST("denoiseprofile_backtransform_v2", "denoiseprofile", "denoiseprofile_backtransform_v2",
         IMG_IN("in", dp_pre), IMG_OUT("out"), INT("width", W), INT("height", H), DP_VST,
         FLOAT("bias", 0.3f), FLOAT4("wb", 2.1f, 1.0f, 1.6f, 0.0f));
    TEST("denoiseprofile_backtransform_Y0U0V0", "denoiseprofile", "denoiseprofile_backtransform_Y0U0V0",
         IMG_IN("in", dp_pre), IMG_OUT("out"), INT("width", W), INT("height", H), DP_VST,
         FLOAT("bias", 0.3f), FLOAT4("wb", 2.1f, 1.0f, 1.6f, 0.0f), BUFFER("toRGB", dp_rgb, sizeof(dp_rgb)));
#undef DP_VST

    // the 5x5 a-trous filter, on scales 0 and 2
    static float dp_filter[25];
    static const float dp_m[5] = { 0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f };
    for(int j = 0; j < 5; j++)
      for(int i = 0; i < 5; i++) dp_filter[5 * j + i] = dp_m[i] * dp_m[j];
    for(int sc = 0; sc < 3; sc += 2)
      TEST_TOL(sc ? "denoiseprofile_decompose scale 2" : "denoiseprofile_decompose", "denoiseprofile",
               "denoiseprofile_decompose", 1e-4,
               IMG_IN("in", dp_pre), IMG_OUT("coarse"), IMG_OUT("detail"), INT("width", W), INT("height", H),
               INT("scale", sc), FLOAT("inv_sigma2", 1.0f / (0.25f * sc + 1.0f)),
               BUFFER("filter", dp_filter, sizeof(dp_filter)));
    static float dp_detail[4 * W * H];
    for(int k = 0; k < 4 * W * H; k++) dp_detail[k] = 0.6f * (image_in[(k + 4 * 29) % (4 * W * H)] - 0.5f);
    TEST("denoiseprofile_synthesize", "denoiseprofile", "denoiseprofile_synthesize",
         IMG_IN("coarse", dp_pre), IMG_IN("detail", dp_detail), IMG_OUT("out"), INT("width", W), INT("height", H),
         FLOAT4_BUF("threshold", 0.1f, 0.05f, 0.15f, 0.0f), SLANG_INT("band", 0),
         FLOAT4("boost", 1.0f, 1.0f, 1.0f, 1.0f));

    // block sums of the squared detail. The OpenCL kernel adds up each
    // block as a tree, the Slang one by columns, so they differ in the last bits
    enum { DP_NBX = (W + 15) / 16, DP_NBY = (H + 15) / 16 };
    TEST_GRID("denoiseprofile_reduce_first", "denoiseprofile", "denoiseprofile_reduce_first", 1e-5,
              16 * DP_NBX, DP_NBY,
              IMG_IN("in", dp_detail), INT("width", W), INT("height", H),
              BUFFER_OUT("accu", NULL, sizeof(float) * 4 * DP_NBX * DP_NBY), LOCAL("buffer", sizeof(float) * 4 * 256));
    t[n - 1].clgw = DP_NBX * 16;
    t[n - 1].clgh = DP_NBY * 16;
    t[n - 1].cllx = 16;
    t[n - 1].clly = 16;
  }

#undef TEST
#undef TEST_TOL
#undef TEST_GRID
  return n;
}

// -------------------------------------------------------------------------------------------------
// helpers

void die(const char *fmt, ...)
{
  // after the device lines, which a pipe buffers
  fflush(stdout);
  va_list ap;
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fputc('\n', stderr);
  exit(2);
}

#define CK(x)                                                                                        \
  do                                                                                                 \
  {                                                                                                  \
    const int _e = (int)(x);                                                                         \
    if(_e) die("%s:%d: %s returned %d", __FILE__, __LINE__, #x, _e);                                 \
  } while(0)

char *slurp(const char *path, size_t *len)
{
  FILE *f = fopen(path, "rb");
  if(!f) die("cannot open %s", path);
  fseek(f, 0, SEEK_END);
  *len = ftell(f);
  rewind(f);
  char *buf = malloc(*len + 1);
  if(fread(buf, 1, *len, f) != *len) die("cannot read %s", path);
  buf[*len] = 0;
  fclose(f);
  return buf;
}

int device_wanted(const char *wanted, const int index, const char *name)
{
  if(!wanted || !*wanted) return 0;
  char *end = NULL;
  const long i = strtol(wanted, &end, 10);
  if(*end == '\0') return i == index;
  const size_t n = strlen(wanted);
  for(const char *c = name; *c; c++)
    if(!strncasecmp(c, wanted, n)) return 1;
  return 0;
}

// -------------------------------------------------------------------------------------------------
// OpenCL

typedef struct cl_env_t
{
  cl_context ctx;
  cl_command_queue queue;
  cl_device_id dev;
  int no_r;           // single channel images have to be stored as RGBA
} cl_env_t;

static int cl_has_format(const cl_env_t *e, const cl_image_format fmt)
{
  cl_image_format formats[256];
  cl_uint n = 0;
  CK(clGetSupportedImageFormats(e->ctx, CL_MEM_READ_WRITE, CL_MEM_OBJECT_IMAGE2D, 256, formats, &n));
  for(cl_uint i = 0; i < n && i < 256; i++)
    if(formats[i].image_channel_order == fmt.image_channel_order
       && formats[i].image_channel_data_type == fmt.image_channel_data_type)
      return 1;
  return 0;
}

// a single channel image, or where the device has no CL_R images (PoCL 6
// only offers RGBA) an RGBA one with the value in x, which reads back the
// same: CL_R images read as (r, 0, 0, 1)
static cl_mem cl_image_1ch(const cl_env_t *e, cl_mem_flags flags, cl_channel_type type,
                           const cl_image_desc *desc, const void *px)
{
  cl_int err;
  if(!e->no_r)
  {
    const cl_image_format fmt = { CL_R, type };
    cl_mem m = clCreateImage(e->ctx, flags | CL_MEM_COPY_HOST_PTR, &fmt, desc, (void *)px, &err);
    CK(err);
    return m;
  }
  const size_t npix = desc->image_width * desc->image_height;
  const cl_image_format fmt = { CL_RGBA, type };
  cl_mem m;
  if(type == CL_FLOAT)
  {
    float *rgba = malloc(npix * 4 * sizeof(float));
    for(size_t i = 0; i < npix; i++)
    {
      rgba[4 * i] = ((const float *)px)[i];
      rgba[4 * i + 1] = rgba[4 * i + 2] = 0.0f;
      rgba[4 * i + 3] = 1.0f;
    }
    m = clCreateImage(e->ctx, flags | CL_MEM_COPY_HOST_PTR, &fmt, desc, rgba, &err);
    free(rgba);
  }
  else
  {
    uint16_t *rgba = malloc(npix * 4 * sizeof(uint16_t));
    for(size_t i = 0; i < npix; i++)
    {
      rgba[4 * i] = ((const uint16_t *)px)[i];
      rgba[4 * i + 1] = rgba[4 * i + 2] = 0;
      rgba[4 * i + 3] = 1;
    }
    m = clCreateImage(e->ctx, flags | CL_MEM_COPY_HOST_PTR, &fmt, desc, rgba, &err);
    free(rgba);
  }
  CK(err);
  return m;
}

// DT_PARITY_OPENCL_DEVICE picks the device by index, counted across all
// platforms, or by part of its name. Without it the first one is used.
static void cl_init(cl_env_t *e, char *name, const size_t size)
{
  cl_platform_id platforms[16];
  cl_uint nplatforms = 0;
  cl_int err;
  // the ICD loader returns CL_PLATFORM_NOT_FOUND_KHR without a driver
  if(clGetPlatformIDs(16, platforms, &nplatforms) != CL_SUCCESS) nplatforms = 0;
  const char *wanted = getenv("DT_PARITY_OPENCL_DEVICE");
  int index = 0, found = 0;
  for(cl_uint p = 0; p < nplatforms && p < 16; p++)
  {
    cl_device_id devs[16];
    cl_uint ndevs = 0;
    if(clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 16, devs, &ndevs) != CL_SUCCESS) continue;
    for(cl_uint d = 0; d < ndevs && d < 16; d++, index++)
    {
      char dev_name[256] = { 0 };
      clGetDeviceInfo(devs[d], CL_DEVICE_NAME, sizeof(dev_name) - 1, dev_name, NULL);
      const int pick = !found && (wanted && *wanted ? device_wanted(wanted, index, dev_name) : 1);
      printf("OpenCL device %d: %s%s\n", index, dev_name, pick ? ", used" : "");
      if(pick)
      {
        e->dev = devs[d];
        snprintf(name, size, "%s", dev_name);
        found = 1;
      }
    }
  }
  if(!found && index > 0) die("no OpenCL device matches DT_PARITY_OPENCL_DEVICE '%s'", wanted);
  if(!found)
  {
    printf("no OpenCL device, skipping\n");
    exit(SKIP);
  }
  e->ctx = clCreateContext(NULL, 1, &e->dev, NULL, NULL, &err);
  CK(err);
  e->queue = clCreateCommandQueue(e->ctx, e->dev, 0, &err);
  CK(err);
  e->no_r = !cl_has_format(e, (cl_image_format){ CL_R, CL_FLOAT })
            || !cl_has_format(e, (cl_image_format){ CL_R, CL_UNSIGNED_INT16 });
  if(e->no_r) printf("OpenCL device has no single channel images, using RGBA\n");
}

size_t out_size(const arg_t *arg)
{
  if(arg->kind == ARG_IMAGE_OUT) return sizeof(float) * arg->w * arg->h * arg->channels;
  if(arg->kind == ARG_BUFFER_OUT) return arg->size;
  return 0;
}

// outputs start zeroed, as some kernels write only part of them
const void *initial_data(const arg_t *arg)
{
  static void *zeros = NULL;
  static size_t nzeros = 0;
  if(arg->kind == ARG_BUFFER_OUT && arg->data) return arg->data;
  const size_t sz = out_size(arg);
  if(sz > nzeros)
  {
    free(zeros);
    zeros = calloc(1, sz);
    nzeros = sz;
  }
  return zeros;
}

static void cl_run(cl_env_t *e, const char *cl_dir, const test_t *t, void **outs)
{
  char path[1024], opts[1100];
  snprintf(path, sizeof(path), "%s/%s.cl", cl_dir, t->program);
  // the same include path darktable passes, plus hdr-transfer.h which lives in src/common
  snprintf(opts, sizeof(opts), "-I%s -I%s/../../src/common", cl_dir, cl_dir);

  size_t len;
  const char *src = slurp(path, &len);
  cl_int err;
  cl_program prog = clCreateProgramWithSource(e->ctx, 1, &src, &len, &err);
  CK(err);
  if(clBuildProgram(prog, 1, &e->dev, opts, NULL, NULL) != CL_SUCCESS)
  {
    static char log[1 << 16];
    clGetProgramBuildInfo(prog, e->dev, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
    die("building %s failed:\n%s", path, log);
  }
  cl_kernel k = clCreateKernel(prog, t->kernel, &err);
  CK(err);

  cl_mem mem[MAX_ARGS] = { 0 };
  for(int a = 0, ca = 0; a < t->nargs; a++)
  {
    const arg_t *arg = &t->args[a];
    if(arg->slang_only) continue;
    const int a_cl = ca++;
    const cl_image_format fmt = { arg->channels == 1 ? CL_R : CL_RGBA, CL_FLOAT };
    const cl_image_desc desc = { .image_type = CL_MEM_OBJECT_IMAGE2D, .image_width = arg->w, .image_height = arg->h };
    switch(arg->kind)
    {
      case ARG_IMAGE_IN:
        if(arg->channels == 1)
          mem[a] = cl_image_1ch(e, CL_MEM_READ_ONLY, CL_FLOAT, &desc, arg->pixels);
        else
        {
          mem[a] = clCreateImage(e->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &fmt, &desc, (void *)arg->pixels, &err);
          CK(err);
        }
        CK(clSetKernelArg(k, a_cl, sizeof(cl_mem), &mem[a]));
        break;
      case ARG_IMAGE_OUT:
        if(arg->channels == 1)
          mem[a] = cl_image_1ch(e, CL_MEM_READ_WRITE, CL_FLOAT, &desc, initial_data(arg));
        else
        {
          mem[a] = clCreateImage(e->ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, &fmt, &desc,
                                 (void *)initial_data(arg), &err);
          CK(err);
        }
        CK(clSetKernelArg(k, a_cl, sizeof(cl_mem), &mem[a]));
        break;
      case ARG_BUFFER_OUT:
        mem[a] = clCreateBuffer(e->ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, arg->size,
                                (void *)initial_data(arg), &err);
        CK(err);
        CK(clSetKernelArg(k, a_cl, sizeof(cl_mem), &mem[a]));
        break;
      case ARG_LOCAL:
        CK(clSetKernelArg(k, a_cl, arg->size, NULL));
        break;
      case ARG_RAW_U16:
        mem[a] = cl_image_1ch(e, CL_MEM_READ_ONLY, CL_UNSIGNED_INT16, &desc, arg->data);
        CK(clSetKernelArg(k, a_cl, sizeof(cl_mem), &mem[a]));
        break;
      case ARG_BUFFER:
        mem[a] = clCreateBuffer(e->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, arg->size, (void *)arg->data, &err);
        CK(err);
        CK(clSetKernelArg(k, a_cl, sizeof(cl_mem), &mem[a]));
        break;
      case ARG_INT:
        CK(clSetKernelArg(k, a_cl, sizeof(int), &arg->i));
        break;
      case ARG_FLOAT:
        CK(clSetKernelArg(k, a_cl, sizeof(float), &arg->f));
        break;
      case ARG_FLOAT4:
      case ARG_FLOAT4_BUF:
        CK(clSetKernelArg(k, a_cl, 4 * sizeof(float), arg->v));
        break;
      case ARG_INT2:
        CK(clSetKernelArg(k, a_cl, 2 * sizeof(int), arg->iv));
        break;
      case ARG_INT4:
        CK(clSetKernelArg(k, a_cl, 4 * sizeof(int), arg->iv));
        break;
      case ARG_STRUCT:
        CK(clSetKernelArg(k, a_cl, arg->size, arg->data));
        break;
      case ARG_FLOAT2:
        CK(clSetKernelArg(k, a_cl, 2 * sizeof(float), arg->v));
        break;
      case ARG_CHAR2:
      {
        const cl_char2 c2 = { { (cl_char)arg->iv[0], (cl_char)arg->iv[1] } };
        CK(clSetKernelArg(k, a_cl, sizeof(c2), &c2));
        break;
      }
    }
  }

  // a single row is a 1D kernel
  const size_t ly = t->clly ? t->clly : t->gh == 1 ? 1 : 16;
  const size_t lx = t->cllx ? t->cllx : 16;
  const size_t gw = t->clgw ? t->clgw : t->gw;
  const size_t gh = t->clgh ? t->clgh : t->gh;
  const size_t global[2] = { (gw + lx - 1) / lx * lx, (gh + ly - 1) / ly * ly };
  const size_t local[2] = { lx, ly };
  CK(clEnqueueNDRangeKernel(e->queue, k, 2, NULL, global, local, 0, NULL, NULL));
  for(int a = 0; a < t->nargs; a++)
  {
    const arg_t *arg = &t->args[a];
    const size_t origin[3] = { 0, 0, 0 }, region[3] = { arg->w, arg->h, 1 };
    if(arg->kind == ARG_IMAGE_OUT && arg->channels == 1 && e->no_r)
    {
      const size_t npix = (size_t)arg->w * arg->h;
      float *rgba = malloc(npix * 4 * sizeof(float));
      CK(clEnqueueReadImage(e->queue, mem[a], CL_TRUE, origin, region, 0, 0, rgba, 0, NULL, NULL));
      for(size_t i = 0; i < npix; i++) ((float *)outs[a])[i] = rgba[4 * i];
      free(rgba);
    }
    else if(arg->kind == ARG_IMAGE_OUT)
      CK(clEnqueueReadImage(e->queue, mem[a], CL_TRUE, origin, region, 0, 0, outs[a], 0, NULL, NULL));
    else if(arg->kind == ARG_BUFFER_OUT)
      CK(clEnqueueReadBuffer(e->queue, mem[a], CL_TRUE, 0, arg->size, outs[a], 0, NULL, NULL));
  }

  for(int a = 0; a < t->nargs; a++)
    if(mem[a]) clReleaseMemObject(mem[a]);
  clReleaseKernel(k);
  clReleaseProgram(prog);
  free((void *)src);
}

#ifndef PARITY_METAL

// -------------------------------------------------------------------------------------------------
// SPIR-V: find each parameter's binding and push constant offset by name, so that
// the test cases only list parameters in OpenCL argument order

typedef struct spv_param_t
{
  char name[128];
  int binding;
  int offset;
} spv_param_t;

typedef struct spv_info_t
{
  int nparams;
  spv_param_t params[MAX_ARGS];
} spv_info_t;

static const char *spv_string(const uint32_t *w)
{
  return (const char *)w;
}

static void spv_reflect(const uint32_t *code, size_t words, const char *entry, spv_info_t *info)
{
  // pass 1: names, decorations, the entry point's interface and push constant type
  enum { MAXID = 1 << 20 };
  const char **names = calloc(MAXID, sizeof(char *));
  int *binding = malloc(MAXID * sizeof(int));
  int *ptr_pointee = calloc(MAXID, sizeof(int));
  int *var_ptrtype = calloc(MAXID, sizeof(int));
  int *var_storage = calloc(MAXID, sizeof(int));
  memset(binding, -1, MAXID * sizeof(int));
  const uint32_t *iface = NULL;
  int niface = 0;
  typedef struct { int type, member, offset; const char *name; } member_t;
  member_t *members = calloc(4096, sizeof(member_t));
  int nmembers = 0;

  if(code[0] != 0x07230203) die("not a SPIR-V module");
  if(code[3] >= MAXID) die("SPIR-V id bound too large");
  for(size_t i = 5; i < words;)
  {
    const uint32_t op = code[i] & 0xffff, n = code[i] >> 16;
    const uint32_t *w = code + i;
    if(n == 0) die("broken SPIR-V");
    switch(op)
    {
      case 5: // OpName
        names[w[1]] = spv_string(w + 2);
        break;
      case 6: // OpMemberName
        members[nmembers++] = (member_t){ .type = w[1], .member = w[2], .offset = -1, .name = spv_string(w + 3) };
        break;
      case 15: // OpEntryPoint
        if(!strcmp(spv_string(w + 3), entry))
        {
          const size_t slen = strlen(spv_string(w + 3)) / 4 + 1;
          iface = w + 3 + slen;
          niface = n - 3 - slen;
        }
        break;
      case 32: // OpTypePointer
        ptr_pointee[w[1]] = w[3];
        break;
      case 59: // OpVariable
        var_ptrtype[w[2]] = w[1];
        var_storage[w[2]] = w[3];
        break;
      case 71: // OpDecorate
        if(w[2] == 33) binding[w[1]] = w[3]; // Binding
        break;
      case 72: // OpMemberDecorate
        if(w[3] == 35) // Offset
        {
          int found = 0;
          for(int m = 0; m < nmembers; m++)
            if(members[m].type == (int)w[1] && members[m].member == (int)w[2])
            {
              members[m].offset = w[4];
              found = 1;
            }
          if(!found) members[nmembers++] = (member_t){ .type = w[1], .member = w[2], .offset = w[4], .name = "" };
        }
        break;
    }
    i += n;
  }
  if(!iface) die("entry point %s not found", entry);

  info->nparams = 0;
  int push_type = -1;
  for(int k = 0; k < niface; k++)
  {
    const uint32_t id = iface[k];
    const char *nm = names[id];
    if(!nm) continue;
    if(var_storage[id] == 9) // PushConstant: the uniform scalars
    {
      const int st = ptr_pointee[var_ptrtype[id]];
      push_type = st;
      for(int m = 0; m < nmembers; m++)
        if(members[m].type == st && members[m].name[0])
        {
          spv_param_t *p = &info->params[info->nparams++];
          snprintf(p->name, sizeof(p->name), "%s", members[m].name);
          p->binding = -1;
          p->offset = members[m].offset;
        }
    }
    else if(binding[id] >= 0 && !strncmp(nm, "entryPointParams.", 17))
    {
      // "entryPointParams.<param>" or "entryPointParams.<param>.tex"
      spv_param_t *p = &info->params[info->nparams++];
      snprintf(p->name, sizeof(p->name), "%s", nm + 17);
      char *dot = strchr(p->name, '.');
      if(dot) *dot = 0;
      p->binding = binding[id];
      p->offset = -1;
    }
  }
  // member offsets are only known after the whole module was read. Each entry
  // point has its own push constant struct, with the same member names in
  // different places, so only this entry point's struct counts
  for(int k = 0; k < info->nparams; k++)
    if(info->params[k].binding < 0)
      for(int m = 0; m < nmembers; m++)
        if(members[m].type == push_type && !strcmp(members[m].name, info->params[k].name)
           && members[m].offset >= 0)
          info->params[k].offset = members[m].offset;

  free(names);
  free(binding);
  free(ptr_pointee);
  free(var_ptrtype);
  free(var_storage);
  free(members);
}

static const spv_param_t *spv_find(const spv_info_t *info, const char *name)
{
  for(int k = 0; k < info->nparams; k++)
    if(!strcmp(info->params[k].name, name)) return &info->params[k];
  return NULL;
}

// -------------------------------------------------------------------------------------------------
// Vulkan

typedef struct vk_env_t
{
  VkInstance instance;
  VkPhysicalDevice pdev;
  VkDevice dev;
  VkQueue queue;
  uint32_t family;
  VkCommandPool pool;
} vk_env_t;

typedef struct vk_buffer_t
{
  VkBuffer buf;
  VkDeviceMemory mem;
  void *map;
} vk_buffer_t;

typedef struct vk_image_t
{
  VkImage img;
  VkDeviceMemory mem;
  VkImageView view;
} vk_image_t;

// DT_PARITY_VULKAN_DEVICE picks the device by index or by part of its name.
// Without it a discrete GPU wins over an integrated one, which wins over the
// rest, as in src/common/slang_gpu_vulkan.c
static void vk_init(vk_env_t *e, char *name, const size_t size)
{
  const VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_2 };
  const VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app };
  uint32_t n = 0;
  if(vkCreateInstance(&ici, NULL, &e->instance) == VK_SUCCESS) vkEnumeratePhysicalDevices(e->instance, &n, NULL);
  if(n == 0)
  {
    printf("no Vulkan device, skipping\n");
    exit(SKIP);
  }
  VkPhysicalDevice *devs = calloc(n, sizeof(*devs));
  vkEnumeratePhysicalDevices(e->instance, &n, devs);
  const char *wanted = getenv("DT_PARITY_VULKAN_DEVICE");
  int best = -1, best_rank = -1;
  for(uint32_t d = 0; d < n; d++)
  {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(devs[d], &props);
    const int rank = device_wanted(wanted, d, props.deviceName) ? 4
      : wanted && *wanted ? -1
      : props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 3
      : props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 2
      : props.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU ? 1
      : 0;
    if(rank > best_rank)
    {
      best = d;
      best_rank = rank;
    }
  }
  if(best < 0) die("no Vulkan device matches DT_PARITY_VULKAN_DEVICE '%s'", wanted);
  for(uint32_t d = 0; d < n; d++)
  {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(devs[d], &props);
    printf("Vulkan device %u: %s%s\n", d, props.deviceName, (int)d == best ? ", used" : "");
    if((int)d == best) snprintf(name, size, "%s", props.deviceName);
  }
  e->pdev = devs[best];
  free(devs);

  uint32_t nf = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(e->pdev, &nf, NULL);
  VkQueueFamilyProperties *qf = calloc(nf, sizeof(*qf));
  vkGetPhysicalDeviceQueueFamilyProperties(e->pdev, &nf, qf);
  e->family = UINT32_MAX;
  for(uint32_t f = 0; f < nf && e->family == UINT32_MAX; f++)
    if(qf[f].queueFlags & VK_QUEUE_COMPUTE_BIT) e->family = f;
  free(qf);
  if(e->family == UINT32_MAX) die("no Vulkan compute queue");

  const float prio = 1.0f;
  const VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                        .queueFamilyIndex = e->family, .queueCount = 1, .pQueuePriorities = &prio };
  // the kernels write images declared without a format, and filmic seeds its
  // noise with 64-bit integers
  const VkPhysicalDeviceFeatures features = { .shaderStorageImageWriteWithoutFormat = VK_TRUE,
                                              .shaderInt64 = VK_TRUE };
  const VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1,
                                   .pQueueCreateInfos = &qci, .pEnabledFeatures = &features };
  CK(vkCreateDevice(e->pdev, &dci, NULL, &e->dev));
  vkGetDeviceQueue(e->dev, e->family, 0, &e->queue);
  const VkCommandPoolCreateInfo cpci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                         .queueFamilyIndex = e->family };
  CK(vkCreateCommandPool(e->dev, &cpci, NULL, &e->pool));
}

static uint32_t vk_memtype(vk_env_t *e, uint32_t bits, VkMemoryPropertyFlags want)
{
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(e->pdev, &mp);
  for(uint32_t i = 0; i < mp.memoryTypeCount; i++)
    if((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
  die("no suitable Vulkan memory type");
  return 0;
}

static vk_buffer_t vk_buffer(vk_env_t *e, VkDeviceSize size, VkBufferUsageFlags usage)
{
  vk_buffer_t b;
  const VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size, .usage = usage };
  CK(vkCreateBuffer(e->dev, &bci, NULL, &b.buf));
  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(e->dev, b.buf, &req);
  const VkMemoryAllocateInfo mai = {
    .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size,
    .memoryTypeIndex = vk_memtype(e, req.memoryTypeBits,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) };
  CK(vkAllocateMemory(e->dev, &mai, NULL, &b.mem));
  CK(vkBindBufferMemory(e->dev, b.buf, b.mem, 0));
  CK(vkMapMemory(e->dev, b.mem, 0, size, 0, &b.map));
  return b;
}

static void vk_buffer_free(vk_env_t *e, vk_buffer_t *b)
{
  vkDestroyBuffer(e->dev, b->buf, NULL);
  vkFreeMemory(e->dev, b->mem, NULL);
}

static vk_image_t vk_image(vk_env_t *e, int w, int h, VkFormat fmt, VkImageUsageFlags usage)
{
  vk_image_t im;
  const VkImageCreateInfo ici = {
    .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = fmt,
    .extent = { w, h, 1 }, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
    .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = usage, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED };
  CK(vkCreateImage(e->dev, &ici, NULL, &im.img));
  VkMemoryRequirements req;
  vkGetImageMemoryRequirements(e->dev, im.img, &req);
  const VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size,
                                     .memoryTypeIndex = vk_memtype(e, req.memoryTypeBits, 0) };
  CK(vkAllocateMemory(e->dev, &mai, NULL, &im.mem));
  CK(vkBindImageMemory(e->dev, im.img, im.mem, 0));
  const VkImageViewCreateInfo vci = { .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = im.img,
                                      .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = fmt,
                                      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
  CK(vkCreateImageView(e->dev, &vci, NULL, &im.view));
  return im;
}

static void vk_image_free(vk_env_t *e, vk_image_t *im)
{
  vkDestroyImageView(e->dev, im->view, NULL);
  vkDestroyImage(e->dev, im->img, NULL);
  vkFreeMemory(e->dev, im->mem, NULL);
}

static void vk_barrier(VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to)
{
  const VkImageMemoryBarrier b = {
    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
    .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, .oldLayout = from, .newLayout = to,
    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = img,
    .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
  vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0,
                       NULL, 1, &b);
}

static void vk_run(vk_env_t *e, const char *spv_dir, const test_t *t, void **outs)
{
  char path[1024];
  snprintf(path, sizeof(path), "%s/%s.spv", spv_dir, t->program);
  size_t len;
  uint32_t *code = (uint32_t *)slurp(path, &len);
  spv_info_t info;
  const char *entry = t->slang_kernel ? t->slang_kernel : t->kernel;
  spv_reflect(code, len / 4, entry, &info);

  VkDescriptorSetLayoutBinding lb[MAX_ARGS];
  VkWriteDescriptorSet wr[MAX_ARGS];
  VkDescriptorImageInfo dii[MAX_ARGS];
  VkDescriptorBufferInfo dbi[MAX_ARGS];
  vk_image_t img[MAX_ARGS] = { 0 };
  vk_buffer_t buf[MAX_ARGS] = { 0 };
  vk_buffer_t stage[MAX_ARGS] = { 0 };
  int nbind = 0, have_img[MAX_ARGS] = { 0 }, have_buf[MAX_ARGS] = { 0 };
  uint8_t push[256] = { 0 };
  int push_size = 0;

  for(int a = 0; a < t->nargs; a++)
  {
    const arg_t *arg = &t->args[a];
    const spv_param_t *p = spv_find(&info, arg->name);
    if(!p)
    {
      // a parameter the kernel never reads is dropped from the module
      if(out_size(arg)) die("%s: output %s not in the SPIR-V", t->kernel, arg->name);
      continue;
    }
    const VkFormat fmt = arg->channels == 1 ? VK_FORMAT_R32_SFLOAT : VK_FORMAT_R32G32B32A32_SFLOAT;
    switch(arg->kind)
    {
      case ARG_IMAGE_IN:
      case ARG_IMAGE_OUT:
      {
        const int is_out = arg->kind == ARG_IMAGE_OUT;
        img[a] = vk_image(e, arg->w, arg->h, fmt,
                          (is_out ? VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                                  : VK_IMAGE_USAGE_SAMPLED_BIT) | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        have_img[a] = 1;
        const size_t bytes = (size_t)arg->w * arg->h * arg->channels * sizeof(float);
        stage[a] = vk_buffer(e, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        memcpy(stage[a].map, is_out ? initial_data(arg) : arg->pixels, bytes);
        const VkDescriptorType type = is_out ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        lb[nbind] = (VkDescriptorSetLayoutBinding){ p->binding, type, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL };
        dii[nbind] = (VkDescriptorImageInfo){ .imageView = img[a].view,
                                              .imageLayout = is_out ? VK_IMAGE_LAYOUT_GENERAL
                                                                    : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        wr[nbind] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = p->binding,
                                            .descriptorCount = 1, .descriptorType = type, .pImageInfo = &dii[nbind] };
        nbind++;
        break;
      }
      case ARG_LOCAL:
        break;
      case ARG_BUFFER:
      case ARG_BUFFER_OUT:
      case ARG_RAW_U16:
      case ARG_FLOAT4_BUF:
      {
        const int is_value = arg->kind == ARG_FLOAT4_BUF;
        const size_t size = is_value ? sizeof(arg->v) : arg->size;
        buf[a] = vk_buffer(e, size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        have_buf[a] = 1;
        memcpy(buf[a].map, is_value ? (const void *)arg->v
                           : arg->kind == ARG_BUFFER_OUT ? initial_data(arg) : arg->data, size);
        lb[nbind] = (VkDescriptorSetLayoutBinding){ p->binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                                    VK_SHADER_STAGE_COMPUTE_BIT, NULL };
        dbi[nbind] = (VkDescriptorBufferInfo){ buf[a].buf, 0, size };
        wr[nbind] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = p->binding,
                                            .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                            .pBufferInfo = &dbi[nbind] };
        nbind++;
        break;
      }
      case ARG_INT:
      case ARG_FLOAT:
      case ARG_FLOAT4:
      case ARG_INT2:
      case ARG_INT4:
      case ARG_STRUCT:
      case ARG_FLOAT2:
      case ARG_CHAR2:
      {
        const size_t sz = arg->kind == ARG_STRUCT ? arg->size
                        : arg->kind == ARG_FLOAT4 || arg->kind == ARG_INT4 ? 16
                        : arg->kind == ARG_INT2 || arg->kind == ARG_FLOAT2 || arg->kind == ARG_CHAR2 ? 8 : 4;
        const void *src = arg->kind == ARG_STRUCT ? arg->data
                        : arg->kind == ARG_INT ? (const void *)&arg->i
                        : arg->kind == ARG_FLOAT ? (const void *)&arg->f
                        : arg->kind == ARG_INT2 || arg->kind == ARG_INT4 || arg->kind == ARG_CHAR2
                          ? (const void *)arg->iv
                        : (const void *)arg->v;
        if(p->offset < 0 || p->offset + sz > sizeof(push)) die("%s: bad offset for %s", t->kernel, arg->name);
        memcpy(push + p->offset, src, sz);
        if(p->offset + (int)sz > push_size) push_size = p->offset + sz;
        break;
      }
    }
  }
  push_size = (push_size + 3) & ~3;

  const VkDescriptorSetLayoutCreateInfo dslci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                                  .bindingCount = nbind, .pBindings = lb };
  VkDescriptorSetLayout dsl;
  CK(vkCreateDescriptorSetLayout(e->dev, &dslci, NULL, &dsl));
  const VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size };
  const VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                            .setLayoutCount = 1, .pSetLayouts = &dsl,
                                            .pushConstantRangeCount = push_size ? 1 : 0, .pPushConstantRanges = &pcr };
  VkPipelineLayout layout;
  CK(vkCreatePipelineLayout(e->dev, &plci, NULL, &layout));
  const VkShaderModuleCreateInfo smci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = len,
                                          .pCode = code };
  VkShaderModule sm;
  CK(vkCreateShaderModule(e->dev, &smci, NULL, &sm));
  const VkComputePipelineCreateInfo cpci = {
    .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT,
               .module = sm, .pName = entry },
    .layout = layout };
  VkPipeline pipe;
  CK(vkCreateComputePipelines(e->dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));

  const VkDescriptorPoolSize ps[3] = { { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, MAX_ARGS },
                                       { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, MAX_ARGS },
                                       { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, MAX_ARGS } };
  const VkDescriptorPoolCreateInfo dpci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1,
                                            .poolSizeCount = 3, .pPoolSizes = ps };
  VkDescriptorPool dp;
  CK(vkCreateDescriptorPool(e->dev, &dpci, NULL, &dp));
  const VkDescriptorSetAllocateInfo dsai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                             .descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &dsl };
  VkDescriptorSet ds;
  CK(vkAllocateDescriptorSets(e->dev, &dsai, &ds));
  for(int b = 0; b < nbind; b++) wr[b].dstSet = ds;
  vkUpdateDescriptorSets(e->dev, nbind, wr, 0, NULL);

  const VkCommandBufferAllocateInfo cbai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                             .commandPool = e->pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                             .commandBufferCount = 1 };
  VkCommandBuffer cb;
  CK(vkAllocateCommandBuffers(e->dev, &cbai, &cb));
  const VkCommandBufferBeginInfo cbbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
  CK(vkBeginCommandBuffer(cb, &cbbi));
  for(int a = 0; a < t->nargs; a++)
  {
    if(!have_img[a]) continue;
    const VkBufferImageCopy cpy = { .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                                    .imageExtent = { t->args[a].w, t->args[a].h, 1 } };
    vk_barrier(cb, img[a].img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    vkCmdCopyBufferToImage(cb, stage[a].buf, img[a].img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cpy);
    vk_barrier(cb, img[a].img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
               t->args[a].kind == ARG_IMAGE_OUT ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  }
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
  vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, NULL);
  if(push_size) vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size, push);
  // the kernels are written for 16x16 work groups, see [numthreads] in the .slang files
  vkCmdDispatch(cb, (t->gw + 15) / 16, (t->gh + 15) / 16, 1);
  for(int a = 0; a < t->nargs; a++)
  {
    if(!have_img[a] || t->args[a].kind != ARG_IMAGE_OUT) continue;
    const VkBufferImageCopy cpy = { .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                                    .imageExtent = { t->args[a].w, t->args[a].h, 1 } };
    vk_barrier(cb, img[a].img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    vkCmdCopyImageToBuffer(cb, img[a].img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, stage[a].buf, 1, &cpy);
  }
  CK(vkEndCommandBuffer(cb));
  const VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb };
  CK(vkQueueSubmit(e->queue, 1, &si, VK_NULL_HANDLE));
  CK(vkQueueWaitIdle(e->queue));
  for(int a = 0; a < t->nargs; a++)
  {
    const arg_t *arg = &t->args[a];
    if(arg->kind == ARG_IMAGE_OUT && have_img[a]) memcpy(outs[a], stage[a].map, out_size(arg));
    else if(arg->kind == ARG_BUFFER_OUT && have_buf[a]) memcpy(outs[a], buf[a].map, out_size(arg));
  }

  vkFreeCommandBuffers(e->dev, e->pool, 1, &cb);
  vkDestroyDescriptorPool(e->dev, dp, NULL);
  vkDestroyPipeline(e->dev, pipe, NULL);
  vkDestroyShaderModule(e->dev, sm, NULL);
  vkDestroyPipelineLayout(e->dev, layout, NULL);
  vkDestroyDescriptorSetLayout(e->dev, dsl, NULL);
  for(int a = 0; a < t->nargs; a++)
  {
    if(have_img[a])
    {
      vk_image_free(e, &img[a]);
      vk_buffer_free(e, &stage[a]);
    }
    if(have_buf[a]) vk_buffer_free(e, &buf[a]);
  }
  free(code);
}

#endif // !PARITY_METAL

// -------------------------------------------------------------------------------------------------

// relative error with an absolute floor, so values near zero do not dominate
#define TOLERANCE 1e-4

#ifdef PARITY_METAL
#define GPU_API "metal"
#else
#define GPU_API "vulkan"
#endif

int main(int argc, char **argv)
{
  if(argc < 3) die("usage: %s <dir with .cl files> <dir with the Slang kernels> [program]", argv[0]);
  if(sizeof(profile_info_t) != 160) die("profile_info_t is %zu bytes, the kernels expect 160", sizeof(profile_info_t));

  make_inputs();
  static test_t tests[1024];
  const int ntests = make_tests(tests);

  char cl_name[256], gpu_name[256];
  cl_env_t cl;
  cl_init(&cl, cl_name, sizeof(cl_name));
#ifdef PARITY_METAL
  mtl_init(gpu_name, sizeof(gpu_name));
#else
  vk_env_t vk;
  vk_init(&vk, gpu_name, sizeof(gpu_name));
#endif
  printf("comparing OpenCL on %s with " GPU_API " on %s\n", cl_name, gpu_name);

  int failed = 0, ran = 0;
  for(int k = 0; k < ntests; k++)
  {
    test_t *t = &tests[k];
    if(argc > 3 && strcmp(argv[3], t->program)) continue;
    void *out_cl[MAX_ARGS] = { 0 }, *out_vk[MAX_ARGS] = { 0 };
    for(int a = 0; a < t->nargs; a++)
      if(out_size(&t->args[a]))
      {
        out_cl[a] = calloc(1, out_size(&t->args[a]));
        out_vk[a] = calloc(1, out_size(&t->args[a]));
      }
    cl_run(&cl, argv[1], t, out_cl);
#ifdef PARITY_METAL
    mtl_run(argv[2], t, out_vk);
#else
    vk_run(&vk, argv[2], t, out_vk);
#endif

    // every output is compared, and the test fails on the worst of them
    const double tolerance = t->tolerance > 0.0 ? t->tolerance : TOLERANCE;
    double max_err = 0.0, max_abs = 0.0;
    size_t worst = 0, nan_mismatch = 0, byte_mismatch = 0, over = 0;
    int worst_arg = -1;
    for(int a = 0; a < t->nargs; a++)
    {
      const arg_t *arg = &t->args[a];
      if(!out_cl[a]) continue;
      if(arg->bytes)
      {
        for(size_t i = 0; i < arg->size; i++)
          byte_mismatch += ((uint8_t *)out_cl[a])[i] != ((uint8_t *)out_vk[a])[i];
        continue;
      }
      const float *fc = out_cl[a], *fv = out_vk[a];
      for(size_t i = 0; i < out_size(arg) / sizeof(float); i++)
      {
        if(arg->ints)
        {
          const double c = ((int32_t *)out_cl[a])[i], v = ((int32_t *)out_vk[a])[i];
          const double err = fabs(c - v) / fmax(1.0, fabs(c));
          if(err > max_err || worst_arg < 0)
          {
            max_err = fmax(err, max_err);
            worst = i;
            worst_arg = a;
          }
          max_abs = fmax(max_abs, fabs(c - v));
          if(err > tolerance) over++;
          continue;
        }
        if(isnan(fc[i]) || isnan(fv[i]))
        {
          if(isnan(fc[i]) != isnan(fv[i])) nan_mismatch++;
          continue;
        }
        const double d = fabs((double)fc[i] - (double)fv[i]);
        const double err = d / fmax(1.0, fabs((double)fc[i]));
        if(err > max_err || worst_arg < 0)
        {
          max_err = fmax(err, max_err);
          worst = i;
          worst_arg = a;
        }
        if(d > max_abs) max_abs = d;
        if(err > tolerance) over++;
      }
    }

    const int ok = over <= (size_t)t->outliers && nan_mismatch == 0 && byte_mismatch == 0;
    printf("%-4s %-45s max error %.2e  max abs %.2e  nan mismatches %zu", ok ? "ok" : "FAIL", t->label, max_err,
           max_abs, nan_mismatch);
    if(over) printf("  %zu past the tolerance", over);
    if(byte_mismatch) printf("  byte mismatches %zu", byte_mismatch);
    if(!ok && worst_arg >= 0 && t->args[worst_arg].ints)
      printf("  (%s element %zu: opencl %d, " GPU_API " %d)", t->args[worst_arg].name, worst,
             ((int32_t *)out_cl[worst_arg])[worst], ((int32_t *)out_vk[worst_arg])[worst]);
    else if(!ok && worst_arg >= 0)
    {
      const arg_t *arg = &t->args[worst_arg];
      const int ch = arg->kind == ARG_IMAGE_OUT ? arg->channels : 1;
      printf("  (%s element %zu channel %zu: opencl %g, " GPU_API " %g)", arg->name, worst / ch, worst % ch,
             ((float *)out_cl[worst_arg])[worst], ((float *)out_vk[worst_arg])[worst]);
    }
    printf("\n");
    failed += !ok;
    ran++;
    for(int a = 0; a < t->nargs; a++)
    {
      free(out_cl[a]);
      free(out_vk[a]);
    }
  }
  printf("%d of %d kernel runs match\n", ran - failed, ran);
  return failed ? 1 : 0;
}
