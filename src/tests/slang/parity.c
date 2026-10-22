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
// through OpenCL and once from the Slang-built SPIR-V through Vulkan, and
// compares the output images. It needs no GPU: Mesa's lavapipe and PoCL are
// enough, which is how it runs in CI.
//
// usage: darktable-test-slang-parity <dir with .cl files> <dir with .spv files>

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <vulkan/vulkan.h>

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define W 517
#define H 263
#define MAX_ARGS 48

// -------------------------------------------------------------------------------------------------
// test cases

typedef enum arg_kind_t
{
  ARG_IMAGE_IN,
  ARG_IMAGE_OUT,
  ARG_INT,
  ARG_FLOAT,
  ARG_FLOAT4,
  ARG_BUFFER,
  ARG_BUFFER_OUT, // read and written by the kernel, and compared
  ARG_LOCAL,      // OpenCL local memory, which the Slang kernels do without
} arg_kind_t;

typedef struct arg_t
{
  arg_kind_t kind;
  const char *name;   // parameter name in the Slang entry point
  int w, h, channels; // images
  const float *pixels;
  int i;
  float f;
  float v[4];
  const void *data; // buffers, and the initial contents of ARG_BUFFER_OUT (zeros if NULL)
  size_t size;
  int bytes;        // ARG_BUFFER_OUT holds bytes, compared exactly
} arg_t;

typedef struct test_t
{
  const char *label;
  const char *program; // .cl file, also the .spv file name
  const char *kernel;
  double tolerance; // 0 for the default
  int gw, gh;       // work items, W x H if 0
  int clgw, clgh, cllx, clly; // OpenCL global and local sizes, if they differ from the above
  int skip_alpha;   // the OpenCL kernel leaves the alpha of 4-channel outputs undefined
  int outliers;     // values allowed past the tolerance, where the math is ill-conditioned
  int nargs;
  arg_t args[MAX_ARGS];
} test_t;

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

  // a sparse mask of 3x3 cells for the opposed highlights, for three colors.
  // highlights_chroma reads a cell past the end of a plane for the last
  // column, as the host code allocates it, so there is some room after.
  const int msize = ((W / 3 + 3) / 4 * 4) * ((H / 3 + 3) / 4 * 4);
  mask_bytes = calloc(1, 3 * msize + 64);
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

#define IMG_IN(n, px) { .kind = ARG_IMAGE_IN, .name = n, .w = W, .h = H, .channels = 4, .pixels = px }
#define LUT(n, px, lw, lh) { .kind = ARG_IMAGE_IN, .name = n, .w = lw, .h = lh, .channels = 1, .pixels = px }
#define IMG_OUT(n) { .kind = ARG_IMAGE_OUT, .name = n, .w = W, .h = H, .channels = 4 }
#define RAW_IN(n, px) { .kind = ARG_IMAGE_IN, .name = n, .w = W, .h = H, .channels = 1, .pixels = px }
#define RAW_OUT(n) { .kind = ARG_IMAGE_OUT, .name = n, .w = W, .h = H, .channels = 1 }
#define BUFFER_OUT(n, ptr, sz) { .kind = ARG_BUFFER_OUT, .name = n, .data = ptr, .size = sz }
#define BYTES_OUT(n, sz) { .kind = ARG_BUFFER_OUT, .name = n, .size = sz, .bytes = 1 }
#define LOCAL(n, sz) { .kind = ARG_LOCAL, .name = n, .size = sz }
#define INT(n, val) { .kind = ARG_INT, .name = n, .i = val }
#define FLOAT(n, val) { .kind = ARG_FLOAT, .name = n, .f = val }
#define FLOAT4(n, a, b, c, d) { .kind = ARG_FLOAT4, .name = n, .v = { a, b, c, d } }
#define BUFFER(n, ptr, sz) { .kind = ARG_BUFFER, .name = n, .data = ptr, .size = sz }

static const float levels[9] = { 0.05f, 0.45f, 0.9f, 0.0f, 0.5f, 1.0f, 0.1f, 0.4f, 0.95f };
static const float inv_gamma[3] = { 1.1f, 0.9f, 1.3f };
static const float coeffs_linear[3] = { 1.0f, 1.0f, 0.6f };
static const float coeffs_off[3] = { -1.0f, 0.0f, 0.0f };
static const float matrix3x3[9] = { 0.9f, 0.05f, 0.05f, 0.1f, 0.8f, 0.1f, 0.0f, 0.2f, 0.8f };

static int make_tests(test_t *t)
{
  int n = 0;
#define TEST_GRID(lbl, prog, kern, tol, w, h, ...)                                        \
  {                                                                                      \
    const arg_t a[] = { __VA_ARGS__ };                                                   \
    t[n] = (test_t){ .label = lbl, .program = prog, .kernel = kern, .tolerance = tol,    \
                     .gw = w, .gh = h, .nargs = sizeof(a) / sizeof(a[0]) };             \
    memcpy(t[n].args, a, sizeof(a));                                                     \
    n++;                                                                                 \
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
      // where Metal lands up to 2.5e-4 (relative) from OpenCL
      TEST_TOL(cm_labels[k][mode], "channelmixer", cm_kernels[k], mode == 1 ? 1e-3 : 0.0,
           IMG_IN("in", image_in), IMG_OUT("out"), INT("width", W), INT("height", H),
           BUFFER("RGB_to_XYZ", cm_rgb_to_xyz, sizeof(cm_rgb_to_xyz)),
           BUFFER("XYZ_to_RGB", cm_xyz_to_rgb, sizeof(cm_xyz_to_rgb)), BUFFER("MIX", cm_mix, sizeof(cm_mix)),
           FLOAT4("illuminant", 0.97f, 1.01f, 0.78f, 0.0f), FLOAT4("saturation", 0.1f, -0.05f, 0.08f, 0.0f),
           FLOAT4("lightness", 0.02f, 0.05f, -0.03f, 0.0f), FLOAT4("grey", 0.3f, 0.6f, 0.1f, 0.0f),
           FLOAT("p", 1.02f), FLOAT("gamut", mode == 0 ? 0.5f : 0.0f), INT("clip", mode == 0),
           INT("apply_grey", mode == 2), INT("version", mode == 0 ? 2 : mode == 1 ? 0 : 1));
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
  static const char *bayer_labels[2][6] = {
    { "whitebalance_1f rggb", "highlights_1f_clip rggb", "highlights_initmask rggb", "highlights_chroma rggb",
      "highlights_opposed rggb", "border_interpolate rggb" },
    { "whitebalance_1f bggr", "highlights_1f_clip bggr", "highlights_initmask bggr", "highlights_chroma bggr",
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
    TEST_GRID(bayer_labels[b][3], "basic", "highlights_chroma", 0.0, H, 1,
         RAW_IN("in", image_raw), BUFFER("mask", mask_bytes, 3 * msize + 64),
         BUFFER_OUT("accu", NULL, sizeof(float) * 8 * H), INT("width", W), INT("height", H),
         INT("msize", msize), INT("mwidth", mwidth), INT("filters", bayer[b]),
         BUFFER("xtrans", xtrans, sizeof(xtrans)), BUFFER("clips", hl_clips, sizeof(hl_clips)),
         BUFFER("correction", hl_correction, sizeof(hl_correction)));
    TEST(bayer_labels[b][4], "basic", "highlights_opposed",
         RAW_IN("in", image_raw), RAW_OUT("out"), INT("owidth", W), INT("oheight", H),
         INT("iwidth", W), INT("iheight", H), INT("dx", 0), INT("dy", 0), INT("filters", bayer[b]),
         BUFFER("xtrans", xtrans, sizeof(xtrans)), BUFFER("clips", hl_clips, sizeof(hl_clips)),
         BUFFER("chroma", hl_chroma, sizeof(hl_chroma)), BUFFER("correction", hl_correction, sizeof(hl_correction)));
    TEST(bayer_labels[b][5], "demosaic_ppg", "border_interpolate",
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
    // the OpenCL poissonian noise reads uninitialized alpha values, the Slang
    // one starts them at zero as the CPU code does
    for(int k = 0; k < 3; k++)
    {
      TEST(fm_noise[k], "filmic", "filmic_inpaint_noise",
           IMG_IN("in", image_in), RAW_IN("mask", rcd_dir), IMG_OUT("out"), INT("width", W), INT("height", H),
           FLOAT("noise_level", 0.2f), FLOAT("threshold", 3.0f), INT("noise_distribution", k));
      t[n - 1].skip_alpha = k == 2;
    }
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

#undef TEST
#undef TEST_TOL
#undef TEST_GRID
  return n;
}

// -------------------------------------------------------------------------------------------------
// helpers

static void die(const char *fmt, ...)
{
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

static char *slurp(const char *path, size_t *len)
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

// -------------------------------------------------------------------------------------------------
// OpenCL

typedef struct cl_env_t
{
  cl_context ctx;
  cl_command_queue queue;
  cl_device_id dev;
} cl_env_t;

static void cl_init(cl_env_t *e)
{
  cl_platform_id platform;
  cl_int err;
  CK(clGetPlatformIDs(1, &platform, NULL));
  CK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &e->dev, NULL));
  e->ctx = clCreateContext(NULL, 1, &e->dev, NULL, NULL, &err);
  CK(err);
  e->queue = clCreateCommandQueue(e->ctx, e->dev, 0, &err);
  CK(err);
}

// bytes a kernel writes to an output argument, 0 for inputs
static size_t out_size(const arg_t *arg)
{
  if(arg->kind == ARG_IMAGE_OUT) return sizeof(float) * arg->w * arg->h * arg->channels;
  if(arg->kind == ARG_BUFFER_OUT) return arg->size;
  return 0;
}

// outputs start zeroed, as some kernels write only part of them
static const void *initial_data(const arg_t *arg)
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
  for(int a = 0; a < t->nargs; a++)
  {
    const arg_t *arg = &t->args[a];
    const cl_image_format fmt = { arg->channels == 1 ? CL_R : CL_RGBA, CL_FLOAT };
    const cl_image_desc desc = { .image_type = CL_MEM_OBJECT_IMAGE2D, .image_width = arg->w, .image_height = arg->h };
    switch(arg->kind)
    {
      case ARG_IMAGE_IN:
        mem[a] = clCreateImage(e->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &fmt, &desc, (void *)arg->pixels, &err);
        CK(err);
        CK(clSetKernelArg(k, a, sizeof(cl_mem), &mem[a]));
        break;
      case ARG_IMAGE_OUT:
        mem[a] = clCreateImage(e->ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, &fmt, &desc,
                               (void *)initial_data(arg), &err);
        CK(err);
        CK(clSetKernelArg(k, a, sizeof(cl_mem), &mem[a]));
        break;
      case ARG_BUFFER_OUT:
        mem[a] = clCreateBuffer(e->ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, arg->size,
                                (void *)initial_data(arg), &err);
        CK(err);
        CK(clSetKernelArg(k, a, sizeof(cl_mem), &mem[a]));
        break;
      case ARG_LOCAL:
        CK(clSetKernelArg(k, a, arg->size, NULL));
        break;
      case ARG_BUFFER:
        mem[a] = clCreateBuffer(e->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, arg->size, (void *)arg->data, &err);
        CK(err);
        CK(clSetKernelArg(k, a, sizeof(cl_mem), &mem[a]));
        break;
      case ARG_INT:
        CK(clSetKernelArg(k, a, sizeof(int), &arg->i));
        break;
      case ARG_FLOAT:
        CK(clSetKernelArg(k, a, sizeof(float), &arg->f));
        break;
      case ARG_FLOAT4:
        CK(clSetKernelArg(k, a, 4 * sizeof(float), arg->v));
        break;
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
    if(arg->kind == ARG_IMAGE_OUT)
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

static void vk_init(vk_env_t *e)
{
  const VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_2 };
  const VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app };
  CK(vkCreateInstance(&ici, NULL, &e->instance));
  uint32_t n = 1;
  vkEnumeratePhysicalDevices(e->instance, &n, &e->pdev);
  if(n == 0) die("no Vulkan device");

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
  spv_reflect(code, len / 4, t->kernel, &info);

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
        buf[a] = vk_buffer(e, arg->size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        have_buf[a] = 1;
        memcpy(buf[a].map, arg->kind == ARG_BUFFER ? arg->data : initial_data(arg), arg->size);
        lb[nbind] = (VkDescriptorSetLayoutBinding){ p->binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                                    VK_SHADER_STAGE_COMPUTE_BIT, NULL };
        dbi[nbind] = (VkDescriptorBufferInfo){ buf[a].buf, 0, arg->size };
        wr[nbind] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = p->binding,
                                            .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                            .pBufferInfo = &dbi[nbind] };
        nbind++;
        break;
      case ARG_INT:
      case ARG_FLOAT:
      case ARG_FLOAT4:
      {
        const size_t sz = arg->kind == ARG_FLOAT4 ? 16 : 4;
        const void *src = arg->kind == ARG_INT ? (const void *)&arg->i
                        : arg->kind == ARG_FLOAT ? (const void *)&arg->f : (const void *)arg->v;
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
               .module = sm, .pName = t->kernel },
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

// -------------------------------------------------------------------------------------------------

// relative error with an absolute floor, so values near zero do not dominate
#define TOLERANCE 1e-4

int main(int argc, char **argv)
{
  if(argc < 3) die("usage: %s <dir with .cl files> <dir with .spv files> [kernel]", argv[0]);
  if(sizeof(profile_info_t) != 160) die("profile_info_t is %zu bytes, the kernels expect 160", sizeof(profile_info_t));

  make_inputs();
  static test_t tests[160];
  const int ntests = make_tests(tests);

  cl_env_t cl;
  vk_env_t vk;
  cl_init(&cl);
  vk_init(&vk);

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
    vk_run(&vk, argv[2], t, out_vk);

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
        if(t->skip_alpha && arg->channels == 4 && i % 4 == 3) continue;
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
    if(!ok && worst_arg >= 0)
    {
      const arg_t *arg = &t->args[worst_arg];
      const int ch = arg->kind == ARG_IMAGE_OUT ? arg->channels : 1;
      printf("  (%s element %zu channel %zu: opencl %g, vulkan %g)", arg->name, worst / ch, worst % ch,
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
