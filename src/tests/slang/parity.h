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

// the test cases and helpers of parity.c, shared with the Metal backend in
// parity_metal.m, which has to be Objective-C

#pragma once

#include <stddef.h>

#define MAX_ARGS 48

typedef enum arg_kind_t
{
  ARG_IMAGE_IN,
  ARG_IMAGE_OUT,
  ARG_INT,
  ARG_FLOAT,
  ARG_FLOAT4,
  ARG_INT2,
  ARG_INT4,
  ARG_STRUCT,     // passed by value, from data and size
  ARG_FLOAT2,
  ARG_BUFFER,
  ARG_BUFFER_OUT, // read and written by the kernel, and compared
  ARG_LOCAL,      // OpenCL local memory, which the Slang kernels do without
  ARG_RAW_U16,    // uint16 raw data: an unsigned integer image in OpenCL, a buffer of value pairs in Slang
  ARG_FLOAT4_BUF, // a float4 value in OpenCL, a buffer holding it in Slang
  ARG_CHAR2,      // a char2 in OpenCL, an int2 in Slang
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
  int iv[4];
  const void *data; // buffers, and the initial contents of ARG_BUFFER_OUT (zeros if NULL)
  size_t size;
  int bytes;        // ARG_BUFFER_OUT holds bytes, compared exactly
  int ints;         // ARG_BUFFER_OUT holds int counts, compared as numbers
  int slang_only;   // a Slang parameter the OpenCL kernel does not have
} arg_t;

typedef struct test_t
{
  const char *label;
  const char *program; // .cl file, also the .spv file name
  const char *kernel;
  const char *slang_kernel; // the Slang entry point, where it differs from the OpenCL kernel
  double tolerance; // 0 for the default
  int gw, gh;       // work items, W x H if 0
  int clgw, clgh, cllx, clly; // OpenCL global and local sizes, if they differ from the above
  int outliers;     // values allowed past the tolerance, where the math is ill-conditioned
  int nargs;
  arg_t args[MAX_ARGS];
} test_t;

void die(const char *fmt, ...) __attribute__((format(printf, 1, 2), noreturn));
char *slurp(const char *path, size_t *len);
// bytes a kernel writes to an output argument, 0 for inputs
size_t out_size(const arg_t *arg);
// what an output starts with: zeros, or the initial contents of ARG_BUFFER_OUT
const void *initial_data(const arg_t *arg);

// the exit code for ctest's SKIP_RETURN_CODE, when a machine has no device to compare
#define SKIP 77

// 1 if wanted, the value of a DT_PARITY_*_DEVICE variable, names the
// device with this index or a part of its name, ignoring case
int device_wanted(const char *wanted, const int index, const char *name);

#ifdef PARITY_METAL
// writes the Metal device's name to name, and exits with SKIP if there is none
void mtl_init(char *name, const size_t size);
// runs the kernel from <dir>/<program>.<entry>.metallib
void mtl_run(const char *dir, const test_t *t, void **outs);
#endif
