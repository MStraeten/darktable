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

#pragma once

// Runs the Slang-compiled kernels from data/kernels/slang on a GPU, through
// Vulkan or Metal. This is a proof of concept for #22732. Host buffers passed
// with SLIMG_IN/SLIMG_OUT/SLBUF are copied to the device and back for every
// call; a dt_slang_gpu_image_t stays on the device, which is how the pixelpipe
// hands images from one module's process_slang() to the next.

#include <glib.h>
#include <stddef.h>

typedef enum dt_slang_arg_kind_t
{
  DT_SLANG_ARG_IMAGE_IN,
  DT_SLANG_ARG_IMAGE_OUT,
  DT_SLANG_ARG_BUFFER,
  DT_SLANG_ARG_VALUE,
  DT_SLANG_ARG_IMAGE_DEV,
  DT_SLANG_ARG_BUFFER_DEV,
} dt_slang_arg_kind_t;

typedef struct dt_slang_gpu_image_t dt_slang_gpu_image_t;
typedef struct dt_slang_gpu_buffer_t dt_slang_gpu_buffer_t;

typedef struct dt_slang_arg_t
{
  dt_slang_arg_kind_t kind;
  const void *ptr; // host data, written to for DT_SLANG_ARG_IMAGE_OUT, or the device image or buffer
  size_t size;     // bytes, for buffers and values
  int width, height, channels;
} dt_slang_arg_t;

// arguments are given in the order of the kernel's parameters, like CLARG()
// for OpenCL. Images are float, with 1 or 4 channels.
#define SLIMG_IN(p, w, h, c) ((dt_slang_arg_t){ DT_SLANG_ARG_IMAGE_IN, (p), 0, (w), (h), (c) })
#define SLIMG_OUT(p, w, h, c) ((dt_slang_arg_t){ DT_SLANG_ARG_IMAGE_OUT, (p), 0, (w), (h), (c) })
#define SLBUF(p, s) ((dt_slang_arg_t){ DT_SLANG_ARG_BUFFER, (p), (s), 0, 0, 0 })
#define SLARG(v) ((dt_slang_arg_t){ DT_SLANG_ARG_VALUE, &(v), sizeof(v), 0, 0, 0 })
// a device image, as the kernel's input or output
#define SLIMG_DEV(img) ((dt_slang_arg_t){ DT_SLANG_ARG_IMAGE_DEV, (img), 0, 0, 0, 0 })
// a device buffer, which kernels can read and write
#define SLBUF_DEV(buf) ((dt_slang_arg_t){ DT_SLANG_ARG_BUFFER_DEV, (buf), 0, 0, 0, 0 })

#ifdef HAVE_SLANG_GPU

// TRUE if the slang_gpu config key is set and a device could be set up
gboolean dt_slang_gpu_enabled(void);

// runs kernel from data/kernels/slang/<program> over width x height work items
// and returns 0 on success. Any failure leaves the output undefined, and the
// caller falls back to its CPU code.
int dt_slang_gpu_run_2d(const char *program,
                        const char *kernel,
                        const int width,
                        const int height,
                        const int nargs,
                        const dt_slang_arg_t *args);

void dt_slang_gpu_cleanup(void);

// sets up the device and loads every kernel in a background thread, so that
// the first module doesn't wait for that. Does nothing unless slang_gpu is set.
void dt_slang_gpu_prepare(void);

// a float image with 1 or 4 channels on the device, filled from host if that
// is not NULL. Returns NULL on failure.
dt_slang_gpu_image_t *dt_slang_gpu_image_new(const int width,
                                             const int height,
                                             const int channels,
                                             const void *host);
// like dt_slang_gpu_image_new(), but the image may use host's memory instead
// of a copy: host must stay valid and unchanged until the image is freed, and
// dt_slang_gpu_image_free() waits for the kernels still using it
dt_slang_gpu_image_t *dt_slang_gpu_image_wrap(const int width,
                                              const int height,
                                              const int channels,
                                              const void *host);
void dt_slang_gpu_image_free(dt_slang_gpu_image_t *img);
int dt_slang_gpu_image_width(const dt_slang_gpu_image_t *img);
int dt_slang_gpu_image_height(const dt_slang_gpu_image_t *img);
// copy between host memory and the whole image. Both return 0 on success, and
// a read waits for the kernels writing the image to finish.
int dt_slang_gpu_image_read(dt_slang_gpu_image_t *img, void *host);
int dt_slang_gpu_image_write(dt_slang_gpu_image_t *img, const void *host);

// a buffer of size bytes on the device, for the intermediate results of a
// chain of kernels. It is filled from host if that is not NULL, and zeroed
// otherwise. Returns NULL on failure.
dt_slang_gpu_buffer_t *dt_slang_gpu_buffer_new(const size_t size, const void *host);
void dt_slang_gpu_buffer_free(dt_slang_gpu_buffer_t *buf);
// copies the first size bytes to host, after the kernels writing the buffer
// have finished. Returns 0 on success.
int dt_slang_gpu_buffer_read(dt_slang_gpu_buffer_t *buf, void *host, const size_t size);

#define dt_slang_gpu_run_2d_args(program, kernel, width, height, ...)                    \
  dt_slang_gpu_run_2d((program), (kernel), (width), (height),                           \
                      sizeof((dt_slang_arg_t[]){ __VA_ARGS__ }) / sizeof(dt_slang_arg_t),  \
                      (dt_slang_arg_t[]){ __VA_ARGS__ })

#else

static inline gboolean dt_slang_gpu_enabled(void) { return FALSE; }
static inline void dt_slang_gpu_cleanup(void) {}
static inline void dt_slang_gpu_prepare(void) {}

#endif
