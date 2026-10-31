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

G_BEGIN_DECLS

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
#ifdef __cplusplus
#define _DT_SLANG_ARG(...) (dt_slang_arg_t{ __VA_ARGS__ })
#else
#define _DT_SLANG_ARG(...) ((dt_slang_arg_t){ __VA_ARGS__ })
#endif
#define SLIMG_IN(p, w, h, c) _DT_SLANG_ARG(DT_SLANG_ARG_IMAGE_IN, (p), 0, (w), (h), (c))
#define SLIMG_OUT(p, w, h, c) _DT_SLANG_ARG(DT_SLANG_ARG_IMAGE_OUT, (p), 0, (w), (h), (c))
#define SLBUF(p, s) _DT_SLANG_ARG(DT_SLANG_ARG_BUFFER, (p), (s), 0, 0, 0)
#define SLARG(v) _DT_SLANG_ARG(DT_SLANG_ARG_VALUE, &(v), sizeof(v), 0, 0, 0)
// a device image, as the kernel's input or output
#define SLIMG_DEV(img) _DT_SLANG_ARG(DT_SLANG_ARG_IMAGE_DEV, (img), 0, 0, 0, 0)
// a device buffer, which kernels can read and write
#define SLBUF_DEV(buf) _DT_SLANG_ARG(DT_SLANG_ARG_BUFFER_DEV, (buf), 0, 0, 0, 0)

#ifdef HAVE_SLANG_GPU

// TRUE if the slang_gpu config key is set, a device could be set up and
// device errors haven't stopped the kernels for this session
gboolean dt_slang_gpu_enabled(void);

// failed pixelpipe runs after which the kernels stay off for the session, as
// with DT_OPENCL_MAX_ERRORS. A lost device stops them at once.
#define DT_SLANG_GPU_MAX_ERRORS 5

// device errors so far: failed submissions, kernels the device couldn't run,
// a lost device. Failing to allocate is not one; the module falls back to its
// CPU code. The pixelpipe restarts on the CPU when this grows during a run,
// since kernels that ran before the error may have left bad results.
int dt_slang_gpu_errors(void);
// counts a pixelpipe run that restarted on the CPU after a device error
void dt_slang_gpu_run_failed(void);

// for the backends: counts a device error, and stops the kernels for the
// session if the device can't be used any more
void dt_slang_gpu_report_error(const char *what, const gboolean fatal);
gboolean dt_slang_gpu_stopped(void);

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
// an image in host's memory, which kernels then write directly, so reading
// it into host only waits for them. host must stay valid until the next read
// or dt_slang_gpu_wait(), and freeing the image does not wait. Returns NULL
// where the device can't use host memory, which is always the case on Vulkan.
dt_slang_gpu_image_t *dt_slang_gpu_image_alias(const int width,
                                               const int height,
                                               const int channels,
                                               void *host);
void dt_slang_gpu_image_free(dt_slang_gpu_image_t *img);
int dt_slang_gpu_image_width(const dt_slang_gpu_image_t *img);
int dt_slang_gpu_image_height(const dt_slang_gpu_image_t *img);
// copy between host memory and the whole image. Both return 0 on success, and
// a read waits for the kernels writing the image to finish.
int dt_slang_gpu_image_read(dt_slang_gpu_image_t *img, void *host);
int dt_slang_gpu_image_write(dt_slang_gpu_image_t *img, const void *host);
// waits for every kernel that has been started
void dt_slang_gpu_wait(void);
// frees the images and buffers kept for reuse, on a GPU with memory of its
// own: OpenCL's check whether a module fits doesn't see them. Images still
// in use by kernels are freed once those finish.
void dt_slang_gpu_release_pools(void);

// the device memory a module may use now, in bytes, counting the pooled
// images and buffers that would be freed for it, and the largest image
// width or height. The pixelpipe tiles a module that needs more. The
// slang_gpu_memory config key (MB) lowers the memory, to test tiling.
void dt_slang_gpu_memory_info(size_t *available, int *max_dim);

// a buffer of size bytes on the device, for the intermediate results of a
// chain of kernels. It is filled from host if that is not NULL, and zeroed
// otherwise. Returns NULL on failure.
dt_slang_gpu_buffer_t *dt_slang_gpu_buffer_new(const size_t size, const void *host);
// the same without the clearing, for a temporary that the kernels write in
// full before they read it. Its contents are whatever its last user left
dt_slang_gpu_buffer_t *dt_slang_gpu_buffer_new_uninit(const size_t size);
void dt_slang_gpu_buffer_free(dt_slang_gpu_buffer_t *buf);
// copies the first size bytes to host, after the kernels writing the buffer
// have finished. Returns 0 on success.
int dt_slang_gpu_buffer_read(dt_slang_gpu_buffer_t *buf, void *host, const size_t size);

#ifdef __cplusplus
// C++ has no compound literals, so the arguments go in a list
G_END_DECLS
#include <initializer_list>
static inline int _dt_slang_gpu_run_2d_list(const char *program, const char *kernel, const int width,
                                            const int height, std::initializer_list<dt_slang_arg_t> args)
{
  return dt_slang_gpu_run_2d(program, kernel, width, height, (int)args.size(), args.begin());
}
G_BEGIN_DECLS
#define dt_slang_gpu_run_2d_args(program, kernel, width, height, ...)                    \
  _dt_slang_gpu_run_2d_list((program), (kernel), (width), (height), { __VA_ARGS__ })
#else
#define dt_slang_gpu_run_2d_args(program, kernel, width, height, ...)                    \
  dt_slang_gpu_run_2d((program), (kernel), (width), (height),                           \
                      sizeof((dt_slang_arg_t[]){ __VA_ARGS__ }) / sizeof(dt_slang_arg_t),  \
                      (dt_slang_arg_t[]){ __VA_ARGS__ })
#endif

// a kernel that walks whole rows or columns needs only one work item per
// line. The kernels are written for 16x16 work groups, so the n items are
// laid out 16 wide and the kernel takes its index as gid.y * 16 + gid.x;
// a plain n x 1 dispatch would leave 15 of every 16 threads idle
#define dt_slang_gpu_run_1d_args(program, kernel, n, ...)                                \
  dt_slang_gpu_run_2d_args((program), (kernel), 16, ((n) + 15) / 16, __VA_ARGS__)

#else

static inline gboolean dt_slang_gpu_enabled(void) { return FALSE; }
static inline void dt_slang_gpu_cleanup(void) {}
static inline void dt_slang_gpu_prepare(void) {}
static inline void dt_slang_gpu_release_pools(void) {}

#endif

G_END_DECLS
