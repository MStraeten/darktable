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

// Metal backend of slang_gpu.h, built with ARC

#include "common/slang_gpu.h"
#include "common/darktable.h"
#include "common/file_location.h"
#include "control/conf.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <malloc/malloc.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#define DT_SLANG_MAX_PARAMS 32
#define DT_SLANG_POOL_SIZE 4
#define DT_SLANG_BUFFER_POOL_SIZE 12

@interface DtSlangKernel : NSObject
{
@public
  id<MTLComputePipelineState> pipeline;
  int ntex;
  NSUInteger tex_index[DT_SLANG_MAX_PARAMS]; // in index order
  gboolean tex_write[DT_SLANG_MAX_PARAMS];
  int nbuf;
  NSUInteger buf_index[DT_SLANG_MAX_PARAMS]; // in index order
  int nvals;
  NSUInteger val_offset[DT_SLANG_MAX_PARAMS]; // in declaration order
  NSUInteger val_space[DT_SLANG_MAX_PARAMS];  // bytes up to the next member
  gboolean has_vals;
  NSUInteger vals_index, vals_size;
}
@end

@implementation DtSlangKernel
@end

typedef struct dt_slang_metal_t
{
  gboolean initialized;
  gboolean usable;
  __strong id<MTLDevice> dev;
  __strong id<MTLCommandQueue> queue;
  __strong NSMutableDictionary<NSString *, NSObject *> *libraries; // program -> MTLLibrary
  __strong NSMutableDictionary<NSString *, DtSlangKernel *> *kernels; // "program:kernel"
  __strong id<MTLCommandBuffer> pending; // committed without waiting
  __strong NSMutableArray *images;        // freed device images, most recent first
  __strong NSMutableArray *buffers;       // freed device buffers, most recent first
  NSUInteger max_image_dim;
} dt_slang_metal_t;

static dt_slang_metal_t _mtl = { 0 };
static GMutex _mtl_lock;
static GThread *_prepare_thread = NULL;

// --------------------------------------------------------------------------------------------
// device

static gboolean _init_device(void)
{
  _mtl.dev = MTLCreateSystemDefaultDevice();
  if(!_mtl.dev)
  {
    dt_print(DT_DEBUG_OPENCL, "[slang_gpu] no Metal device");
    return FALSE;
  }
  _mtl.queue = [_mtl.dev newCommandQueue];
  if(!_mtl.queue)
  {
    dt_print(DT_DEBUG_OPENCL, "[slang_gpu] can't create a command queue on %s", _mtl.dev.name.UTF8String);
    return FALSE;
  }
  // the 2D texture limit of the Metal feature set tables
  _mtl.max_image_dim = [_mtl.dev supportsFamily:MTLGPUFamilyApple3] || [_mtl.dev supportsFamily:MTLGPUFamilyMetal3]
                         ? 16384 : 8192;
  _mtl.libraries = [NSMutableDictionary dictionary];
  _mtl.kernels = [NSMutableDictionary dictionary];
  _mtl.images = [NSMutableArray array];
  _mtl.buffers = [NSMutableArray array];
  dt_print(DT_DEBUG_OPENCL, "[slang_gpu] using Metal device %s%s", _mtl.dev.name.UTF8String,
           _mtl.dev.hasUnifiedMemory ? " (unified memory)" : "");
  return TRUE;
}

gboolean dt_slang_gpu_enabled(void)
{
  if(!dt_conf_get_bool("slang_gpu")) return FALSE;

  g_mutex_lock(&_mtl_lock);
  if(!_mtl.initialized)
  {
    _mtl.initialized = TRUE;
    @autoreleasepool
    {
      _mtl.usable = _init_device();
    }
  }
  const gboolean usable = _mtl.usable;
  g_mutex_unlock(&_mtl_lock);
  return usable;
}

void dt_slang_gpu_cleanup(void)
{
  if(_prepare_thread) g_thread_join(_prepare_thread);
  _prepare_thread = NULL;
  g_mutex_lock(&_mtl_lock);
  if(_mtl.pending) [_mtl.pending waitUntilCompleted];
  _mtl.pending = nil;
  _mtl.images = nil;
  _mtl.buffers = nil;
  _mtl.kernels = nil;
  _mtl.libraries = nil;
  _mtl.queue = nil;
  _mtl.dev = nil;
  _mtl.initialized = _mtl.usable = FALSE;
  g_mutex_unlock(&_mtl_lock);
}

// --------------------------------------------------------------------------------------------
// kernels

// Slang turns a Metal entry point's scalars into one struct in a constant
// buffer, named entryPointParams_<n>, and gives the images and buffers their
// own textures and buffers, in declaration order but numbered across all
// entry points of the file. Textures and buffers have separate index spaces,
// so arguments are matched by kind: images to textures and buffers to buffers,
// each in index order, and values to the struct members in offset order.
static gboolean _is_vals_struct(NSString *name)
{
  if(![name hasPrefix:@"entryPointParams_"]) return FALSE;
  NSString *rest = [name substringFromIndex:17];
  return rest.length > 0
         && [rest rangeOfCharacterFromSet:[[NSCharacterSet decimalDigitCharacterSet] invertedSet]].location
                == NSNotFound;
}

static void _sort_indices(NSUInteger *idx, gboolean *flag, const int n)
{
  for(int a = 0; a < n; a++)
    for(int b = a + 1; b < n; b++)
      if(idx[b] < idx[a])
      {
        const NSUInteger t = idx[a];
        idx[a] = idx[b];
        idx[b] = t;
        if(flag)
        {
          const gboolean f = flag[a];
          flag[a] = flag[b];
          flag[b] = f;
        }
      }
}

static gboolean _reflect(MTLComputePipelineReflection *refl, DtSlangKernel *k)
{
  for(id<MTLBinding> b in refl.bindings)
  {
    if(b.type == MTLBindingTypeTexture)
    {
      if(k->ntex == DT_SLANG_MAX_PARAMS) return FALSE;
      k->tex_index[k->ntex] = b.index;
      k->tex_write[k->ntex] = b.access != MTLBindingAccessReadOnly;
      k->ntex++;
    }
    else if(b.type == MTLBindingTypeBuffer && _is_vals_struct(b.name))
    {
      id<MTLBufferBinding> bb = (id<MTLBufferBinding>)b;
      MTLStructType *st = bb.bufferStructType;
      if(k->has_vals || !st || st.members.count > DT_SLANG_MAX_PARAMS) return FALSE;
      k->has_vals = TRUE;
      k->vals_index = b.index;
      k->vals_size = bb.bufferDataSize;
      NSArray<MTLStructMember *> *members =
        [st.members sortedArrayUsingComparator:^NSComparisonResult(MTLStructMember *x, MTLStructMember *y) {
          return x.offset < y.offset ? NSOrderedAscending : x.offset > y.offset ? NSOrderedDescending : NSOrderedSame;
        }];
      k->nvals = (int)members.count;
      for(int m = 0; m < k->nvals; m++)
      {
        k->val_offset[m] = members[m].offset;
        k->val_space[m] = (m + 1 < k->nvals ? members[m + 1].offset : k->vals_size) - members[m].offset;
      }
    }
    else if(b.type == MTLBindingTypeBuffer)
    {
      if(k->nbuf == DT_SLANG_MAX_PARAMS) return FALSE;
      k->buf_index[k->nbuf++] = b.index;
    }
    else
      return FALSE;
  }
  _sort_indices(k->tex_index, k->tex_write, k->ntex);
  _sort_indices(k->buf_index, NULL, k->nbuf);
  return TRUE;
}

static DtSlangKernel *_get_kernel(const char *program, const char *entry)
{
  NSString *key = [NSString stringWithFormat:@"%s:%s", program, entry];
  DtSlangKernel *k = _mtl.kernels[key];
  if(k) return k;

  NSString *prog = @(program);
  id<MTLLibrary> lib = (id<MTLLibrary>)_mtl.libraries[prog];
  NSError *error = nil;
  if(!lib)
  {
    char datadir[PATH_MAX] = { 0 };
    dt_loc_get_datadir(datadir, sizeof(datadir));
    NSString *path = [NSString stringWithFormat:@"%s/kernels/slang/%s.metallib", datadir, program];
    lib = [_mtl.dev newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];
    if(!lib)
    {
      dt_print(DT_DEBUG_OPENCL, "[slang_gpu] can't load %s: %s", path.UTF8String,
               error.localizedDescription.UTF8String);
      return nil;
    }
    _mtl.libraries[prog] = lib;
  }

  id<MTLFunction> fn = [lib newFunctionWithName:@(entry)];
  if(!fn)
  {
    dt_print(DT_DEBUG_OPENCL, "[slang_gpu] no kernel %s in %s.metallib", entry, program);
    return nil;
  }
  MTLComputePipelineReflection *refl = nil;
  k = [DtSlangKernel new];
  k->pipeline = [_mtl.dev newComputePipelineStateWithFunction:fn
                                                      options:MTLPipelineOptionBindingInfo
                                                   reflection:&refl
                                                        error:&error];
  if(!k->pipeline)
  {
    dt_print(DT_DEBUG_OPENCL, "[slang_gpu] can't create the pipeline for %s: %s", entry,
             error.localizedDescription.UTF8String);
    return nil;
  }
  if(!_reflect(refl, k))
  {
    dt_print(DT_DEBUG_OPENCL, "[slang_gpu] can't find the parameters of %s in %s.metallib", entry, program);
    return nil;
  }
  _mtl.kernels[key] = k;
  return k;
}

// loads every .metallib in kernels/slang and builds the pipelines of all their
// kernels, which takes longer than running them the first time
static void _preload_kernels(void)
{
  char datadir[PATH_MAX] = { 0 };
  dt_loc_get_datadir(datadir, sizeof(datadir));
  gchar *dirname = g_build_filename(datadir, "kernels", "slang", NULL);
  GDir *dir = g_dir_open(dirname, 0, NULL);
  const gchar *file;
  while(dir && (file = g_dir_read_name(dir)))
  {
    if(!g_str_has_suffix(file, ".metallib")) continue;
    gchar *program = g_strndup(file, strlen(file) - strlen(".metallib"));
    NSString *path = [NSString stringWithFormat:@"%s/%s", dirname, file];
    id<MTLLibrary> lib = [_mtl.dev newLibraryWithURL:[NSURL fileURLWithPath:path] error:NULL];
    if(lib)
    {
      _mtl.libraries[@(program)] = lib;
      for(NSString *fn in lib.functionNames) _get_kernel(program, fn.UTF8String);
    }
    g_free(program);
  }
  if(dir) g_dir_close(dir);
  g_free(dirname);
}

static gpointer _prepare(gpointer data)
{
  const double start = dt_get_debug_wtime();
  g_mutex_lock(&_mtl_lock);
  @autoreleasepool
  {
    if(!_mtl.initialized)
    {
      _mtl.initialized = TRUE;
      _mtl.usable = _init_device();
    }
    if(_mtl.usable) _preload_kernels();
  }
  g_mutex_unlock(&_mtl_lock);
  dt_print(DT_DEBUG_PERF, "[slang_gpu] preparing the Metal device and kernels took %.3f secs",
           dt_get_debug_wtime() - start);
  return NULL;
}

void dt_slang_gpu_prepare(void)
{
  if(!dt_conf_get_bool("slang_gpu") || _prepare_thread) return;
  _prepare_thread = g_thread_new("slang_gpu", _prepare, NULL);
}

// --------------------------------------------------------------------------------------------
// memory

// Every image is a linear texture on an MTLBuffer in shared storage, which
// the CPU reads and writes directly on unified memory. Host buffers that are
// whole VM pages, as the large blocks from posix_memalign() are on macOS, are
// wrapped without a copy; anything else is copied into a buffer of our own.
@interface DtSlangImage : NSObject
{
@public
  id<MTLBuffer> buffer;
  id<MTLTexture> texture;
  int width, height, channels;
  size_t pitch;
  gboolean wrapped; // buffer is the caller's memory
}
@end

@implementation DtSlangImage
@end

static size_t _linear_pitch(const int width, const int channels)
{
  const MTLPixelFormat format = channels == 1 ? MTLPixelFormatR32Float : MTLPixelFormatRGBA32Float;
  const size_t align = [_mtl.dev minimumLinearTextureAlignmentForPixelFormat:format];
  const size_t row = sizeof(float) * channels * width;
  return (row + align - 1) / align * align;
}

static gboolean _image_fits(const int width, const int height, const int channels)
{
  return width > 0 && height > 0 && width <= _mtl.max_image_dim && height <= _mtl.max_image_dim
         && (channels == 1 || channels == 4);
}

static DtSlangImage *_image_on(id<MTLBuffer> buffer, const int width, const int height, const int channels,
                               const size_t pitch)
{
  MTLTextureDescriptor *d = [MTLTextureDescriptor
    texture2DDescriptorWithPixelFormat:channels == 1 ? MTLPixelFormatR32Float : MTLPixelFormatRGBA32Float
                                 width:width
                                height:height
                             mipmapped:NO];
  d.storageMode = MTLStorageModeShared;
  d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  id<MTLTexture> t = [buffer newTextureWithDescriptor:d offset:0 bytesPerRow:pitch];
  if(!t) return nil;
  DtSlangImage *img = [DtSlangImage new];
  img->buffer = buffer;
  img->texture = t;
  img->width = width;
  img->height = height;
  img->channels = channels;
  img->pitch = pitch;
  return img;
}

static void _copy_rows(void *dst, const size_t dst_pitch, const void *src, const size_t src_pitch,
                       const size_t row, const int height)
{
  if(dst_pitch == row && src_pitch == row)
    memcpy(dst, src, row * height);
  else
    for(int y = 0; y < height; y++)
      memcpy((char *)dst + y * dst_pitch, (const char *)src + y * src_pitch, row);
}

static DtSlangImage *_image_new(const int width, const int height, const int channels, const void *host)
{
  const size_t pitch = _linear_pitch(width, channels);
  if(pitch * height > _mtl.dev.maxBufferLength) return nil;
  id<MTLBuffer> buffer = [_mtl.dev newBufferWithLength:pitch * height options:MTLResourceStorageModeShared];
  if(!buffer) return nil;
  DtSlangImage *img = _image_on(buffer, width, height, channels, pitch);
  if(img && host)
    _copy_rows(buffer.contents, pitch, host, sizeof(float) * channels * width, sizeof(float) * channels * width,
               height);
  return img;
}

static DtSlangImage *_image_wrap(const void *host, const int width, const int height, const int channels)
{
  const size_t row = sizeof(float) * channels * width;
  const size_t page = sysconf(_SC_PAGESIZE);
  const size_t length = (row * height + page - 1) / page * page;
  // malloc_size() is 0 for a pointer into the middle of a block
  if((uintptr_t)host % page || row != _linear_pitch(width, channels)
     || malloc_size(host) < length || length > _mtl.dev.maxBufferLength)
    return nil;
  id<MTLBuffer> buffer = [_mtl.dev newBufferWithBytesNoCopy:(void *)host
                                                     length:length
                                                    options:MTLResourceStorageModeShared
                                                deallocator:nil];
  if(!buffer) return nil;
  DtSlangImage *img = _image_on(buffer, width, height, channels, row);
  if(img) img->wrapped = TRUE;
  return img;
}

// a device buffer, in shared storage like the images
@interface DtSlangBuffer : NSObject
{
@public
  id<MTLBuffer> buffer;
  size_t size;
}
@end

@implementation DtSlangBuffer
@end

// waits for the last command buffer that ran without waiting
static void _finish(void)
{
  if(_mtl.pending)
  {
    [_mtl.pending waitUntilCompleted];
    _mtl.pending = nil;
  }
}

// --------------------------------------------------------------------------------------------
// dispatch

static int _run(const char *program,
                const char *entry,
                const int width,
                const int height,
                const int nargs,
                const dt_slang_arg_t *args)
{
  DtSlangKernel *k = _get_kernel(program, entry);
  if(!k) return 1;

  // ARC objects are declared before the first goto, which must not skip their initialization
  DtSlangImage *images[DT_SLANG_MAX_PARAMS];
  id<MTLCommandBuffer> cb = nil;
  id<MTLComputeCommandEncoder> enc = nil;

  // match the arguments to the kernel's parameters, see _reflect()
  int tex_arg[DT_SLANG_MAX_PARAMS], buf_arg[DT_SLANG_MAX_PARAMS];
  int ntex = 0, nbuf = 0, nvals = 0;
  gboolean host_images = FALSE;
  uint8_t vals[256] = { 0 };
  if(k->vals_size > sizeof(vals)) goto mismatch;
  for(int a = 0; a < nargs; a++)
  {
    const dt_slang_arg_t *arg = &args[a];
    if(arg->kind == DT_SLANG_ARG_VALUE)
    {
      if(nvals >= k->nvals || arg->size > k->val_space[nvals]) goto mismatch;
      memcpy(vals + k->val_offset[nvals], arg->ptr, arg->size);
      nvals++;
    }
    else if(arg->kind == DT_SLANG_ARG_BUFFER || arg->kind == DT_SLANG_ARG_BUFFER_DEV)
    {
      if(nbuf >= k->nbuf || !arg->ptr) goto mismatch;
      buf_arg[nbuf++] = a;
    }
    else
    {
      // a device image can be read or written
      if(ntex >= k->ntex || !arg->ptr
         || (arg->kind != DT_SLANG_ARG_IMAGE_DEV && k->tex_write[ntex] != (arg->kind == DT_SLANG_ARG_IMAGE_OUT)))
        goto mismatch;
      if(arg->kind != DT_SLANG_ARG_IMAGE_DEV)
      {
        if(!_image_fits(arg->width, arg->height, arg->channels))
        {
          dt_print(DT_DEBUG_OPENCL, "[slang_gpu] %dx%d image too large for %s", arg->width, arg->height, entry);
          return 1;
        }
        host_images = TRUE;
      }
      tex_arg[ntex++] = a;
    }
  }
  if(ntex != k->ntex || nbuf != k->nbuf || nvals != k->nvals) goto mismatch;

  // host buffers can be freed or reused once we return
  if(host_images) _finish();

  cb = [_mtl.queue commandBuffer];
  enc = [cb computeCommandEncoder];
  if(!cb || !enc) goto failed;
  [enc setComputePipelineState:k->pipeline];
  if(k->has_vals) [enc setBytes:vals length:k->vals_size atIndex:k->vals_index];
  for(int b = 0; b < nbuf; b++)
  {
    const dt_slang_arg_t *arg = &args[buf_arg[b]];
    id<MTLBuffer> buf = arg->kind == DT_SLANG_ARG_BUFFER_DEV
      ? ((__bridge DtSlangBuffer *)arg->ptr)->buffer
      : [_mtl.dev newBufferWithBytes:arg->ptr
                              length:arg->size
                             options:MTLResourceStorageModeShared];
    if(!buf) goto failed;
    [enc setBuffer:buf offset:0 atIndex:k->buf_index[b]];
  }
  for(int t = 0; t < ntex; t++)
  {
    const dt_slang_arg_t *arg = &args[tex_arg[t]];
    if(arg->kind == DT_SLANG_ARG_IMAGE_DEV)
      images[t] = (__bridge DtSlangImage *)arg->ptr;
    else
    {
      images[t] = _image_wrap(arg->ptr, arg->width, arg->height, arg->channels);
      if(!images[t])
        images[t] = _image_new(arg->width, arg->height, arg->channels,
                               arg->kind == DT_SLANG_ARG_IMAGE_IN ? arg->ptr : NULL);
    }
    if(!images[t]) goto failed;
    [enc setTexture:images[t]->texture atIndex:k->tex_index[t]];
  }

  // the kernels are written for 16x16 work groups, see [numthreads] in the .slang files
  NSUInteger tw = 16, th = 16;
  if(k->pipeline.maxTotalThreadsPerThreadgroup < tw * th)
  {
    tw = k->pipeline.threadExecutionWidth;
    th = MAX(1, k->pipeline.maxTotalThreadsPerThreadgroup / tw);
  }
  [enc dispatchThreadgroups:MTLSizeMake((width + tw - 1) / tw, (height + th - 1) / th, 1)
      threadsPerThreadgroup:MTLSizeMake(tw, th, 1)];
  [enc endEncoding];
  [cb commit];

  // with only device images nothing needs to wait: the queue runs command
  // buffers in order, and a read waits in _finish()
  if(!host_images)
  {
    _mtl.pending = cb;
    return 0;
  }

  [cb waitUntilCompleted];
  if(cb.status != MTLCommandBufferStatusCompleted)
  {
    dt_print(DT_DEBUG_OPENCL, "[slang_gpu] running %s failed: %s", entry,
             cb.error.localizedDescription.UTF8String);
    return 1;
  }
  for(int t = 0; t < ntex; t++)
  {
    const dt_slang_arg_t *arg = &args[tex_arg[t]];
    if(arg->kind != DT_SLANG_ARG_IMAGE_OUT || images[t]->wrapped) continue;
    const size_t row = sizeof(float) * arg->channels * arg->width;
    _copy_rows((void *)arg->ptr, row, images[t]->buffer.contents, images[t]->pitch, row, arg->height);
  }
  return 0;

failed:
  if(enc) [enc endEncoding];
  dt_print(DT_DEBUG_OPENCL, "[slang_gpu] running %s failed", entry);
  return 1;

mismatch:
  dt_print(DT_DEBUG_OPENCL, "[slang_gpu] arguments don't match the parameters of %s:"
           " %d images, %d buffers and %d values expected", entry, k->ntex, k->nbuf, k->nvals);
  return 1;
}

int dt_slang_gpu_run_2d(const char *program,
                        const char *kernel,
                        const int width,
                        const int height,
                        const int nargs,
                        const dt_slang_arg_t *args)
{
  if(!dt_slang_gpu_enabled()) return 1;

  const double start = dt_get_debug_wtime();
  // one queue, one submission at a time: pipes running in parallel wait here
  g_mutex_lock(&_mtl_lock);
  int err;
  @autoreleasepool
  {
    err = _run(program, kernel, width, height, nargs, args);
  }
  g_mutex_unlock(&_mtl_lock);
  if(!err)
    dt_print(DT_DEBUG_PERF, "[slang_gpu] %s %dx%d took %.3f secs (Metal)",
             kernel, width, height, dt_get_debug_wtime() - start);
  return err;
}

// --------------------------------------------------------------------------------------------
// device images

dt_slang_gpu_image_t *dt_slang_gpu_image_new(const int width,
                                             const int height,
                                             const int channels,
                                             const void *host)
{
  if(!dt_slang_gpu_enabled() || !_image_fits(width, height, channels)) return NULL;

  g_mutex_lock(&_mtl_lock);
  DtSlangImage *img = nil;
  @autoreleasepool
  {
    // a fresh 160 MB buffer costs more than the kernels that use it, mostly
    // for the pages the GPU touches the first time
    for(NSUInteger i = 0; i < _mtl.images.count; i++)
    {
      DtSlangImage *pooled = _mtl.images[i];
      if(pooled->width == width && pooled->height == height && pooled->channels == channels)
      {
        img = pooled;
        [_mtl.images removeObjectAtIndex:i];
        break;
      }
    }
    if(img && host)
    {
      // the image may still be in use by a pending kernel
      _finish();
      const size_t row = sizeof(float) * channels * width;
      _copy_rows(img->buffer.contents, img->pitch, host, row, row, height);
    }
    else if(!img)
      img = _image_new(width, height, channels, host);
  }
  g_mutex_unlock(&_mtl_lock);
  if(!img) dt_print(DT_DEBUG_OPENCL, "[slang_gpu] can't allocate a %dx%d device image", width, height);
  return img ? (__bridge_retained dt_slang_gpu_image_t *)img : NULL;
}

dt_slang_gpu_image_t *dt_slang_gpu_image_wrap(const int width,
                                              const int height,
                                              const int channels,
                                              const void *host)
{
  if(!host || !dt_slang_gpu_enabled() || !_image_fits(width, height, channels)) return NULL;

  g_mutex_lock(&_mtl_lock);
  DtSlangImage *img;
  @autoreleasepool
  {
    img = _image_wrap(host, width, height, channels);
  }
  g_mutex_unlock(&_mtl_lock);
  return img ? (__bridge_retained dt_slang_gpu_image_t *)img
             : dt_slang_gpu_image_new(width, height, channels, host);
}

void dt_slang_gpu_image_free(dt_slang_gpu_image_t *image)
{
  if(!image) return;
  DtSlangImage *img = CFBridgingRelease(image);
  g_mutex_lock(&_mtl_lock);
  // the caller may reuse the memory of a wrapped image as soon as we return,
  // and it belongs to the caller, so it never goes to the pool
  if(img->wrapped)
  {
    _finish();
    g_mutex_unlock(&_mtl_lock);
    return;
  }
  // enough for a pipe's input and output, and those of a second pipe. A
  // pending command buffer keeps its own reference to the texture.
  if(_mtl.images)
  {
    [_mtl.images insertObject:img atIndex:0];
    while(_mtl.images.count > DT_SLANG_POOL_SIZE) [_mtl.images removeLastObject];
  }
  g_mutex_unlock(&_mtl_lock);
}

int dt_slang_gpu_image_width(const dt_slang_gpu_image_t *image)
{
  return ((__bridge DtSlangImage *)image)->width;
}

int dt_slang_gpu_image_height(const dt_slang_gpu_image_t *image)
{
  return ((__bridge DtSlangImage *)image)->height;
}

int dt_slang_gpu_image_read(dt_slang_gpu_image_t *image, void *host)
{
  DtSlangImage *img = (__bridge DtSlangImage *)image;
  if(!img || !host) return 1;
  const double start = dt_get_debug_wtime();
  g_mutex_lock(&_mtl_lock);
  id<MTLCommandBuffer> last = _mtl.pending;
  _finish();
  g_mutex_unlock(&_mtl_lock);
  if(last && last.status != MTLCommandBufferStatusCompleted)
  {
    dt_print(DT_DEBUG_OPENCL, "[slang_gpu] a kernel writing this image failed: %s",
             last.error.localizedDescription.UTF8String);
    return 1;
  }
  const size_t row = sizeof(float) * img->channels * img->width;
  _copy_rows(host, row, img->buffer.contents, img->pitch, row, img->height);
  dt_print(DT_DEBUG_PERF, "[slang_gpu] reading %dx%d from the device took %.3f secs",
           img->width, img->height, dt_get_debug_wtime() - start);
  return 0;
}

int dt_slang_gpu_image_write(dt_slang_gpu_image_t *image, const void *host)
{
  DtSlangImage *img = (__bridge DtSlangImage *)image;
  if(!img || !host) return 1;
  const double start = dt_get_debug_wtime();
  // a pending kernel may still read the image
  g_mutex_lock(&_mtl_lock);
  _finish();
  g_mutex_unlock(&_mtl_lock);
  const size_t row = sizeof(float) * img->channels * img->width;
  _copy_rows(img->buffer.contents, img->pitch, host, row, row, img->height);
  dt_print(DT_DEBUG_PERF, "[slang_gpu] writing %dx%d to the device took %.3f secs",
           img->width, img->height, dt_get_debug_wtime() - start);
  return 0;
}

// --------------------------------------------------------------------------------------------
// device buffers

dt_slang_gpu_buffer_t *dt_slang_gpu_buffer_new(const size_t size, const void *host)
{
  if(!dt_slang_gpu_enabled() || size == 0) return NULL;

  g_mutex_lock(&_mtl_lock);
  DtSlangBuffer *buf = nil;
  @autoreleasepool
  {
    for(NSUInteger i = 0; i < _mtl.buffers.count; i++)
    {
      DtSlangBuffer *pooled = _mtl.buffers[i];
      if(pooled->size == size)
      {
        buf = pooled;
        [_mtl.buffers removeObjectAtIndex:i];
        break;
      }
    }
    if(buf && host)
    {
      // the buffer may still be in use by a pending kernel
      _finish();
      memcpy(buf->buffer.contents, host, size);
    }
    else if(!buf && size <= _mtl.dev.maxBufferLength)
    {
      id<MTLBuffer> mb = host
        ? [_mtl.dev newBufferWithBytes:host length:size options:MTLResourceStorageModeShared]
        : [_mtl.dev newBufferWithLength:size options:MTLResourceStorageModeShared];
      if(mb)
      {
        buf = [DtSlangBuffer new];
        buf->buffer = mb;
        buf->size = size;
      }
    }
  }
  g_mutex_unlock(&_mtl_lock);
  if(!buf) dt_print(DT_DEBUG_OPENCL, "[slang_gpu] can't allocate a device buffer of %zu bytes", size);
  return buf ? (__bridge_retained dt_slang_gpu_buffer_t *)buf : NULL;
}

void dt_slang_gpu_buffer_free(dt_slang_gpu_buffer_t *buffer)
{
  if(!buffer) return;
  DtSlangBuffer *buf = CFBridgingRelease(buffer);
  g_mutex_lock(&_mtl_lock);
  // a demosaic chain holds eight buffers of the same size. A pending command
  // buffer keeps its own reference to the MTLBuffer.
  if(_mtl.buffers)
  {
    [_mtl.buffers insertObject:buf atIndex:0];
    while(_mtl.buffers.count > DT_SLANG_BUFFER_POOL_SIZE) [_mtl.buffers removeLastObject];
  }
  g_mutex_unlock(&_mtl_lock);
}

int dt_slang_gpu_buffer_read(dt_slang_gpu_buffer_t *buffer, void *host, const size_t size)
{
  DtSlangBuffer *buf = (__bridge DtSlangBuffer *)buffer;
  if(!buf || !host || size > buf->size) return 1;
  g_mutex_lock(&_mtl_lock);
  id<MTLCommandBuffer> last = _mtl.pending;
  _finish();
  g_mutex_unlock(&_mtl_lock);
  if(last && last.status != MTLCommandBufferStatusCompleted)
  {
    dt_print(DT_DEBUG_OPENCL, "[slang_gpu] a kernel writing this buffer failed: %s",
             last.error.localizedDescription.UTF8String);
    return 1;
  }
  memcpy(host, buf->buffer.contents, size);
  return 0;
}
