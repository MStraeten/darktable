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

// the macOS 15 SDK renamed MTLPipelineOptionArgumentInfo, keeping its value;
// Xcode 15 (SDK 14) has only the old name
#if !defined(MAC_OS_VERSION_15_0)
#define MTLPipelineOptionBindingInfo MTLPipelineOptionArgumentInfo
#endif

#define DT_SLANG_MAX_PARAMS 32
#define DT_SLANG_POOL_SIZE 4
#define DT_SLANG_BUFFER_POOL_SIZE 12
#define DT_SLANG_STAGING_POOL_SIZE 4

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
  __strong NSMutableArray<id<MTLCommandBuffer>> *inflight; // committed, not yet checked
  __strong NSMutableArray *images;        // freed device images, most recent first
  __strong NSMutableArray *buffers;       // freed device buffers, most recent first
  __strong NSMutableArray *staging;       // transfer buffers, see _staging_get()
  NSUInteger max_image_dim;
  gboolean private_storage; // device images and buffers in private storage
} dt_slang_metal_t;

static dt_slang_metal_t _mtl = { 0 };
static GMutex _mtl_lock;
// guards _mtl.staging only, which command buffer completion handlers refill
static GMutex _staging_lock;
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
  _mtl.staging = [NSMutableArray array];
  // shared storage is system memory on a discrete GPU, so kernels working
  // there would read and write every pixel over PCIe
  _mtl.private_storage = !_mtl.dev.hasUnifiedMemory || dt_conf_get_bool("slang_gpu_private_memory");
  dt_print(DT_DEBUG_OPENCL, "[slang_gpu] using Metal device %s%s%s", _mtl.dev.name.UTF8String,
           _mtl.dev.hasUnifiedMemory ? " (unified memory)" : "",
           _mtl.private_storage ? ", images in private storage" : "");
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
  const gboolean usable = _mtl.usable && !dt_slang_gpu_stopped();
  g_mutex_unlock(&_mtl_lock);
  return usable;
}

void dt_slang_gpu_cleanup(void)
{
  if(_prepare_thread) g_thread_join(_prepare_thread);
  _prepare_thread = NULL;
  g_mutex_lock(&_mtl_lock);
  for(id<MTLCommandBuffer> cb in _mtl.inflight) [cb waitUntilCompleted];
  _mtl.inflight = nil;
  _mtl.images = nil;
  _mtl.buffers = nil;
  g_mutex_lock(&_staging_lock);
  _mtl.staging = nil;
  g_mutex_unlock(&_staging_lock);
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

  // one library per entry point, see data/kernels/slang/CMakeLists.txt
  NSString *name = [NSString stringWithFormat:@"%s.%s", program, entry];
  id<MTLLibrary> lib = (id<MTLLibrary>)_mtl.libraries[name];
  NSError *error = nil;
  if(!lib)
  {
    char datadir[PATH_MAX] = { 0 };
    dt_loc_get_datadir(datadir, sizeof(datadir));
    NSString *path = [NSString stringWithFormat:@"%s/kernels/slang/%@.metallib", datadir, name];
    lib = [_mtl.dev newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];
    if(!lib)
    {
      dt_print(DT_DEBUG_OPENCL, "[slang_gpu] can't load %s: %s", path.UTF8String,
               error.localizedDescription.UTF8String);
      return nil;
    }
    _mtl.libraries[name] = lib;
  }

  id<MTLFunction> fn = [lib newFunctionWithName:@(entry)];
  if(!fn)
  {
    dt_print(DT_DEBUG_OPENCL, "[slang_gpu] no kernel %s in %s.metallib", entry, name.UTF8String);
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
    dt_print(DT_DEBUG_OPENCL, "[slang_gpu] can't find the parameters of %s in %s.metallib", entry,
             name.UTF8String);
    return nil;
  }
  _mtl.kernels[key] = k;
  return k;
}

// loads every <program>.<entry>.metallib in kernels/slang and builds its
// pipeline, which takes longer than running the kernel the first time
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
    gchar **parts = g_strsplit(file, ".", 3);
    if(g_strv_length(parts) == 3) _get_kernel(parts[0], parts[1]);
    g_strfreev(parts);
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

// On unified memory every image is a linear texture on an MTLBuffer in
// shared storage, which the CPU reads and writes directly. Host buffers that
// are whole VM pages, as the large blocks from posix_memalign() are on macOS,
// are wrapped without a copy; anything else is copied into a buffer of our
// own. With private storage, device images are textures of their own with no
// buffer, and blits copy them through staging buffers; host images passed to
// a kernel stay in shared storage, as the kernel reads them once anyway.
@interface DtSlangImage : NSObject
{
@public
  id<MTLBuffer> buffer; // nil for a private image
  id<MTLTexture> texture;
  int width, height, channels;
  size_t pitch;
  gboolean wrapped; // buffer is the caller's memory
  gboolean aliased; // and the caller keeps it valid, see dt_slang_gpu_image_alias()
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

// a shared buffer on the caller's memory, if it is whole VM pages
static id<MTLBuffer> _host_buffer(const void *host, const size_t size)
{
  const size_t page = sysconf(_SC_PAGESIZE);
  const size_t length = (size + page - 1) / page * page;
  // malloc_size() is 0 for a pointer into the middle of a block
  if((uintptr_t)host % page || malloc_size(host) < length || length > _mtl.dev.maxBufferLength)
    return nil;
  return [_mtl.dev newBufferWithBytesNoCopy:(void *)host
                                     length:length
                                    options:MTLResourceStorageModeShared
                                deallocator:nil];
}

static DtSlangImage *_image_wrap(const void *host, const int width, const int height, const int channels)
{
  const size_t row = sizeof(float) * channels * width;
  if(row != _linear_pitch(width, channels)) return nil;
  id<MTLBuffer> buffer = _host_buffer(host, row * height);
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

// checks the command buffers that have finished, oldest first, or with wait
// all of them. Each one is checked: the queue runs them in order but goes on
// after one fails, so a read that only looked at the last one would miss a
// kernel failing before it. Returns FALSE if one failed.
static gboolean _check(const gboolean wait)
{
  gboolean ok = TRUE;
  while(_mtl.inflight.count)
  {
    id<MTLCommandBuffer> cb = _mtl.inflight.firstObject;
    if(wait)
      [cb waitUntilCompleted];
    else if(cb.status != MTLCommandBufferStatusCompleted && cb.status != MTLCommandBufferStatusError)
      break;
    if(cb.status == MTLCommandBufferStatusError)
    {
      gchar *what = g_strdup_printf("a command buffer failed: %s", cb.error.localizedDescription.UTF8String);
      // Apple Silicon has no removable GPU, so a failed command buffer
      // leaves the device usable and only counts towards the failed runs
      dt_slang_gpu_report_error(what, FALSE);
      g_free(what);
      ok = FALSE;
    }
    [_mtl.inflight removeObjectAtIndex:0];
  }
  return ok;
}

// commits cb without waiting for it
static void _commit(id<MTLCommandBuffer> cb)
{
  [cb commit];
  if(!_mtl.inflight) _mtl.inflight = [NSMutableArray array];
  [_mtl.inflight addObject:cb];
  _check(FALSE);
}

// waits for every committed command buffer. Returns FALSE if one failed.
static gboolean _finish(void)
{
  return _check(TRUE);
}

// --------------------------------------------------------------------------------------------
// private storage

static DtSlangImage *_image_private(const int width, const int height, const int channels)
{
  MTLTextureDescriptor *d = [MTLTextureDescriptor
    texture2DDescriptorWithPixelFormat:channels == 1 ? MTLPixelFormatR32Float : MTLPixelFormatRGBA32Float
                                 width:width
                                height:height
                             mipmapped:NO];
  d.storageMode = MTLStorageModePrivate;
  d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  id<MTLTexture> t = [_mtl.dev newTextureWithDescriptor:d];
  if(!t) return nil;
  DtSlangImage *img = [DtSlangImage new];
  img->texture = t;
  img->width = width;
  img->height = height;
  img->channels = channels;
  img->pitch = sizeof(float) * channels * width;
  return img;
}

static size_t _image_bytes(DtSlangImage *img)
{
  return img->buffer ? img->buffer.length : img->texture.allocatedSize;
}

// the smallest pooled transfer buffer that holds size bytes, or a new one.
// Allocating them per transfer would cost about as much as the copy.
static id<MTLBuffer> _staging_get(const size_t size)
{
  id<MTLBuffer> found = nil;
  g_mutex_lock(&_staging_lock);
  NSUInteger pick = NSNotFound;
  for(NSUInteger i = 0; i < _mtl.staging.count; i++)
  {
    id<MTLBuffer> b = _mtl.staging[i];
    if(b.length >= size && (pick == NSNotFound || b.length < ((id<MTLBuffer>)_mtl.staging[pick]).length))
      pick = i;
  }
  if(pick != NSNotFound)
  {
    found = _mtl.staging[pick];
    [_mtl.staging removeObjectAtIndex:pick];
  }
  g_mutex_unlock(&_staging_lock);
  return found ? found
               : size <= _mtl.dev.maxBufferLength
                 ? [_mtl.dev newBufferWithLength:size options:MTLResourceStorageModeShared]
                 : nil;
}

static void _staging_put(id<MTLBuffer> b)
{
  g_mutex_lock(&_staging_lock);
  if(_mtl.staging)
  {
    [_mtl.staging insertObject:b atIndex:0];
    while(_mtl.staging.count > DT_SLANG_STAGING_POOL_SIZE) [_mtl.staging removeLastObject];
  }
  g_mutex_unlock(&_staging_lock);
}

static size_t _staging_bytes(void)
{
  size_t bytes = 0;
  g_mutex_lock(&_staging_lock);
  for(id<MTLBuffer> b in _mtl.staging) bytes += b.length;
  g_mutex_unlock(&_staging_lock);
  return bytes;
}

// copies host memory into a private image (img) or buffer (dst) on the queue,
// after the kernels already queued, without waiting. The host memory is
// copied to a staging buffer first, as the caller may change it once we
// return, and the staging buffer goes back to the pool when the blit is done.
static gboolean _upload(DtSlangImage *img, id<MTLBuffer> dst, const void *host, const size_t size)
{
  id<MTLBuffer> staging = _staging_get(size);
  id<MTLCommandBuffer> cb = staging ? [_mtl.queue commandBuffer] : nil;
  id<MTLBlitCommandEncoder> blit = cb ? [cb blitCommandEncoder] : nil;
  if(!blit) return FALSE;
  memcpy(staging.contents, host, size);
  if(img)
    [blit copyFromBuffer:staging
            sourceOffset:0
       sourceBytesPerRow:img->pitch
     sourceBytesPerImage:size
              sourceSize:MTLSizeMake(img->width, img->height, 1)
               toTexture:img->texture
        destinationSlice:0
        destinationLevel:0
       destinationOrigin:MTLOriginMake(0, 0, 0)];
  else
    [blit copyFromBuffer:staging sourceOffset:0 toBuffer:dst destinationOffset:0 size:size];
  [blit endEncoding];
  [cb addCompletedHandler:^(id<MTLCommandBuffer> done) { _staging_put(staging); }];
  _commit(cb);
  return TRUE;
}

// copies a private image or buffer to host memory once the queue is done,
// straight into the host memory if it can be wrapped. Called with the lock held.
static gboolean _download(DtSlangImage *img, id<MTLBuffer> src, void *host, const size_t size)
{
  id<MTLBuffer> wrapped = _host_buffer(host, size);
  id<MTLBuffer> dst = wrapped ? wrapped : _staging_get(size);
  id<MTLCommandBuffer> cb = dst ? [_mtl.queue commandBuffer] : nil;
  id<MTLBlitCommandEncoder> blit = cb ? [cb blitCommandEncoder] : nil;
  if(!blit) return FALSE;
  if(img)
    [blit copyFromTexture:img->texture
                   sourceSlice:0
                   sourceLevel:0
                  sourceOrigin:MTLOriginMake(0, 0, 0)
                    sourceSize:MTLSizeMake(img->width, img->height, 1)
                      toBuffer:dst
             destinationOffset:0
        destinationBytesPerRow:img->pitch
      destinationBytesPerImage:size];
  else
    [blit copyFromBuffer:src sourceOffset:0 toBuffer:dst destinationOffset:0 size:size];
  [blit endEncoding];
  // the blit runs after the kernels writing the source, so waiting for every
  // command buffer also tells whether one of those failed
  _commit(cb);
  if(!_finish())
  {
    dt_print(DT_DEBUG_OPENCL, "[slang_gpu] a kernel writing this %s or the copy from the device failed",
             img ? "image" : "buffer");
    if(!wrapped) _staging_put(dst);
    return FALSE;
  }
  if(!wrapped)
  {
    memcpy(host, dst.contents, size);
    _staging_put(dst);
  }
  return TRUE;
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
  gboolean host_out = FALSE;
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
        host_out |= arg->kind == DT_SLANG_ARG_IMAGE_OUT;
      }
      tex_arg[ntex++] = a;
    }
  }
  if(ntex != k->ntex || nbuf != k->nbuf || nvals != k->nvals) goto mismatch;

  // host outputs are read back below, so this call waits anyway
  if(host_out) _finish();

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
      // a host input is copied unless we wait: the caller may free or change
      // it once we return, and the command buffer keeps the copy alive. This
      // keeps lookup tables such as colorin's from stalling the queue.
      images[t] = host_out ? _image_wrap(arg->ptr, arg->width, arg->height, arg->channels) : nil;
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
  _commit(cb);

  // without host outputs nothing needs to wait: the queue runs command
  // buffers in order, and a read waits in _finish()
  if(!host_out) return 0;

  if(!_finish())
  {
    dt_print(DT_DEBUG_OPENCL, "[slang_gpu] running %s failed", entry);
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
    if(img && host && !img->buffer)
    {
      if(!_upload(img, nil, host, img->pitch * height)) img = nil;
    }
    else if(img && host)
    {
      // the image may still be in use by a pending kernel
      _finish();
      const size_t row = sizeof(float) * channels * width;
      _copy_rows(img->buffer.contents, img->pitch, host, row, row, height);
    }
    else if(!img && _mtl.private_storage)
    {
      img = _image_private(width, height, channels);
      if(img && host && !_upload(img, nil, host, img->pitch * height)) img = nil;
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
  DtSlangImage *img = nil;
  @autoreleasepool
  {
    // a private image takes a copy, so that kernels don't read the host memory over PCIe
    if(!_mtl.private_storage) img = _image_wrap(host, width, height, channels);
  }
  g_mutex_unlock(&_mtl_lock);
  return img ? (__bridge_retained dt_slang_gpu_image_t *)img
             : dt_slang_gpu_image_new(width, height, channels, host);
}

dt_slang_gpu_image_t *dt_slang_gpu_image_alias(const int width,
                                               const int height,
                                               const int channels,
                                               void *host)
{
  if(!dt_slang_gpu_enabled() || !host || !_image_fits(width, height, channels)) return NULL;
  // the caller copies a private image back when it needs the host memory
  if(_mtl.private_storage) return NULL;
  DtSlangImage *img;
  g_mutex_lock(&_mtl_lock);
  @autoreleasepool
  {
    img = _image_wrap(host, width, height, channels);
  }
  g_mutex_unlock(&_mtl_lock);
  if(!img) return NULL;
  img->aliased = TRUE;
  return (__bridge_retained dt_slang_gpu_image_t *)img;
}

// Apple GPUs share the system memory, so the pools take nothing from a
// separate device heap that OpenCL would run out of
void dt_slang_gpu_release_pools(void)
{
}

void dt_slang_gpu_wait(void)
{
  // kernels started before an error stopped them may still write to host memory
  if(!_mtl.usable) return;
  g_mutex_lock(&_mtl_lock);
  _finish();
  g_mutex_unlock(&_mtl_lock);
}

void dt_slang_gpu_memory_info(size_t *available, int *max_dim)
{
  *available = 0;
  *max_dim = 0;
  if(!dt_slang_gpu_enabled()) return;
  g_mutex_lock(&_mtl_lock);
  // what the pools keep counts as allocated although it would be freed for
  // the module
  size_t pooled = 0;
  for(DtSlangImage *img in _mtl.images) pooled += _image_bytes(img);
  for(DtSlangBuffer *buf in _mtl.buffers) pooled += buf->buffer.length;
  pooled += _staging_bytes();
  const size_t limit = _mtl.dev.recommendedMaxWorkingSetSize;
  const size_t used = _mtl.dev.currentAllocatedSize > pooled ? _mtl.dev.currentAllocatedSize - pooled : 0;
  size_t best = limit > used ? limit - used : 0;
  const int cap = dt_conf_get_int("slang_gpu_memory");
  if(cap > 0) best = MIN(best, (size_t)cap << 20);
  *available = best;
  *max_dim = (int)_mtl.max_image_dim;
  g_mutex_unlock(&_mtl_lock);
}

void dt_slang_gpu_image_free(dt_slang_gpu_image_t *image)
{
  if(!image) return;
  DtSlangImage *img = CFBridgingRelease(image);
  // a pending command buffer keeps the buffer, and the caller the memory
  if(img->aliased) return;
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
  if(!img->buffer)
  {
    gboolean ok;
    @autoreleasepool
    {
      ok = _download(img, nil, host, img->pitch * img->height);
    }
    g_mutex_unlock(&_mtl_lock);
    if(!ok) return 1;
    dt_print(DT_DEBUG_PERF, "[slang_gpu] reading %dx%d from the device took %.3f secs",
             img->width, img->height, dt_get_debug_wtime() - start);
    return 0;
  }
  const gboolean ok = _finish();
  g_mutex_unlock(&_mtl_lock);
  if(!ok)
  {
    dt_print(DT_DEBUG_OPENCL, "[slang_gpu] a kernel writing this image failed");
    return 1;
  }
  const size_t row = sizeof(float) * img->channels * img->width;
  if(host != img->buffer.contents)
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
  g_mutex_lock(&_mtl_lock);
  if(!img->buffer)
  {
    gboolean ok;
    @autoreleasepool
    {
      ok = _upload(img, nil, host, img->pitch * img->height);
    }
    g_mutex_unlock(&_mtl_lock);
    if(!ok) return 1;
    dt_print(DT_DEBUG_PERF, "[slang_gpu] writing %dx%d to the device took %.3f secs",
             img->width, img->height, dt_get_debug_wtime() - start);
    return 0;
  }
  // a pending kernel may still read the image
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

// zeroes a buffer on the queue, after the kernels already queued
static gboolean _clear(id<MTLBuffer> buffer)
{
  id<MTLCommandBuffer> cb = [_mtl.queue commandBuffer];
  id<MTLBlitCommandEncoder> blit = cb ? [cb blitCommandEncoder] : nil;
  if(!blit) return FALSE;
  [blit fillBuffer:buffer range:NSMakeRange(0, buffer.length) value:0];
  [blit endEncoding];
  _commit(cb);
  return TRUE;
}

static dt_slang_gpu_buffer_t *_buffer_new(const size_t size, const void *host, const gboolean zero)
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
    if(buf && host && _mtl.private_storage)
    {
      if(!_upload(nil, buf->buffer, host, size)) buf = nil;
    }
    else if(buf && (host || zero))
    {
      // the buffer may still be in use by a pending kernel. A new one starts
      // zeroed, a pooled one has to be cleared. Clearing it on the queue
      // orders the fill after those kernels without waiting for them: blurs
      // take and return temporaries per pass, and a wait here stalled every one
      const gboolean cleared = !host && _clear(buf->buffer);
      if(!cleared && _mtl.private_storage)
        buf = nil;
      else if(!cleared)
      {
        _finish();
        if(host)
          memcpy(buf->buffer.contents, host, size);
        else
          memset(buf->buffer.contents, 0, size);
      }
    }
    else if(!buf && size <= _mtl.dev.maxBufferLength && _mtl.private_storage)
    {
      id<MTLBuffer> mb = [_mtl.dev newBufferWithLength:size options:MTLResourceStorageModePrivate];
      // private memory isn't guaranteed to start zeroed
      if(mb && (host ? _upload(nil, mb, host, size) : !zero || _clear(mb)))
      {
        buf = [DtSlangBuffer new];
        buf->buffer = mb;
        buf->size = size;
      }
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

dt_slang_gpu_buffer_t *dt_slang_gpu_buffer_new(const size_t size, const void *host)
{
  return _buffer_new(size, host, TRUE);
}

dt_slang_gpu_buffer_t *dt_slang_gpu_buffer_new_uninit(const size_t size)
{
  return _buffer_new(size, NULL, FALSE);
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
  if(buf->buffer.storageMode == MTLStorageModePrivate)
  {
    gboolean ok;
    @autoreleasepool
    {
      ok = _download(nil, buf->buffer, host, size);
    }
    g_mutex_unlock(&_mtl_lock);
    return ok ? 0 : 1;
  }
  const gboolean ok = _finish();
  g_mutex_unlock(&_mtl_lock);
  if(!ok)
  {
    dt_print(DT_DEBUG_OPENCL, "[slang_gpu] a kernel writing this buffer failed");
    return 1;
  }
  memcpy(host, buf->buffer.contents, size);
  return 0;
}
