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

#include <stdint.h>
#include <string.h>

#define DT_SLANG_MAX_PARAMS 32

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
  NSUInteger max_image_dim;
} dt_slang_metal_t;

static dt_slang_metal_t _mtl = { 0 };
static GMutex _mtl_lock;

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
  g_mutex_lock(&_mtl_lock);
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

// --------------------------------------------------------------------------------------------
// dispatch

static id<MTLTexture> _texture(const dt_slang_arg_t *arg)
{
  MTLTextureDescriptor *d = [MTLTextureDescriptor
    texture2DDescriptorWithPixelFormat:arg->channels == 1 ? MTLPixelFormatR32Float : MTLPixelFormatRGBA32Float
                                 width:arg->width
                                height:arg->height
                             mipmapped:NO];
  // with unified memory the CPU reads and writes the texture directly;
  // otherwise a managed texture keeps a copy on each side. macOS 27 deprecates
  // managed storage because Apple silicon has no use for it, Intel Macs still do.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
  d.storageMode = _mtl.dev.hasUnifiedMemory ? MTLStorageModeShared : MTLStorageModeManaged;
#pragma clang diagnostic pop
  d.usage = arg->kind == DT_SLANG_ARG_IMAGE_OUT ? MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite
                                                : MTLTextureUsageShaderRead;
  id<MTLTexture> t = [_mtl.dev newTextureWithDescriptor:d];
  if(t && arg->kind == DT_SLANG_ARG_IMAGE_IN)
    [t replaceRegion:MTLRegionMake2D(0, 0, arg->width, arg->height)
         mipmapLevel:0
           withBytes:arg->ptr
         bytesPerRow:sizeof(float) * arg->channels * arg->width];
  return t;
}

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
  id<MTLTexture> textures[DT_SLANG_MAX_PARAMS];
  id<MTLCommandBuffer> cb = nil;
  id<MTLComputeCommandEncoder> enc = nil;

  // match the arguments to the kernel's parameters, see _reflect()
  int tex_arg[DT_SLANG_MAX_PARAMS], buf_arg[DT_SLANG_MAX_PARAMS];
  int ntex = 0, nbuf = 0, nvals = 0;
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
    else if(arg->kind == DT_SLANG_ARG_BUFFER)
    {
      if(nbuf >= k->nbuf) goto mismatch;
      buf_arg[nbuf++] = a;
    }
    else
    {
      if(ntex >= k->ntex || k->tex_write[ntex] != (arg->kind == DT_SLANG_ARG_IMAGE_OUT)) goto mismatch;
      if(arg->width > _mtl.max_image_dim || arg->height > _mtl.max_image_dim
         || (arg->channels != 1 && arg->channels != 4))
      {
        dt_print(DT_DEBUG_OPENCL, "[slang_gpu] %dx%d image too large for %s", arg->width, arg->height, entry);
        return 1;
      }
      tex_arg[ntex++] = a;
    }
  }
  if(ntex != k->ntex || nbuf != k->nbuf || nvals != k->nvals) goto mismatch;

  cb = [_mtl.queue commandBuffer];
  enc = [cb computeCommandEncoder];
  if(!cb || !enc) goto failed;
  [enc setComputePipelineState:k->pipeline];
  if(k->has_vals) [enc setBytes:vals length:k->vals_size atIndex:k->vals_index];
  for(int b = 0; b < nbuf; b++)
  {
    const dt_slang_arg_t *arg = &args[buf_arg[b]];
    id<MTLBuffer> buf = [_mtl.dev newBufferWithBytes:arg->ptr
                                              length:arg->size
                                             options:MTLResourceStorageModeShared];
    if(!buf) goto failed;
    [enc setBuffer:buf offset:0 atIndex:k->buf_index[b]];
  }
  for(int t = 0; t < ntex; t++)
  {
    textures[t] = _texture(&args[tex_arg[t]]);
    if(!textures[t]) goto failed;
    [enc setTexture:textures[t] atIndex:k->tex_index[t]];
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

  if(!_mtl.dev.hasUnifiedMemory)
  {
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    for(int t = 0; t < ntex; t++)
      if(args[tex_arg[t]].kind == DT_SLANG_ARG_IMAGE_OUT) [blit synchronizeResource:textures[t]];
#pragma clang diagnostic pop
    [blit endEncoding];
  }
  [cb commit];
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
    if(arg->kind != DT_SLANG_ARG_IMAGE_OUT) continue;
    [textures[t] getBytes:(void *)arg->ptr
              bytesPerRow:sizeof(float) * arg->channels * arg->width
               fromRegion:MTLRegionMake2D(0, 0, arg->width, arg->height)
              mipmapLevel:0];
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
    dt_print(DT_DEBUG_PERF, "[slang_gpu] %s %dx%d took %.3f secs (Metal, with copies)",
             kernel, width, height, dt_get_debug_wtime() - start);
  return err;
}
