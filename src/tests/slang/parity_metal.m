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

// The Metal backend of parity.c: runs a test case's kernel from the
// <program>.<entry>.metallib that data/kernels/slang/CMakeLists.txt builds,
// binding the arguments as src/common/slang_gpu_metal.m does, with images as
// linear textures on shared buffers.

#import <Metal/Metal.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "parity.h"

// the macOS 15 SDK renamed MTLPipelineOptionArgumentInfo, see slang_gpu_metal.m
#if !defined(MAC_OS_VERSION_15_0)
#define MTLPipelineOptionBindingInfo MTLPipelineOptionArgumentInfo
#endif

static id<MTLDevice> dev;
static id<MTLCommandQueue> queue;

// DT_PARITY_METAL_DEVICE picks the device by index or by part of its name.
// Without it the system default is used.
void mtl_init(char *name, const size_t size)
{
  NSArray<id<MTLDevice>> *devs = MTLCopyAllDevices();
  if(devs.count == 0)
  {
    printf("no Metal device, skipping\n");
    exit(SKIP);
  }
  const char *wanted = getenv("DT_PARITY_METAL_DEVICE");
  id<MTLDevice> def = MTLCreateSystemDefaultDevice();
  for(NSUInteger d = 0; d < devs.count; d++)
  {
    const int pick = !dev && (wanted && *wanted ? device_wanted(wanted, (int)d, devs[d].name.UTF8String)
                                                : devs[d].registryID == def.registryID);
    if(pick) dev = devs[d];
    printf("Metal device %lu: %s%s\n", (unsigned long)d, devs[d].name.UTF8String, pick ? ", used" : "");
  }
  if(!dev && wanted && *wanted) die("no Metal device matches DT_PARITY_METAL_DEVICE '%s'", wanted);
  if(!dev) dev = def ? def : devs[0];
  queue = [dev newCommandQueue];
  if(!queue) die("can't create a Metal command queue");
  snprintf(name, size, "%s", dev.name.UTF8String);
}

// Slang names the Metal parameters of an entry point entryPointParams_<name>_<n>,
// with _tex before the suffix for the texture of an ImageIn or ImageOut, and
// gathers its scalars in a struct entryPointParams_<n>, whose members are
// <name>_<n>. This returns <name>, or "" for that struct.
static NSString *param_name(NSString *s)
{
  if([s hasPrefix:@"entryPointParams_"]) s = [s substringFromIndex:17];
  const NSRange r = [s rangeOfString:@"_" options:NSBackwardsSearch];
  NSString *suffix = r.location == NSNotFound ? s : [s substringFromIndex:r.location + 1];
  if(suffix.length
     && [suffix rangeOfCharacterFromSet:[[NSCharacterSet decimalDigitCharacterSet] invertedSet]].location
          == NSNotFound)
    s = r.location == NSNotFound ? @"" : [s substringToIndex:r.location];
  if([s hasSuffix:@"_tex"]) s = [s substringToIndex:s.length - 4];
  return s;
}

static id<MTLTexture> image(const int w, const int h, const int channels, const void *px,
                            __strong id<MTLBuffer> *buffer, size_t *pitch)
{
  const MTLPixelFormat format = channels == 1 ? MTLPixelFormatR32Float : MTLPixelFormatRGBA32Float;
  const size_t row = sizeof(float) * channels * w;
  const size_t align = [dev minimumLinearTextureAlignmentForPixelFormat:format];
  *pitch = (row + align - 1) / align * align;
  *buffer = [dev newBufferWithLength:*pitch * h options:MTLResourceStorageModeShared];
  if(!*buffer) die("can't allocate a %dx%d image", w, h);
  for(int y = 0; y < h; y++)
    memcpy((char *)(*buffer).contents + y * *pitch, (const char *)px + y * row, row);
  MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                               width:w
                                                                              height:h
                                                                           mipmapped:NO];
  d.storageMode = MTLStorageModeShared;
  d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  id<MTLTexture> t = [*buffer newTextureWithDescriptor:d offset:0 bytesPerRow:*pitch];
  if(!t) die("can't create a %dx%d texture", w, h);
  return t;
}

void mtl_run(const char *dir, const test_t *t, void **outs)
{
  @autoreleasepool
  {
    const char *entry = t->slang_kernel ? t->slang_kernel : t->kernel;
    NSString *path = [NSString stringWithFormat:@"%s/%s.%s.metallib", dir, t->program, entry];
    NSError *error = nil;
    id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];
    if(!lib) die("can't load %s: %s", path.UTF8String, error.localizedDescription.UTF8String);
    id<MTLFunction> fn = [lib newFunctionWithName:@(entry)];
    if(!fn) die("no kernel %s in %s", entry, path.UTF8String);
    MTLComputePipelineReflection *refl = nil;
    id<MTLComputePipelineState> pipe = [dev newComputePipelineStateWithFunction:fn
                                                                        options:MTLPipelineOptionBindingInfo
                                                                     reflection:&refl
                                                                          error:&error];
    if(!pipe) die("can't create the pipeline for %s: %s", entry, error.localizedDescription.UTF8String);

    // a parameter the kernel never reads has no binding
    NSMutableDictionary<NSString *, id<MTLBinding>> *bindings = [NSMutableDictionary dictionary];
    NSMutableSet<NSString *> *unbound = [NSMutableSet set];
    id<MTLBufferBinding> vals_binding = nil;
    for(id<MTLBinding> b in refl.bindings)
    {
      NSString *pn = param_name(b.name);
      if(b.type == MTLBindingTypeBuffer && pn.length == 0)
        vals_binding = (id<MTLBufferBinding>)b;
      else if(b.type == MTLBindingTypeBuffer || b.type == MTLBindingTypeTexture)
      {
        bindings[pn] = b;
        [unbound addObject:pn];
      }
      else
        die("%s: unexpected parameter %s", entry, b.name.UTF8String);
    }
    // each member of the scalars' struct, and the bytes up to the next one
    NSMutableDictionary<NSString *, NSArray<NSNumber *> *> *members = [NSMutableDictionary dictionary];
    NSUInteger vals_size = 0;
    if(vals_binding)
    {
      vals_size = vals_binding.bufferDataSize;
      NSArray<MTLStructMember *> *sorted = [vals_binding.bufferStructType.members
        sortedArrayUsingComparator:^NSComparisonResult(MTLStructMember *x, MTLStructMember *y) {
          return x.offset < y.offset ? NSOrderedAscending : x.offset > y.offset ? NSOrderedDescending : NSOrderedSame;
        }];
      for(NSUInteger m = 0; m < sorted.count; m++)
      {
        const NSUInteger end = m + 1 < sorted.count ? sorted[m + 1].offset : vals_size;
        members[param_name(sorted[m].name)] = @[ @(sorted[m].offset), @(end - sorted[m].offset) ];
      }
    }
    uint8_t vals[512] = { 0 };
    if(vals_size > sizeof(vals)) die("%s: %lu bytes of values", entry, (unsigned long)vals_size);

    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipe];
    id<MTLBuffer> bufs[MAX_ARGS] = { nil };
    size_t pitch[MAX_ARGS] = { 0 };
    for(int a = 0; a < t->nargs; a++)
    {
      const arg_t *arg = &t->args[a];
      NSString *an = @(arg->name);
      if(arg->kind == ARG_LOCAL || arg->cl_only) continue;
      id<MTLBinding> b = bindings[an];
      NSArray<NSNumber *> *member = members[an];
      if(!b && !member)
      {
        if(out_size(arg)) die("%s: output %s not in the Metal library", entry, arg->name);
        continue;
      }
      switch(arg->kind)
      {
        case ARG_IMAGE_IN:
        case ARG_IMAGE_OUT:
        {
          if(!b || b.type != MTLBindingTypeTexture) die("%s: %s is not a texture", entry, arg->name);
          const void *px = arg->kind == ARG_IMAGE_OUT ? initial_data(arg) : arg->pixels;
          [enc setTexture:image(arg->w, arg->h, arg->channels, px, &bufs[a], &pitch[a]) atIndex:b.index];
          break;
        }
        case ARG_BUFFER:
        case ARG_BUFFER_OUT:
        case ARG_RAW_U16:
        case ARG_FLOAT4_BUF:
        {
          if(!b || b.type != MTLBindingTypeBuffer) die("%s: %s is not a buffer", entry, arg->name);
          const int is_value = arg->kind == ARG_FLOAT4_BUF;
          const size_t size = is_value ? sizeof(arg->v) : arg->size;
          const void *src = is_value ? (const void *)arg->v
                          : arg->kind == ARG_BUFFER_OUT ? initial_data(arg) : arg->data;
          bufs[a] = [dev newBufferWithBytes:src length:size options:MTLResourceStorageModeShared];
          if(!bufs[a]) die("%s: can't allocate %zu bytes for %s", entry, size, arg->name);
          [enc setBuffer:bufs[a] offset:0 atIndex:b.index];
          break;
        }
        case ARG_LOCAL:
          break;
        case ARG_INT:
        case ARG_FLOAT:
        case ARG_FLOAT4:
        case ARG_INT2:
        case ARG_INT4:
        case ARG_STRUCT:
        case ARG_FLOAT2:
        case ARG_CHAR2:
        {
          if(!member) die("%s: %s is not a value", entry, arg->name);
          const size_t sz = arg->kind == ARG_STRUCT ? arg->size
                          : arg->kind == ARG_FLOAT4 || arg->kind == ARG_INT4 ? 16
                          : arg->kind == ARG_INT2 || arg->kind == ARG_FLOAT2 || arg->kind == ARG_CHAR2 ? 8 : 4;
          const void *src = arg->kind == ARG_STRUCT ? arg->data
                          : arg->kind == ARG_INT ? (const void *)&arg->i
                          : arg->kind == ARG_FLOAT ? (const void *)&arg->f
                          : arg->kind == ARG_INT2 || arg->kind == ARG_INT4 || arg->kind == ARG_CHAR2
                            ? (const void *)arg->iv
                          : (const void *)arg->v;
          if(sz > member[1].unsignedIntegerValue) die("%s: %s takes %zu bytes, has room for %lu", entry,
                                                      arg->name, sz, (unsigned long)member[1].unsignedIntegerValue);
          memcpy(vals + member[0].unsignedIntegerValue, src, sz);
          break;
        }
      }
      [unbound removeObject:an];
    }
    if(unbound.count) die("%s: no argument for %s", entry, unbound.anyObject.UTF8String);
    if(vals_binding) [enc setBytes:vals length:vals_size atIndex:vals_binding.index];

    // the kernels are written for 16x16 work groups, see [numthreads] in the
    // .slang files; smaller groups as in slang_gpu_metal.m where a kernel
    // can't have that many threads
    NSUInteger tw = 16, th = 16;
    if(pipe.maxTotalThreadsPerThreadgroup < tw * th)
    {
      tw = pipe.threadExecutionWidth;
      th = MAX(1, pipe.maxTotalThreadsPerThreadgroup / tw);
    }
    [enc dispatchThreadgroups:MTLSizeMake((t->gw + tw - 1) / tw, (t->gh + th - 1) / th, 1)
        threadsPerThreadgroup:MTLSizeMake(tw, th, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if(cb.status != MTLCommandBufferStatusCompleted)
      die("running %s failed: %s", entry, cb.error.localizedDescription.UTF8String);

    for(int a = 0; a < t->nargs; a++)
    {
      const arg_t *arg = &t->args[a];
      if(!bufs[a] || !out_size(arg)) continue;
      if(arg->kind == ARG_BUFFER_OUT)
        memcpy(outs[a], bufs[a].contents, arg->size);
      else
      {
        const size_t row = sizeof(float) * arg->channels * arg->w;
        for(int y = 0; y < arg->h; y++)
          memcpy((char *)outs[a] + y * row, (const char *)bufs[a].contents + y * pitch[a], row);
      }
    }
  }
}
