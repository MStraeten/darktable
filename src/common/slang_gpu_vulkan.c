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

// Vulkan backend of slang_gpu.h

#include "common/slang_gpu.h"
#include "common/darktable.h"
#include "common/file_location.h"
#include "control/conf.h"

#include <vulkan/vulkan.h>

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DT_SLANG_MAX_PARAMS 32
// Vulkan guarantees 128 bytes of push constants, most devices have 256
#define DT_SLANG_PUSH_SIZE 256
// denoiseprofile holds ten full-size images at once
#define DT_SLANG_POOL_SIZE 12
#define DT_SLANG_BUFFER_POOL_SIZE 12
#define DT_SLANG_STAGING_POOL_SIZE 8
// submissions queued before the oldest is waited for
#define DT_SLANG_MAX_INFLIGHT 64
// host buffers up to this size are bound in host memory, larger ones copied to the device
#define DT_SLANG_DIRECT_BUFFER (256 * 1024)

typedef struct dt_slang_resource_t
{
  uint32_t binding;
  VkDescriptorType type;
} dt_slang_resource_t;

typedef struct dt_slang_kernel_t
{
  VkShaderModule module;
  VkDescriptorSetLayout set_layout;
  VkPipelineLayout layout;
  VkPipeline pipeline;
  uint32_t local_size[2];
  int nres;
  dt_slang_resource_t res[DT_SLANG_MAX_PARAMS]; // in binding order
  int nvals;
  uint32_t val_offset[DT_SLANG_MAX_PARAMS];     // in declaration order
} dt_slang_kernel_t;

typedef struct dt_slang_vulkan_t
{
  gboolean initialized;
  gboolean usable;
  VkInstance instance;
  VkPhysicalDevice pdev;
  VkPhysicalDeviceMemoryProperties memprops;
  uint32_t max_image_dim;
  uint32_t push_size; // up to DT_SLANG_PUSH_SIZE
  VkDevice dev;
  VkQueue queue;
  VkCommandPool pool;
  GHashTable *kernels; // "program:kernel" -> dt_slang_kernel_t
  GList *images;       // freed device images, for reuse
  GList *buffers;      // freed device buffers, for reuse
  GList *staging;      // staging buffers no batch uses, for reuse
  GList *doomed_images;  // evicted from the pools, freed once the queue is idle
  GList *doomed_buffers;
  GQueue inflight;     // submitted batches, oldest first
  GList *spare;        // finished batches, for reuse
  gboolean timestamps; // the queue can time kernels
  float timestamp_period; // ns per timestamp tick
  VkDeviceSize pool_budget; // bytes the image and buffer pools may keep together
} dt_slang_vulkan_t;

static dt_slang_vulkan_t _vk = { 0 };
static GMutex _vk_lock;
static GThread *_prepare_thread = NULL;

// --------------------------------------------------------------------------------------------
// device

static void _destroy_kernel(gpointer data)
{
  dt_slang_kernel_t *k = data;
  if(k->pipeline) vkDestroyPipeline(_vk.dev, k->pipeline, NULL);
  if(k->layout) vkDestroyPipelineLayout(_vk.dev, k->layout, NULL);
  if(k->set_layout) vkDestroyDescriptorSetLayout(_vk.dev, k->set_layout, NULL);
  if(k->module) vkDestroyShaderModule(_vk.dev, k->module, NULL);
  g_free(k);
}

static gboolean _init_device(void)
{
  const VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                                  .pApplicationName = "darktable",
                                  .apiVersion = VK_API_VERSION_1_2 };
  const VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                     .pApplicationInfo = &app };
  if(vkCreateInstance(&ici, NULL, &_vk.instance) != VK_SUCCESS)
  {
    dt_print(DT_DEBUG_OPENCL, "[slang_gpu] no Vulkan instance");
    return FALSE;
  }

  uint32_t ndev = 0;
  vkEnumeratePhysicalDevices(_vk.instance, &ndev, NULL);
  if(ndev == 0)
  {
    dt_print(DT_DEBUG_OPENCL, "[slang_gpu] no Vulkan device");
    return FALSE;
  }
  VkPhysicalDevice *devs = g_new(VkPhysicalDevice, ndev);
  vkEnumeratePhysicalDevices(_vk.instance, &ndev, devs);

  // slang_gpu_device picks a device by index or by part of its name. Without
  // it a discrete GPU wins over an integrated one, which wins over the rest:
  // on hybrid laptops the integrated GPU is usually enumerated first, and a
  // CPU implementation such as lavapipe is only a last resort
  const char *wanted = dt_conf_get_string_const("slang_gpu_device");
  char *end = NULL;
  const long wanted_index = wanted && *wanted ? strtol(wanted, &end, 10) : -1;
  const gboolean by_index = wanted && *wanted && end && *end == '\0';
  gchar *wanted_name = wanted && *wanted && !by_index ? g_utf8_casefold(wanted, -1) : NULL;

  int best = -1, best_rank = -1;
  for(uint32_t d = 0; d < ndev; d++)
  {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(devs[d], &props);
    gchar *name = g_utf8_casefold(props.deviceName, -1);
    const gboolean match = by_index ? wanted_index == (long)d : wanted_name && strstr(name, wanted_name) != NULL;
    g_free(name);
    const int rank = match ? 4
      : props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 3
      : props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 2
      : props.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU ? 1
      : 0;
    dt_print(DT_DEBUG_OPENCL, "[slang_gpu] Vulkan device %u: %s, %s, vendor 0x%04x%s", d, props.deviceName,
             props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? "discrete"
             : props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? "integrated"
             : props.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU ? "virtual"
             : props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU ? "cpu" : "other",
             props.vendorID, match ? ", requested" : "");
    if(rank > best_rank)
    {
      best = d;
      best_rank = rank;
    }
  }
  if((wanted_name || by_index) && best_rank < 4)
    dt_print(DT_DEBUG_OPENCL, "[slang_gpu] no Vulkan device matches slang_gpu_device '%s'", wanted);
  g_free(wanted_name);
  _vk.pdev = devs[best];
  g_free(devs);

  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(_vk.pdev, &props);
  _vk.max_image_dim = props.limits.maxImageDimension2D;
  _vk.push_size = MIN(DT_SLANG_PUSH_SIZE, props.limits.maxPushConstantsSize);
  vkGetPhysicalDeviceMemoryProperties(_vk.pdev, &_vk.memprops);
  for(uint32_t h = 0; h < _vk.memprops.memoryHeapCount; h++)
    dt_print(DT_DEBUG_OPENCL, "[slang_gpu] memory heap %u: %" PRIu64 " MB%s", h,
             (uint64_t)(_vk.memprops.memoryHeaps[h].size >> 20),
             _vk.memprops.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT ? ", device local" : "");
  for(uint32_t h = 0; h < _vk.memprops.memoryHeapCount; h++)
    if(_vk.memprops.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
      _vk.pool_budget = MAX(_vk.pool_budget, _vk.memprops.memoryHeaps[h].size / 4);

  VkPhysicalDeviceFeatures features;
  vkGetPhysicalDeviceFeatures(_vk.pdev, &features);
  if(!features.shaderStorageImageWriteWithoutFormat)
  {
    dt_print(DT_DEBUG_OPENCL, "[slang_gpu] %s can't write images without a format", props.deviceName);
    return FALSE;
  }

  uint32_t nfam = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(_vk.pdev, &nfam, NULL);
  VkQueueFamilyProperties *fam = g_new(VkQueueFamilyProperties, nfam);
  vkGetPhysicalDeviceQueueFamilyProperties(_vk.pdev, &nfam, fam);
  uint32_t family = UINT32_MAX;
  for(uint32_t f = 0; f < nfam && family == UINT32_MAX; f++)
    if(fam[f].queueFlags & VK_QUEUE_COMPUTE_BIT) family = f;
  _vk.timestamps = family != UINT32_MAX && fam[family].timestampValidBits > 0
                   && props.limits.timestampPeriod > 0.0f;
  _vk.timestamp_period = props.limits.timestampPeriod;
  g_free(fam);
  if(family == UINT32_MAX)
  {
    dt_print(DT_DEBUG_OPENCL, "[slang_gpu] %s has no compute queue", props.deviceName);
    return FALSE;
  }

  const float priority = 1.0f;
  const VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                        .queueFamilyIndex = family,
                                        .queueCount = 1,
                                        .pQueuePriorities = &priority };
  // 64-bit integers seed filmic's noise, as in OpenCL; nearly every device has them
  const VkPhysicalDeviceFeatures enable = { .shaderStorageImageWriteWithoutFormat = VK_TRUE,
                                            .shaderInt64 = features.shaderInt64 };
  const VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                                   .queueCreateInfoCount = 1,
                                   .pQueueCreateInfos = &qci,
                                   .pEnabledFeatures = &enable };
  if(vkCreateDevice(_vk.pdev, &dci, NULL, &_vk.dev) != VK_SUCCESS)
  {
    dt_print(DT_DEBUG_OPENCL, "[slang_gpu] can't create a device on %s", props.deviceName);
    return FALSE;
  }
  vkGetDeviceQueue(_vk.dev, family, 0, &_vk.queue);

  const VkCommandPoolCreateInfo cpci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                         .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT
                                                  | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                         .queueFamilyIndex = family };
  if(vkCreateCommandPool(_vk.dev, &cpci, NULL, &_vk.pool) != VK_SUCCESS) return FALSE;

  _vk.kernels = g_hash_table_new_full(g_str_hash, g_str_equal, g_free, _destroy_kernel);
  dt_print(DT_DEBUG_OPENCL, "[slang_gpu] using Vulkan device %s", props.deviceName);
  return TRUE;
}

gboolean dt_slang_gpu_enabled(void)
{
  if(!dt_conf_get_bool("slang_gpu")) return FALSE;

  g_mutex_lock(&_vk_lock);
  if(!_vk.initialized)
  {
    _vk.initialized = TRUE;
    _vk.usable = _init_device();
  }
  const gboolean usable = _vk.usable;
  g_mutex_unlock(&_vk_lock);
  return usable;
}

// --------------------------------------------------------------------------------------------
// kernels

// Slang lays out an entry point's parameters as push constants (the scalars,
// in declaration order) and descriptor bindings (the images and buffers, in
// declaration order but numbered across all entry points of the file). Both
// are read from the SPIR-V, so callers pass arguments positionally.
static gboolean _reflect(const uint32_t *code,
                         const size_t words,
                         const char *entry,
                         dt_slang_kernel_t *k)
{
  if(words < 5 || code[0] != 0x07230203) return FALSE;
  const uint32_t bound = code[3];

  int32_t *binding = g_new(int32_t, bound);
  uint32_t *pointee = g_new0(uint32_t, bound);   // pointer type -> pointee
  uint32_t *var_type = g_new0(uint32_t, bound);  // variable -> pointer type
  uint32_t *var_class = g_new0(uint32_t, bound); // variable -> storage class
  uint32_t *image_sampled = g_new0(uint32_t, bound);
  uint32_t *member_offset = NULL; // filled for the push constant struct only
  memset(binding, -1, sizeof(int32_t) * bound);

  uint32_t entry_id = 0;
  const uint32_t *iface = NULL;
  int niface = 0;

  // first pass: everything but member offsets, which need the struct id
  for(size_t i = 5; i < words;)
  {
    const uint32_t op = code[i] & 0xffff, n = code[i] >> 16;
    const uint32_t *w = code + i;
    if(n == 0 || i + n > words) goto error;
    switch(op)
    {
      case 15: // OpEntryPoint
        if(!strcmp((const char *)(w + 3), entry))
        {
          const size_t slen = strlen((const char *)(w + 3)) / 4 + 1;
          entry_id = w[2];
          iface = w + 3 + slen;
          niface = n - 3 - slen;
        }
        break;
      case 16: // OpExecutionMode
        if(entry_id && w[1] == entry_id && w[2] == 17) // LocalSize
        {
          k->local_size[0] = w[3];
          k->local_size[1] = w[4];
        }
        break;
      case 25: // OpTypeImage
        if(w[1] < bound) image_sampled[w[1]] = w[7];
        break;
      case 32: // OpTypePointer
        if(w[1] < bound) pointee[w[1]] = w[3];
        break;
      case 59: // OpVariable
        if(w[2] < bound)
        {
          var_type[w[2]] = w[1];
          var_class[w[2]] = w[3];
        }
        break;
      case 71: // OpDecorate
        if(w[2] == 33 && w[1] < bound) binding[w[1]] = w[3]; // Binding
        break;
    }
    i += n;
  }
  if(!iface || !k->local_size[0]) goto error;

  uint32_t push_struct = 0;
  k->nres = 0;
  for(int j = 0; j < niface; j++)
  {
    const uint32_t id = iface[j];
    if(id >= bound) continue;
    if(var_class[id] == 9) // PushConstant
      push_struct = pointee[var_type[id]];
    else if(binding[id] >= 0)
    {
      if(k->nres == DT_SLANG_MAX_PARAMS) goto error;
      const uint32_t type = pointee[var_type[id]];
      VkDescriptorType dtype;
      if(var_class[id] == 12) // StorageBuffer
        dtype = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      else if(image_sampled[type] == 1)
        dtype = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
      else if(image_sampled[type] == 2)
        dtype = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      else
        goto error;
      k->res[k->nres++] = (dt_slang_resource_t){ binding[id], dtype };
    }
  }
  // the entry point lists its variables in no particular order
  for(int a = 0; a < k->nres; a++)
    for(int b = a + 1; b < k->nres; b++)
      if(k->res[b].binding < k->res[a].binding)
      {
        const dt_slang_resource_t t = k->res[a];
        k->res[a] = k->res[b];
        k->res[b] = t;
      }

  k->nvals = 0;
  if(push_struct)
  {
    member_offset = g_new(uint32_t, DT_SLANG_MAX_PARAMS);
    memset(member_offset, 0xff, sizeof(uint32_t) * DT_SLANG_MAX_PARAMS);
    for(size_t i = 5; i < words;)
    {
      const uint32_t op = code[i] & 0xffff, n = code[i] >> 16;
      const uint32_t *w = code + i;
      if(op == 72 && w[1] == push_struct && w[3] == 35 && w[2] < DT_SLANG_MAX_PARAMS) // OpMemberDecorate Offset
      {
        member_offset[w[2]] = w[4];
        if((int)w[2] + 1 > k->nvals) k->nvals = w[2] + 1;
      }
      i += n;
    }
    // a kernel with more values than this device's push constants falls
    // back to the CPU
    for(int m = 0; m < k->nvals; m++)
    {
      if(member_offset[m] == UINT32_MAX || member_offset[m] + sizeof(float) > _vk.push_size) goto error;
      k->val_offset[m] = member_offset[m];
    }
  }

  g_free(binding);
  g_free(pointee);
  g_free(var_type);
  g_free(var_class);
  g_free(image_sampled);
  g_free(member_offset);
  return TRUE;

error:
  g_free(binding);
  g_free(pointee);
  g_free(var_type);
  g_free(var_class);
  g_free(image_sampled);
  g_free(member_offset);
  return FALSE;
}

static dt_slang_kernel_t *_get_kernel(const char *program, const char *entry)
{
  gchar *key = g_strdup_printf("%s:%s", program, entry);
  dt_slang_kernel_t *k = g_hash_table_lookup(_vk.kernels, key);
  if(k)
  {
    g_free(key);
    return k;
  }

  char datadir[PATH_MAX] = { 0 };
  dt_loc_get_datadir(datadir, sizeof(datadir));
  gchar *spv = g_strdup_printf("%s/kernels/slang/%s.spv", datadir, program);
  gchar *code = NULL;
  gsize len = 0;
  if(!g_file_get_contents(spv, &code, &len, NULL) || len % 4)
  {
    dt_print(DT_DEBUG_OPENCL, "[slang_gpu] can't read %s", spv);
    g_free(spv);
    g_free(key);
    g_free(code);
    return NULL;
  }

  k = g_new0(dt_slang_kernel_t, 1);
  if(!_reflect((const uint32_t *)code, len / 4, entry, k))
  {
    dt_print(DT_DEBUG_OPENCL, "[slang_gpu] can't find the parameters of %s in %s", entry, spv);
    goto error;
  }

  VkDescriptorSetLayoutBinding lb[DT_SLANG_MAX_PARAMS];
  for(int r = 0; r < k->nres; r++)
    lb[r] = (VkDescriptorSetLayoutBinding){ .binding = k->res[r].binding,
                                            .descriptorType = k->res[r].type,
                                            .descriptorCount = 1,
                                            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
  const VkDescriptorSetLayoutCreateInfo dslci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                                  .bindingCount = k->nres,
                                                  .pBindings = lb };
  const VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, _vk.push_size };
  const VkShaderModuleCreateInfo smci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                          .codeSize = len,
                                          .pCode = (const uint32_t *)code };
  if(vkCreateDescriptorSetLayout(_vk.dev, &dslci, NULL, &k->set_layout) != VK_SUCCESS
     || vkCreateShaderModule(_vk.dev, &smci, NULL, &k->module) != VK_SUCCESS)
    goto error;
  const VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                            .setLayoutCount = 1,
                                            .pSetLayouts = &k->set_layout,
                                            .pushConstantRangeCount = 1,
                                            .pPushConstantRanges = &pcr };
  if(vkCreatePipelineLayout(_vk.dev, &plci, NULL, &k->layout) != VK_SUCCESS) goto error;
  const VkComputePipelineCreateInfo cpci = {
    .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
               .stage = VK_SHADER_STAGE_COMPUTE_BIT,
               .module = k->module,
               .pName = entry },
    .layout = k->layout
  };
  if(vkCreateComputePipelines(_vk.dev, VK_NULL_HANDLE, 1, &cpci, NULL, &k->pipeline) != VK_SUCCESS)
  {
    dt_print(DT_DEBUG_OPENCL, "[slang_gpu] can't create the pipeline for %s", entry);
    goto error;
  }

  g_hash_table_insert(_vk.kernels, key, k);
  g_free(spv);
  g_free(code);
  return k;

error:
  _destroy_kernel(k);
  g_free(spv);
  g_free(key);
  g_free(code);
  return NULL;
}

// creates the pipelines of every entry point in every .spv in kernels/slang,
// which takes longer than running them the first time
static void _preload_kernels(void)
{
  char datadir[PATH_MAX] = { 0 };
  dt_loc_get_datadir(datadir, sizeof(datadir));
  gchar *dirname = g_build_filename(datadir, "kernels", "slang", NULL);
  GDir *dir = g_dir_open(dirname, 0, NULL);
  const gchar *file;
  while(dir && (file = g_dir_read_name(dir)))
  {
    if(!g_str_has_suffix(file, ".spv")) continue;
    gchar *path = g_build_filename(dirname, file, NULL);
    gchar *code = NULL;
    gsize len = 0;
    if(g_file_get_contents(path, &code, &len, NULL) && len % 4 == 0 && len >= 20)
    {
      gchar *program = g_strndup(file, strlen(file) - strlen(".spv"));
      const uint32_t *w = (const uint32_t *)code;
      const size_t words = len / 4;
      for(size_t i = 5; i < words;)
      {
        const uint32_t op = w[i] & 0xffff, n = w[i] >> 16;
        if(n == 0 || i + n > words) break;
        if(op == 15 && n > 3) // OpEntryPoint
        {
          gchar *entry = g_strndup((const char *)(w + i + 3), sizeof(uint32_t) * (n - 3));
          _get_kernel(program, entry);
          g_free(entry);
        }
        i += n;
      }
      g_free(program);
    }
    g_free(code);
    g_free(path);
  }
  if(dir) g_dir_close(dir);
  g_free(dirname);
}

static gpointer _prepare(gpointer data)
{
  const double start = dt_get_debug_wtime();
  g_mutex_lock(&_vk_lock);
  if(!_vk.initialized)
  {
    _vk.initialized = TRUE;
    _vk.usable = _init_device();
  }
  if(_vk.usable) _preload_kernels();
  g_mutex_unlock(&_vk_lock);
  dt_print(DT_DEBUG_PERF, "[slang_gpu] preparing the Vulkan device and kernels took %.3f secs",
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

typedef struct dt_slang_mem_t
{
  VkBuffer buffer;       // storage buffer, or staging buffer of an image
  VkDeviceMemory buffer_mem;
  void *map;
  VkImage image;
  VkDeviceMemory image_mem;
  VkImageView view;
  gboolean coherent;     // the mapped memory needs no invalidation before host reads
} dt_slang_mem_t;

static gboolean _reclaim(void);

// device memory, after freeing the pools if the device is out of it. Pooled
// images and buffers otherwise filled a 4 GB GPU after a few exports
static gboolean _allocate(const VkMemoryAllocateInfo *mai, VkDeviceMemory *mem)
{
  VkResult res = vkAllocateMemory(_vk.dev, mai, NULL, mem);
  if(res == VK_ERROR_OUT_OF_DEVICE_MEMORY && _reclaim())
    res = vkAllocateMemory(_vk.dev, mai, NULL, mem);
  return res == VK_SUCCESS;
}

static gboolean _memtype(const uint32_t bits, const VkMemoryPropertyFlags want, uint32_t *type)
{
  for(uint32_t i = 0; i < _vk.memprops.memoryTypeCount; i++)
    if((bits & (1u << i)) && (_vk.memprops.memoryTypes[i].propertyFlags & want) == want)
    {
      *type = i;
      return TRUE;
    }
  return FALSE;
}

// a mapped buffer in the first memory type matching one of the n wanted
// property sets, in order
static gboolean _alloc_mapped(dt_slang_mem_t *m,
                              const VkDeviceSize size,
                              const VkBufferUsageFlags usage,
                              const VkMemoryPropertyFlags *want,
                              const int n)
{
  const VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                   .size = size,
                                   .usage = usage };
  if(vkCreateBuffer(_vk.dev, &bci, NULL, &m->buffer) != VK_SUCCESS) return FALSE;
  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(_vk.dev, m->buffer, &req);
  uint32_t type;
  int i = 0;
  while(i < n && !_memtype(req.memoryTypeBits, want[i], &type)) i++;
  if(i == n) return FALSE;
  m->coherent = (_vk.memprops.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
  const VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                     .allocationSize = req.size,
                                     .memoryTypeIndex = type };
  // the whole allocation is mapped, which is what an invalidation of a
  // non-coherent type has to cover
  return vkAllocateMemory(_vk.dev, &mai, NULL, &m->buffer_mem) == VK_SUCCESS
         && vkBindBufferMemory(_vk.dev, m->buffer, m->buffer_mem, 0) == VK_SUCCESS
         && vkMapMemory(_vk.dev, m->buffer_mem, 0, VK_WHOLE_SIZE, 0, &m->map) == VK_SUCCESS;
}

// a mapped buffer the host writes for the device
static gboolean _alloc_upload(dt_slang_mem_t *m, const VkDeviceSize size)
{
  const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  return _alloc_mapped(m, size,
                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &want, 1);
}

// a mapped buffer the device copies results into for the host. Host reads
// from uncached memory are slow: on an NVIDIA laptop GPU, reading a
// 3908x2602 image took 600 ms from the coherent type the uploads use
static gboolean _alloc_readback(dt_slang_mem_t *m, const VkDeviceSize size)
{
  const VkMemoryPropertyFlags want[3]
      = { VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT
              | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT };
  return _alloc_mapped(m, size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, want, 3);
}

// a storage buffer in device memory, which the host never maps. Mapped
// device memory is limited to a 256 MB window on discrete GPUs without
// resizable BAR, so the large buffers used to end up in host memory, read
// over PCIe and cleared by the CPU
static gboolean _alloc_device(dt_slang_mem_t *m, const VkDeviceSize size)
{
  const VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                   .size = size,
                                   .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                                            | VK_BUFFER_USAGE_TRANSFER_DST_BIT };
  if(vkCreateBuffer(_vk.dev, &bci, NULL, &m->buffer) != VK_SUCCESS) return FALSE;
  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(_vk.dev, m->buffer, &req);
  uint32_t type;
  if(!_memtype(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &type)
     && !_memtype(req.memoryTypeBits, 0, &type))
    return FALSE;
  const VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                     .allocationSize = req.size,
                                     .memoryTypeIndex = type };
  return _allocate(&mai, &m->buffer_mem)
         && vkBindBufferMemory(_vk.dev, m->buffer, m->buffer_mem, 0) == VK_SUCCESS;
}

// makes what the device wrote to m visible to the host, after the commands
// writing it have finished
static void _invalidate(const dt_slang_mem_t *m)
{
  if(m->coherent) return;
  const VkMappedMemoryRange range = { .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                                      .memory = m->buffer_mem,
                                      .offset = 0,
                                      .size = VK_WHOLE_SIZE };
  vkInvalidateMappedMemoryRanges(_vk.dev, 1, &range);
}

static void _host_read_barrier(VkCommandBuffer cb)
{
  const VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                               .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                               .dstAccessMask = VK_ACCESS_HOST_READ_BIT };
  vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0,
                       NULL);
}

// orders everything recorded so far in cb before what follows
static void _memory_barrier(VkCommandBuffer cb)
{
  const VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                               .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                               .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT };
  vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                       0, 1, &mb, 0, NULL, 0, NULL);
}

static gboolean _create_image(VkImage *image,
                              VkDeviceMemory *mem,
                              VkImageView *view,
                              const int width,
                              const int height,
                              const int channels,
                              const VkImageUsageFlags usage)
{
  const VkFormat format = channels == 1 ? VK_FORMAT_R32_SFLOAT : VK_FORMAT_R32G32B32A32_SFLOAT;
  const VkImageCreateInfo ici = {
    .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
    .imageType = VK_IMAGE_TYPE_2D,
    .format = format,
    .extent = { width, height, 1 },
    .mipLevels = 1,
    .arrayLayers = 1,
    .samples = VK_SAMPLE_COUNT_1_BIT,
    .tiling = VK_IMAGE_TILING_OPTIMAL,
    .usage = usage,
    .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
  };
  if(vkCreateImage(_vk.dev, &ici, NULL, image) != VK_SUCCESS) return FALSE;
  VkMemoryRequirements req;
  vkGetImageMemoryRequirements(_vk.dev, *image, &req);
  uint32_t type;
  if(!_memtype(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &type)
     && !_memtype(req.memoryTypeBits, 0, &type))
    return FALSE;
  const VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                     .allocationSize = req.size,
                                     .memoryTypeIndex = type };
  if(!_allocate(&mai, mem)
     || vkBindImageMemory(_vk.dev, *image, *mem, 0) != VK_SUCCESS)
    return FALSE;
  const VkImageViewCreateInfo vci = { .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                      .image = *image,
                                      .viewType = VK_IMAGE_VIEW_TYPE_2D,
                                      .format = format,
                                      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
  return vkCreateImageView(_vk.dev, &vci, NULL, view) == VK_SUCCESS;
}

static void _free_mem(dt_slang_mem_t *m)
{
  if(m->view) vkDestroyImageView(_vk.dev, m->view, NULL);
  if(m->image) vkDestroyImage(_vk.dev, m->image, NULL);
  if(m->image_mem) vkFreeMemory(_vk.dev, m->image_mem, NULL);
  if(m->buffer) vkDestroyBuffer(_vk.dev, m->buffer, NULL);
  if(m->buffer_mem) vkFreeMemory(_vk.dev, m->buffer_mem, NULL);
}

static void _barrier(VkCommandBuffer cb, VkImage image, VkImageLayout from, VkImageLayout to)
{
  const VkImageMemoryBarrier b = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                   .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                                   .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                                   .oldLayout = from,
                                   .newLayout = to,
                                   .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .image = image,
                                   .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
  vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                       0, 0, NULL, 0, NULL, 1, &b);
}

// device images stay in the GENERAL layout, which allows sampling, storage
// writes and transfers alike, so kernels can take them as input or output
struct dt_slang_gpu_image_t
{
  VkImage image;
  VkDeviceMemory mem;
  VkImageView view;
  int width, height, channels;
};

// device buffers live in device memory; the host fills, clears and reads
// them with transfer commands
struct dt_slang_gpu_buffer_t
{
  dt_slang_mem_t mem;
  size_t size;
};

static void _image_free(dt_slang_gpu_image_t *img)
{
  if(img->view) vkDestroyImageView(_vk.dev, img->view, NULL);
  if(img->image) vkDestroyImage(_vk.dev, img->image, NULL);
  if(img->mem) vkFreeMemory(_vk.dev, img->mem, NULL);
  g_free(img);
}

static void _buffer_free(dt_slang_gpu_buffer_t *buf)
{
  _free_mem(&buf->mem);
  g_free(buf);
}

// a mapped buffer for one copy between host and device, kept for reuse
typedef struct dt_slang_staging_t
{
  dt_slang_mem_t mem;
  size_t size;
  gboolean to_device;
} dt_slang_staging_t;

static void _staging_destroy(dt_slang_staging_t *s)
{
  _free_mem(&s->mem);
  g_free(s);
}

// --------------------------------------------------------------------------------------------
// submissions

// Every call records its commands into a batch of its own and submits it
// without waiting, as OpenCL queues kernels: waiting for each one left the
// GPU idle between kernels and let it clock down. A batch keeps what its
// commands use (staging buffers, temporary images and buffers) until its
// fence shows it has finished. The queue runs batches in submission order,
// and each starts with a full memory barrier, so a kernel sees the results of
// those before it.
typedef struct dt_slang_batch_t
{
  VkCommandBuffer cb;
  VkFence fence;
  VkDescriptorPool dpool;
  VkQueryPool queries;          // start and end timestamp, with -d perf
  GList *staging;               // dt_slang_staging_t, back to the staging pool when done
  GList *buffers;               // dt_slang_gpu_buffer_t, back to the buffer pool when done
  GList *mems;                  // dt_slang_mem_t, freed when done
  const char *entry;            // the kernel, for the timing
  int width, height;
} dt_slang_batch_t;

static void _batch_destroy(dt_slang_batch_t *b)
{
  if(b->queries) vkDestroyQueryPool(_vk.dev, b->queries, NULL);
  if(b->dpool) vkDestroyDescriptorPool(_vk.dev, b->dpool, NULL);
  if(b->fence) vkDestroyFence(_vk.dev, b->fence, NULL);
  if(b->cb) vkFreeCommandBuffers(_vk.dev, _vk.pool, 1, &b->cb);
  g_free(b);
}

static void _staging_release(dt_slang_staging_t *s)
{
  _vk.staging = g_list_prepend(_vk.staging, s);
  GList *last = g_list_nth(_vk.staging, DT_SLANG_STAGING_POOL_SIZE);
  if(last)
  {
    last->prev->next = NULL;
    last->prev = NULL;
    g_list_free_full(last, (GDestroyNotify)_staging_destroy);
  }
}

// images and buffers that still may be in use are freed once the queue is idle
static void _free_doomed(void)
{
  g_list_free_full(_vk.doomed_images, (GDestroyNotify)_image_free);
  g_list_free_full(_vk.doomed_buffers, (GDestroyNotify)_buffer_free);
  _vk.doomed_images = NULL;
  _vk.doomed_buffers = NULL;
}

// cuts a pool list after the first max entries or once *bytes, counting
// on from the other pool, exceeds the budget. The cut entries may still be
// in use, so they are freed once the queue is idle
static GList *_trim(GList *list, const int max, VkDeviceSize *bytes, GList **doomed, const gboolean images)
{
  int n = 0;
  for(GList *l = list; l; l = g_list_next(l), n++)
  {
    if(images)
    {
      const dt_slang_gpu_image_t *i = l->data;
      *bytes += sizeof(float) * i->channels * i->width * i->height;
    }
    else
      *bytes += ((const dt_slang_gpu_buffer_t *)l->data)->size;
    if(n >= max || *bytes > _vk.pool_budget)
    {
      if(l->prev) l->prev->next = NULL;
      else list = NULL;
      l->prev = NULL;
      *doomed = g_list_concat(*doomed, l);
      break;
    }
  }
  return list;
}

// the pools keep what a module chain uses, within a quarter of the device
// memory: allocating images on every run left the GPU waiting
static void _trim_pools(void)
{
  VkDeviceSize bytes = 0;
  _vk.images = _trim(_vk.images, DT_SLANG_POOL_SIZE, &bytes, &_vk.doomed_images, TRUE);
  _vk.buffers = _trim(_vk.buffers, DT_SLANG_BUFFER_POOL_SIZE, &bytes, &_vk.doomed_buffers, FALSE);
}

static void _buffer_release(dt_slang_gpu_buffer_t *buf)
{
  // a demosaic chain holds eight buffers of the same size
  _vk.buffers = g_list_prepend(_vk.buffers, buf);
  _trim_pools();
}

// hands back what b used and keeps b for reuse. Pooled images and buffers
// may still be in use by earlier batches, which is fine: only later batches
// get them again
static void _batch_release(dt_slang_batch_t *b)
{
  b->entry = NULL;
  g_list_free_full(b->staging, (GDestroyNotify)_staging_release);
  g_list_free_full(b->buffers, (GDestroyNotify)_buffer_release);
  for(GList *l = b->mems; l; l = g_list_next(l))
  {
    _free_mem(l->data);
    g_free(l->data);
  }
  g_list_free(b->mems);
  b->staging = b->buffers = b->mems = NULL;
  vkResetCommandBuffer(b->cb, 0);
  vkResetDescriptorPool(_vk.dev, b->dpool, 0);
  _vk.spare = g_list_prepend(_vk.spare, b);
}

// a finished batch: logs its time and hands back what it used
static void _retire(dt_slang_batch_t *b)
{
  if(b->queries && b->entry)
  {
    uint64_t ts[2] = { 0 };
    if(vkGetQueryPoolResults(_vk.dev, b->queries, 0, 2, sizeof(ts), ts, sizeof(uint64_t),
                             VK_QUERY_RESULT_64_BIT) == VK_SUCCESS)
      dt_print(DT_DEBUG_PERF, "[slang_gpu] %s %dx%d took %.3f secs (Vulkan)", b->entry, b->width, b->height,
               (double)(ts[1] - ts[0]) * _vk.timestamp_period * 1e-9);
  }
  vkResetFences(_vk.dev, 1, &b->fence);
  _batch_release(b);
}

// retires the batches that have finished, oldest first
static void _retire_finished(void)
{
  dt_slang_batch_t *b;
  while((b = g_queue_peek_head(&_vk.inflight)) && vkGetFenceStatus(_vk.dev, b->fence) == VK_SUCCESS)
    _retire(g_queue_pop_head(&_vk.inflight));
  if(g_queue_is_empty(&_vk.inflight)) _free_doomed();
}

// waits for everything submitted so far
static void _wait_all(void)
{
  dt_slang_batch_t *b;
  while((b = g_queue_pop_head(&_vk.inflight)))
  {
    vkWaitForFences(_vk.dev, 1, &b->fence, VK_TRUE, UINT64_MAX);
    _retire(b);
  }
  _free_doomed();
}

// frees the pooled images and buffers for an allocation the device had no
// memory for. TRUE if there was something to free
static gboolean _reclaim(void)
{
  if(!_vk.images && !_vk.buffers && !_vk.doomed_images && !_vk.doomed_buffers) return FALSE;
  dt_print(DT_DEBUG_OPENCL, "[slang_gpu] out of device memory, freeing the pooled images and buffers");
  _wait_all();
  _vk.doomed_images = g_list_concat(_vk.doomed_images, _vk.images);
  _vk.doomed_buffers = g_list_concat(_vk.doomed_buffers, _vk.buffers);
  _vk.images = NULL;
  _vk.buffers = NULL;
  _free_doomed();
  return TRUE;
}

static dt_slang_batch_t *_batch_begin(const char *entry, const int width, const int height)
{
  _retire_finished();
  // a long chain of kernels without a read would otherwise hold its
  // staging buffers and temporaries all at once
  if(g_queue_get_length(&_vk.inflight) >= DT_SLANG_MAX_INFLIGHT)
  {
    dt_slang_batch_t *oldest = g_queue_pop_head(&_vk.inflight);
    vkWaitForFences(_vk.dev, 1, &oldest->fence, VK_TRUE, UINT64_MAX);
    _retire(oldest);
  }

  dt_slang_batch_t *b = NULL;
  if(_vk.spare)
  {
    b = _vk.spare->data;
    _vk.spare = g_list_delete_link(_vk.spare, _vk.spare);
  }
  else
  {
    b = g_new0(dt_slang_batch_t, 1);
    const VkCommandBufferAllocateInfo cbai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                               .commandPool = _vk.pool,
                                               .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                               .commandBufferCount = 1 };
    const VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    const VkDescriptorPoolSize sizes[3] = { { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, DT_SLANG_MAX_PARAMS },
                                            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, DT_SLANG_MAX_PARAMS },
                                            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, DT_SLANG_MAX_PARAMS } };
    const VkDescriptorPoolCreateInfo dpci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                              .maxSets = 1,
                                              .poolSizeCount = 3,
                                              .pPoolSizes = sizes };
    if(vkAllocateCommandBuffers(_vk.dev, &cbai, &b->cb) != VK_SUCCESS
       || vkCreateFence(_vk.dev, &fci, NULL, &b->fence) != VK_SUCCESS
       || vkCreateDescriptorPool(_vk.dev, &dpci, NULL, &b->dpool) != VK_SUCCESS)
    {
      _batch_destroy(b);
      return NULL;
    }
  }
  if(_vk.timestamps && !b->queries && (darktable.unmuted & DT_DEBUG_PERF))
  {
    const VkQueryPoolCreateInfo qpci = { .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                                         .queryType = VK_QUERY_TYPE_TIMESTAMP,
                                         .queryCount = 2 };
    if(vkCreateQueryPool(_vk.dev, &qpci, NULL, &b->queries) != VK_SUCCESS) b->queries = VK_NULL_HANDLE;
  }
  b->entry = entry;
  b->width = width;
  b->height = height;

  const VkCommandBufferBeginInfo begin = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                           .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
  vkBeginCommandBuffer(b->cb, &begin);
  // device images and buffers carry data from one batch to the next
  _memory_barrier(b->cb);
  return b;
}

// starts the timing of a batch's kernel, after its uploads
static void _batch_timestamp(dt_slang_batch_t *b, const int which)
{
  if(!b->queries || !b->entry) return;
  if(which == 0) vkCmdResetQueryPool(b->cb, b->queries, 0, 2);
  vkCmdWriteTimestamp(b->cb, which == 0 ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                      b->queries, which);
}

// submits b, and with wait waits for it and everything before it
static gboolean _batch_submit(dt_slang_batch_t *b, const gboolean wait)
{
  const VkSubmitInfo submit = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                .commandBufferCount = 1,
                                .pCommandBuffers = &b->cb };
  if(vkEndCommandBuffer(b->cb) != VK_SUCCESS
     || vkQueueSubmit(_vk.queue, 1, &submit, b->fence) != VK_SUCCESS)
  {
    _batch_release(b);
    return FALSE;
  }
  g_queue_push_tail(&_vk.inflight, b);
  if(wait) _wait_all();
  return TRUE;
}

// a staging buffer of at least bytes that no batch is using
static dt_slang_staging_t *_staging_acquire(const size_t bytes, const gboolean to_device)
{
  dt_slang_staging_t *best = NULL;
  GList *best_link = NULL;
  for(GList *l = _vk.staging; l; l = g_list_next(l))
  {
    dt_slang_staging_t *s = l->data;
    if(s->to_device == to_device && s->size >= bytes && (!best || s->size < best->size))
    {
      best = s;
      best_link = l;
    }
  }
  if(best)
  {
    _vk.staging = g_list_delete_link(_vk.staging, best_link);
    return best;
  }
  dt_slang_staging_t *s = g_new0(dt_slang_staging_t, 1);
  s->size = bytes;
  s->to_device = to_device;
  if(to_device ? !_alloc_upload(&s->mem, bytes) : !_alloc_readback(&s->mem, bytes))
  {
    _staging_destroy(s);
    return NULL;
  }
  return s;
}

// a device buffer of exactly size bytes, from the pool or new; its contents
// are whatever its last user left
static dt_slang_gpu_buffer_t *_buffer_acquire(const size_t size)
{
  for(GList *l = _vk.buffers; l; l = g_list_next(l))
  {
    dt_slang_gpu_buffer_t *pooled = l->data;
    if(pooled->size == size)
    {
      _vk.buffers = g_list_delete_link(_vk.buffers, l);
      return pooled;
    }
  }
  dt_slang_gpu_buffer_t *buf = g_new0(dt_slang_gpu_buffer_t, 1);
  buf->size = size;
  if(!_alloc_device(&buf->mem, size))
  {
    dt_print(DT_DEBUG_OPENCL, "[slang_gpu] can't allocate a device buffer of %zu bytes", size);
    _buffer_free(buf);
    return NULL;
  }
  return buf;
}

// records an upload of bytes from host into dst at offset 0 in b
static gboolean _record_upload(dt_slang_batch_t *b, const void *host, const size_t bytes, VkBuffer dst)
{
  dt_slang_staging_t *s = _staging_acquire(bytes, TRUE);
  if(!s) return FALSE;
  memcpy(s->mem.map, host, bytes);
  b->staging = g_list_prepend(b->staging, s);
  const VkBufferCopy copy = { 0, 0, bytes };
  vkCmdCopyBuffer(b->cb, s->mem.buffer, dst, 1, &copy);
  return TRUE;
}

// copies between host memory and a device image through a staging buffer.
// Writes don't wait; reads wait for the kernels before them.
static int _image_transfer(dt_slang_gpu_image_t *img, void *host, const gboolean to_device)
{
  const size_t bytes = sizeof(float) * img->channels * img->width * img->height;
  dt_slang_staging_t *s = _staging_acquire(bytes, to_device);
  if(!s) return 1;
  dt_slang_batch_t *b = _batch_begin(NULL, 0, 0);
  if(!b)
  {
    _staging_release(s);
    return 1;
  }
  const VkBufferImageCopy copy = { .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                                   .imageExtent = { img->width, img->height, 1 } };
  if(to_device)
  {
    memcpy(s->mem.map, host, bytes);
    b->staging = g_list_prepend(b->staging, s);
    vkCmdCopyBufferToImage(b->cb, s->mem.buffer, img->image, VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
    return _batch_submit(b, FALSE) ? 0 : 1;
  }

  vkCmdCopyImageToBuffer(b->cb, img->image, VK_IMAGE_LAYOUT_GENERAL, s->mem.buffer, 1, &copy);
  _host_read_barrier(b->cb);
  const gboolean ok = _batch_submit(b, TRUE);
  if(ok)
  {
    _invalidate(&s->mem);
    memcpy(host, s->mem.map, bytes);
  }
  _staging_release(s);
  return ok ? 0 : 1;
}

void dt_slang_gpu_cleanup(void)
{
  if(_prepare_thread) g_thread_join(_prepare_thread);
  _prepare_thread = NULL;
  g_mutex_lock(&_vk_lock);
  if(_vk.dev)
  {
    _wait_all();
    vkDeviceWaitIdle(_vk.dev);
  }
  g_list_free_full(_vk.spare, (GDestroyNotify)_batch_destroy);
  g_list_free_full(_vk.images, (GDestroyNotify)_image_free);
  g_list_free_full(_vk.buffers, (GDestroyNotify)_buffer_free);
  g_list_free_full(_vk.staging, (GDestroyNotify)_staging_destroy);
  if(_vk.kernels) g_hash_table_destroy(_vk.kernels);
  if(_vk.pool) vkDestroyCommandPool(_vk.dev, _vk.pool, NULL);
  if(_vk.dev) vkDestroyDevice(_vk.dev, NULL);
  if(_vk.instance) vkDestroyInstance(_vk.instance, NULL);
  memset(&_vk, 0, sizeof(_vk));
  g_mutex_unlock(&_vk_lock);
}

// --------------------------------------------------------------------------------------------
// images

dt_slang_gpu_image_t *dt_slang_gpu_image_new(const int width,
                                             const int height,
                                             const int channels,
                                             const void *host)
{
  if(!dt_slang_gpu_enabled()) return NULL;
  if(width <= 0 || height <= 0 || (channels != 1 && channels != 4)) return NULL;

  g_mutex_lock(&_vk_lock);
  // allocating and clearing a full image costs more than the kernels running
  // on it, and the pipe asks for the same sizes on every run
  for(GList *l = _vk.images; l; l = g_list_next(l))
  {
    dt_slang_gpu_image_t *img = l->data;
    if(img->width == width && img->height == height && img->channels == channels)
    {
      _vk.images = g_list_delete_link(_vk.images, l);
      if(host && _image_transfer(img, (void *)host, TRUE))
      {
        _vk.doomed_images = g_list_prepend(_vk.doomed_images, img);
        img = NULL;
      }
      g_mutex_unlock(&_vk_lock);
      return img;
    }
  }

  dt_slang_gpu_image_t *img = g_new0(dt_slang_gpu_image_t, 1);
  img->width = width;
  img->height = height;
  img->channels = channels;
  gboolean ok = (uint32_t)width <= _vk.max_image_dim && (uint32_t)height <= _vk.max_image_dim
    && _create_image(&img->image, &img->mem, &img->view, width, height, channels,
                     VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT
                     | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
  if(ok)
  {
    dt_slang_batch_t *b = _batch_begin(NULL, 0, 0);
    ok = b != NULL;
    if(ok)
    {
      _barrier(b->cb, img->image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
      ok = _batch_submit(b, FALSE);
    }
  }
  if(ok && host) ok = _image_transfer(img, (void *)host, TRUE) == 0;
  if(!ok)
  {
    dt_print(DT_DEBUG_OPENCL, "[slang_gpu] can't allocate a %dx%d device image", width, height);
    _wait_all();
    _image_free(img);
    img = NULL;
  }
  g_mutex_unlock(&_vk_lock);
  return img;
}

// a device image always holds a copy here
dt_slang_gpu_image_t *dt_slang_gpu_image_wrap(const int width,
                                              const int height,
                                              const int channels,
                                              const void *host)
{
  return host ? dt_slang_gpu_image_new(width, height, channels, host) : NULL;
}

dt_slang_gpu_image_t *dt_slang_gpu_image_alias(const int width,
                                               const int height,
                                               const int channels,
                                               void *host)
{
  return NULL;
}

void dt_slang_gpu_wait(void)
{
  if(!_vk.usable) return;
  g_mutex_lock(&_vk_lock);
  _wait_all();
  g_mutex_unlock(&_vk_lock);
}

void dt_slang_gpu_image_free(dt_slang_gpu_image_t *img)
{
  if(!img) return;
  g_mutex_lock(&_vk_lock);
  // kernels still running may use the image, so a pooled one is reused only
  // by later kernels
  _vk.images = g_list_prepend(_vk.images, img);
  _trim_pools();
  _retire_finished();
  g_mutex_unlock(&_vk_lock);
}

int dt_slang_gpu_image_width(const dt_slang_gpu_image_t *img)
{
  return img->width;
}

int dt_slang_gpu_image_height(const dt_slang_gpu_image_t *img)
{
  return img->height;
}

int dt_slang_gpu_image_read(dt_slang_gpu_image_t *img, void *host)
{
  const double start = dt_get_debug_wtime();
  g_mutex_lock(&_vk_lock);
  const int err = _image_transfer(img, host, FALSE);
  g_mutex_unlock(&_vk_lock);
  dt_print(DT_DEBUG_PERF, "[slang_gpu] reading %dx%d from the device took %.3f secs",
           img->width, img->height, dt_get_debug_wtime() - start);
  return err;
}

int dt_slang_gpu_image_write(dt_slang_gpu_image_t *img, const void *host)
{
  const double start = dt_get_debug_wtime();
  g_mutex_lock(&_vk_lock);
  const int err = _image_transfer(img, (void *)host, TRUE);
  g_mutex_unlock(&_vk_lock);
  dt_print(DT_DEBUG_PERF, "[slang_gpu] writing %dx%d to the device took %.3f secs",
           img->width, img->height, dt_get_debug_wtime() - start);
  return err;
}

// --------------------------------------------------------------------------------------------
// buffers

static dt_slang_gpu_buffer_t *_buffer_new(const size_t size, const void *host, const gboolean zero)
{
  if(!dt_slang_gpu_enabled() || size == 0) return NULL;

  g_mutex_lock(&_vk_lock);
  dt_slang_gpu_buffer_t *buf = _buffer_acquire(size);
  // neither a pooled buffer nor fresh device memory is zeroed
  if(buf && (host || zero))
  {
    dt_slang_batch_t *b = _batch_begin(NULL, 0, 0);
    gboolean ok = b != NULL;
    if(ok)
    {
      if(host)
        ok = _record_upload(b, host, size, buf->mem.buffer);
      else
        vkCmdFillBuffer(b->cb, buf->mem.buffer, 0, VK_WHOLE_SIZE, 0);
      ok = _batch_submit(b, FALSE) && ok;
    }
    if(!ok)
    {
      _buffer_release(buf);
      buf = NULL;
    }
  }
  g_mutex_unlock(&_vk_lock);
  return buf;
}

dt_slang_gpu_buffer_t *dt_slang_gpu_buffer_new(const size_t size, const void *host)
{
  return _buffer_new(size, host, TRUE);
}

dt_slang_gpu_buffer_t *dt_slang_gpu_buffer_new_uninit(const size_t size)
{
  return _buffer_new(size, NULL, FALSE);
}

void dt_slang_gpu_buffer_free(dt_slang_gpu_buffer_t *buf)
{
  if(!buf) return;
  g_mutex_lock(&_vk_lock);
  // as for images, kernels still running may use the buffer
  _buffer_release(buf);
  _retire_finished();
  g_mutex_unlock(&_vk_lock);
}

int dt_slang_gpu_buffer_read(dt_slang_gpu_buffer_t *buf, void *host, const size_t size)
{
  if(!buf || !host || size > buf->size) return 1;
  g_mutex_lock(&_vk_lock);
  int err = 1;
  dt_slang_staging_t *s = _staging_acquire(size, FALSE);
  dt_slang_batch_t *b = s ? _batch_begin(NULL, 0, 0) : NULL;
  if(b)
  {
    const VkBufferCopy copy = { 0, 0, size };
    vkCmdCopyBuffer(b->cb, buf->mem.buffer, s->mem.buffer, 1, &copy);
    _host_read_barrier(b->cb);
    if(_batch_submit(b, TRUE))
    {
      _invalidate(&s->mem);
      memcpy(host, s->mem.map, size);
      err = 0;
    }
  }
  if(s) _staging_release(s);
  g_mutex_unlock(&_vk_lock);
  return err;
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
  dt_slang_kernel_t *k = _get_kernel(program, entry);
  if(!k) return 1;

  // match the arguments to the kernel's parameters, images and buffers to
  // descriptor bindings and values to push constants, each in order
  int res_arg[DT_SLANG_MAX_PARAMS], nres = 0, nvals = 0;
  uint8_t push[DT_SLANG_PUSH_SIZE] = { 0 };
  gboolean host_out = FALSE;
  for(int a = 0; a < nargs; a++)
  {
    const dt_slang_arg_t *arg = &args[a];
    if(arg->kind == DT_SLANG_ARG_VALUE)
    {
      if(nvals >= k->nvals || k->val_offset[nvals] + arg->size > _vk.push_size) goto mismatch;
      memcpy(push + k->val_offset[nvals], arg->ptr, arg->size);
      nvals++;
      continue;
    }
    if(nres >= k->nres) goto mismatch;
    if(arg->kind == DT_SLANG_ARG_IMAGE_DEV || arg->kind == DT_SLANG_ARG_BUFFER_DEV)
    {
      if((k->res[nres].type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) != (arg->kind == DT_SLANG_ARG_BUFFER_DEV)
         || !arg->ptr)
        goto mismatch;
      res_arg[nres++] = a;
      continue;
    }
    const VkDescriptorType want = arg->kind == DT_SLANG_ARG_IMAGE_IN ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE
                                : arg->kind == DT_SLANG_ARG_IMAGE_OUT ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    if(k->res[nres].type != want) goto mismatch;
    if(arg->kind != DT_SLANG_ARG_BUFFER
       && (arg->width > _vk.max_image_dim || arg->height > _vk.max_image_dim
           || (arg->channels != 1 && arg->channels != 4)))
    {
      dt_print(DT_DEBUG_OPENCL, "[slang_gpu] %dx%d image too large for %s", arg->width, arg->height, entry);
      return 1;
    }
    if(arg->kind == DT_SLANG_ARG_IMAGE_OUT) host_out = TRUE;
    res_arg[nres++] = a;
  }
  if(nres != k->nres || nvals != k->nvals) goto mismatch;

  dt_slang_batch_t *b = _batch_begin(entry, width, height);
  if(!b) return 1;

  int err = 1;
  // host outputs are copied out after the wait, so their staging buffers
  // stay out of the batch
  dt_slang_staging_t *readback[DT_SLANG_MAX_PARAMS] = { NULL };
  VkImage images[DT_SLANG_MAX_PARAMS] = { VK_NULL_HANDLE };

  VkWriteDescriptorSet writes[DT_SLANG_MAX_PARAMS];
  VkDescriptorImageInfo image_info[DT_SLANG_MAX_PARAMS];
  VkDescriptorBufferInfo buffer_info[DT_SLANG_MAX_PARAMS];
  for(int r = 0; r < nres; r++)
  {
    const dt_slang_arg_t *arg = &args[res_arg[r]];
    writes[r] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                        .dstBinding = k->res[r].binding,
                                        .descriptorCount = 1,
                                        .descriptorType = k->res[r].type };
    if(arg->kind == DT_SLANG_ARG_IMAGE_DEV)
    {
      const dt_slang_gpu_image_t *img = arg->ptr;
      image_info[r] = (VkDescriptorImageInfo){ .imageView = img->view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
      writes[r].pImageInfo = &image_info[r];
    }
    else if(arg->kind == DT_SLANG_ARG_BUFFER_DEV)
    {
      const dt_slang_gpu_buffer_t *buf = arg->ptr;
      buffer_info[r] = (VkDescriptorBufferInfo){ buf->mem.buffer, 0, buf->size };
      writes[r].pBufferInfo = &buffer_info[r];
    }
    else if(arg->kind == DT_SLANG_ARG_BUFFER)
    {
      // small tables are read from the staging buffer itself; large ones,
      // such as lens's per-pixel coordinates, get copied to device memory
      if(arg->size <= DT_SLANG_DIRECT_BUFFER)
      {
        dt_slang_staging_t *s = _staging_acquire(arg->size, TRUE);
        if(!s) goto done;
        memcpy(s->mem.map, arg->ptr, arg->size);
        b->staging = g_list_prepend(b->staging, s);
        buffer_info[r] = (VkDescriptorBufferInfo){ s->mem.buffer, 0, arg->size };
      }
      else
      {
        dt_slang_gpu_buffer_t *buf = _buffer_acquire(arg->size);
        if(!buf) goto done;
        b->buffers = g_list_prepend(b->buffers, buf);
        if(!_record_upload(b, arg->ptr, arg->size, buf->mem.buffer)) goto done;
        buffer_info[r] = (VkDescriptorBufferInfo){ buf->mem.buffer, 0, arg->size };
      }
      writes[r].pBufferInfo = &buffer_info[r];
    }
    else
    {
      const gboolean out = arg->kind == DT_SLANG_ARG_IMAGE_OUT;
      dt_slang_mem_t *m = g_new0(dt_slang_mem_t, 1);
      b->mems = g_list_prepend(b->mems, m);
      if(!_create_image(&m->image, &m->image_mem, &m->view, arg->width, arg->height, arg->channels,
                        out ? VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                            : VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT))
        goto done;
      images[r] = m->image;
      const size_t bytes = sizeof(float) * arg->channels * arg->width * arg->height;
      const VkBufferImageCopy copy = { .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                                       .imageExtent = { arg->width, arg->height, 1 } };
      if(out)
      {
        readback[r] = _staging_acquire(bytes, FALSE);
        if(!readback[r]) goto done;
        _barrier(b->cb, m->image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
      }
      else
      {
        dt_slang_staging_t *s = _staging_acquire(bytes, TRUE);
        if(!s) goto done;
        memcpy(s->mem.map, arg->ptr, bytes);
        b->staging = g_list_prepend(b->staging, s);
        _barrier(b->cb, m->image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        vkCmdCopyBufferToImage(b->cb, s->mem.buffer, m->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        _barrier(b->cb, m->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
      }
      image_info[r] = (VkDescriptorImageInfo){
        .imageView = m->view,
        .imageLayout = out ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
      writes[r].pImageInfo = &image_info[r];
    }
  }

  const VkDescriptorSetAllocateInfo dsai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                             .descriptorPool = b->dpool,
                                             .descriptorSetCount = 1,
                                             .pSetLayouts = &k->set_layout };
  VkDescriptorSet set;
  if(vkAllocateDescriptorSets(_vk.dev, &dsai, &set) != VK_SUCCESS) goto done;
  for(int r = 0; r < nres; r++) writes[r].dstSet = set;
  vkUpdateDescriptorSets(_vk.dev, nres, writes, 0, NULL);

  // the uploads before the kernel
  _memory_barrier(b->cb);
  _batch_timestamp(b, 0);
  vkCmdBindPipeline(b->cb, VK_PIPELINE_BIND_POINT_COMPUTE, k->pipeline);
  vkCmdBindDescriptorSets(b->cb, VK_PIPELINE_BIND_POINT_COMPUTE, k->layout, 0, 1, &set, 0, NULL);
  vkCmdPushConstants(b->cb, k->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, _vk.push_size, push);
  vkCmdDispatch(b->cb, (width + k->local_size[0] - 1) / k->local_size[0],
                (height + k->local_size[1] - 1) / k->local_size[1], 1);
  _batch_timestamp(b, 1);
  for(int r = 0; r < nres; r++)
  {
    const dt_slang_arg_t *arg = &args[res_arg[r]];
    if(arg->kind != DT_SLANG_ARG_IMAGE_OUT) continue;
    const VkBufferImageCopy copy = { .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                                     .imageExtent = { arg->width, arg->height, 1 } };
    _barrier(b->cb, images[r], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    vkCmdCopyImageToBuffer(b->cb, images[r], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback[r]->mem.buffer,
                           1, &copy);
  }
  if(host_out) _host_read_barrier(b->cb);
  // a kernel with host outputs waits for them; the others only queue
  const gboolean submitted = _batch_submit(b, host_out);
  b = NULL;
  if(!submitted) goto done;

  for(int r = 0; r < nres; r++)
  {
    const dt_slang_arg_t *arg = &args[res_arg[r]];
    if(arg->kind != DT_SLANG_ARG_IMAGE_OUT) continue;
    _invalidate(&readback[r]->mem);
    memcpy((void *)arg->ptr, readback[r]->mem.map, sizeof(float) * arg->channels * arg->width * arg->height);
  }
  err = 0;

done:
  // a batch that failed while recording never runs, but hands back what it took
  if(b) _batch_release(b);
  for(int r = 0; r < nres; r++)
    if(readback[r]) _staging_release(readback[r]);
  if(err) dt_print(DT_DEBUG_OPENCL, "[slang_gpu] running %s failed", entry);
  return err;

mismatch:
  dt_print(DT_DEBUG_OPENCL, "[slang_gpu] arguments don't match the parameters of %s:"
           " %d resources and %d values expected", entry, k->nres, k->nvals);
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

  // one queue for all pipes: pipes running in parallel take turns here. The
  // time of each kernel is logged from GPU timestamps once it has finished
  g_mutex_lock(&_vk_lock);
  const int err = _run(program, kernel, width, height, nargs, args);
  g_mutex_unlock(&_vk_lock);
  return err;
}
