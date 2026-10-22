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

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DT_SLANG_MAX_PARAMS 32
// every Vulkan device supports at least 128 bytes of push constants
#define DT_SLANG_PUSH_SIZE 128
#define DT_SLANG_POOL_SIZE 4
#define DT_SLANG_BUFFER_POOL_SIZE 12

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
  VkDevice dev;
  VkQueue queue;
  VkCommandPool pool;
  GHashTable *kernels; // "program:kernel" -> dt_slang_kernel_t
  GList *images;       // freed device images, for reuse
  GList *buffers;      // freed device buffers, for reuse
} dt_slang_vulkan_t;

static dt_slang_vulkan_t _vk = { 0 };
static GMutex _vk_lock;
static GThread *_prepare_thread = NULL;

// --------------------------------------------------------------------------------------------
// device

static void _image_free(struct dt_slang_gpu_image_t *img);
static void _buffer_free(struct dt_slang_gpu_buffer_t *buf);

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
  // prefer a real GPU over a CPU implementation such as lavapipe
  _vk.pdev = devs[0];
  for(uint32_t d = 0; d < ndev; d++)
  {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(devs[d], &props);
    if(props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
       || props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
    {
      _vk.pdev = devs[d];
      break;
    }
  }
  g_free(devs);

  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(_vk.pdev, &props);
  _vk.max_image_dim = props.limits.maxImageDimension2D;
  vkGetPhysicalDeviceMemoryProperties(_vk.pdev, &_vk.memprops);

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
  const VkPhysicalDeviceFeatures enable = { .shaderStorageImageWriteWithoutFormat = VK_TRUE };
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
                                         .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
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

void dt_slang_gpu_cleanup(void)
{
  if(_prepare_thread) g_thread_join(_prepare_thread);
  _prepare_thread = NULL;
  g_mutex_lock(&_vk_lock);
  if(_vk.dev) vkDeviceWaitIdle(_vk.dev);
  g_list_free_full(_vk.images, (GDestroyNotify)_image_free);
  g_list_free_full(_vk.buffers, (GDestroyNotify)_buffer_free);
  if(_vk.kernels) g_hash_table_destroy(_vk.kernels);
  if(_vk.pool) vkDestroyCommandPool(_vk.dev, _vk.pool, NULL);
  if(_vk.dev) vkDestroyDevice(_vk.dev, NULL);
  if(_vk.instance) vkDestroyInstance(_vk.instance, NULL);
  memset(&_vk, 0, sizeof(_vk));
  g_mutex_unlock(&_vk_lock);
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
    for(int m = 0; m < k->nvals; m++)
    {
      if(member_offset[m] == UINT32_MAX) goto error;
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
  const VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, DT_SLANG_PUSH_SIZE };
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
} dt_slang_mem_t;

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

// a mapped buffer, in device local memory if prefer_device and the device
// has some that the host can map
static gboolean _alloc_buffer_in(dt_slang_mem_t *m,
                                 const VkDeviceSize size,
                                 const VkBufferUsageFlags usage,
                                 const gboolean prefer_device)
{
  const VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                   .size = size,
                                   .usage = usage };
  if(vkCreateBuffer(_vk.dev, &bci, NULL, &m->buffer) != VK_SUCCESS) return FALSE;
  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(_vk.dev, m->buffer, &req);
  const VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  uint32_t type;
  if(!(prefer_device && _memtype(req.memoryTypeBits, host | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &type))
     && !_memtype(req.memoryTypeBits, host, &type))
    return FALSE;
  const VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                     .allocationSize = req.size,
                                     .memoryTypeIndex = type };
  return vkAllocateMemory(_vk.dev, &mai, NULL, &m->buffer_mem) == VK_SUCCESS
         && vkBindBufferMemory(_vk.dev, m->buffer, m->buffer_mem, 0) == VK_SUCCESS
         && vkMapMemory(_vk.dev, m->buffer_mem, 0, size, 0, &m->map) == VK_SUCCESS;
}

static gboolean _alloc_buffer(dt_slang_mem_t *m, const VkDeviceSize size, const VkBufferUsageFlags usage)
{
  return _alloc_buffer_in(m, size, usage, FALSE);
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
  if(vkAllocateMemory(_vk.dev, &mai, NULL, mem) != VK_SUCCESS
     || vkBindImageMemory(_vk.dev, *image, *mem, 0) != VK_SUCCESS)
    return FALSE;
  const VkImageViewCreateInfo vci = { .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                      .image = *image,
                                      .viewType = VK_IMAGE_VIEW_TYPE_2D,
                                      .format = format,
                                      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
  return vkCreateImageView(_vk.dev, &vci, NULL, view) == VK_SUCCESS;
}

static gboolean _alloc_image(dt_slang_mem_t *m, const dt_slang_arg_t *arg)
{
  const gboolean out = arg->kind == DT_SLANG_ARG_IMAGE_OUT;
  if(!_create_image(&m->image, &m->image_mem, &m->view, arg->width, arg->height, arg->channels,
                    out ? VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                        : VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT))
    return FALSE;

  const size_t bytes = sizeof(float) * arg->channels * arg->width * arg->height;
  if(!_alloc_buffer(m, bytes, out ? VK_BUFFER_USAGE_TRANSFER_DST_BIT : VK_BUFFER_USAGE_TRANSFER_SRC_BIT))
    return FALSE;
  if(!out) memcpy(m->map, arg->ptr, bytes);
  return TRUE;
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

static VkCommandBuffer _begin_commands(void)
{
  const VkCommandBufferAllocateInfo cbai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                             .commandPool = _vk.pool,
                                             .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                             .commandBufferCount = 1 };
  VkCommandBuffer cb = VK_NULL_HANDLE;
  if(vkAllocateCommandBuffers(_vk.dev, &cbai, &cb) != VK_SUCCESS) return VK_NULL_HANDLE;
  const VkCommandBufferBeginInfo begin = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                           .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
  vkBeginCommandBuffer(cb, &begin);
  // device images carry data from one submission to the next
  const VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                               .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                               .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT };
  vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                       0, 1, &mb, 0, NULL, 0, NULL);
  return cb;
}

// ends, submits and frees cb, and waits for it
static gboolean _submit_commands(VkCommandBuffer cb)
{
  const VkSubmitInfo submit = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                .commandBufferCount = 1,
                                .pCommandBuffers = &cb };
  const gboolean ok = vkEndCommandBuffer(cb) == VK_SUCCESS
                      && vkQueueSubmit(_vk.queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS
                      && vkQueueWaitIdle(_vk.queue) == VK_SUCCESS;
  vkFreeCommandBuffers(_vk.dev, _vk.pool, 1, &cb);
  return ok;
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

static void _image_free(dt_slang_gpu_image_t *img)
{
  if(img->view) vkDestroyImageView(_vk.dev, img->view, NULL);
  if(img->image) vkDestroyImage(_vk.dev, img->image, NULL);
  if(img->mem) vkFreeMemory(_vk.dev, img->mem, NULL);
  g_free(img);
}

// copies between host memory and a device image through a staging buffer
static int _image_transfer(dt_slang_gpu_image_t *img, void *host, const gboolean to_device)
{
  const size_t bytes = sizeof(float) * img->channels * img->width * img->height;
  dt_slang_mem_t staging = { 0 };
  int err = 1;
  if(!_alloc_buffer(&staging, bytes, to_device ? VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                                                : VK_BUFFER_USAGE_TRANSFER_DST_BIT))
    goto done;
  if(to_device) memcpy(staging.map, host, bytes);

  VkCommandBuffer cb = _begin_commands();
  if(!cb) goto done;
  const VkBufferImageCopy copy = { .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                                   .imageExtent = { img->width, img->height, 1 } };
  if(to_device)
    vkCmdCopyBufferToImage(cb, staging.buffer, img->image, VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
  else
    vkCmdCopyImageToBuffer(cb, img->image, VK_IMAGE_LAYOUT_GENERAL, staging.buffer, 1, &copy);
  if(!_submit_commands(cb)) goto done;

  if(!to_device) memcpy(host, staging.map, bytes);
  err = 0;

done:
  _free_mem(&staging);
  return err;
}

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
        _image_free(img);
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
    VkCommandBuffer cb = _begin_commands();
    ok = cb != VK_NULL_HANDLE;
    if(ok)
    {
      _barrier(cb, img->image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
      ok = _submit_commands(cb);
    }
  }
  if(ok && host) ok = _image_transfer(img, (void *)host, TRUE) == 0;
  if(!ok)
  {
    dt_print(DT_DEBUG_OPENCL, "[slang_gpu] can't allocate a %dx%d device image", width, height);
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

// kernels run synchronously here
void dt_slang_gpu_wait(void)
{
}

void dt_slang_gpu_image_free(dt_slang_gpu_image_t *img)
{
  if(!img) return;
  g_mutex_lock(&_vk_lock);
  // enough for a pipe's input and output, and those of a second pipe
  _vk.images = g_list_prepend(_vk.images, img);
  GList *last = g_list_nth(_vk.images, DT_SLANG_POOL_SIZE);
  if(last)
  {
    last->prev->next = NULL;
    last->prev = NULL;
    g_list_free_full(last, (GDestroyNotify)_image_free);
  }
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

// device buffers are mapped storage buffers, so the host reads them directly
// once the queue is idle, which it is between runs
struct dt_slang_gpu_buffer_t
{
  dt_slang_mem_t mem;
  size_t size;
};

static void _buffer_free(dt_slang_gpu_buffer_t *buf)
{
  _free_mem(&buf->mem);
  g_free(buf);
}

dt_slang_gpu_buffer_t *dt_slang_gpu_buffer_new(const size_t size, const void *host)
{
  if(!dt_slang_gpu_enabled() || size == 0) return NULL;

  g_mutex_lock(&_vk_lock);
  dt_slang_gpu_buffer_t *buf = NULL;
  for(GList *l = _vk.buffers; l; l = g_list_next(l))
  {
    dt_slang_gpu_buffer_t *pooled = l->data;
    if(pooled->size == size)
    {
      _vk.buffers = g_list_delete_link(_vk.buffers, l);
      buf = pooled;
      break;
    }
  }
  if(!buf)
  {
    buf = g_new0(dt_slang_gpu_buffer_t, 1);
    buf->size = size;
    if(!_alloc_buffer_in(&buf->mem, size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, TRUE))
    {
      dt_print(DT_DEBUG_OPENCL, "[slang_gpu] can't allocate a device buffer of %zu bytes", size);
      _buffer_free(buf);
      buf = NULL;
    }
  }
  if(buf && host) memcpy(buf->mem.map, host, size);
  g_mutex_unlock(&_vk_lock);
  return buf;
}

void dt_slang_gpu_buffer_free(dt_slang_gpu_buffer_t *buf)
{
  if(!buf) return;
  g_mutex_lock(&_vk_lock);
  // a demosaic chain holds eight buffers of the same size
  _vk.buffers = g_list_prepend(_vk.buffers, buf);
  GList *last = g_list_nth(_vk.buffers, DT_SLANG_BUFFER_POOL_SIZE);
  if(last)
  {
    last->prev->next = NULL;
    last->prev = NULL;
    g_list_free_full(last, (GDestroyNotify)_buffer_free);
  }
  g_mutex_unlock(&_vk_lock);
}

int dt_slang_gpu_buffer_read(dt_slang_gpu_buffer_t *buf, void *host, const size_t size)
{
  if(!buf || !host || size > buf->size) return 1;
  g_mutex_lock(&_vk_lock);
  memcpy(host, buf->mem.map, size);
  g_mutex_unlock(&_vk_lock);
  return 0;
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
  for(int a = 0; a < nargs; a++)
  {
    const dt_slang_arg_t *arg = &args[a];
    if(arg->kind == DT_SLANG_ARG_VALUE)
    {
      if(nvals >= k->nvals || k->val_offset[nvals] + arg->size > DT_SLANG_PUSH_SIZE) goto mismatch;
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
    res_arg[nres++] = a;
  }
  if(nres != k->nres || nvals != k->nvals) goto mismatch;

  int err = 1;
  dt_slang_mem_t mem[DT_SLANG_MAX_PARAMS] = { { 0 } };
  VkDescriptorPool dpool = VK_NULL_HANDLE;
  VkCommandBuffer cb = VK_NULL_HANDLE;

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
      if(!_alloc_buffer(&mem[r], arg->size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) goto done;
      memcpy(mem[r].map, arg->ptr, arg->size);
      buffer_info[r] = (VkDescriptorBufferInfo){ mem[r].buffer, 0, arg->size };
      writes[r].pBufferInfo = &buffer_info[r];
    }
    else
    {
      if(!_alloc_image(&mem[r], arg)) goto done;
      image_info[r] = (VkDescriptorImageInfo){
        .imageView = mem[r].view,
        .imageLayout = arg->kind == DT_SLANG_ARG_IMAGE_OUT ? VK_IMAGE_LAYOUT_GENERAL
                                                           : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
      writes[r].pImageInfo = &image_info[r];
    }
  }

  const VkDescriptorPoolSize sizes[3] = { { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, DT_SLANG_MAX_PARAMS },
                                          { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, DT_SLANG_MAX_PARAMS },
                                          { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, DT_SLANG_MAX_PARAMS } };
  const VkDescriptorPoolCreateInfo dpci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                            .maxSets = 1,
                                            .poolSizeCount = 3,
                                            .pPoolSizes = sizes };
  if(vkCreateDescriptorPool(_vk.dev, &dpci, NULL, &dpool) != VK_SUCCESS) goto done;
  const VkDescriptorSetAllocateInfo dsai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                             .descriptorPool = dpool,
                                             .descriptorSetCount = 1,
                                             .pSetLayouts = &k->set_layout };
  VkDescriptorSet set;
  if(vkAllocateDescriptorSets(_vk.dev, &dsai, &set) != VK_SUCCESS) goto done;
  for(int r = 0; r < nres; r++) writes[r].dstSet = set;
  vkUpdateDescriptorSets(_vk.dev, nres, writes, 0, NULL);

  cb = _begin_commands();
  if(!cb) goto done;
  for(int r = 0; r < nres; r++)
  {
    const dt_slang_arg_t *arg = &args[res_arg[r]];
    if(arg->kind == DT_SLANG_ARG_IMAGE_IN)
    {
      const VkBufferImageCopy copy = { .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                                       .imageExtent = { arg->width, arg->height, 1 } };
      _barrier(cb, mem[r].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
      vkCmdCopyBufferToImage(cb, mem[r].buffer, mem[r].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
      _barrier(cb, mem[r].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    else if(arg->kind == DT_SLANG_ARG_IMAGE_OUT)
      _barrier(cb, mem[r].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
  }
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, k->pipeline);
  vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, k->layout, 0, 1, &set, 0, NULL);
  vkCmdPushConstants(cb, k->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, DT_SLANG_PUSH_SIZE, push);
  vkCmdDispatch(cb, (width + k->local_size[0] - 1) / k->local_size[0],
                (height + k->local_size[1] - 1) / k->local_size[1], 1);
  for(int r = 0; r < nres; r++)
  {
    const dt_slang_arg_t *arg = &args[res_arg[r]];
    if(arg->kind != DT_SLANG_ARG_IMAGE_OUT) continue;
    const VkBufferImageCopy copy = { .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                                     .imageExtent = { arg->width, arg->height, 1 } };
    _barrier(cb, mem[r].image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    vkCmdCopyImageToBuffer(cb, mem[r].image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, mem[r].buffer, 1, &copy);
  }
  const gboolean submitted = _submit_commands(cb);
  cb = VK_NULL_HANDLE;
  if(!submitted) goto done;

  for(int r = 0; r < nres; r++)
  {
    const dt_slang_arg_t *arg = &args[res_arg[r]];
    if(arg->kind == DT_SLANG_ARG_IMAGE_OUT)
      memcpy((void *)arg->ptr, mem[r].map, sizeof(float) * arg->channels * arg->width * arg->height);
  }
  err = 0;

done:
  if(cb) vkFreeCommandBuffers(_vk.dev, _vk.pool, 1, &cb);
  if(dpool) vkDestroyDescriptorPool(_vk.dev, dpool, NULL);
  for(int r = 0; r < nres; r++) _free_mem(&mem[r]);
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

  const double start = dt_get_debug_wtime();
  // one queue, one submission at a time: pipes running in parallel wait here
  g_mutex_lock(&_vk_lock);
  const int err = _run(program, kernel, width, height, nargs, args);
  g_mutex_unlock(&_vk_lock);
  if(!err)
    dt_print(DT_DEBUG_PERF, "[slang_gpu] %s %dx%d took %.3f secs (Vulkan)",
             kernel, width, height, dt_get_debug_wtime() - start);
  return err;
}
