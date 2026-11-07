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
// denoiseprofile holds ten full-size images at once, demosaic eight buffers
// and colorequal six. The pools' bytes are capped by pool_budget; capped at
// 12 entries, the buffers of one export pushed out those of the next
#define DT_SLANG_POOL_SIZE 32
#define DT_SLANG_BUFFER_POOL_SIZE 32
#define DT_SLANG_STAGING_POOL_SIZE 8
// staging buffers larger than DT_SLANG_DIRECT_BUFFER, for images and for
// tables such as lens's coordinates
#define DT_SLANG_LARGE_STAGING_POOL_SIZE 8
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
  gboolean lost;       // the device failed, and nothing more runs on it
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
  GList *doomed_images;  // evicted from the pools, freed once no batch uses them
  GList *doomed_buffers;
  GList *stale;        // dt_slang_stale_t, handles of reused image memory, destroyed once no batch uses them
  GQueue inflight;     // submitted batches, oldest first
  uint64_t submitted;  // serial of the last submitted batch
  uint64_t retired;    // serial of the last finished one
  GList *spare;        // finished batches, for reuse
  gboolean timestamps; // the queue can time kernels
  float timestamp_period; // ns per timestamp tick
  VkDeviceSize pool_budget; // bytes the image and buffer pools may keep together
  VkDeviceSize host_cap;    // integrated GPU: bytes of system memory the host heaps may hold, 0 for no cap
  uint32_t host_heaps;      // bit mask of the heaps that are system memory
  gboolean memory_budget;   // VK_EXT_memory_budget reports the free device memory
  gboolean discrete;        // the device has memory of its own, which OpenCL shares
  gboolean host_import;     // VK_EXT_external_memory_host: copies read and write host memory directly
  VkDeviceSize import_align; // its minImportedHostPointerAlignment
  PFN_vkGetMemoryHostPointerPropertiesEXT host_pointer_props;
} dt_slang_vulkan_t;

static dt_slang_vulkan_t _vk = { 0 };
static GMutex _vk_lock;
static GThread *_prepare_thread = NULL;
// guards _vk.kernels and _building. Kernels are built outside _vk_lock, so
// that the background preload does not hold up pipes, which Vulkan allows:
// without a pipeline cache, creating pipelines and layouts needs no
// external synchronization
static GMutex _kernels_lock;
static GCond _kernels_built;
// keys of the kernels whose pipeline a thread is building
static GHashTable *_building = NULL;
// the background preload makes way for pipes: it waits while one is
// building a kernel and until none has asked for one for a while
static gint _pipes_waiting = 0;
static gint64 _last_pipe_request = 0;
static gint _preload_stop = 0;
#define DT_SLANG_PRELOAD_IDLE_US 250000

// whether device images and buffers may live in heap h. An integrated GPU,
// such as an AMD APU, has only a BIOS carve-out of 0.5-2 GB as device local
// memory, and reaches the rest of system memory through the host heap
// (GTT) at the same speed
static gboolean _device_heap(const uint32_t h)
{
  return !_vk.discrete || (_vk.memprops.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT);
}

// darktable's use of the host heaps, see host_cap
static VkDeviceSize _host_usage(const VkPhysicalDeviceMemoryBudgetPropertiesEXT *budget)
{
  VkDeviceSize used = 0;
  for(uint32_t h = 0; h < _vk.memprops.memoryHeapCount; h++)
    if(_vk.host_heaps & (1u << h)) used += budget->heapUsage[h];
  return used;
}

// the part of heap h's budget darktable may fill. An integrated GPU shares
// its memory with the desktop, so 2% of each heap stays free there
static VkDeviceSize _usable_budget(const VkPhysicalDeviceMemoryBudgetPropertiesEXT *budget, const uint32_t h)
{
  const VkDeviceSize reserve = _vk.discrete ? 0 : _vk.memprops.memoryHeaps[h].size / 50;
  return budget->heapBudget[h] > reserve ? budget->heapBudget[h] - reserve : 0;
}

// bytes by which size more on heap h would exceed its budget or host_cap
static VkDeviceSize _shortfall(const VkPhysicalDeviceMemoryBudgetPropertiesEXT *budget, const uint32_t h,
                               const VkDeviceSize size)
{
  const VkDeviceSize usable = _usable_budget(budget, h);
  VkDeviceSize over = budget->heapUsage[h] + size > usable ? budget->heapUsage[h] + size - usable : 0;
  if(_vk.host_cap && (_vk.host_heaps & (1u << h)))
  {
    const VkDeviceSize used = _host_usage(budget) + size;
    if(used > _vk.host_cap) over = MAX(over, used - _vk.host_cap);
  }
  return over;
}

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

static gboolean _create_instance(VkInstance *instance)
{
  const VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                                  .pApplicationName = "darktable",
                                  .apiVersion = VK_API_VERSION_1_2 };
  const VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                     .pApplicationInfo = &app };
  return vkCreateInstance(&ici, NULL, instance) == VK_SUCCESS;
}

static const char *_device_type(const VkPhysicalDeviceType type)
{
  return type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? "discrete"
       : type == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? "integrated"
       : type == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU ? "virtual"
       : type == VK_PHYSICAL_DEVICE_TYPE_CPU ? "cpu" : "other";
}

// the device wanted asks for, by index or by part of its name. Without one,
// or if none matches, a discrete GPU wins over an integrated one, which wins
// over the rest: on hybrid laptops the integrated GPU is usually enumerated
// first, and a CPU implementation such as lavapipe is only a last resort
static int _pick_device(const VkPhysicalDevice *devs,
                        const uint32_t ndev,
                        const char *wanted,
                        const gboolean log)
{
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
    if(log)
      dt_print(DT_DEBUG_SLANG, "[slang_gpu] Vulkan device %u: %s, %s, vendor 0x%04x%s", d, props.deviceName,
               _device_type(props.deviceType), props.vendorID, match ? ", requested" : "");
    if(rank > best_rank)
    {
      best = d;
      best_rank = rank;
    }
  }
  if(log && (wanted_name || by_index) && best_rank < 4)
    dt_print(DT_DEBUG_SLANG, "[slang_gpu] no Vulkan device matches slang_gpu_device '%s'", wanted);
  g_free(wanted_name);
  return best;
}

static gboolean _init_device(void)
{
  if(!_create_instance(&_vk.instance))
  {
    dt_print(DT_DEBUG_SLANG, "[slang_gpu] no Vulkan instance");
    return FALSE;
  }

  uint32_t ndev = 0;
  vkEnumeratePhysicalDevices(_vk.instance, &ndev, NULL);
  if(ndev == 0)
  {
    dt_print(DT_DEBUG_SLANG, "[slang_gpu] no Vulkan device");
    return FALSE;
  }
  VkPhysicalDevice *devs = g_new(VkPhysicalDevice, ndev);
  vkEnumeratePhysicalDevices(_vk.instance, &ndev, devs);
  _vk.pdev = devs[_pick_device(devs, ndev, dt_conf_get_string_const("slang_gpu_device"), TRUE)];
  g_free(devs);

  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(_vk.pdev, &props);
  _vk.max_image_dim = props.limits.maxImageDimension2D;
  _vk.discrete = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
  _vk.push_size = MIN(DT_SLANG_PUSH_SIZE, props.limits.maxPushConstantsSize);
  vkGetPhysicalDeviceMemoryProperties(_vk.pdev, &_vk.memprops);
  for(uint32_t h = 0; h < _vk.memprops.memoryHeapCount; h++)
    dt_print(DT_DEBUG_SLANG, "[slang_gpu] memory heap %u: %" PRIu64 " MB%s", h,
             (uint64_t)(_vk.memprops.memoryHeaps[h].size >> 20),
             _vk.memprops.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT ? ", device local" : "");
  // an integrated GPU's images come from all heaps, and _make_room() evicts
  // the pools when an allocation needs the memory. Half of it holds the
  // nine 380 MB images of a 64 MP denoiseprofile tile: within a quarter of
  // the largest heap they were freed after each tile, and allocating them
  // again, which has the kernel clear the pages, cost 0.1-0.15 s per tile
  VkDeviceSize all_heaps = 0;
  for(uint32_t h = 0; h < _vk.memprops.memoryHeapCount; h++)
    if(_device_heap(h))
    {
      _vk.pool_budget = MAX(_vk.pool_budget, _vk.memprops.memoryHeaps[h].size / 4);
      all_heaps += _vk.memprops.memoryHeaps[h].size;
    }
  if(!_vk.discrete) _vk.pool_budget = all_heaps / 2;

  // an integrated GPU's memory is system memory, except a carve-out the
  // firmware reserves, which shows as a device local heap next to a host
  // one. Like OpenCL's unified_fraction, a quarter of it is the default cap:
  // without one the pools grew until other programs were killed for memory,
  // which the kernel does not account to darktable
  if(!_vk.discrete)
  {
    uint32_t local = 0;
    for(uint32_t h = 0; h < _vk.memprops.memoryHeapCount; h++)
      if(_vk.memprops.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) local |= 1u << h;
    const uint32_t all = (1u << _vk.memprops.memoryHeapCount) - 1;
    _vk.host_heaps = local == all ? all : all & ~local;
    _vk.host_cap = darktable.dtresources.total_memory * dt_slang_gpu_unified_fraction(0.25f);
  }
  const int cap = dt_conf_get_int("slang_gpu_memory");
  if(cap > 0)
  {
    if(!_vk.discrete) _vk.host_cap = MIN(_vk.host_cap, (VkDeviceSize)cap << 20);
    _vk.pool_budget = MIN(_vk.pool_budget, (VkDeviceSize)cap << 20);
  }
  if(_vk.host_cap) _vk.pool_budget = MIN(_vk.pool_budget, _vk.host_cap / 2);
  dt_print(DT_DEBUG_SLANG, "[slang_gpu] pools keep up to %" PRIu64 " MB, system memory capped at %" PRIu64 " MB",
           (uint64_t)(_vk.pool_budget >> 20), (uint64_t)(_vk.host_cap >> 20));

  uint32_t next = 0;
  vkEnumerateDeviceExtensionProperties(_vk.pdev, NULL, &next, NULL);
  VkExtensionProperties *ext = g_new(VkExtensionProperties, next);
  vkEnumerateDeviceExtensionProperties(_vk.pdev, NULL, &next, ext);
  gboolean host_import = FALSE;
  for(uint32_t e = 0; e < next; e++)
  {
    if(!strcmp(ext[e].extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) _vk.memory_budget = TRUE;
    if(!strcmp(ext[e].extensionName, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME)) host_import = TRUE;
  }
  g_free(ext);

  VkPhysicalDeviceFeatures features;
  vkGetPhysicalDeviceFeatures(_vk.pdev, &features);
  if(!features.shaderStorageImageWriteWithoutFormat)
  {
    dt_print(DT_DEBUG_SLANG, "[slang_gpu] %s can't write images without a format", props.deviceName);
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
    dt_print(DT_DEBUG_SLANG, "[slang_gpu] %s has no compute queue", props.deviceName);
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
  const char *const dev_ext[] = { VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME };
  const VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                                   .queueCreateInfoCount = 1,
                                   .pQueueCreateInfos = &qci,
                                   .enabledExtensionCount = host_import ? 1 : 0,
                                   .ppEnabledExtensionNames = dev_ext,
                                   .pEnabledFeatures = &enable };
  if(vkCreateDevice(_vk.pdev, &dci, NULL, &_vk.dev) != VK_SUCCESS)
  {
    dt_print(DT_DEBUG_SLANG, "[slang_gpu] can't create a device on %s", props.deviceName);
    return FALSE;
  }
  vkGetDeviceQueue(_vk.dev, family, 0, &_vk.queue);

  // copies between host and device images then run on the GPU's copy
  // engine straight from the pipe's buffers, instead of the CPU copying
  // every byte through a staging buffer
  if(host_import)
  {
    VkPhysicalDeviceExternalMemoryHostPropertiesEXT hp
        = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT };
    VkPhysicalDeviceProperties2 props2 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &hp };
    vkGetPhysicalDeviceProperties2(_vk.pdev, &props2);
    _vk.import_align = hp.minImportedHostPointerAlignment;
    _vk.host_pointer_props = (PFN_vkGetMemoryHostPointerPropertiesEXT)
      vkGetDeviceProcAddr(_vk.dev, "vkGetMemoryHostPointerPropertiesEXT");
    _vk.host_import = _vk.host_pointer_props && _vk.import_align > 0
                      && (_vk.import_align & (_vk.import_align - 1)) == 0;
  }
  dt_print(DT_DEBUG_SLANG, "[slang_gpu] %s", _vk.host_import ? "copies use host memory directly"
                                                              : "copies go through staging buffers");

  const VkCommandPoolCreateInfo cpci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                         .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT
                                                  | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                         .queueFamilyIndex = family };
  if(vkCreateCommandPool(_vk.dev, &cpci, NULL, &_vk.pool) != VK_SUCCESS) return FALSE;

  _vk.kernels = g_hash_table_new_full(g_str_hash, g_str_equal, g_free, _destroy_kernel);
  dt_print(DT_DEBUG_SLANG, "[slang_gpu] using Vulkan device %s", props.deviceName);
  return TRUE;
}

GList *dt_slang_gpu_devices(int *automatic)
{
  *automatic = -1;
  g_mutex_lock(&_vk_lock);
  // with slang_gpu off there is no instance yet
  VkInstance instance = _vk.instance;
  if(!instance && !_create_instance(&instance))
  {
    g_mutex_unlock(&_vk_lock);
    return NULL;
  }
  uint32_t ndev = 0;
  vkEnumeratePhysicalDevices(instance, &ndev, NULL);
  VkPhysicalDevice *devs = g_new(VkPhysicalDevice, MAX(ndev, 1));
  vkEnumeratePhysicalDevices(instance, &ndev, devs);
  GList *list = NULL;
  for(uint32_t d = 0; d < ndev; d++)
  {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(devs[d], &props);
    dt_slang_gpu_device_t *device = g_new0(dt_slang_gpu_device_t, 1);
    device->name = g_strdup(props.deviceName);
    device->unified_memory = props.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
    list = g_list_prepend(list, device);
  }
  if(ndev) *automatic = _pick_device(devs, ndev, NULL, FALSE);
  g_free(devs);
  if(instance != _vk.instance) vkDestroyInstance(instance, NULL);
  g_mutex_unlock(&_vk_lock);
  return g_list_reverse(list);
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
  const gboolean usable = _vk.usable && !dt_slang_gpu_stopped();
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

static dt_slang_kernel_t *_build_kernel(const char *program, const char *entry)
{
  char datadir[PATH_MAX] = { 0 };
  dt_loc_get_datadir(datadir, sizeof(datadir));
  gchar *spv = g_strdup_printf("%s/kernels/slang/%s.spv", datadir, program);
  gchar *code = NULL;
  gsize len = 0;
  if(!g_file_get_contents(spv, &code, &len, NULL) || len % 4)
  {
    dt_print(DT_DEBUG_SLANG, "[slang_gpu] can't read %s", spv);
    g_free(spv);
    g_free(code);
    return NULL;
  }

  dt_slang_kernel_t *k = g_new0(dt_slang_kernel_t, 1);
  if(!_reflect((const uint32_t *)code, len / 4, entry, k))
  {
    dt_print(DT_DEBUG_SLANG, "[slang_gpu] can't find the parameters of %s in %s", entry, spv);
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
    dt_print(DT_DEBUG_SLANG, "[slang_gpu] can't create the pipeline for %s", entry);
    goto error;
  }

  g_free(spv);
  g_free(code);
  return k;

error:
  _destroy_kernel(k);
  g_free(spv);
  g_free(code);
  return NULL;
}

static dt_slang_kernel_t *_get_kernel(const char *program, const char *entry, const gboolean from_pipe)
{
  gchar *key = g_strdup_printf("%s:%s", program, entry);
  if(from_pipe) g_atomic_int_inc(&_pipes_waiting);
  // the preload and a pipe needing the same kernel build it once
  g_mutex_lock(&_kernels_lock);
  if(from_pipe) _last_pipe_request = g_get_monotonic_time();
  if(!_building) _building = g_hash_table_new_full(g_str_hash, g_str_equal, g_free, NULL);
  while(g_hash_table_contains(_building, key)) g_cond_wait(&_kernels_built, &_kernels_lock);
  dt_slang_kernel_t *k = g_hash_table_lookup(_vk.kernels, key);
  if(!k) g_hash_table_add(_building, g_strdup(key));
  g_mutex_unlock(&_kernels_lock);
  if(!k)
  {
    k = _build_kernel(program, entry);
    g_mutex_lock(&_kernels_lock);
    if(k) g_hash_table_insert(_vk.kernels, g_strdup(key), k);
    g_hash_table_remove(_building, key);
    g_cond_broadcast(&_kernels_built);
    g_mutex_unlock(&_kernels_lock);
  }
  if(from_pipe) g_atomic_int_add(&_pipes_waiting, -1);
  g_free(key);
  return k;
}

static gboolean _pipe_busy(void)
{
  g_mutex_lock(&_kernels_lock);
  const gboolean busy = g_atomic_int_get(&_pipes_waiting)
                        || g_get_monotonic_time() - _last_pipe_request < DT_SLANG_PRELOAD_IDLE_US;
  g_mutex_unlock(&_kernels_lock);
  return busy;
}

// creates the pipelines of every entry point in every .spv in kernels/slang,
// which takes longer than running them the first time. It builds one at a
// time and only while no pipe is dispatching, so that a pipe's own builds
// do not queue behind it
static void _preload_kernels(void)
{
  char datadir[PATH_MAX] = { 0 };
  dt_loc_get_datadir(datadir, sizeof(datadir));
  gchar *dirname = g_build_filename(datadir, "kernels", "slang", NULL);
  GDir *dir = g_dir_open(dirname, 0, NULL);
  const gchar *file;
  while(dir && (file = g_dir_read_name(dir)) && !g_atomic_int_get(&_preload_stop))
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
        if(g_atomic_int_get(&_preload_stop)) break;
        if(op == 15 && n > 3) // OpEntryPoint
        {
          gchar *entry = g_strndup((const char *)(w + i + 3), sizeof(uint32_t) * (n - 3));
          while(!g_atomic_int_get(&_preload_stop) && _pipe_busy()) g_usleep(10000);
          if(!g_atomic_int_get(&_preload_stop)) _get_kernel(program, entry, FALSE);
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
  const gboolean usable = _vk.usable;
  g_mutex_unlock(&_vk_lock);
  dt_print(DT_DEBUG_PERF, "[slang_gpu] preparing the Vulkan device took %.3f secs",
           dt_get_debug_wtime() - start);
  // there are about 400 pipelines, and without a warm shader cache a driver
  // can take tens of milliseconds for each, so they are built in the
  // background and a module builds what it needs itself
  if(usable)
  {
    _preload_kernels();
    dt_print(DT_DEBUG_PERF, "[slang_gpu] building the Vulkan kernels in the background %s after %.3f secs",
             g_atomic_int_get(&_preload_stop) ? "stopped" : "finished", dt_get_debug_wtime() - start);
  }
  return NULL;
}

void dt_slang_gpu_prepare(void)
{
  if(!dt_conf_get_bool("slang_gpu") || _prepare_thread) return;
  g_atomic_int_set(&_preload_stop, 0);
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
  uint32_t heap;         // of a mapped buffer's memory
} dt_slang_mem_t;

static gboolean _reclaim(void);
static void _make_room(const uint32_t heap, const VkDeviceSize size);

// device memory, after freeing the pools if the device is out of it. Pooled
// images and buffers otherwise filled a 4 GB GPU after a few exports. An
// integrated GPU's host heap is the same memory as its device heap, so
// staging buffers there have to fit the budget too
static gboolean _allocate(const VkMemoryAllocateInfo *mai, VkDeviceMemory *mem)
{
  const uint32_t heap = _vk.memprops.memoryTypes[mai->memoryTypeIndex].heapIndex;
  const double start = dt_get_debug_wtime();
  if(!_vk.discrete || (_vk.memprops.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT))
    _make_room(heap, mai->allocationSize);
  const double made_room = dt_get_debug_wtime();
  VkResult res = vkAllocateMemory(_vk.dev, mai, NULL, mem);
  if(res == VK_ERROR_OUT_OF_DEVICE_MEMORY && _reclaim())
    res = vkAllocateMemory(_vk.dev, mai, NULL, mem);
  // a 61 MP export on a Radeon 780M stalled for 1.9 s around an allocation
  // in a nearly full heap; this tells the pool's share from the driver's
  const double end = dt_get_debug_wtime();
  if(end - start > 0.05)
    dt_print(DT_DEBUG_PERF, "[slang_gpu] allocating %" PRIu64 " MB on heap %u took %.3f secs"
             " (%.3f making room, %.3f in the driver)", (uint64_t)(mai->allocationSize >> 20), heap,
             end - start, made_room - start, end - made_room);
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

// whether size more bytes fit the budget of heap h; without
// VK_EXT_memory_budget only a failed allocation tells
static gboolean _heap_has_room(const uint32_t h, const VkDeviceSize size)
{
  if(!_vk.memory_budget) return TRUE;
  VkPhysicalDeviceMemoryBudgetPropertiesEXT budget
      = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT };
  VkPhysicalDeviceMemoryProperties2 props2
      = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, .pNext = &budget };
  vkGetPhysicalDeviceMemoryProperties2(_vk.pdev, &props2);
  return _shortfall(&budget, h, size) == 0;
}

// memory for a device image or buffer, device local if possible, in the
// memory type stored in *type. On an integrated GPU it goes to another heap
// once the carve-out is full, instead of evicting the pools to squeeze into it
static gboolean _alloc_on_device(const VkMemoryRequirements *req, VkDeviceMemory *mem, uint32_t *type)
{
  uint32_t local;
  if(!_memtype(req->memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &local)
     && !_memtype(req->memoryTypeBits, 0, &local))
    return FALSE;
  const uint32_t local_heap = _vk.memprops.memoryTypes[local].heapIndex;
  uint32_t other = UINT32_MAX;
  if(!_vk.discrete)
    for(uint32_t i = 0; i < _vk.memprops.memoryTypeCount && other == UINT32_MAX; i++)
      if((req->memoryTypeBits & (1u << i)) && _vk.memprops.memoryTypes[i].heapIndex != local_heap)
        other = i;

  VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                               .allocationSize = req->size,
                               .memoryTypeIndex = local };
  if(other != UINT32_MAX && !_heap_has_room(local_heap, req->size)
     && _heap_has_room(_vk.memprops.memoryTypes[other].heapIndex, req->size))
    mai.memoryTypeIndex = other;
  *type = mai.memoryTypeIndex;
  if(_allocate(&mai, mem)) return TRUE;
  if(other == UINT32_MAX || mai.memoryTypeIndex == other) return FALSE;
  mai.memoryTypeIndex = *type = other;
  return _allocate(&mai, mem);
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
  uint32_t type = 0;
  int i = 0;
  while(i < n && !_memtype(req.memoryTypeBits, want[i], &type)) i++;
  if(i == n) return FALSE;
  m->coherent = (_vk.memprops.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
  m->heap = _vk.memprops.memoryTypes[type].heapIndex;
  const VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                     .allocationSize = req.size,
                                     .memoryTypeIndex = type };
  // the whole allocation is mapped, which is what an invalidation of a
  // non-coherent type has to cover
  return _allocate(&mai, &m->buffer_mem)
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
  return _alloc_on_device(&req, &m->buffer_mem, &type)
         && vkBindBufferMemory(_vk.dev, m->buffer, m->buffer_mem, 0) == VK_SUCCESS;
}

// copies to and from mapped memory. Large ones are split over the cores: a
// single core takes about 30 ms for lens's 244 MB of per-pixel coordinates
static void _copy(void *dst, const void *src, const size_t bytes)
{
  const size_t chunk = (size_t)1 << 20;
  if(bytes < 16 * chunk)
  {
    memcpy(dst, src, bytes);
    return;
  }
  const size_t n = (bytes + chunk - 1) / chunk;
  DT_OMP_FOR()
  for(size_t i = 0; i < n; i++)
    memcpy((char *)dst + i * chunk, (const char *)src + i * chunk, MIN(chunk, bytes - i * chunk));
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

static VkFormat _image_format(const int channels)
{
  return channels == 1 ? VK_FORMAT_R32_SFLOAT : VK_FORMAT_R32G32B32A32_SFLOAT;
}

#define DT_SLANG_IMAGE_USAGE (VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT \
                              | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT)

// an image without memory
static gboolean _create_image(VkImage *image,
                              const int width,
                              const int height,
                              const int channels,
                              const VkImageUsageFlags usage)
{
  const VkFormat format = _image_format(channels);
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
  return vkCreateImage(_vk.dev, &ici, NULL, image) == VK_SUCCESS;
}

static gboolean _create_view(VkImageView *view, const VkImage image, const int channels)
{
  const VkImageViewCreateInfo vci = { .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                      .image = image,
                                      .viewType = VK_IMAGE_VIEW_TYPE_2D,
                                      .format = _image_format(channels),
                                      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
  return vkCreateImageView(_vk.dev, &vci, NULL, view) == VK_SUCCESS;
}

// image memory is allocated in 16 sizes per power of two, so that images a
// few rows or columns apart share pooled memory: a 61 MP export uses
// 9692x6636 and 9624x6626 images, 988 and 981 MB, and allocated four to six
// of them anew on every export on a Radeon 780M, at 50-200 ms each
static VkDeviceSize _size_class(const VkDeviceSize size)
{
  VkDeviceSize step = 64 << 10;
  while(step << 5 <= size) step <<= 1;
  return (size + step - 1) & ~(step - 1);
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
  VkDeviceSize mem_size; // of mem, a _size_class()
  uint32_t mem_type;
  int width, height, channels;
  uint64_t last_use; // serial of the last batch that may use it, once freed
};

// the image and view of a pooled image whose memory went to another one,
// destroyed once the batches that may use them have finished
typedef struct dt_slang_stale_t
{
  VkImage image;
  VkImageView view;
  uint64_t last_use;
} dt_slang_stale_t;

static void _stale_free(dt_slang_stale_t *st)
{
  if(st->view) vkDestroyImageView(_vk.dev, st->view, NULL);
  if(st->image) vkDestroyImage(_vk.dev, st->image, NULL);
  g_free(st);
}

// device buffers live in device memory; the host fills, clears and reads
// them with transfer commands
struct dt_slang_gpu_buffer_t
{
  dt_slang_mem_t mem;
  size_t size;
  uint64_t last_use;
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
  uint64_t serial;              // its place in the queue, from 1
} dt_slang_batch_t;

static void _batch_destroy(dt_slang_batch_t *b)
{
  if(b->queries) vkDestroyQueryPool(_vk.dev, b->queries, NULL);
  if(b->dpool) vkDestroyDescriptorPool(_vk.dev, b->dpool, NULL);
  if(b->fence) vkDestroyFence(_vk.dev, b->fence, NULL);
  if(b->cb) vkFreeCommandBuffers(_vk.dev, _vk.pool, 1, &b->cb);
  g_free(b);
}

// small and large staging buffers are pooled apart: the small ones of
// every kernel's tables pushed the large ones out, so an export allocated
// and faulted in its 232 MB of lens coordinates and its 155 MB readback anew
static void _staging_release(dt_slang_staging_t *s)
{
  _vk.staging = g_list_prepend(_vk.staging, s);
  const gboolean large = s->size > DT_SLANG_DIRECT_BUFFER;
  int n = 0;
  for(GList *l = _vk.staging; l;)
  {
    GList *next = g_list_next(l);
    const dt_slang_staging_t *p = l->data;
    if((p->size > DT_SLANG_DIRECT_BUFFER) == large
       && ++n > (large ? DT_SLANG_LARGE_STAGING_POOL_SIZE : DT_SLANG_STAGING_POOL_SIZE))
    {
      _staging_destroy(l->data);
      _vk.staging = g_list_delete_link(_vk.staging, l);
    }
    l = next;
  }
}

static size_t _image_bytes(const dt_slang_gpu_image_t *img)
{
  return img->mem_size;
}

// images and buffers evicted from the pools are freed once the batches that
// may use them have finished. Waiting for an idle queue instead let them pile
// up during a chain of kernels without reads.
static void _free_doomed(void)
{
  for(GList *l = _vk.stale; l;)
  {
    GList *next = g_list_next(l);
    dt_slang_stale_t *st = l->data;
    if(st->last_use <= _vk.retired)
    {
      _stale_free(st);
      _vk.stale = g_list_delete_link(_vk.stale, l);
    }
    l = next;
  }
  for(GList *l = _vk.doomed_images; l;)
  {
    GList *next = g_list_next(l);
    dt_slang_gpu_image_t *img = l->data;
    if(img->last_use <= _vk.retired)
    {
      _image_free(img);
      _vk.doomed_images = g_list_delete_link(_vk.doomed_images, l);
    }
    l = next;
  }
  for(GList *l = _vk.doomed_buffers; l;)
  {
    GList *next = g_list_next(l);
    dt_slang_gpu_buffer_t *buf = l->data;
    if(buf->last_use <= _vk.retired)
    {
      _buffer_free(buf);
      _vk.doomed_buffers = g_list_delete_link(_vk.doomed_buffers, l);
    }
    l = next;
  }
}

// cuts a pool list after the first max entries or once *bytes, counting
// on from the other pool, exceeds the budget. The cut entries may still be
// in use, so they are freed once their batches have finished
static GList *_trim(GList *list, const int max, VkDeviceSize *bytes, GList **doomed, const gboolean images)
{
  int n = 0;
  for(GList *l = list; l; l = g_list_next(l), n++)
  {
    if(images)
      *bytes += _image_bytes(l->data);
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
  buf->last_use = _vk.submitted;
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
  _vk.retired = b->serial;
  _batch_release(b);
}

// counts a failed submission or wait, once per lost device: after the first
// error every batch still queued on it fails too
static void _device_error(const char *what, const VkResult r, const gboolean fatal)
{
  if(_vk.lost) return;
  _vk.lost = fatal;
  gchar *msg = g_strdup_printf("%s (VkResult %d)", what, r);
  dt_slang_gpu_report_error(msg, fatal);
  g_free(msg);
}

// waits for everything submitted so far. Returns FALSE if the device failed
// any of it, in which case the results of every kernel since the last wait
// are undefined
static gboolean _wait_all(void)
{
  gboolean ok = TRUE;
  dt_slang_batch_t *b;
  while((b = g_queue_pop_head(&_vk.inflight)))
  {
    // a wait fails only for a lost device or one out of memory, and neither
    // leaves a way to tell which commands ran, so both count as lost
    const VkResult r = vkWaitForFences(_vk.dev, 1, &b->fence, VK_TRUE, UINT64_MAX);
    if(r != VK_SUCCESS)
    {
      if(ok) _device_error("waiting for the kernels", r, TRUE);
      ok = FALSE;
    }
    _retire(b);
  }
  _free_doomed();
  return ok && !_vk.lost;
}

// retires the batches that have finished, oldest first
static void _retire_finished(void)
{
  dt_slang_batch_t *b;
  while((b = g_queue_peek_head(&_vk.inflight)))
  {
    const VkResult r = vkGetFenceStatus(_vk.dev, b->fence);
    if(r == VK_NOT_READY) break;
    if(r != VK_SUCCESS)
    {
      // a lost device finishes nothing still queued
      _wait_all();
      return;
    }
    _retire(g_queue_pop_head(&_vk.inflight));
  }
  _free_doomed();
}

// waits for the batches up to serial
static void _wait_until(const uint64_t serial)
{
  dt_slang_batch_t *b;
  while((b = g_queue_peek_head(&_vk.inflight)) && b->serial <= serial)
  {
    const VkResult r = vkWaitForFences(_vk.dev, 1, &b->fence, VK_TRUE, UINT64_MAX);
    if(r != VK_SUCCESS) _device_error("waiting for the kernels", r, TRUE);
    _retire(g_queue_pop_head(&_vk.inflight));
  }
  _free_doomed();
}

// frees the pooled images and buffers for an allocation the device had no
// memory for. TRUE if there was something to free
static gboolean _reclaim(void)
{
  if(!_vk.images && !_vk.buffers && !_vk.doomed_images && !_vk.doomed_buffers) return FALSE;
  dt_print(DT_DEBUG_SLANG, "[slang_gpu] out of device memory, freeing the pooled images and buffers");
  _wait_all();
  _vk.doomed_images = g_list_concat(_vk.doomed_images, _vk.images);
  _vk.doomed_buffers = g_list_concat(_vk.doomed_buffers, _vk.buffers);
  _vk.images = NULL;
  _vk.buffers = NULL;
  _free_doomed();
  return TRUE;
}

// the oldest pooled image or buffer, moved to the doomed ones. Returns its size
static size_t _evict_oldest(uint64_t *last_use)
{
  GList *li = g_list_last(_vk.images);
  GList *lb = g_list_last(_vk.buffers);
  const dt_slang_gpu_image_t *img = li ? li->data : NULL;
  const dt_slang_gpu_buffer_t *buf = lb ? lb->data : NULL;
  if(img && (!buf || img->last_use <= buf->last_use))
  {
    _vk.images = g_list_remove_link(_vk.images, li);
    _vk.doomed_images = g_list_concat(_vk.doomed_images, li);
    *last_use = MAX(*last_use, img->last_use);
    return _image_bytes(img);
  }
  if(buf)
  {
    _vk.buffers = g_list_remove_link(_vk.buffers, lb);
    _vk.doomed_buffers = g_list_concat(_vk.doomed_buffers, lb);
    *last_use = MAX(*last_use, buf->last_use);
    return buf->size;
  }
  return 0;
}

// frees doomed and then the oldest pooled images and buffers until size bytes
// fit the heap's budget, waiting only for the batches that use them. Running
// out of device memory first made the reclaim above wait for the whole queue
// and free every pool, on each export after the first on a 4 GB GPU.
static void _make_room(const uint32_t heap, const VkDeviceSize size)
{
  if(!_vk.memory_budget) return;
  VkPhysicalDeviceMemoryBudgetPropertiesEXT budget
      = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT };
  VkPhysicalDeviceMemoryProperties2 props2
      = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, .pNext = &budget };
  vkGetPhysicalDeviceMemoryProperties2(_vk.pdev, &props2);
  if(_shortfall(&budget, heap, size) == 0) return;

  // an integrated GPU's heaps share one pool of memory, and the large
  // staging buffers kept for reuse grew to more than the host heap during
  // a 64 MP export, shrinking the device heap's budget with it. No batch
  // uses an idle one, so the oldest go first and without a wait. Only those
  // counted against this heap's budget or host_cap help: on a Radeon 780M
  // the device heap freed the host heap's staging on every export, and the
  // next export allocated it again
  if(!_vk.discrete && _vk.staging)
  {
    VkDeviceSize staged = 0;
    for(GList *l = g_list_last(_vk.staging); l && _shortfall(&budget, heap, size) > 0;)
    {
      GList *prev = g_list_previous(l);
      const dt_slang_staging_t *s = l->data;
      const uint32_t from = s->mem.heap;
      if(from == heap
         || (_vk.host_cap && (_vk.host_heaps & (1u << from)) && (_vk.host_heaps & (1u << heap))))
      {
        staged += s->size;
        _staging_destroy(l->data);
        _vk.staging = g_list_delete_link(_vk.staging, l);
        vkGetPhysicalDeviceMemoryProperties2(_vk.pdev, &props2);
      }
      l = prev;
    }
    if(staged)
      dt_print(DT_DEBUG_SLANG, "[slang_gpu] freeing %" PRIu64 " MB of idle staging buffers for %" PRIu64 " MB",
               (uint64_t)(staged >> 20), (uint64_t)(size >> 20));
    if(_shortfall(&budget, heap, size) == 0) return;
  }

  const VkDeviceSize need = _shortfall(&budget, heap, size);
  VkDeviceSize found = 0;
  uint64_t until = 0;
  for(GList *l = _vk.doomed_images; l; l = g_list_next(l))
  {
    const dt_slang_gpu_image_t *img = l->data;
    found += _image_bytes(img);
    until = MAX(until, img->last_use);
  }
  for(GList *l = _vk.doomed_buffers; l; l = g_list_next(l))
  {
    const dt_slang_gpu_buffer_t *buf = l->data;
    found += buf->size;
    until = MAX(until, buf->last_use);
  }
  size_t evicted;
  while(found < need && (evicted = _evict_oldest(&until))) found += evicted;
  if(found == 0) return;

  dt_print(DT_DEBUG_SLANG, "[slang_gpu] freeing %" PRIu64 " MB of pooled memory for %" PRIu64 " MB,"
           " %" PRIu64 " of %" PRIu64 " MB used", (uint64_t)(found >> 20), (uint64_t)(size >> 20),
           (uint64_t)(budget.heapUsage[heap] >> 20), (uint64_t)(budget.heapBudget[heap] >> 20));
  _wait_until(until);
}

static dt_slang_batch_t *_batch_begin(const char *entry, const int width, const int height)
{
  _retire_finished();
  // a long chain of kernels without a read would otherwise hold its
  // staging buffers and temporaries all at once
  if(g_queue_get_length(&_vk.inflight) >= DT_SLANG_MAX_INFLIGHT)
  {
    dt_slang_batch_t *oldest = g_queue_pop_head(&_vk.inflight);
    const VkResult r = vkWaitForFences(_vk.dev, 1, &oldest->fence, VK_TRUE, UINT64_MAX);
    if(r != VK_SUCCESS) _device_error("waiting for the kernels", r, TRUE);
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

// submits b, and with wait waits for it and everything before it. Returns
// FALSE if b can't be submitted, or with wait if the device failed
static gboolean _batch_submit(dt_slang_batch_t *b, const gboolean wait)
{
  const VkSubmitInfo submit = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                .commandBufferCount = 1,
                                .pCommandBuffers = &b->cb };
  VkResult r = vkEndCommandBuffer(b->cb);
  if(r == VK_SUCCESS) r = vkQueueSubmit(_vk.queue, 1, &submit, b->fence);
  if(r != VK_SUCCESS)
  {
    // out of memory leaves the device usable
    _device_error("submitting kernels", r, r == VK_ERROR_DEVICE_LOST);
    _batch_release(b);
    return FALSE;
  }
  b->serial = ++_vk.submitted;
  g_queue_push_tail(&_vk.inflight, b);
  return wait ? _wait_all() : TRUE;
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
    dt_print(DT_DEBUG_SLANG, "[slang_gpu] can't allocate a device buffer of %zu bytes", size);
    _buffer_free(buf);
    return NULL;
  }
  return buf;
}

// records an upload of bytes from host into dst at offset 0 in b
static gboolean _record_upload(dt_slang_batch_t *b, const void *host, const size_t bytes, VkBuffer dst)
{
  const double start = dt_get_debug_wtime();
  dt_slang_staging_t *s = _staging_acquire(bytes, TRUE);
  if(!s) return FALSE;
  _copy(s->mem.map, host, bytes);
  if(bytes >= (16 << 20))
    dt_print(DT_DEBUG_PERF, "[slang_gpu] staging %zu MB for the device took %.3f secs",
             bytes >> 20, dt_get_debug_wtime() - start);
  b->staging = g_list_prepend(b->staging, s);
  const VkBufferCopy copy = { 0, 0, bytes };
  vkCmdCopyBuffer(b->cb, s->mem.buffer, dst, 1, &copy);
  return TRUE;
}

// host memory the device copies from and to directly. An import lives only
// for its one copy: with the 1 GB input and output of a tiled
// denoiseprofile imported for the whole module, each of its hundreds of
// submissions got slower on amdgpu, and the module took 5.4 instead of
// 2.8 seconds
typedef struct dt_slang_import_t
{
  const char *base;   // the host address rounded down to the import alignment
  dt_slang_mem_t mem; // a transfer buffer on the imported pages
} dt_slang_import_t;

// pinning the pages costs about 0.1 ms per MB, and below this the copy
// through a staging buffer is cheaper
#define DT_SLANG_IMPORT_MIN (4 << 20)
// importing a whole 61 MP readback, 981 MB, on a Radeon 780M pushed the host
// heap over its budget, and _make_room() evicted 2.9 GB of pooled images
// that the next export allocated again. Larger uploads are imported in
// bands of this size, larger reads go through _read_banded()
#define DT_SLANG_IMPORT_MAX (128 << 20)

static void _import_free(dt_slang_import_t *imp)
{
  _free_mem(&imp->mem);
  g_free(imp);
}

static dt_slang_import_t *_import_new(const void *host, const size_t bytes)
{
  if(!_vk.host_import || !host || bytes == 0) return NULL;
  const uintptr_t start = (uintptr_t)host & ~(uintptr_t)(_vk.import_align - 1);
  const uintptr_t end = ((uintptr_t)host + bytes + _vk.import_align - 1) & ~(uintptr_t)(_vk.import_align - 1);
  const VkDeviceSize span = end - start;

  VkMemoryHostPointerPropertiesEXT hpp = { .sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT };
  if(_vk.host_pointer_props(_vk.dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
                            (const void *)start, &hpp) != VK_SUCCESS)
    return NULL;

  dt_slang_import_t *imp = g_new0(dt_slang_import_t, 1);
  imp->base = (const char *)start;
  const VkExternalMemoryBufferCreateInfo ebci
      = { .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
          .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT };
  const VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                   .pNext = &ebci,
                                   .size = span,
                                   .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT };
  VkMemoryRequirements req = { 0 };
  uint32_t type = 0;
  // host reads after a device write need no invalidation in a coherent type
  gboolean ok = vkCreateBuffer(_vk.dev, &bci, NULL, &imp->mem.buffer) == VK_SUCCESS;
  if(ok)
  {
    vkGetBufferMemoryRequirements(_vk.dev, imp->mem.buffer, &req);
    ok = req.size <= span
         && _memtype(req.memoryTypeBits & hpp.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &type);
  }
  if(ok)
  {
    const VkImportMemoryHostPointerInfoEXT import
        = { .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
            .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
            .pHostPointer = (void *)start };
    const VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                       .pNext = &import,
                                       .allocationSize = span,
                                       .memoryTypeIndex = type };
    ok = _allocate(&mai, &imp->mem.buffer_mem)
         && vkBindBufferMemory(_vk.dev, imp->mem.buffer, imp->mem.buffer_mem, 0) == VK_SUCCESS;
  }
  if(!ok)
  {
    dt_print(DT_DEBUG_SLANG, "[slang_gpu] can't import %zu MB of host memory, copying it instead", bytes >> 20);
    _import_free(imp);
    return NULL;
  }
  return imp;
}

static void _copy_rows(void *dst, const size_t dst_pitch, const void *src, const size_t src_pitch,
                       const size_t row, const int height)
{
  if(dst_pitch == row && src_pitch == row)
  {
    _copy(dst, src, row * height);
    return;
  }
  DT_OMP_FOR()
  for(int y = 0; y < height; y++)
    memcpy((char *)dst + y * dst_pitch, (const char *)src + y * src_pitch, row);
}

// reads above DT_SLANG_IMPORT_MAX go through two staging buffers of this
// size, the CPU copying one out while the device fills the other. Importing
// the host memory instead pinned it at about 0.5 ms per MB on a Radeon 780M,
// 0.4 of the 1.2 s that the final read of a 61 MP export took
#define DT_SLANG_READ_BAND (64 << 20)

static int _read_banded(dt_slang_gpu_image_t *img,
                        void *host,
                        const size_t pitch,
                        const int x,
                        const int y,
                        const int w,
                        const int h)
{
  const size_t row = sizeof(float) * img->channels * w;
  const int band = MAX(1, (int)MIN(DT_SLANG_READ_BAND / row, (size_t)h));
  const int bands = (h + band - 1) / band;
  dt_slang_staging_t *s[2] = { _staging_acquire(row * band, FALSE), NULL };
  if(s[0]) s[1] = _staging_acquire(row * band, FALSE);
  if(!s[1])
  {
    if(s[0]) _staging_release(s[0]);
    return 1;
  }
  uint64_t serial[2] = { 0, 0 };
  gboolean ok = TRUE;
  // band k goes to s[k & 1], and is queued before band k - 1 is copied out
  for(int k = 0; k <= bands && ok; k++)
  {
    if(k < bands)
    {
      dt_slang_batch_t *b = _batch_begin(NULL, 0, 0);
      ok = b != NULL;
      if(ok)
      {
        const VkBufferImageCopy copy = { .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                                         .imageOffset = { x, y + k * band, 0 },
                                         .imageExtent = { w, MIN(band, h - k * band), 1 } };
        vkCmdCopyImageToBuffer(b->cb, img->image, VK_IMAGE_LAYOUT_GENERAL, s[k & 1]->mem.buffer, 1, &copy);
        _host_read_barrier(b->cb);
        ok = _batch_submit(b, FALSE);
        serial[k & 1] = _vk.submitted;
      }
    }
    if(ok && k > 0)
    {
      const int j = k - 1;
      _wait_until(serial[j & 1]);
      ok = !_vk.lost;
      if(ok)
      {
        _invalidate(&s[j & 1]->mem);
        _copy_rows((char *)host + (size_t)j * band * pitch, pitch, s[j & 1]->mem.map, row, row,
                   MIN(band, h - j * band));
      }
    }
  }
  if(!ok) _wait_all();
  _staging_release(s[0]);
  _staging_release(s[1]);
  return ok ? 0 : 1;
}

// copies between the rectangle x, y, w x h of img and host, whose rows are
// pitch bytes apart. With the host memory imported the GPU copies it
// directly, and both directions wait for the copy, as the caller may change
// or free host afterwards; otherwise the CPU copies it through a staging
// buffer, and only reads wait.
static int _image_transfer_rect(dt_slang_gpu_image_t *img,
                                void *host,
                                const size_t pitch,
                                const int x,
                                const int y,
                                const int w,
                                const int h,
                                const gboolean to_device)
{
  const size_t bpp = sizeof(float) * img->channels;
  const size_t row = bpp * w;
  const size_t span = pitch * (h - 1) + row;
  if(!to_device && span > DT_SLANG_IMPORT_MAX && h > 1) return _read_banded(img, host, pitch, x, y, w, h);
  if(_vk.host_import && span > DT_SLANG_IMPORT_MAX && h > 1)
  {
    const int band = MAX(1, DT_SLANG_IMPORT_MAX / pitch);
    for(int b = 0; b < h; b += band)
      if(_image_transfer_rect(img, (char *)host + b * pitch, pitch, x, y + b, w, MIN(band, h - b), to_device))
        return 1;
    return 0;
  }

  VkBufferImageCopy copy = { .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                             .imageOffset = { x, y, 0 },
                             .imageExtent = { w, h, 1 } };

  // the copy engine addresses the host rows in whole pixels
  dt_slang_import_t *imp = _vk.host_import && span >= DT_SLANG_IMPORT_MIN
                           && pitch % bpp == 0 && (uintptr_t)host % bpp == 0
    ? _import_new(host, span)
    : NULL;
  if(imp)
  {
    dt_slang_batch_t *b = _batch_begin(NULL, 0, 0);
    if(!b)
    {
      _import_free(imp);
      return 1;
    }
    copy.bufferOffset = (const char *)host - imp->base;
    copy.bufferRowLength = pitch / bpp;
    if(to_device)
      vkCmdCopyBufferToImage(b->cb, imp->mem.buffer, img->image, VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
    else
    {
      vkCmdCopyImageToBuffer(b->cb, img->image, VK_IMAGE_LAYOUT_GENERAL, imp->mem.buffer, 1, &copy);
      _host_read_barrier(b->cb);
    }
    const gboolean ok = _batch_submit(b, TRUE);
    _import_free(imp);
    return ok ? 0 : 1;
  }

  dt_slang_staging_t *s = _staging_acquire(row * h, to_device);
  if(!s) return 1;
  dt_slang_batch_t *b = _batch_begin(NULL, 0, 0);
  if(!b)
  {
    _staging_release(s);
    return 1;
  }
  if(to_device)
  {
    _copy_rows(s->mem.map, row, host, pitch, row, h);
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
    _copy_rows(host, pitch, s->mem.map, row, row, h);
  }
  _staging_release(s);
  return ok ? 0 : 1;
}

// copies between host memory and the whole of img
static int _image_transfer(dt_slang_gpu_image_t *img, void *host, const gboolean to_device)
{
  const size_t row = sizeof(float) * img->channels * img->width;
  return _image_transfer_rect(img, host, row, 0, 0, img->width, img->height, to_device);
}

void dt_slang_gpu_cleanup(void)
{
  g_atomic_int_set(&_preload_stop, 1);
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
  g_list_free_full(_vk.stale, (GDestroyNotify)_stale_free);
  g_list_free_full(_vk.staging, (GDestroyNotify)_staging_destroy);
  if(_vk.kernels) g_hash_table_destroy(_vk.kernels);
  if(_building) g_hash_table_destroy(_building);
  _building = NULL;
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
    && _create_image(&img->image, width, height, channels, DT_SLANG_IMAGE_USAGE);
  if(ok)
  {
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(_vk.dev, img->image, &req);
    req.size = _size_class(req.size);
    // the memory of a pooled image of another size in the same class; its
    // own image may still be in use, so it is destroyed later
    dt_slang_gpu_image_t *pooled = NULL;
    for(GList *l = _vk.images; l && !pooled; l = g_list_next(l))
    {
      dt_slang_gpu_image_t *p = l->data;
      if(p->mem_size == req.size && (req.memoryTypeBits & (1u << p->mem_type)))
      {
        pooled = p;
        _vk.images = g_list_delete_link(_vk.images, l);
      }
    }
    if(pooled)
    {
      img->mem = pooled->mem;
      img->mem_type = pooled->mem_type;
      dt_slang_stale_t *st = g_new0(dt_slang_stale_t, 1);
      st->image = pooled->image;
      st->view = pooled->view;
      st->last_use = pooled->last_use;
      _vk.stale = g_list_prepend(_vk.stale, st);
      g_free(pooled);
    }
    else
      ok = _alloc_on_device(&req, &img->mem, &img->mem_type);
    img->mem_size = req.size;
    ok = ok && vkBindImageMemory(_vk.dev, img->image, img->mem, 0) == VK_SUCCESS
         && _create_view(&img->view, img->image, channels);
  }
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
    dt_print(DT_DEBUG_SLANG, "[slang_gpu] can't allocate a %dx%d device image", width, height);
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

void dt_slang_gpu_release_pools(void)
{
  if(!_vk.usable || !_vk.discrete) return;
  g_mutex_lock(&_vk_lock);
  _retire_finished();
  _vk.doomed_images = g_list_concat(_vk.doomed_images, _vk.images);
  _vk.doomed_buffers = g_list_concat(_vk.doomed_buffers, _vk.buffers);
  _vk.images = NULL;
  _vk.buffers = NULL;
  _free_doomed();
  g_mutex_unlock(&_vk_lock);
}

void dt_slang_gpu_wait(void)
{
  if(!_vk.usable) return;
  g_mutex_lock(&_vk_lock);
  _wait_all();
  g_mutex_unlock(&_vk_lock);
}

void dt_slang_gpu_free_align(void *mem)
{
  // Vulkan never aliases host memory
  dt_free_align(mem);
}

void dt_slang_gpu_image_free(dt_slang_gpu_image_t *img)
{
  if(!img) return;
  g_mutex_lock(&_vk_lock);
  // kernels still running may use the image, so a pooled one is reused only
  // by later kernels
  img->last_use = _vk.submitted;
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

int dt_slang_gpu_image_write_rect(dt_slang_gpu_image_t *img, const void *host, const size_t pitch)
{
  if(!img || !host) return 1;
  g_mutex_lock(&_vk_lock);
  const int err = _image_transfer_rect(img, (void *)host, pitch, 0, 0, img->width, img->height, TRUE);
  g_mutex_unlock(&_vk_lock);
  return err;
}

int dt_slang_gpu_image_read_rect(dt_slang_gpu_image_t *img,
                                 const int x,
                                 const int y,
                                 const int width,
                                 const int height,
                                 void *host,
                                 const size_t pitch)
{
  if(!img || !host || x < 0 || y < 0 || width <= 0 || height <= 0
     || x + width > img->width || y + height > img->height)
    return 1;
  g_mutex_lock(&_vk_lock);
  const int err = _image_transfer_rect(img, host, pitch, x, y, width, height, FALSE);
  g_mutex_unlock(&_vk_lock);
  return err;
}

int dt_slang_gpu_image_copy(dt_slang_gpu_image_t *src,
                            const int sx,
                            const int sy,
                            dt_slang_gpu_image_t *dst,
                            const int dx,
                            const int dy,
                            const int width,
                            const int height)
{
  if(!src || !dst || src->channels != dst->channels || width <= 0 || height <= 0
     || sx < 0 || sy < 0 || dx < 0 || dy < 0
     || sx + width > src->width || sy + height > src->height
     || dx + width > dst->width || dy + height > dst->height)
    return 1;
  g_mutex_lock(&_vk_lock);
  dt_slang_batch_t *b = _batch_begin(NULL, 0, 0);
  gboolean ok = b != NULL;
  if(ok)
  {
    const VkImageCopy copy = { .srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                               .srcOffset = { sx, sy, 0 },
                               .dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                               .dstOffset = { dx, dy, 0 },
                               .extent = { width, height, 1 } };
    vkCmdCopyImage(b->cb, src->image, VK_IMAGE_LAYOUT_GENERAL, dst->image, VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
    ok = _batch_submit(b, FALSE);
  }
  g_mutex_unlock(&_vk_lock);
  return ok ? 0 : 1;
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
      _copy(host, s->mem.map, size);
      err = 0;
    }
  }
  if(s) _staging_release(s);
  g_mutex_unlock(&_vk_lock);
  return err;
}

// --------------------------------------------------------------------------------------------
// dispatch

static int _run(dt_slang_kernel_t *k,
                const char *entry,
                const int width,
                const int height,
                const int nargs,
                const dt_slang_arg_t *args)
{
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
      dt_print(DT_DEBUG_SLANG, "[slang_gpu] %dx%d image too large for %s", arg->width, arg->height, entry);
      return 1;
    }
    if(arg->kind == DT_SLANG_ARG_IMAGE_OUT) host_out = TRUE;
    res_arg[nres++] = a;
  }
  if(nres != k->nres || nvals != k->nvals) goto mismatch;

  // the timing is printed once the batch has finished, after the caller's
  // entry name may be gone: gaussian.c builds its names on the stack
  dt_slang_batch_t *b = _batch_begin(g_intern_string(entry), width, height);
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
      VkMemoryRequirements req;
      uint32_t type;
      if(!_create_image(&m->image, arg->width, arg->height, arg->channels,
                        out ? VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                            : VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT))
        goto done;
      vkGetImageMemoryRequirements(_vk.dev, m->image, &req);
      if(!_alloc_on_device(&req, &m->image_mem, &type)
         || vkBindImageMemory(_vk.dev, m->image, m->image_mem, 0) != VK_SUCCESS
         || !_create_view(&m->view, m->image, arg->channels))
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
        _copy(s->mem.map, arg->ptr, bytes);
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
    _copy((void *)arg->ptr, readback[r]->mem.map, sizeof(float) * arg->channels * arg->width * arg->height);
  }
  err = 0;

done:
  // a batch that failed while recording never runs, but hands back what it took
  if(b) _batch_release(b);
  for(int r = 0; r < nres; r++)
    if(readback[r]) _staging_release(readback[r]);
  if(err) dt_print(DT_DEBUG_SLANG, "[slang_gpu] running %s failed", entry);
  return err;

mismatch:
  dt_print(DT_DEBUG_SLANG, "[slang_gpu] arguments don't match the parameters of %s:"
           " %d resources and %d values expected", entry, k->nres, k->nvals);
  return 1;
}

// what the pools hold, including what was evicted and waits for its
// batches, and with staging the idle staging buffers
static VkDeviceSize _pooled_bytes(const gboolean staging)
{
  VkDeviceSize pooled = 0;
  for(GList *l = _vk.images; l; l = g_list_next(l))
    pooled += _image_bytes(l->data);
  for(GList *l = _vk.doomed_images; l; l = g_list_next(l))
    pooled += _image_bytes(l->data);
  for(GList *l = _vk.buffers; l; l = g_list_next(l))
    pooled += ((const dt_slang_gpu_buffer_t *)l->data)->size;
  for(GList *l = _vk.doomed_buffers; l; l = g_list_next(l))
    pooled += ((const dt_slang_gpu_buffer_t *)l->data)->size;
  if(staging)
    for(GList *l = _vk.staging; l; l = g_list_next(l))
      pooled += ((const dt_slang_staging_t *)l->data)->size;
  return pooled;
}

size_t dt_slang_gpu_pooled_memory(void)
{
  if(!dt_slang_gpu_enabled()) return 0;
  g_mutex_lock(&_vk_lock);
  // dt_slang_gpu_memory_info() counts the idle staging buffers only on an
  // integrated GPU, where they share the heaps with the images
  const VkDeviceSize pooled = _pooled_bytes(!_vk.discrete);
  g_mutex_unlock(&_vk_lock);
  return pooled;
}

void dt_slang_gpu_memory_info(size_t *available, int *max_dim)
{
  *available = 0;
  *max_dim = 0;
  if(!dt_slang_gpu_enabled()) return;

  g_mutex_lock(&_vk_lock);
  VkDeviceSize best = 0;
  if(_vk.memory_budget)
  {
    // the budget covers other applications too, and what the pools keep
    // counts as used although it would be freed for the module. So does what
    // was evicted from them and waits for its batches: after a tiled
    // denoiseprofile on a Radeon 780M that left sharpen and colorequal
    // 150-200 MB short, and both ran in tiles
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget
        = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT };
    VkPhysicalDeviceMemoryProperties2 props2
        = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, .pNext = &budget };
    vkGetPhysicalDeviceMemoryProperties2(_vk.pdev, &props2);
    VkDeviceSize pooled = _pooled_bytes(FALSE);
    if(_vk.discrete)
    {
      for(uint32_t h = 0; h < _vk.memprops.memoryHeapCount; h++)
        if(_device_heap(h))
        {
          const VkDeviceSize used = budget.heapUsage[h] > pooled ? budget.heapUsage[h] - pooled : 0;
          if(budget.heapBudget[h] > used) best = MAX(best, budget.heapBudget[h] - used);
        }
    }
    else
    {
      // a module's images may be spread over all heaps, see _alloc_on_device(),
      // and _make_room() frees the idle staging buffers for them
      VkDeviceSize total = 0, total_host = 0, free_host = 0;
      for(uint32_t h = 0; h < _vk.memprops.memoryHeapCount; h++)
      {
        const VkDeviceSize usable = _usable_budget(&budget, h);
        const VkDeviceSize free = usable > budget.heapUsage[h] ? usable - budget.heapUsage[h] : 0;
        if(_vk.host_heaps & (1u << h))
        {
          total_host += usable;
          free_host += free;
        }
        else
        {
          total += usable;
          best += free;
        }
      }
      if(_vk.host_cap)
      {
        const VkDeviceSize used = _host_usage(&budget);
        total_host = MIN(total_host, _vk.host_cap);
        // the usage counts more than darktable's images on some drivers
        // (lavapipe reports the whole process), so tiles get at least a
        // quarter of the cap rather than none
        free_host = MIN(free_host, MAX(_vk.host_cap > used ? _vk.host_cap - used : 0, _vk.host_cap / 4));
      }
      total += total_host;
      best += free_host;
      pooled = _pooled_bytes(TRUE);
      best = MIN(total, best + pooled);
    }
  }
  else
    for(uint32_t h = 0; h < _vk.memprops.memoryHeapCount; h++)
      if(_device_heap(h))
      {
        VkDeviceSize usable = _vk.memprops.memoryHeaps[h].size / 4 * 3;
        if(_vk.host_cap && (_vk.host_heaps & (1u << h))) usable = MIN(usable, _vk.host_cap);
        best = _vk.discrete ? MAX(best, usable) : best + usable;
      }
  const int cap = dt_conf_get_int("slang_gpu_memory");
  if(cap > 0) best = MIN(best, (VkDeviceSize)cap << 20);
  *available = best;
  *max_dim = _vk.max_image_dim;
  g_mutex_unlock(&_vk_lock);
}

int dt_slang_gpu_run_2d(const char *program,
                        const char *kernel,
                        const int width,
                        const int height,
                        const int nargs,
                        const dt_slang_arg_t *args)
{
  if(!dt_slang_gpu_enabled()) return 1;

  // built outside _vk_lock, so that another pipe can dispatch meanwhile
  dt_slang_kernel_t *k = _get_kernel(program, kernel, TRUE);
  if(!k) return 1;

  // one queue for all pipes: pipes running in parallel take turns here. The
  // time of each kernel is logged from GPU timestamps once it has finished
  g_mutex_lock(&_vk_lock);
  const int err = _run(k, kernel, width, height, nargs, args);
  g_mutex_unlock(&_vk_lock);
  return err;
}
