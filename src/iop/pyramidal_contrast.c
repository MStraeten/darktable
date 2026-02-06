/*
    This file is part of darktable,
    Copyright (C) 2018-2025 darktable developers.

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

 * This module provides dual local contrast enhancement modes:
 * - GLOBAL : Pyramidal contrast: multi-scale enhancement with independent controls
 * - EXPRT : Local RGB contrast: multi-scale enhancement with shared parameters
 *
 * Both operate in scene-referred linear RGB space and should be placed early in the pipe.
    */
/*** DOCUMENTATION
 *
 * This module performs local contrast enhancement in scene-referred linear RGB space.
 *
 * It works by computing two luminance masks:
 * 1. A pixel-wise luminance (unblurred)
 * 2. A smoothed luminance using edge-aware filters (guided filter or EIGF)
 *
 * The difference between these two masks represents the local detail/contrast.
 * The local contrast is then enhanced by scaling this difference and applying
 * it as an exposure correction to each pixel.
 *
 * The module should be placed early in the pipe (before color profile)
 * as it operates on scene-linear RGB data.
 * A Modifier
 ***/


#include "common/extra_optimizations.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "bauhaus/bauhaus.h"
#include "common/darktable.h"
#include "common/fast_guided_filter.h"
#include "common/eigf.h"
#include "common/luminance_mask.h"
#include "common/opencl.h"
#include "control/conf.h"
#include "control/control.h"
#include "develop/blend.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "develop/imageop_math.h"
#include "develop/imageop_gui.h"
#include "gui/accelerators.h"
#include "gui/draw.h"
#include "dtgtk/paint.h"
#include "dtgtk/togglebutton.h"
#include "dtgtk/expander.h"
#include "gui/gtk.h"
#include "gui/presets.h"
#include "iop/iop_api.h"
#include "common/iop_group.h"

#ifdef _OPENMP
#include <omp.h>
#endif


DT_MODULE_INTROSPECTION(1, dt_iop_pyramidal_contrast_params_t)


#define MIN_FLOAT exp2f(-16.0f)

/**
 * Filter types for detail preservation / smoothing.
 * DT_TONEEQ_NONE is intentionally omitted as it produces no blur,
 * which would result in no local contrast extraction.
 **/
typedef enum dt_iop_pyramidal_contrast_filter_t
{
  DT_PYR_AVG_GUIDED = 0, // $DESCRIPTION: "averaged guided filter"
  DT_PYR_GUIDED,         // $DESCRIPTION: "guided filter"
  DT_PYR_AVG_EIGF,       // $DESCRIPTION: "averaged EIGF"
  DT_PYR_EIGF            // $DESCRIPTION: "EIGF"
} dt_iop_pyramidal_contrast_filter_t;

typedef enum dt_iop_pyramidal_contrast_expert_filter_t
{
  DT_EXP_AVG_GUIDED = 0, // $DESCRIPTION: "averaged guided filter"
  DT_EXP_GUIDED,         // $DESCRIPTION: "guided filter"
  DT_EXP_AVG_EIGF,       // $DESCRIPTION: "averaged EIGF"
  DT_EXP_EIGF            // $DESCRIPTION: "EIGF"
} dt_iop_pyramidal_contrast_expert_filter_t;

typedef enum dt_iop_pyramidal_contrast_mode_t
{
  DT_PYR_MODE_GLOBAL = 0, // $DESCRIPTION: "global"
  DT_PYR_MODE_EXPERT = 1  // $DESCRIPTION: "expert"
} dt_iop_pyramidal_contrast_mode_t;

#define N_SCALES 3

typedef struct dt_iop_pyramidal_contrast_params_t
{
  dt_iop_pyramidal_contrast_mode_t mode; // $DEFAULT: DT_PYR_MODE_GLOBAL $DESCRIPTION: "mode"

  // Local contrast scaling factor
  // Global mode params
  float pyr_micro_scale;    // $MIN: 0.0 $MAX: 5.0 $DEFAULT: 1.0 $DESCRIPTION: "micro contrast"
  float pyr_fine_scale;     // $MIN: 0.0 $MAX: 5.0 $DEFAULT: 1.0 $DESCRIPTION: "fine contrast"
  float pyr_detail_scale;   // $MIN: 0.0 $MAX: 5.0 $DEFAULT: 1.5 $DESCRIPTION: "local contrast"
  float pyr_medium_scale;   // $MIN: 0.0 $MAX: 5.0 $DEFAULT: 1.0 $DESCRIPTION: "broad contrast"
  float pyr_broad_scale;    // $MIN: 0.0 $MAX: 5.0 $DEFAULT: 1.0 $DESCRIPTION: "extended contrast"
  float pyr_global_scale;   // $MIN: 0.0 $MAX: 5.0 $DEFAULT: 1.0 $DESCRIPTION: "global contrast"

  // Masking parameters
  // Blending is log-encoded because changes in small values are more noticeable
  float pyr_blending;       // $MIN: 1.0 $MAX: 4.0 $DEFAULT: 1.2 $DESCRIPTION: "feature scale"
  float pyr_feathering;     // $MIN: 0.01 $MAX: 10000.0 $DEFAULT: 5.0 $DESCRIPTION: "edges refinement/feathering"

  float pyr_f_mult_micro;  // $MIN: 0.1 $MAX: 5.0 $DEFAULT: 0.5 $DESCRIPTION: "micro contrast feathering"
  float pyr_f_mult_fine;   // $MIN: 0.1 $MAX: 5.0 $DEFAULT: 0.75 $DESCRIPTION: "fine contrast feathering"
  float pyr_f_mult_detail; // $MIN: 0.1 $MAX: 5.0 $DEFAULT: 1.0 $DESCRIPTION: "local contrast feathering"
  float pyr_f_mult_medium; // $MIN: 0.1 $MAX: 5.0 $DEFAULT: 1.25 $DESCRIPTION: "broad contrast feathering"
  float pyr_f_mult_broad;  // $MIN: 0.1 $MAX: 5.0 $DEFAULT: 1.50 $DESCRIPTION: "extended contrast feathering"

  float pyr_s_mult_micro;  // $MIN: 0.1 $MAX: 5.0 $DEFAULT: 0.25 $DESCRIPTION: "micro contrast scale mult."
  float pyr_s_mult_fine;   // $MIN: 0.1 $MAX: 5.0 $DEFAULT: 0.625 $DESCRIPTION: "fine contrast scale mult."
  float pyr_s_mult_detail; // $MIN: 0.1 $MAX: 5.0 $DEFAULT: 1.0 $DESCRIPTION: "local contrast scale mult."
  float pyr_s_mult_medium; // $MIN: 0.1 $MAX: 5.0 $DEFAULT: 1.8 $DESCRIPTION: "broad contrast scale mult."
  float pyr_s_mult_broad;  // $MIN: 0.1 $MAX: 5.0 $DEFAULT: 2.8 $DESCRIPTION: "extended contrast scale mult."

  dt_iop_pyramidal_contrast_filter_t pyr_details; // $DEFAULT: DT_PYR_EIGF $DESCRIPTION: "feature extractor"
  dt_iop_luminance_mask_method_t pyr_method;      // $DEFAULT: DT_TONEEQ_NORM_2 $DESCRIPTION: "luminance estimator"
  int pyr_iterations;       // $MIN: 1 $MAX: 20 $DEFAULT: 1 $DESCRIPTION: "filter diffusion"

  // Expert mode params
  float exp_detail_boost[N_SCALES];   // $MIN: 0.0 $MAX: 500.0 $DEFAULT: 100.0 $DESCRIPTION: "detail boost"
  float exp_feature_scale[N_SCALES];       // $MIN: 0.01 $MAX: 100.0 $DEFAULT: 12.0 $DESCRIPTION: "feature scale"
  float exp_feathering[N_SCALES];     // $MIN: 0.01 $MAX: 10000.0 $DEFAULT: 5.0 $DESCRIPTION: "edges refinement"

  dt_iop_pyramidal_contrast_expert_filter_t exp_details; // $DEFAULT: DT_EXP_EIGF $DESCRIPTION: "feature extractor"
  dt_iop_luminance_mask_method_t exp_method;      // $DEFAULT: DT_TONEEQ_NORM_2 $DESCRIPTION: "luminance estimator"
  int exp_iterations;                             // $MIN: 1 $MAX: 20 $DEFAULT: 1 $DESCRIPTION: "filter diffusion"
} dt_iop_pyramidal_contrast_params_t;

typedef struct dt_iop_pyramidal_contrast_scale_data_t
{
  float exp_detail_boost;
  float exp_feature_scale;
  float exp_feathering;
  int exp_radius;    // derived from feature_scale and image size
} dt_iop_pyramidal_contrast_scale_data_t;

typedef struct dt_iop_pyramidal_contrast_data_t
{
  dt_iop_pyramidal_contrast_mode_t mode;

  float pyr_broad_scale;
  float pyr_medium_scale;
  float pyr_detail_scale;
  float pyr_fine_scale;
  float pyr_micro_scale;
  float pyr_global_scale;
  float pyr_blending, pyr_feathering;
  float pyr_f_mult_micro, pyr_f_mult_fine, pyr_f_mult_detail, pyr_f_mult_medium, pyr_f_mult_broad;
  float pyr_s_mult_micro, pyr_s_mult_fine, pyr_s_mult_detail, pyr_s_mult_medium, pyr_s_mult_broad;
  float pyr_scale;
  int pyr_radius;
  int pyr_radius_broad;
  int pyr_radius_medium;
  int pyr_radius_fine;
  int pyr_radius_micro;
  int pyr_iterations;
  dt_iop_luminance_mask_method_t pyr_method;
  dt_iop_pyramidal_contrast_filter_t pyr_details;

  dt_iop_pyramidal_contrast_scale_data_t exp_scales[N_SCALES];
  int exp_iterations;
  dt_iop_luminance_mask_method_t exp_method;
  dt_iop_pyramidal_contrast_expert_filter_t exp_details;
} dt_iop_pyramidal_contrast_data_t;


typedef struct dt_iop_pyramidal_contrast_global_data_t
{
  // Reserved for OpenCL kernels
} dt_iop_pyramidal_contrast_global_data_t;


typedef enum dt_iop_pyramidal_contrast_mask_t
{
  DT_PYR_MASK_OFF = 0,
  DT_PYR_MASK_BROAD = 1,
  DT_PYR_MASK_MEDIUM = 2,
  DT_PYR_MASK_DETAIL = 3,
  DT_PYR_MASK_FINE = 4,
  DT_PYR_MASK_MICRO = 5
} dt_iop_pyramidal_contrast_mask_t;

typedef struct dt_iop_pyramidal_contrast_gui_data_t
{
  // Flags
  dt_iop_pyramidal_contrast_mask_t pyr_mask_display;

  // Buffer dimensions
  int pyr_buf_width;
  int pyr_buf_height;
  int pyr_pipe_order;

  // Hash for cache invalidation
  dt_hash_t pyr_ui_preview_hash;
  dt_hash_t pyr_thumb_preview_hash;
  size_t pyr_full_preview_buf_width, pyr_full_preview_buf_height;
  size_t pyr_thumb_preview_buf_width, pyr_thumb_preview_buf_height;

  // Cached luminance buffers
  float *pyr_thumb_preview_buf_pixel;     // pixel-wise luminance (no blur)
  float *pyr_thumb_preview_buf_smoothed_broad;
  float *pyr_thumb_preview_buf_smoothed_medium;
  float *pyr_thumb_preview_buf_smoothed;  // smoothed luminance
  float *pyr_thumb_preview_buf_smoothed_fine;
  float *pyr_thumb_preview_buf_smoothed_micro;
  float *pyr_full_preview_buf_pixel;
  float *pyr_full_preview_buf_smoothed_broad;
  float *pyr_full_preview_buf_smoothed_medium;
  float *pyr_full_preview_buf_smoothed;
  float *pyr_full_preview_buf_smoothed_fine;
  float *pyr_full_preview_buf_smoothed_micro;

  // Cache validity
  gboolean luminance_valid;

  // GTK widgets
  GtkWidget *pyr_broad_scale, *pyr_medium_scale, *pyr_detail_scale, *pyr_fine_scale, *pyr_micro_scale, *pyr_global_scale;
  GtkWidget *pyr_blending;
  GtkWidget *pyr_feathering;
  dt_gui_collapsible_section_t pyr_advanced_expander;
  GtkWidget *pyr_f_mult_micro, *pyr_f_mult_fine, *pyr_f_mult_detail, *pyr_f_mult_medium, *pyr_f_mult_broad;

  // New buttons for mask display in expanders
  GtkWidget *pyr_f_view_broad, *pyr_f_view_medium, *pyr_f_view_detail, *pyr_f_view_fine, *pyr_f_view_micro;

  // Expert mode widgets
  GtkNotebook *notebook;
  GtkWidget *global_box;
  GtkWidget *expert_box;

  int exp_mask_display_scale;
  int exp_buf_width;
  int exp_buf_height;
  int exp_pipe_order;
  dt_hash_t exp_ui_preview_hash;
  dt_hash_t exp_thumb_preview_hash;
  size_t exp_full_preview_buf_width, exp_full_preview_buf_height;
  size_t exp_thumb_preview_buf_width, exp_thumb_preview_buf_height;
  float *exp_thumb_preview_buf_pixel;
  float *exp_thumb_preview_buf_smoothed[N_SCALES];
  float *exp_full_preview_buf_pixel;
  float *exp_full_preview_buf_smoothed[N_SCALES];
  gboolean exp_luminance_valid;
  GtkWidget *exp_detail_boost[N_SCALES];
  GtkWidget *exp_feature_scale[N_SCALES];
  GtkWidget *exp_feathering[N_SCALES];
  GtkWidget *exp_show_mask[N_SCALES];
  GtkWidget *exp_details;
  GtkWidget *exp_iterations;
} dt_iop_pyramidal_contrast_gui_data_t;
static void mode_tab_switch_callback(GtkNotebook *notebook, GtkWidget *page, guint page_num, dt_iop_module_t *self);

static void exp_invalidate_luminance_cache(dt_iop_module_t *const self);
static void pyr_invalidate_luminance_cache(dt_iop_module_t *const self);

const char *name()
{
  return _("pyramidal contrast");
}

const char *aliases()
{
  return _("local contrast|clarity|detail enhancement");
}

const char **description(dt_iop_module_t *self)
{
  return dt_iop_set_description
    (self, _("enhance local contrast by boosting contrast while preserving edges"),
     _("creative"),
     _("linear, RGB, scene-referred"),
     _("linear, RGB"),
     _("linear, RGB, scene-referred"));
}

int default_group()
{
  return IOP_GROUP_BASIC | IOP_GROUP_EFFECTS;
}

int flags()
{
  return IOP_FLAGS_INCLUDE_IN_STYLES | IOP_FLAGS_SUPPORTS_BLENDING;
}

dt_iop_colorspace_type_t default_colorspace(dt_iop_module_t *self,
                                            dt_dev_pixelpipe_t *pipe,
                                            dt_dev_pixelpipe_iop_t *piece)
{
  return IOP_CS_RGB;
}

int legacy_params(dt_iop_module_t *self,
                  const void *const old_params,
                  const int old_version,
                  void **new_params,
                  int32_t *new_params_size,
                  int *new_version)
{
  return 1;
}

/**
 * Helper functions
 **/

static void pyr_hash_set_get(const dt_hash_t *hash_in,
                         dt_hash_t *hash_out,
                         dt_pthread_mutex_t *lock)
{
  dt_pthread_mutex_lock(lock);
  *hash_out = *hash_in;
  dt_pthread_mutex_unlock(lock);
}

static void exp_invalidate_luminance_cache(dt_iop_module_t *const self)
{
  dt_iop_pyramidal_contrast_gui_data_t *const restrict g = self->gui_data;

  dt_iop_gui_enter_critical_section(self);
  g->exp_luminance_valid = FALSE;
  g->exp_thumb_preview_hash = DT_INVALID_HASH;
  g->exp_ui_preview_hash = DT_INVALID_HASH;
  dt_iop_gui_leave_critical_section(self);
  dt_iop_refresh_all(self);
}

static void pyr_invalidate_luminance_cache(dt_iop_module_t *const self)
{
  dt_iop_pyramidal_contrast_gui_data_t *const restrict g = self->gui_data;

  dt_iop_gui_enter_critical_section(self);
  g->luminance_valid = FALSE;
  g->pyr_thumb_preview_hash = DT_INVALID_HASH;
  g->pyr_ui_preview_hash = DT_INVALID_HASH;
  g->exp_luminance_valid = FALSE;
  g->exp_thumb_preview_hash = DT_INVALID_HASH;
  g->exp_ui_preview_hash = DT_INVALID_HASH;
  dt_iop_gui_leave_critical_section(self);
  dt_iop_refresh_all(self);
}


/**
 * Compute pixel-wise luminance mask (no blur)
 **/
__DT_CLONE_TARGETS__
static inline void pyr_compute_pixel_luminance_mask(const float *const restrict in,
                                                float *const restrict luminance,
                                                const size_t width,
                                                const size_t height,
                                                const dt_iop_luminance_mask_method_t method)
{
  // No exposure/contrast boost, just compute raw luminance
  luminance_mask(in, luminance, width, height, method, 1.0f, 0.0f, 1.0f);
}


/**
 * Compute smoothed luminance mask using edge-aware filters
 **/
__DT_CLONE_TARGETS__
static inline void pyr_compute_smoothed_luminance_mask(const float *const restrict in,
                                                   float *const restrict luminance,
                                                   const size_t width,
                                                   const size_t height,
                                                const dt_iop_pyramidal_contrast_data_t *const d,
                                                const int radius,
                                                const float feathering)
{
  // First compute pixel-wise luminance (no boost)
  luminance_mask(in, luminance, width, height, d->pyr_method, 1.0f, 0.0f, 1.0f);

  // Then apply the smoothing filter
  switch(d->pyr_details)
  {
    case(DT_PYR_AVG_GUIDED):
    {
      fast_surface_blur(luminance, width, height, radius, feathering, d->pyr_iterations,
                        DT_GF_BLENDING_GEOMEAN, d->pyr_scale, 0.0f,
                        exp2f(-14.0f), 4.0f);
      break;
    }

    case(DT_PYR_GUIDED):
    {
      fast_surface_blur(luminance, width, height, radius, feathering, d->pyr_iterations,
                        DT_GF_BLENDING_LINEAR, d->pyr_scale, 0.0f,
                        exp2f(-14.0f), 4.0f);
      break;
    }

    case(DT_PYR_AVG_EIGF):
    {
      fast_eigf_surface_blur(luminance, width, height,
                             radius, feathering, d->pyr_iterations,
                             DT_GF_BLENDING_GEOMEAN, d->pyr_scale,
                             0.0f, exp2f(-14.0f), 4.0f);
      break;
    }

    case(DT_PYR_EIGF):
    {
      fast_eigf_surface_blur(luminance, width, height,
                             radius, feathering, d->pyr_iterations,
                             DT_GF_BLENDING_LINEAR, d->pyr_scale,
                             0.0f, exp2f(-14.0f), 4.0f);
      break;
    }
  }
}


/**
 * Apply local contrast enhancement
 *
 * The detail (local contrast) is the log-space difference between pixel luminance
 * and smoothed luminance. Boosting this difference amplifies local details.
 **/
__DT_CLONE_TARGETS__
static inline void pyr_apply_local_contrast(const float *const restrict in,
                                        const float *const restrict luminance_pixel,
                                        const float *const restrict luminance_smoothed,
                                        const float *const restrict luminance_smoothed_broad,
                                        const float *const restrict luminance_smoothed_medium,
                                        const float *const restrict luminance_smoothed_fine,
                                        const float *const restrict luminance_smoothed_micro,
                                        float *const restrict out,
                                        const dt_iop_roi_t *const roi_in,
                                        const dt_iop_roi_t *const roi_out,
                                        const dt_iop_pyramidal_contrast_data_t *const d)
{
  const size_t npixels = (size_t)roi_in->width * roi_in->height;

  DT_OMP_FOR()
  for(size_t k = 0; k < npixels; k++)
  {
    // Detail in log space (EV): how much brighter/darker is this pixel
    // compared to its local neighborhood
    // detail = log2(pixel_lum / smoothed_lum) = log2(pixel_lum) - log2(smoothed_lum)
    const float lum_pixel = fmaxf(luminance_pixel[k], MIN_FLOAT);
    const float lum_smoothed = fmaxf(luminance_smoothed[k], MIN_FLOAT);
    const float detail_ev = log2f(lum_pixel / lum_smoothed);

    // Scale the detail: detail_scale = 1.0 means no change
    // > 1.0 boosts local contrast, < 1.0 reduces it
    const float scaled_detail_ev = d->pyr_detail_scale * detail_ev;

    // The correction is the difference between scaled and original detail
    float correction_ev = scaled_detail_ev - detail_ev;

    if(luminance_smoothed_broad)
    {
      const float lum_smoothed_broad = fmaxf(luminance_smoothed_broad[k], MIN_FLOAT);
      const float detail_ev_broad = log2f(lum_pixel / lum_smoothed_broad);
      const float scaled_detail_ev_broad = d->pyr_broad_scale * detail_ev_broad;
      correction_ev += scaled_detail_ev_broad - detail_ev_broad;
    }

    if(luminance_smoothed_medium)
    {
      const float lum_smoothed_medium = fmaxf(luminance_smoothed_medium[k], MIN_FLOAT);
      const float detail_ev_medium = log2f(lum_pixel / lum_smoothed_medium);
      const float scaled_detail_ev_medium = d->pyr_medium_scale * detail_ev_medium;
      correction_ev += scaled_detail_ev_medium - detail_ev_medium;
    }

    if(luminance_smoothed_fine)
    {
      const float lum_smoothed_fine = fmaxf(luminance_smoothed_fine[k], MIN_FLOAT);
      const float detail_ev_fine = log2f(lum_pixel / lum_smoothed_fine);
      const float scaled_detail_ev_fine = d->pyr_fine_scale * detail_ev_fine;
      correction_ev += scaled_detail_ev_fine - detail_ev_fine;
    }

    if(luminance_smoothed_micro)
    {
      const float lum_smoothed_micro = fmaxf(luminance_smoothed_micro[k], MIN_FLOAT);
      const float detail_ev_micro = log2f(lum_pixel / lum_smoothed_micro);
      const float scaled_detail_ev_micro = d->pyr_micro_scale * detail_ev_micro;
      correction_ev += scaled_detail_ev_micro - detail_ev_micro;
    }

    // Apply correction in linear space
    // global_scale has the same range as detail_scale.
    const float multiplier = exp2f(correction_ev) * powf(lum_smoothed / 0.1845f, d->pyr_global_scale) * 0.1845f / lum_smoothed;

    for_each_channel(c)
      out[4 * k + c] = in[4 * k + c] * multiplier;
  }
}


/**
 * Display the detail mask (difference between pixel and smoothed luminance)
 * Output is a grayscale image normalized to [0, 1] where:
 * - 0.5 = no local detail (pixel matches neighborhood)
 * - < 0.5 = pixel darker than neighborhood
 * - > 0.5 = pixel brighter than neighborhood
 **/
__DT_CLONE_TARGETS__
static inline void pyr_display_detail_mask(const float *const restrict luminance_pixel,
                                       const float *const restrict luminance_smoothed,
                                       float *const restrict out,
                                       const size_t width,
                                       const size_t height)
{
  const size_t npixels = width * height;

  DT_OMP_FOR()
  for(size_t k = 0; k < npixels; k++)
  {
    const float lum_pixel = fmaxf(luminance_pixel[k], MIN_FLOAT);
    const float lum_smoothed = fmaxf(luminance_smoothed[k], MIN_FLOAT);

    // Detail in log space, mapped to [0, 1] for display
    // Detail range roughly [-2, +2] EV mapped to [0, 1]
    const float detail_ev = log2f(lum_pixel / lum_smoothed);
    const float intensity = fminf(fmaxf(detail_ev / 4.0f + 0.5f, 0.0f), 1.0f);

    // Set all RGB channels to the same intensity (grayscale)
    out[4 * k + 0] = intensity;
    out[4 * k + 1] = intensity;
    out[4 * k + 2] = intensity;
    // Full opacity
    out[4 * k + 3] = 1.0f;
  }
}


/**
 * Check if any scale is active (detail_boost != 1.0)
 **/
static inline gboolean exp_has_active_scales(const dt_iop_pyramidal_contrast_data_t *const d)
{
  for(int s = 0; s < N_SCALES; s++)
  {
    if(d->exp_scales[s].exp_detail_boost != 1.0f) return TRUE;
  }
  return FALSE;
}


/**
 * Check if a specific scale is active
 **/
static inline gboolean exp_scale_is_active(const float detail_boost)
{
  return detail_boost != 1.0f;
}


/**
 * Compute pixel-wise luminance mask (no blur)
 **/
__DT_CLONE_TARGETS__
static inline void exp_compute_pixel_luminance_mask(const float *const restrict in,
                                                float *const restrict luminance,
                                                const size_t width,
                                                const size_t height,
                                                const dt_iop_luminance_mask_method_t method)
{
  // No exposure/contrast boost, just compute raw luminance
  luminance_mask(in, luminance, width, height, method, 1.0f, 0.0f, 1.0f);
}


/**
 * Compute smoothed luminance mask for a single scale
 **/
__DT_CLONE_TARGETS__
static inline void exp_compute_smoothed_luminance_for_scale(
    const float *const restrict in,
    float *const restrict luminance,
    const size_t width,
    const size_t height,
    const dt_iop_pyramidal_contrast_scale_data_t *const scale_data,
    const dt_iop_pyramidal_contrast_expert_filter_t details,
    const int iterations,
    const dt_iop_luminance_mask_method_t method)
{
  // First compute pixel-wise luminance (no boost)
  luminance_mask(in, luminance, width, height, method, 1.0f, 0.0f, 1.0f);

  // Then apply the smoothing filter
  switch(details)
  {
    case(DT_EXP_AVG_GUIDED):
    {
      fast_surface_blur(luminance, width, height,
                        scale_data->exp_radius, scale_data->exp_feathering, iterations,
                        DT_GF_BLENDING_GEOMEAN, 1.0f, 0.0f,
                        exp2f(-14.0f), 4.0f);
      break;
    }

    case(DT_EXP_GUIDED):
    {
      fast_surface_blur(luminance, width, height,
                        scale_data->exp_radius, scale_data->exp_feathering, iterations,
                        DT_GF_BLENDING_LINEAR, 1.0f, 0.0f,
                        exp2f(-14.0f), 4.0f);
      break;
    }

    case(DT_EXP_AVG_EIGF):
    {
      fast_eigf_surface_blur(luminance, width, height,
                             scale_data->exp_radius, scale_data->exp_feathering, iterations,
                             DT_GF_BLENDING_GEOMEAN, 1.0f,
                             0.0f, exp2f(-14.0f), 4.0f);
      break;
    }

    case(DT_EXP_EIGF):
    {
      fast_eigf_surface_blur(luminance, width, height,
                             scale_data->exp_radius, scale_data->exp_feathering, iterations,
                             DT_GF_BLENDING_LINEAR, 1.0f,
                             0.0f, exp2f(-14.0f), 4.0f);
      break;
    }
  }
}


/**
 * Apply multi-scale local contrast enhancement
 *
 * For each pixel:
 * 1. Sum correction_ev contributions from all active scales
 * 2. Apply single exp2f(total_correction) multiplier
 *
 * This is more efficient than applying each scale sequentially and
 * allows the effects to combine additively in log space.
 **/
__DT_CLONE_TARGETS__
static inline void exp_apply_multiscale_local_contrast(
    const float *const restrict in,
    const float *const restrict luminance_pixel,
    float *const restrict *const restrict luminance_smoothed,
    float *const restrict out,
    const size_t width,
    const size_t height,
    const dt_iop_pyramidal_contrast_data_t *const d)
{
  const size_t npixels = width * height;

  // Unpack scale data for vectorization
  float detail_boosts[N_SCALES] DT_ALIGNED_PIXEL;
  gboolean active[N_SCALES];
  int n_active = 0;

  for(int s = 0; s < N_SCALES; s++)
  {
    detail_boosts[s] = d->exp_scales[s].exp_detail_boost;
    active[s] = exp_scale_is_active(detail_boosts[s]);
    if(active[s]) n_active++;
  }

  // Early exit if no scales are active
  if(n_active == 0)
  {
    dt_iop_image_copy_by_size(out, in, width, height, 4);
    return;
  }

  DT_OMP_FOR()
  for(size_t k = 0; k < npixels; k++)
  {
    const float lum_pixel = fmaxf(luminance_pixel[k], MIN_FLOAT);
    float total_correction_ev = 0.0f;

    // Sum correction contributions from all active scales
    for(int s = 0; s < N_SCALES; s++)
    {
      if(!active[s]) continue;

      const float lum_smoothed = fmaxf(luminance_smoothed[s][k], MIN_FLOAT);

      // Detail in log space (EV): how much brighter/darker is this pixel
      // compared to its local neighborhood at this scale
      const float detail_ev = log2f(lum_pixel / lum_smoothed);

      // Scale the detail: detail_boost = 1.0 means no change
      // > 1.0 boosts local contrast, < 1.0 reduces it
      const float scaled_detail_ev = detail_boosts[s] * detail_ev;

      // The correction is the difference between scaled and original detail
      total_correction_ev += scaled_detail_ev - detail_ev;
    }

    // Apply combined correction in linear space
    const float multiplier = exp2f(total_correction_ev);

    for_each_channel(c)
      out[4 * k + c] = in[4 * k + c] * multiplier;
  }
}


/**
 * Display the detail mask for a specific scale
 * Output is a grayscale image normalized to [0, 1] where:
 * - 0.5 = no local detail (pixel matches neighborhood)
 * - < 0.5 = pixel darker than neighborhood
 * - > 0.5 = pixel brighter than neighborhood
 **/
__DT_CLONE_TARGETS__
static inline void exp_display_detail_mask_for_scale(
    const float *const restrict luminance_pixel,
    const float *const restrict luminance_smoothed,
    float *const restrict out,
    const size_t width,
    const size_t height)
{
  const size_t npixels = width * height;

  DT_OMP_FOR()
  for(size_t k = 0; k < npixels; k++)
  {
    const float lum_pixel = fmaxf(luminance_pixel[k], MIN_FLOAT);
    const float lum_smoothed = fmaxf(luminance_smoothed[k], MIN_FLOAT);

    // Detail in log space, mapped to [0, 1] for display
    // Detail range roughly [-2, +2] EV mapped to [0, 1]
    const float detail_ev = log2f(lum_pixel / lum_smoothed);
    const float intensity = fminf(fmaxf(detail_ev / 4.0f + 0.5f, 0.0f), 1.0f);

    // Set all RGB channels to the same intensity (grayscale)
    out[4 * k + 0] = intensity;
    out[4 * k + 1] = intensity;
    out[4 * k + 2] = intensity;
    out[4 * k + 3] = 1.0f;
  }
}


/**
 * Allocate or reallocate smoothed buffers for all scales
 **/
static inline void exp_alloc_smoothed_buffers(float **buffers,
                                          const size_t num_elem)
{
  for(int s = 0; s < N_SCALES; s++)
  {
    dt_free_align(buffers[s]);
    buffers[s] = dt_alloc_align_float(num_elem);
  }
}


/**
 * Free smoothed buffers for all scales
 **/
static inline void exp_free_smoothed_buffers(float **buffers)
{
  for(int s = 0; s < N_SCALES; s++)
  {
    dt_free_align(buffers[s]);
    buffers[s] = NULL;
  }
}


/**
 * Check if all smoothed buffers are allocated
 **/
static inline gboolean exp_smoothed_buffers_valid(float **buffers)
{
  for(int s = 0; s < N_SCALES; s++)
  {
    if(!buffers[s]) return FALSE;
  }
  return TRUE;
}


/**
 * Main processing function
 **/
__DT_CLONE_TARGETS__
static void exp_process(dt_iop_module_t *self,
                                   dt_dev_pixelpipe_iop_t *piece,
                                   const void *const restrict ivoid,
                                   void *const restrict ovoid,
                                   const dt_iop_roi_t *const roi_in,
                                   const dt_iop_roi_t *const roi_out)
{
  const dt_iop_pyramidal_contrast_data_t *const d = piece->data;
  dt_iop_pyramidal_contrast_gui_data_t *const g = self->gui_data;

  const float *const restrict in = (float *const)ivoid;
  float *const restrict out = (float *const)ovoid;
  float *restrict luminance_pixel = NULL;
  float *luminance_smoothed[N_SCALES] = { NULL };

  const size_t width = roi_in->width;
  const size_t height = roi_in->height;
  const size_t num_elem = width * height;

  // Get the hash of the upstream pipe to track changes
  const dt_hash_t hash = dt_dev_pixelpipe_piece_hash(piece, roi_out, TRUE);

  // Sanity checks
  if(width < 1 || height < 1) return;
  if(roi_in->width < roi_out->width || roi_in->height < roi_out->height) return;
  if(piece->colors != 4) return;

  // Fast path: if no scales are active, just copy
  if(!exp_has_active_scales(d) && (!g || g->exp_mask_display_scale == -1))
  {
    dt_iop_image_copy_by_size(out, in, width, height, 4);
    return;
  }

  // Init the luminance mask buffers
  gboolean cached = FALSE;

  if(self->dev->gui_attached)
  {
    // If the module instance has changed order in the pipe, invalidate caches
    if(g->exp_pipe_order != piece->module->iop_order)
    {
      dt_iop_gui_enter_critical_section(self);
      g->exp_ui_preview_hash = DT_INVALID_HASH;
      g->exp_thumb_preview_hash = DT_INVALID_HASH;
      g->exp_pipe_order = piece->module->iop_order;
      g->exp_luminance_valid = FALSE;
      dt_iop_gui_leave_critical_section(self);
    }

    if(piece->pipe->type & DT_DEV_PIXELPIPE_FULL)
    {
      // Re-allocate buffers if size changed
      if(g->exp_full_preview_buf_width != width || g->exp_full_preview_buf_height != height)
      {
        dt_free_align(g->exp_full_preview_buf_pixel);
        g->exp_full_preview_buf_pixel = dt_alloc_align_float(num_elem);
        exp_alloc_smoothed_buffers(g->exp_full_preview_buf_smoothed, num_elem);
        g->exp_full_preview_buf_width = width;
        g->exp_full_preview_buf_height = height;
      }

      luminance_pixel = g->exp_full_preview_buf_pixel;
      for(int s = 0; s < N_SCALES; s++)
        luminance_smoothed[s] = g->exp_full_preview_buf_smoothed[s];
      cached = TRUE;
    }
    else if(piece->pipe->type & DT_DEV_PIXELPIPE_PREVIEW)
    {
      dt_iop_gui_enter_critical_section(self);
      if(g->exp_thumb_preview_buf_width != width || g->exp_thumb_preview_buf_height != height)
      {
        dt_free_align(g->exp_thumb_preview_buf_pixel);
        g->exp_thumb_preview_buf_pixel = dt_alloc_align_float(num_elem);
        exp_alloc_smoothed_buffers(g->exp_thumb_preview_buf_smoothed, num_elem);
        g->exp_thumb_preview_buf_width = width;
        g->exp_thumb_preview_buf_height = height;
        g->exp_luminance_valid = FALSE;
      }

      luminance_pixel = g->exp_thumb_preview_buf_pixel;
      for(int s = 0; s < N_SCALES; s++)
        luminance_smoothed[s] = g->exp_thumb_preview_buf_smoothed[s];
      cached = TRUE;
      dt_iop_gui_leave_critical_section(self);
    }
    else
    {
      luminance_pixel = dt_alloc_align_float(num_elem);
      exp_alloc_smoothed_buffers(luminance_smoothed, num_elem);
    }
  }
  else
  {
    // No interactive editing: allocate local temp buffers
    luminance_pixel = dt_alloc_align_float(num_elem);
    exp_alloc_smoothed_buffers(luminance_smoothed, num_elem);
  }

  // Check buffer allocation
  if(!luminance_pixel || !exp_smoothed_buffers_valid(luminance_smoothed))
  {
    dt_control_log(_("local contrast failed to allocate memory, check your RAM settings"));
    if(!cached)
    {
      dt_free_align(luminance_pixel);
      exp_free_smoothed_buffers(luminance_smoothed);
    }
    return;
  }

  // Compute luminance masks
  if(cached)
  {
    if(piece->pipe->type & DT_DEV_PIXELPIPE_FULL)
    {
      dt_hash_t saved_hash;
      pyr_hash_set_get(&g->exp_ui_preview_hash, &saved_hash, &self->gui_lock);

      dt_iop_gui_enter_critical_section(self);
      const gboolean luminance_valid = g->exp_luminance_valid;
      dt_iop_gui_leave_critical_section(self);

      if(hash != saved_hash || !luminance_valid)
      {
        // Compute pixel luminance once
        exp_compute_pixel_luminance_mask(in, luminance_pixel, width, height, d->exp_method);

        // Compute smoothed luminance for each active scale
        for(int s = 0; s < N_SCALES; s++)
        {
          if(exp_scale_is_active(d->exp_scales[s].exp_detail_boost) || (g && g->exp_mask_display_scale == s))
          {
            exp_compute_smoothed_luminance_for_scale(in, luminance_smoothed[s],
                                                 width, height,
                                                 &d->exp_scales[s],
                                                 d->exp_details, d->exp_iterations,
                                                 d->exp_method);
          }
        }
        pyr_hash_set_get(&hash, &g->exp_ui_preview_hash, &self->gui_lock);
      }
    }
    else if(piece->pipe->type & DT_DEV_PIXELPIPE_PREVIEW)
    {
      dt_hash_t saved_hash;
      pyr_hash_set_get(&g->exp_thumb_preview_hash, &saved_hash, &self->gui_lock);

      dt_iop_gui_enter_critical_section(self);
      const gboolean luminance_valid = g->exp_luminance_valid;
      dt_iop_gui_leave_critical_section(self);

      if(saved_hash != hash || !luminance_valid)
      {
        dt_iop_gui_enter_critical_section(self);
        g->exp_thumb_preview_hash = hash;

        // Compute pixel luminance once
        exp_compute_pixel_luminance_mask(in, luminance_pixel, width, height, d->exp_method);

        // Compute smoothed luminance for each active scale
        for(int s = 0; s < N_SCALES; s++)
        {
          if(exp_scale_is_active(d->exp_scales[s].exp_detail_boost) || (g && g->exp_mask_display_scale == s))
          {
            exp_compute_smoothed_luminance_for_scale(in, luminance_smoothed[s],
                                                 width, height,
                                                 &d->exp_scales[s],
                                                 d->exp_details, d->exp_iterations,
                                                 d->exp_method);
          }
        }

        g->exp_luminance_valid = TRUE;
        dt_iop_gui_leave_critical_section(self);
        dt_dev_pixelpipe_cache_invalidate_later(piece->pipe, self->iop_order);
      }
    }
    else
    {
      // Non-cached pipe: compute everything
      exp_compute_pixel_luminance_mask(in, luminance_pixel, width, height, d->exp_method);
      for(int s = 0; s < N_SCALES; s++)
      {
        if(exp_scale_is_active(d->exp_scales[s].exp_detail_boost))
        {
          exp_compute_smoothed_luminance_for_scale(in, luminance_smoothed[s],
                                               width, height,
                                               &d->exp_scales[s],
                                               d->exp_details, d->exp_iterations,
                                               d->exp_method);
        }
      }
    }
  }
  else
  {
    // Non-GUI: compute everything
    exp_compute_pixel_luminance_mask(in, luminance_pixel, width, height, d->exp_method);
    for(int s = 0; s < N_SCALES; s++)
    {
      if(exp_scale_is_active(d->exp_scales[s].exp_detail_boost) || (g && g->exp_mask_display_scale == s))
      {
        exp_compute_smoothed_luminance_for_scale(in, luminance_smoothed[s],
                                             width, height,
                                             &d->exp_scales[s],
                                             d->exp_details, d->exp_iterations,
                                             d->exp_method);
      }
    }
  }

  // Display output
  if(self->dev->gui_attached && (piece->pipe->type & DT_DEV_PIXELPIPE_FULL))
  {
    const int display_scale = g->exp_mask_display_scale;
    if(display_scale >= 0 && display_scale < N_SCALES
 // && exp_scale_is_active(d->exp_scales[display_scale].exp_detail_boost)
       && luminance_smoothed[display_scale])
    {
      exp_display_detail_mask_for_scale(luminance_pixel, luminance_smoothed[display_scale],
                                    out, width, height);
      piece->pipe->mask_display = DT_DEV_PIXELPIPE_DISPLAY_PASSTHRU;
    }
    else
    {
      exp_apply_multiscale_local_contrast(in, luminance_pixel, luminance_smoothed,
                                      out, width, height, d);
    }
  }
  else
  {
    exp_apply_multiscale_local_contrast(in, luminance_pixel, luminance_smoothed,
                                    out, width, height, d);
  }

  if(!cached)
  {
    dt_free_align(luminance_pixel);
    exp_free_smoothed_buffers(luminance_smoothed);
  }
}


/**
 * Main processing function
 **/
__DT_CLONE_TARGETS__
static void pyr_process(dt_iop_module_t *self,
                                   dt_dev_pixelpipe_iop_t *piece,
                                   const void *const restrict ivoid,
                                   void *const restrict ovoid,
                                   const dt_iop_roi_t *const roi_in,
                                   const dt_iop_roi_t *const roi_out)
{
  const dt_iop_pyramidal_contrast_data_t *const d = piece->data;
  dt_iop_pyramidal_contrast_gui_data_t *const g = self->gui_data;

  const float *const restrict in = (float *const)ivoid;
  float *const restrict out = (float *const)ovoid;
  float *restrict luminance_pixel = NULL;
  float *restrict luminance_smoothed_broad = NULL;
  float *restrict luminance_smoothed_medium = NULL;
  float *restrict luminance_smoothed = NULL;
  float *restrict luminance_smoothed_fine = NULL;
  float *restrict luminance_smoothed_micro = NULL;

  const size_t width = roi_in->width;
  const size_t height = roi_in->height;
  const size_t num_elem = width * height;

  // Get the hash of the upstream pipe to track changes
  const dt_hash_t hash = dt_dev_pixelpipe_piece_hash(piece, roi_out, TRUE);

  // Sanity checks
  if(width < 1 || height < 1) return;
  if(roi_in->width < roi_out->width || roi_in->height < roi_out->height) return;
  if(piece->colors != 4) return;

  // Init the luminance mask buffers
  gboolean cached = FALSE;

  if(self->dev->gui_attached)
  {
    // If the module instance has changed order in the pipe, invalidate caches
    if(g->pyr_pipe_order != piece->module->iop_order)
    {
      dt_iop_gui_enter_critical_section(self);
      g->pyr_ui_preview_hash = DT_INVALID_HASH;
      g->pyr_thumb_preview_hash = DT_INVALID_HASH;
      g->pyr_pipe_order = piece->module->iop_order;
      g->luminance_valid = FALSE;
      dt_iop_gui_leave_critical_section(self);
    }

    if(piece->pipe->type & DT_DEV_PIXELPIPE_FULL)
    {
      // Re-allocate buffers if size changed
      if(g->pyr_full_preview_buf_width != width || g->pyr_full_preview_buf_height != height)
      {
        dt_free_align(g->pyr_full_preview_buf_pixel);
        dt_free_align(g->pyr_full_preview_buf_smoothed_broad);
        dt_free_align(g->pyr_full_preview_buf_smoothed_medium);
        dt_free_align(g->pyr_full_preview_buf_smoothed);
        dt_free_align(g->pyr_full_preview_buf_smoothed_fine);
        dt_free_align(g->pyr_full_preview_buf_smoothed_micro);
        g->pyr_full_preview_buf_pixel = dt_alloc_align_float(num_elem);
        g->pyr_full_preview_buf_smoothed_broad = dt_alloc_align_float(num_elem);
        g->pyr_full_preview_buf_smoothed_medium = dt_alloc_align_float(num_elem);
        g->pyr_full_preview_buf_smoothed = dt_alloc_align_float(num_elem);
        g->pyr_full_preview_buf_smoothed_fine = dt_alloc_align_float(num_elem);
        g->pyr_full_preview_buf_smoothed_micro = dt_alloc_align_float(num_elem);
        g->pyr_full_preview_buf_width = width;
        g->pyr_full_preview_buf_height = height;
      }

      luminance_pixel = g->pyr_full_preview_buf_pixel;
      luminance_smoothed_broad = g->pyr_full_preview_buf_smoothed_broad;
      luminance_smoothed_medium = g->pyr_full_preview_buf_smoothed_medium;
      luminance_smoothed = g->pyr_full_preview_buf_smoothed;
      luminance_smoothed_fine = g->pyr_full_preview_buf_smoothed_fine;
      luminance_smoothed_micro = g->pyr_full_preview_buf_smoothed_micro;
      cached = TRUE;
    }
    else if(piece->pipe->type & DT_DEV_PIXELPIPE_PREVIEW)
    {
      dt_iop_gui_enter_critical_section(self);
      if(g->pyr_thumb_preview_buf_width != width || g->pyr_thumb_preview_buf_height != height)
      {
        dt_free_align(g->pyr_thumb_preview_buf_pixel);
        dt_free_align(g->pyr_thumb_preview_buf_smoothed_broad);
        dt_free_align(g->pyr_thumb_preview_buf_smoothed_medium);
        dt_free_align(g->pyr_thumb_preview_buf_smoothed);
        dt_free_align(g->pyr_thumb_preview_buf_smoothed_fine);
        dt_free_align(g->pyr_thumb_preview_buf_smoothed_micro);
        g->pyr_thumb_preview_buf_pixel = dt_alloc_align_float(num_elem);
        g->pyr_thumb_preview_buf_smoothed_broad = dt_alloc_align_float(num_elem);
        g->pyr_thumb_preview_buf_smoothed_medium = dt_alloc_align_float(num_elem);
        g->pyr_thumb_preview_buf_smoothed = dt_alloc_align_float(num_elem);
        g->pyr_thumb_preview_buf_smoothed_fine = dt_alloc_align_float(num_elem);
        g->pyr_thumb_preview_buf_smoothed_micro = dt_alloc_align_float(num_elem);
        g->pyr_thumb_preview_buf_width = width;
        g->pyr_thumb_preview_buf_height = height;
        g->luminance_valid = FALSE;
      }

      luminance_pixel = g->pyr_thumb_preview_buf_pixel;
      luminance_smoothed_broad = g->pyr_thumb_preview_buf_smoothed_broad;
      luminance_smoothed_medium = g->pyr_thumb_preview_buf_smoothed_medium;
      luminance_smoothed = g->pyr_thumb_preview_buf_smoothed;
      luminance_smoothed_fine = g->pyr_thumb_preview_buf_smoothed_fine;
      luminance_smoothed_micro = g->pyr_thumb_preview_buf_smoothed_micro;
      cached = TRUE;
      dt_iop_gui_leave_critical_section(self);
    }
    else
    {
      luminance_pixel = dt_alloc_align_float(num_elem);
      luminance_smoothed = dt_alloc_align_float(num_elem);
      luminance_smoothed_broad = dt_alloc_align_float(num_elem);
      luminance_smoothed_medium = dt_alloc_align_float(num_elem);
      luminance_smoothed_fine = dt_alloc_align_float(num_elem);
      luminance_smoothed_micro = dt_alloc_align_float(num_elem);
    }
  }
  else
  {
    // No interactive editing: allocate local temp buffers
    luminance_pixel = dt_alloc_align_float(num_elem);
    luminance_smoothed_broad = dt_alloc_align_float(num_elem);
    luminance_smoothed_medium = dt_alloc_align_float(num_elem);
    luminance_smoothed = dt_alloc_align_float(num_elem);
    luminance_smoothed_fine = dt_alloc_align_float(num_elem);
    luminance_smoothed_micro = dt_alloc_align_float(num_elem);
  }

  // Check buffer allocation
  if(!luminance_pixel || !luminance_smoothed_broad || !luminance_smoothed_medium || !luminance_smoothed || !luminance_smoothed_fine || !luminance_smoothed_micro)
  {
    dt_control_log(_("local contrast failed to allocate memory, check your RAM settings"));
    if(!cached)
    {
      dt_free_align(luminance_pixel);
      dt_free_align(luminance_smoothed_broad);
      dt_free_align(luminance_smoothed_medium);
      dt_free_align(luminance_smoothed);
      dt_free_align(luminance_smoothed_fine);
      dt_free_align(luminance_smoothed_micro);
    }
    return;
  }

  // Compute luminance masks
  if(cached)
  {
    if(piece->pipe->type & DT_DEV_PIXELPIPE_FULL)
    {
      dt_hash_t saved_hash;
      pyr_hash_set_get(&g->pyr_ui_preview_hash, &saved_hash, &self->gui_lock);

      dt_iop_gui_enter_critical_section(self);
      const gboolean luminance_valid = g->luminance_valid;
      dt_iop_gui_leave_critical_section(self);

      if(hash != saved_hash || !luminance_valid)
      {
        pyr_compute_pixel_luminance_mask(in, luminance_pixel, width, height, d->pyr_method);
        if(d->pyr_broad_scale != 1.0f || g->pyr_mask_display == DT_PYR_MASK_BROAD)
          pyr_compute_smoothed_luminance_mask(in, luminance_smoothed_broad, width, height, d, d->pyr_radius_broad, d->pyr_feathering * d->pyr_f_mult_broad);
        if(d->pyr_medium_scale != 1.0f || g->pyr_mask_display == DT_PYR_MASK_MEDIUM)
          pyr_compute_smoothed_luminance_mask(in, luminance_smoothed_medium, width, height, d, d->pyr_radius_medium, d->pyr_feathering * d->pyr_f_mult_medium);
        if(d->pyr_detail_scale != 1.0f || g->pyr_mask_display == DT_PYR_MASK_DETAIL)
          pyr_compute_smoothed_luminance_mask(in, luminance_smoothed, width, height, d, d->pyr_radius, d->pyr_feathering * d->pyr_f_mult_detail);
        if(d->pyr_fine_scale != 1.0f || g->pyr_mask_display == DT_PYR_MASK_FINE)
          pyr_compute_smoothed_luminance_mask(in, luminance_smoothed_fine, width, height, d, d->pyr_radius_fine, d->pyr_feathering * d->pyr_f_mult_fine);
        if(d->pyr_micro_scale != 1.0f || g->pyr_mask_display == DT_PYR_MASK_MICRO)
          pyr_compute_smoothed_luminance_mask(in, luminance_smoothed_micro, width, height, d, d->pyr_radius_micro, d->pyr_feathering * d->pyr_f_mult_micro);
        pyr_hash_set_get(&hash, &g->pyr_ui_preview_hash, &self->gui_lock);
      }
    }
    else if(piece->pipe->type & DT_DEV_PIXELPIPE_PREVIEW)
    {
      dt_hash_t saved_hash;
      pyr_hash_set_get(&g->pyr_thumb_preview_hash, &saved_hash, &self->gui_lock);

      dt_iop_gui_enter_critical_section(self);
      const gboolean luminance_valid = g->luminance_valid;
      dt_iop_gui_leave_critical_section(self);

      if(saved_hash != hash || !luminance_valid)
      {
        dt_iop_gui_enter_critical_section(self);
        g->pyr_thumb_preview_hash = hash;
        pyr_compute_pixel_luminance_mask(in, luminance_pixel, width, height, d->pyr_method);
        if(d->pyr_broad_scale != 1.0f || g->pyr_mask_display == DT_PYR_MASK_BROAD)
          pyr_compute_smoothed_luminance_mask(in, luminance_smoothed_broad, width, height, d, d->pyr_radius_broad, d->pyr_feathering * d->pyr_f_mult_broad);
        if(d->pyr_medium_scale != 1.0f || g->pyr_mask_display == DT_PYR_MASK_MEDIUM)
          pyr_compute_smoothed_luminance_mask(in, luminance_smoothed_medium, width, height, d, d->pyr_radius_medium, d->pyr_feathering * d->pyr_f_mult_medium);
        if(d->pyr_detail_scale != 1.0f || g->pyr_mask_display == DT_PYR_MASK_DETAIL)
          pyr_compute_smoothed_luminance_mask(in, luminance_smoothed, width, height, d, d->pyr_radius, d->pyr_feathering * d->pyr_f_mult_detail);
        if(d->pyr_fine_scale != 1.0f || g->pyr_mask_display == DT_PYR_MASK_FINE)
          pyr_compute_smoothed_luminance_mask(in, luminance_smoothed_fine, width, height, d, d->pyr_radius_fine, d->pyr_feathering * d->pyr_f_mult_fine);
        if(d->pyr_micro_scale != 1.0f || g->pyr_mask_display == DT_PYR_MASK_MICRO)
          pyr_compute_smoothed_luminance_mask(in, luminance_smoothed_micro, width, height, d, d->pyr_radius_micro, d->pyr_feathering * d->pyr_f_mult_micro);
        g->luminance_valid = TRUE;
        dt_iop_gui_leave_critical_section(self);
        dt_dev_pixelpipe_cache_invalidate_later(piece->pipe, self->iop_order);
      }
    }
    else
    {
      pyr_compute_pixel_luminance_mask(in, luminance_pixel, width, height, d->pyr_method);
      pyr_compute_smoothed_luminance_mask(in, luminance_smoothed_fine, width, height, d, d->pyr_radius / 2, d->pyr_feathering * 0.75f);
      pyr_compute_smoothed_luminance_mask(in, luminance_smoothed_micro, width, height, d, d->pyr_radius / 4, d->pyr_feathering * 0.5f);
    }
  }
  else
  {
    pyr_compute_pixel_luminance_mask(in, luminance_pixel, width, height, d->pyr_method);
    pyr_compute_smoothed_luminance_mask(in, luminance_smoothed_broad, width, height, d, d->pyr_radius_broad, d->pyr_feathering * 1.5f);
    pyr_compute_smoothed_luminance_mask(in, luminance_smoothed_medium, width, height, d, d->pyr_radius_medium, d->pyr_feathering * 1.25f);
    pyr_compute_smoothed_luminance_mask(in, luminance_smoothed, width, height, d, d->pyr_radius, d->pyr_feathering);
    pyr_compute_smoothed_luminance_mask(in, luminance_smoothed_fine, width, height, d, d->pyr_radius_fine, d->pyr_feathering * 0.75f);
    pyr_compute_smoothed_luminance_mask(in, luminance_smoothed_micro, width, height, d, d->pyr_radius_micro, d->pyr_feathering * 0.5f);
  }

  // Display output
  if(g && g->pyr_mask_display != DT_PYR_MASK_OFF)
  {
    float *lum_smooth = luminance_smoothed;
    if(g->pyr_mask_display == DT_PYR_MASK_BROAD) lum_smooth = luminance_smoothed_broad;
    else if(g->pyr_mask_display == DT_PYR_MASK_MEDIUM) lum_smooth = luminance_smoothed_medium;
    if(g->pyr_mask_display == DT_PYR_MASK_FINE) lum_smooth = luminance_smoothed_fine;
    else if(g->pyr_mask_display == DT_PYR_MASK_MICRO) lum_smooth = luminance_smoothed_micro;

    pyr_display_detail_mask(luminance_pixel, lum_smooth, out, width, height);
    piece->pipe->mask_display = DT_DEV_PIXELPIPE_DISPLAY_PASSTHRU;
  }
  else
  {
    pyr_apply_local_contrast(in, luminance_pixel, luminance_smoothed, 
                         d->pyr_broad_scale != 1.0f ? luminance_smoothed_broad : NULL,
                         d->pyr_medium_scale != 1.0f ? luminance_smoothed_medium : NULL,
                         d->pyr_fine_scale != 1.0f ? luminance_smoothed_fine : NULL,
                         d->pyr_micro_scale != 1.0f ? luminance_smoothed_micro : NULL,
                         out, roi_in, roi_out, d);
  }

  if(!cached)
  {
    dt_free_align(luminance_pixel);
    dt_free_align(luminance_smoothed_broad);
    dt_free_align(luminance_smoothed_medium);
    dt_free_align(luminance_smoothed);
    dt_free_align(luminance_smoothed_fine);
    dt_free_align(luminance_smoothed_micro);
  }
}


void process(dt_iop_module_t *self,
             dt_dev_pixelpipe_iop_t *piece,
             const void *const restrict ivoid,
             void *const restrict ovoid,
             const dt_iop_roi_t *const roi_in,
             const dt_iop_roi_t *const roi_out)
{
  const dt_iop_pyramidal_contrast_data_t *const d = piece->data;
  if(d->mode == DT_PYR_MODE_GLOBAL)
  {
    pyr_process(self, piece, ivoid, ovoid, roi_in, roi_out);
  }
  else
  {
    exp_process(self, piece, ivoid, ovoid, roi_in, roi_out);
  }
}


void modify_roi_in(dt_iop_module_t *self,
                   dt_dev_pixelpipe_iop_t *piece,
                   const dt_iop_roi_t *roi_out,
                   dt_iop_roi_t *roi_in)
{
  dt_iop_pyramidal_contrast_data_t *const d = piece->data;

  // Get the scaled window radius for the box average
  const float max_size = (float)((piece->iwidth > piece->iheight) ? piece->iwidth : piece->iheight);

  for(int s = 0; s < N_SCALES; s++)
  {
    const float diameter = d->exp_scales[s].exp_feature_scale * max_size * roi_in->scale;
    const int radius = (int)((diameter - 1.0f) / 2.0f);
    d->exp_scales[s].exp_radius = radius;
  }
  const float base_diameter = d->pyr_blending * max_size * roi_in->scale;

  const float diameter_broad = base_diameter * d->pyr_s_mult_broad;
  d->pyr_radius_broad = (int)((diameter_broad - 1.0f) / 2.0f);

  const float diameter_medium = base_diameter * d->pyr_s_mult_medium;
  d->pyr_radius_medium = (int)((diameter_medium - 1.0f) / 2.0f);

  const float diameter_detail = base_diameter * d->pyr_s_mult_detail;
  d->pyr_radius = (int)((diameter_detail - 1.0f) / 2.0f);

  const float diameter_fine = base_diameter * d->pyr_s_mult_fine;
  d->pyr_radius_fine = (int)((diameter_fine - 1.0f) / 2.0f);

  const float diameter_micro = base_diameter * d->pyr_s_mult_micro;
  d->pyr_radius_micro = (int)((diameter_micro - 1.0f) / 2.0f);
}

void init(dt_iop_module_t *self)
{
  dt_iop_default_init(self);

  dt_iop_pyramidal_contrast_params_t *d = self->default_params;

  d->mode = DT_PYR_MODE_GLOBAL;

  d->pyr_micro_scale = 1.0f;
  d->pyr_fine_scale = 1.0f;
  d->pyr_detail_scale = 1.5f;
  d->pyr_medium_scale = 1.0f;
  d->pyr_broad_scale = 1.0f;
  d->pyr_global_scale = 1.0f;

  d->pyr_blending = 1.2f;
  d->pyr_feathering = 5.0f;

  d->pyr_f_mult_micro = 0.5f;
  d->pyr_f_mult_fine = 0.75f;
  d->pyr_f_mult_detail = 1.0f;
  d->pyr_f_mult_medium = 1.25f;
  d->pyr_f_mult_broad = 1.5f;

  d->pyr_s_mult_micro = 0.25f;
  d->pyr_s_mult_fine = 0.625f;
  d->pyr_s_mult_detail = 1.0f;
  d->pyr_s_mult_medium = 1.8f;
  d->pyr_s_mult_broad = 2.8f;

  d->pyr_details = DT_PYR_EIGF;
  d->pyr_method = DT_TONEEQ_NORM_2;
  d->pyr_iterations = 1;

  for(int i = 0; i < N_SCALES; i++)
  {
    d->exp_detail_boost[i] = 100.0f;
    d->exp_feature_scale[i] = 12.0f;
    d->exp_feathering[i] = 5.0f;
  }

  d->exp_feature_scale[0] = 4.0f;
  d->exp_feature_scale[1] = 12.0f;
  d->exp_feature_scale[2] = 25.0f;

  d->exp_details = DT_EXP_EIGF;
  d->exp_method = DT_TONEEQ_NORM_2;
  d->exp_iterations = 1;
}


void init_global(dt_iop_module_so_t *self)
{
  dt_iop_pyramidal_contrast_global_data_t *gd = malloc(sizeof(dt_iop_pyramidal_contrast_global_data_t));
  self->data = gd;
}


void cleanup_global(dt_iop_module_so_t *self)
{
  free(self->data);
  self->data = NULL;
}


void commit_params(dt_iop_module_t *self,
                   dt_iop_params_t *p1,
                   dt_dev_pixelpipe_t *pipe,
                   dt_dev_pixelpipe_iop_t *piece)
{
  const dt_iop_pyramidal_contrast_params_t *p = (dt_iop_pyramidal_contrast_params_t *)p1;
  dt_iop_pyramidal_contrast_data_t *d = piece->data;

  d->mode = p->mode;

  d->pyr_method = DT_TONEEQ_NORM_2;
  d->pyr_details = DT_PYR_EIGF;
  d->pyr_iterations = 1;
  d->pyr_scale = 1.0f;
  d->pyr_micro_scale = p->pyr_micro_scale;
  d->pyr_fine_scale = p->pyr_fine_scale;
  d->pyr_detail_scale = p->pyr_detail_scale;
  d->pyr_medium_scale = p->pyr_medium_scale;
  d->pyr_broad_scale = p->pyr_broad_scale; 
  d->pyr_global_scale = p->pyr_global_scale;

  // UI blending param is the square root of the actual blending parameter
  // to make it more sensitive to small values that represent the most important value domain.
  // UI parameter is given in percentage of maximum blending value.
  // The actual blending parameter represents the fraction of the largest image dimension.
  d->pyr_blending = p->pyr_blending * p->pyr_blending / 100.0f;

  // UI guided filter feathering param increases edge taping
  // but actual regularization behaves inversely
  d->pyr_feathering = 1.0f / p->pyr_feathering;

  d->pyr_f_mult_micro = p->pyr_f_mult_micro;
  d->pyr_f_mult_fine = p->pyr_f_mult_fine;
  d->pyr_f_mult_detail = p->pyr_f_mult_detail;
  d->pyr_f_mult_medium = p->pyr_f_mult_medium;
  d->pyr_f_mult_broad = p->pyr_f_mult_broad;

  d->pyr_s_mult_micro = p->pyr_s_mult_micro;
  d->pyr_s_mult_fine = p->pyr_s_mult_fine;
  d->pyr_s_mult_detail = p->pyr_s_mult_detail;
  d->pyr_s_mult_medium = p->pyr_s_mult_medium;
  d->pyr_s_mult_broad = p->pyr_s_mult_broad;

  // Copy shared params
  d->exp_method = p->exp_method;
  d->exp_details = p->exp_details;
  d->exp_iterations = p->exp_iterations;

  // Copy per-scale params and compute derived values
  for(int s = 0; s < N_SCALES; s++)
  {
    // UI parameter is given in percentage of detail strength, where 100% means no change 
    // and 0% means that detail is removed (multiplier 0). Internal math is a multiplier of relative detail EV
    // so that 100% means no change, 200% means double the detail, 50% means half the detail, 
    d->exp_scales[s].exp_detail_boost = p->exp_detail_boost[s] / 100.0f;

    // UI feature_scale param is the square root of the actual feature_scale parameter
    // to make it more sensitive to small values that represent the most important value domain.
    // UI parameter is given in percentage of maximum feature_scale value.
    // The actual feature_scale parameter represents the fraction of the largest image dimension.
    d->exp_scales[s].exp_feature_scale = p->exp_feature_scale[s] * p->exp_feature_scale[s] / 10000.0f;

    // UI guided filter feathering param increases edge taping
    // but actual regularization behaves inversely
    d->exp_scales[s].exp_feathering = 1.0f / p->exp_feathering[s];
  }
}


void init_pipe(dt_iop_module_t *self,
               dt_dev_pixelpipe_t *pipe,
               dt_dev_pixelpipe_iop_t *piece)
{
  piece->data = dt_calloc1_align_type(dt_iop_pyramidal_contrast_data_t);
}


void cleanup_pipe(dt_iop_module_t *self,
                  dt_dev_pixelpipe_t *pipe,
                  dt_dev_pixelpipe_iop_t *piece)
{
  dt_free_align(piece->data);
  piece->data = NULL;
}


static void pyr_gui_cache_init(dt_iop_module_t *self)
{
  dt_iop_pyramidal_contrast_gui_data_t *g = self->gui_data;
  if(g == NULL) return;

  dt_iop_gui_enter_critical_section(self);
  g->pyr_ui_preview_hash = DT_INVALID_HASH;
  g->pyr_thumb_preview_hash = DT_INVALID_HASH;
  g->pyr_mask_display = DT_PYR_MASK_OFF;
  g->luminance_valid = FALSE;

  g->exp_ui_preview_hash = DT_INVALID_HASH;
  g->exp_thumb_preview_hash = DT_INVALID_HASH;
  g->exp_mask_display_scale = -1;  // no mask displayed
  g->exp_luminance_valid = FALSE;

  g->exp_full_preview_buf_pixel = NULL;
  for(int s = 0; s < N_SCALES; s++)
    g->exp_full_preview_buf_smoothed[s] = NULL;
  g->exp_full_preview_buf_width = 0;
  g->exp_full_preview_buf_height = 0;

  g->exp_thumb_preview_buf_pixel = NULL;
  for(int s = 0; s < N_SCALES; s++)
    g->exp_thumb_preview_buf_smoothed[s] = NULL;
  g->exp_thumb_preview_buf_width = 0;
  g->exp_thumb_preview_buf_height = 0;

  g->exp_pipe_order = 0;

  g->pyr_full_preview_buf_pixel = NULL;
  g->pyr_full_preview_buf_smoothed_broad = NULL;
  g->pyr_full_preview_buf_smoothed_medium = NULL;
  g->pyr_full_preview_buf_smoothed = NULL;
  g->pyr_full_preview_buf_smoothed_fine = NULL;
  g->pyr_full_preview_buf_smoothed_micro = NULL;
  g->pyr_full_preview_buf_width = 0;
  g->pyr_full_preview_buf_height = 0;

  g->pyr_thumb_preview_buf_pixel = NULL;
  g->pyr_thumb_preview_buf_smoothed_broad = NULL;
  g->pyr_thumb_preview_buf_smoothed_medium = NULL;
  g->pyr_thumb_preview_buf_smoothed = NULL;
  g->pyr_thumb_preview_buf_smoothed_fine = NULL;
  g->pyr_thumb_preview_buf_smoothed_micro = NULL;
  g->pyr_thumb_preview_buf_width = 0;
  g->pyr_thumb_preview_buf_height = 0;

  g->pyr_pipe_order = 0;
  dt_iop_gui_leave_critical_section(self);
}

static void exp_gui_cache_init(dt_iop_module_t *self)
{
  dt_iop_pyramidal_contrast_gui_data_t *g = self->gui_data;
  if(g == NULL) return;

  dt_iop_gui_enter_critical_section(self);
  g->exp_ui_preview_hash = DT_INVALID_HASH;
  g->exp_thumb_preview_hash = DT_INVALID_HASH;
  g->exp_mask_display_scale = -1;  // no mask displayed
  g->exp_luminance_valid = FALSE;

  g->exp_full_preview_buf_pixel = NULL;
  for(int s = 0; s < N_SCALES; s++)
    g->exp_full_preview_buf_smoothed[s] = NULL;
  g->exp_full_preview_buf_width = 0;
  g->exp_full_preview_buf_height = 0;

  g->exp_thumb_preview_buf_pixel = NULL;
  for(int s = 0; s < N_SCALES; s++)
    g->exp_thumb_preview_buf_smoothed[s] = NULL;
  g->exp_thumb_preview_buf_width = 0;
  g->exp_thumb_preview_buf_height = 0;

  g->exp_pipe_order = 0;
  dt_iop_gui_leave_critical_section(self);
}

static void pyr_update_mask_buttons_state(dt_iop_pyramidal_contrast_gui_data_t *g)
{
  if(darktable.gui->reset) return;
  ++darktable.gui->reset;

  dt_bauhaus_widget_set_quad_active(g->pyr_broad_scale, g->pyr_mask_display == DT_PYR_MASK_BROAD);
  dt_bauhaus_widget_set_quad_active(g->pyr_medium_scale, g->pyr_mask_display == DT_PYR_MASK_MEDIUM);
  dt_bauhaus_widget_set_quad_active(g->pyr_detail_scale, g->pyr_mask_display == DT_PYR_MASK_DETAIL);
  dt_bauhaus_widget_set_quad_active(g->pyr_fine_scale, g->pyr_mask_display == DT_PYR_MASK_FINE);
  dt_bauhaus_widget_set_quad_active(g->pyr_micro_scale, g->pyr_mask_display == DT_PYR_MASK_MICRO);

  if(g->pyr_f_view_broad) gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->pyr_f_view_broad), g->pyr_mask_display == DT_PYR_MASK_BROAD);
  if(g->pyr_f_view_medium) gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->pyr_f_view_medium), g->pyr_mask_display == DT_PYR_MASK_MEDIUM);
  if(g->pyr_f_view_detail) gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->pyr_f_view_detail), g->pyr_mask_display == DT_PYR_MASK_DETAIL);
  if(g->pyr_f_view_fine) gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->pyr_f_view_fine), g->pyr_mask_display == DT_PYR_MASK_FINE);
  if(g->pyr_f_view_micro) gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->pyr_f_view_micro), g->pyr_mask_display == DT_PYR_MASK_MICRO);

  // Update mask toggle buttons
  for(int s = 0; s < N_SCALES; s++)
  {
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->exp_show_mask[s]),
                                 g->exp_mask_display_scale == s);
  }
  --darktable.gui->reset;
}

static void pyr_set_mask_display(dt_iop_module_t *self, dt_iop_pyramidal_contrast_mask_t mask_type)
{
  dt_iop_pyramidal_contrast_gui_data_t *g = self->gui_data;

  if(darktable.gui->reset) return;

  // If blend module is displaying mask, don't display here
  if(self->request_mask_display)
  {
    dt_control_log(_("cannot display masks when the blending mask is displayed"));
    g->pyr_mask_display = DT_PYR_MASK_OFF;
  }
  else
  {
    // Toggle logic
    if(g->pyr_mask_display == mask_type)
    {
      g->pyr_mask_display = DT_PYR_MASK_OFF;
    }
    else
    {
      g->pyr_mask_display = mask_type;
    }
  }

  g->exp_mask_display_scale = -1;
  pyr_update_mask_buttons_state(g);

  pyr_invalidate_luminance_cache(self);
}

static gboolean pyr_mask_toggle_callback(GtkWidget *togglebutton, GdkEventButton *event, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return FALSE;
  dt_iop_pyramidal_contrast_mask_t mask_type = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(togglebutton), "mask-type"));
  pyr_set_mask_display(self, mask_type);
  return TRUE;
}

static void pyr_create_slider_with_mask_button(dt_iop_module_t *self, GtkWidget *container, GtkWidget **slider_widget,
                                            GtkWidget **button_widget, const char *param_name, const char *tooltip,
                                            dt_iop_pyramidal_contrast_mask_t mask_type)
{
  GtkWidget *hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);

  *slider_widget = dt_bauhaus_slider_from_params(self, param_name);
  dt_bauhaus_slider_set_digits(*slider_widget, 2);
  dt_bauhaus_slider_set_soft_range(*slider_widget, 0.1, 3.0);
  dt_bauhaus_slider_set_format(*slider_widget, "%");
  dt_bauhaus_slider_set_factor(*slider_widget, 100.0);

  g_object_ref(*slider_widget);
  gtk_container_remove(GTK_CONTAINER(self->widget), *slider_widget);

  gtk_box_pack_start(GTK_BOX(hbox), *slider_widget, TRUE, TRUE, 0);
  g_object_unref(*slider_widget);

  *button_widget = dt_iop_togglebutton_new(self, NULL, tooltip, NULL, G_CALLBACK(pyr_mask_toggle_callback), TRUE, 0, 0,
                                           dtgtk_cairo_paint_showmask, hbox);
  g_object_set_data(G_OBJECT(*button_widget), "mask-type", GINT_TO_POINTER(mask_type));
  dt_gui_add_class(*button_widget, "dt_transparent_background");

  dt_gui_box_add(container, hbox);
}

static void pyr_show_guiding_controls(const dt_iop_module_t *self)
{
  const dt_iop_pyramidal_contrast_gui_data_t *g = self->gui_data;

  // All filters need these controls
  gtk_widget_set_visible(g->pyr_blending, TRUE);
  gtk_widget_set_visible(g->pyr_feathering, TRUE);
}


void gui_update(dt_iop_module_t *self)
{
  dt_iop_pyramidal_contrast_gui_data_t *g = self->gui_data;

  exp_invalidate_luminance_cache(self);
  pyr_show_guiding_controls(self);
  pyr_invalidate_luminance_cache(self);
  pyr_update_mask_buttons_state(g);

  dt_gui_update_collapsible_section(&g->pyr_advanced_expander);

  dt_iop_pyramidal_contrast_params_t *p = self->params;
  g_signal_handlers_block_by_func(g->notebook, mode_tab_switch_callback, self);
  gtk_notebook_set_current_page(g->notebook, p->mode);
  g_signal_handlers_unblock_by_func(g->notebook, mode_tab_switch_callback, self);
}


void gui_changed(dt_iop_module_t *self,
                 GtkWidget *w,
                 void *previous)
{
  const dt_iop_pyramidal_contrast_gui_data_t *g = self->gui_data;

  if(w == g->pyr_blending || w == g->pyr_feathering
     || w == g->pyr_f_mult_micro || w == g->pyr_f_mult_fine || w == g->pyr_f_mult_detail
     || w == g->pyr_f_mult_medium || w == g->pyr_f_mult_broad)
  {
    pyr_invalidate_luminance_cache(self);
  }

  // Check per-scale widgets
  for(int s = 0; s < N_SCALES; s++)
  {
    if(w == g->exp_feature_scale[s] || w == g->exp_feathering[s])
      exp_invalidate_luminance_cache(self);
  }
}


static void pyr_quad_callback(GtkWidget *quad, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return;
  dt_iop_pyramidal_contrast_gui_data_t *g = self->gui_data;
  dt_iop_pyramidal_contrast_mask_t mask_type = DT_PYR_MASK_OFF;

  if(quad == g->pyr_broad_scale) mask_type = DT_PYR_MASK_BROAD;
  else if(quad == g->pyr_medium_scale) mask_type = DT_PYR_MASK_MEDIUM;
  else if(quad == g->pyr_detail_scale) mask_type = DT_PYR_MASK_DETAIL;
  else if(quad == g->pyr_fine_scale) mask_type = DT_PYR_MASK_FINE;
  else if(quad == g->pyr_micro_scale) mask_type = DT_PYR_MASK_MICRO;

  if(mask_type != DT_PYR_MASK_OFF)
  {
    pyr_set_mask_display(self, mask_type);
  }
}


static void _develop_ui_pipe_started_callback(gpointer instance,
                                              dt_iop_module_t *self)
{
  dt_iop_pyramidal_contrast_gui_data_t *g = self->gui_data;
  if(g == NULL) return;

  if(!self->expanded || !self->enabled)
  {
    dt_iop_gui_enter_critical_section(self);
    g->pyr_mask_display = DT_PYR_MASK_OFF;
    g->exp_mask_display_scale = -1;
    dt_iop_gui_leave_critical_section(self);
  }

  ++darktable.gui->reset;
  dt_iop_gui_enter_critical_section(self);
  pyr_update_mask_buttons_state(g);
  dt_iop_gui_leave_critical_section(self);
  --darktable.gui->reset;
}


static void _develop_preview_pipe_finished_callback(gpointer instance,
                                                    dt_iop_module_t *self)
{
  const dt_iop_pyramidal_contrast_gui_data_t *g = self->gui_data;
  if(g == NULL) return;
}


static void _develop_ui_pipe_finished_callback(gpointer instance,
                                               dt_iop_module_t *self)
{
  const dt_iop_pyramidal_contrast_gui_data_t *g = self->gui_data;
  if(g == NULL) return;
}


void gui_focus(dt_iop_module_t *self, gboolean in)
{
  dt_iop_pyramidal_contrast_gui_data_t *g = self->gui_data;
  if(!in)
  {
    const gboolean mask_was_shown = (g->pyr_mask_display != DT_PYR_MASK_OFF);
    g->pyr_mask_display = DT_PYR_MASK_OFF;
    g->exp_mask_display_scale = -1;

    pyr_update_mask_buttons_state(g);
    if(mask_was_shown) dt_dev_reprocess_center(self->dev);
  }
}


void gui_reset(dt_iop_module_t *self)
{
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

/**
 * Callback for mask display toggle buttons.
 * Ensures only one mask can be displayed at a time.
 **/
static void exp_show_mask_callback(GtkWidget *togglebutton,
                               GdkEventButton *event,
                               dt_iop_module_t *self)
{
  if(darktable.gui->reset) return;
  dt_iop_request_focus(self);

  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(self->off), TRUE);

  dt_iop_pyramidal_contrast_gui_data_t *g = self->gui_data;

  // If blend module is displaying mask, don't display here
  if(self->request_mask_display)
  {
    dt_control_log(_("cannot display masks when the feature_scale mask is displayed"));
    for(int s = 0; s < N_SCALES; s++)
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->exp_show_mask[s]), FALSE);
    g->exp_mask_display_scale = -1;
    return;
  }

  // Find which scale's button was clicked
  int clicked_scale = -1;
  for(int s = 0; s < N_SCALES; s++)
  {
    if(togglebutton == g->exp_show_mask[s])
    {
      clicked_scale = s;
      break;
    }
  }

  // Toggle: if same scale was active, turn off; otherwise switch to new scale
  if(g->exp_mask_display_scale == clicked_scale)
  {
    g->exp_mask_display_scale = -1;  // turn off
  }
  else
  {
    g->exp_mask_display_scale = clicked_scale;  // switch to new
  }

  g->pyr_mask_display = DT_PYR_MASK_OFF;
  pyr_update_mask_buttons_state(g);

  // Update all toggle buttons
  for(int s = 0; s < N_SCALES; s++)
  {
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->exp_show_mask[s]),
                                 g->exp_mask_display_scale == s);
  }

  exp_invalidate_luminance_cache(self);
}

/**
 * Create GUI widgets for a single scale section
 **/
static void exp_create_scale_section(dt_iop_module_t *self,
                                 dt_iop_pyramidal_contrast_gui_data_t *g,
                                 GtkWidget *container,
                                 const int scale_idx)
{
  // Section label
  char section_name[64];
  snprintf(section_name, sizeof(section_name), _("detail scale %d"), scale_idx + 1);

  gtk_widget_set_margin_top(dt_ui_section_label_new(section_name), DT_PIXEL_APPLY_DPI(10));
  dt_gui_box_add(container, dt_ui_section_label_new(section_name));

  // Detail boost slider
  char param_name[64];
  snprintf(param_name, sizeof(param_name), "exp_detail_boost[%d]", scale_idx);
  g->exp_detail_boost[scale_idx] = dt_bauhaus_slider_from_params(self, param_name);
  dt_bauhaus_slider_set_soft_range(g->exp_detail_boost[scale_idx], 0.0, 300.0);
  dt_bauhaus_slider_set_format(g->exp_detail_boost[scale_idx], "%");
  dt_bauhaus_slider_set_digits(g->exp_detail_boost[scale_idx], 2);
  dt_bauhaus_widget_set_label(g->exp_detail_boost[scale_idx], NULL, _("detail strength"));
  gtk_widget_set_tooltip_text
    (g->exp_detail_boost[scale_idx],
     _("amount of local contrast for this scale\n"
       "100% = no change\n"
       "> 100% = increase local contrast\n"
       "< 100% = decrease local contrast"));
  if(self->widget != container)
  {
    g_object_ref(g->exp_detail_boost[scale_idx]);
    gtk_container_remove(GTK_CONTAINER(self->widget), g->exp_detail_boost[scale_idx]);
    dt_gui_box_add(container, g->exp_detail_boost[scale_idx]);
    g_object_unref(g->exp_detail_boost[scale_idx]);
  }
  else
  {
    // Already in the right container, just ensure packing order if needed, 
    // but dt_bauhaus_slider_from_params appends, so it's fine.
    // dt_gui_box_add would fail here.
  }

  // Feature scale slider
  snprintf(param_name, sizeof(param_name), "exp_feature_scale[%d]", scale_idx);
  g->exp_feature_scale[scale_idx] = dt_bauhaus_slider_from_params(self, param_name);
  dt_bauhaus_slider_set_soft_range(g->exp_feature_scale[scale_idx], 0.1, 100.0);
  dt_bauhaus_slider_set_format(g->exp_feature_scale[scale_idx], "%");
  dt_bauhaus_widget_set_label(g->exp_feature_scale[scale_idx], NULL, _("feature scale"));
  gtk_widget_set_tooltip_text
    (g->exp_feature_scale[scale_idx],
     _("size of the smoothing area as percentage of image size\n"
       "larger = affects broader features\n"
       "smaller = affects finer details"));
  if(self->widget != container)
  {
    g_object_ref(g->exp_feature_scale[scale_idx]);
    gtk_container_remove(GTK_CONTAINER(self->widget), g->exp_feature_scale[scale_idx]);
    dt_gui_box_add(container, g->exp_feature_scale[scale_idx]);
    g_object_unref(g->exp_feature_scale[scale_idx]);
  }

  // Edge refinement slider
  snprintf(param_name, sizeof(param_name), "exp_feathering[%d]", scale_idx);
  g->exp_feathering[scale_idx] = dt_bauhaus_slider_from_params(self, param_name);
  dt_bauhaus_slider_set_soft_range(g->exp_feathering[scale_idx], 0.1, 50.0);
  dt_bauhaus_widget_set_label(g->exp_feathering[scale_idx], NULL, _("edges refinement"));
  gtk_widget_set_tooltip_text
    (g->exp_feathering[scale_idx],
     _("edge sensitivity of the filter\n"
       "higher = better edge preservation\n"
       "lower = smoother transitions, but may lead to halos around edges"));
  if(self->widget != container)
  {
    g_object_ref(g->exp_feathering[scale_idx]);
    gtk_container_remove(GTK_CONTAINER(self->widget), g->exp_feathering[scale_idx]);
    dt_gui_box_add(container, g->exp_feathering[scale_idx]);
    g_object_unref(g->exp_feathering[scale_idx]);
  }

  // Mask display toggle
  g->exp_show_mask[scale_idx] = dt_iop_togglebutton_new
    (self, NULL,
     N_("display detail mask"), NULL, G_CALLBACK(exp_show_mask_callback),
     FALSE, 0, 0, dtgtk_cairo_paint_showmask, NULL);
  dt_gui_add_class(g->exp_show_mask[scale_idx], "dt_transparent_background");
  dtgtk_togglebutton_set_paint(DTGTK_TOGGLEBUTTON(g->exp_show_mask[scale_idx]),
                               dtgtk_cairo_paint_showmask, 0, NULL);
  dt_gui_add_class(g->exp_show_mask[scale_idx], "dt_bauhaus_alignment");

  GtkWidget *hbox = dt_gui_hbox(dt_gui_expand(dt_ui_label_new(_("display detail mask"))),
                                g->exp_show_mask[scale_idx]);
  dt_gui_box_add(container, hbox);
}

static void mode_tab_switch_callback(GtkNotebook *notebook,
                                     GtkWidget *page,
                                     guint page_num,
                                     dt_iop_module_t *self)
{
  if(darktable.gui->reset) return;
  dt_iop_pyramidal_contrast_params_t *p = self->params;
  
  p->mode = (dt_iop_pyramidal_contrast_mode_t)page_num;
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

void gui_init(dt_iop_module_t *self)
{
  dt_iop_pyramidal_contrast_gui_data_t *g = IOP_GUI_ALLOC(pyramidal_contrast);

  pyr_gui_cache_init(self);
  exp_gui_cache_init(self);

  // Main container
  self->widget = dt_gui_vbox();
  GtkWidget *root = self->widget;
  
  static dt_action_def_t notebook_def = { };
  g->notebook = GTK_NOTEBOOK(dt_ui_notebook_new(&notebook_def));
  dt_action_define_iop(self, NULL, N_("mode"), GTK_WIDGET(g->notebook), &notebook_def);
  dt_gui_box_add(self->widget, GTK_WIDGET(g->notebook));

  g->global_box = dt_ui_notebook_page(g->notebook, _("global"), _("global contrast settings"));
  g->expert_box = dt_ui_notebook_page(g->notebook, _("expert"), _("expert contrast settings"));

  g_signal_connect(G_OBJECT(g->notebook), "switch-page", G_CALLBACK(mode_tab_switch_callback), self);

  // --- Global Mode ---
  // Set target to global_box
  self->widget = g->global_box;

  // Micro detail slider
  g->pyr_micro_scale = dt_bauhaus_slider_from_params(self, "pyr_micro_scale");
  dt_bauhaus_slider_set_soft_range(g->pyr_micro_scale, 0.25, 3.0);
  dt_bauhaus_slider_set_digits(g->pyr_micro_scale, 2);
  dt_bauhaus_slider_set_format(g->pyr_micro_scale, "%");
  dt_bauhaus_slider_set_factor(g->pyr_micro_scale, 100.0);
  gtk_widget_set_tooltip_text(g->pyr_micro_scale, _("amount of micro contrast enhancement"));
  dt_bauhaus_widget_set_quad(g->pyr_micro_scale, self, dtgtk_cairo_paint_showmask, TRUE, pyr_quad_callback,
                             _("visualize micro contrast mask"));

  // Fine detail slider
  g->pyr_fine_scale = dt_bauhaus_slider_from_params(self, "pyr_fine_scale");
  dt_bauhaus_slider_set_soft_range(g->pyr_fine_scale, 0.25, 3.0);
  dt_bauhaus_slider_set_digits(g->pyr_fine_scale, 2);
  dt_bauhaus_slider_set_format(g->pyr_fine_scale, "%");
  dt_bauhaus_slider_set_factor(g->pyr_fine_scale, 100.0);
  gtk_widget_set_tooltip_text(g->pyr_fine_scale, _("amount of fine contrast enhancement"));
  dt_bauhaus_widget_set_quad(g->pyr_fine_scale, self, dtgtk_cairo_paint_showmask, TRUE, pyr_quad_callback,
                             _("visualize fine contrast mask"));

  // Detail boost slider
  g->pyr_detail_scale = dt_bauhaus_slider_from_params(self, "pyr_detail_scale");
  dt_bauhaus_slider_set_soft_range(g->pyr_detail_scale, 0.25, 3.0);
  dt_bauhaus_slider_set_digits(g->pyr_detail_scale, 2);
  dt_bauhaus_slider_set_format(g->pyr_detail_scale, "%");
  dt_bauhaus_slider_set_factor(g->pyr_detail_scale, 100.0);
  gtk_widget_set_tooltip_text
    (g->pyr_detail_scale,
     _("amount of local contrast enhancement\n"
       "1.0 = no change\n"
       "> 1.0 = boost local contrast\n"
       "< 1.0 = reduce local contrast"));
  dt_bauhaus_widget_set_quad(g->pyr_detail_scale, self, dtgtk_cairo_paint_showmask, TRUE, pyr_quad_callback,
                             _("visualize local contrast mask"));

  // Medium detail slider
  g->pyr_medium_scale = dt_bauhaus_slider_from_params(self, "pyr_medium_scale");
  dt_bauhaus_slider_set_soft_range(g->pyr_medium_scale, 0.25, 3.0);
  dt_bauhaus_slider_set_digits(g->pyr_medium_scale, 2);
  dt_bauhaus_slider_set_format(g->pyr_medium_scale, "%");
  dt_bauhaus_slider_set_factor(g->pyr_medium_scale, 100.0);
  gtk_widget_set_tooltip_text(g->pyr_medium_scale, _("amount of broad contrast enhancement"));
  dt_bauhaus_widget_set_quad(g->pyr_medium_scale, self, dtgtk_cairo_paint_showmask, TRUE, pyr_quad_callback,
                             _("visualize broad contrast mask"));

  // Broad detail slider
  g->pyr_broad_scale = dt_bauhaus_slider_from_params(self, "pyr_broad_scale");
  dt_bauhaus_slider_set_soft_range(g->pyr_broad_scale, 0.25, 3.0);
  dt_bauhaus_slider_set_digits(g->pyr_broad_scale, 2);
  dt_bauhaus_slider_set_format(g->pyr_broad_scale, "%");
  dt_bauhaus_slider_set_factor(g->pyr_broad_scale, 100.0);
  gtk_widget_set_tooltip_text(g->pyr_broad_scale, _("amount of extended contrast enhancement"));
  dt_bauhaus_widget_set_quad(g->pyr_broad_scale, self, dtgtk_cairo_paint_showmask, TRUE, pyr_quad_callback,
                             _("visualize extended contrast mask"));

  // Global contrast slider
  g->pyr_global_scale = dt_bauhaus_slider_from_params(self, "pyr_global_scale");
  dt_bauhaus_slider_set_soft_range(g->pyr_global_scale, 0.25, 3.0);
  dt_bauhaus_slider_set_digits(g->pyr_global_scale, 2);
  dt_bauhaus_slider_set_format(g->pyr_global_scale, "%");
  dt_bauhaus_slider_set_factor(g->pyr_global_scale, 100.0);
  gtk_widget_set_tooltip_text
    (g->pyr_global_scale,
     _("amount of global contrast enhancement"));

  // Separator
  GtkWidget *label = dt_ui_section_label_new(C_("section", "masking"));
  gtk_widget_set_margin_top(label, DT_PIXEL_APPLY_DPI(10));
  dt_gui_box_add(g->global_box, label);

  g->pyr_blending = dt_bauhaus_slider_from_params(self, "pyr_blending");
  dt_bauhaus_slider_set_soft_range(g->pyr_blending, 1.0, 4.0);
  dt_bauhaus_slider_set_format(g->pyr_blending, "%");
  dt_bauhaus_slider_set_factor(g->pyr_blending, 10.0);
  gtk_widget_set_tooltip_text
    (g->pyr_blending,
     _("size of the smoothing area as percentage of image size\n"
       "larger = affects broader features\n"
       "smaller = affects finer details"));

  g->pyr_feathering = dt_bauhaus_slider_from_params(self, "pyr_feathering");
  dt_bauhaus_slider_set_soft_range(g->pyr_feathering, 0.1, 50.0);
  gtk_widget_set_tooltip_text(g->pyr_feathering, _("edges refinement"));

  // Create section
  dt_gui_new_collapsible_section(&g->pyr_advanced_expander, "plugins/darkroom/pyramidal_contrast/expanded_advanced",
                                 _("feathering fine tuning"), GTK_BOX(g->global_box), DT_ACTION(self));
  
  // Switch self->widget to the section container
  self->widget = GTK_WIDGET(g->pyr_advanced_expander.container);

  pyr_create_slider_with_mask_button(self, self->widget, &g->pyr_f_mult_micro, &g->pyr_f_view_micro, "pyr_f_mult_micro", _("visualize micro contrast mask"), DT_PYR_MASK_MICRO);
  pyr_create_slider_with_mask_button(self, self->widget, &g->pyr_f_mult_fine, &g->pyr_f_view_fine, "pyr_f_mult_fine", _("visualize fine contrast mask"), DT_PYR_MASK_FINE);
  pyr_create_slider_with_mask_button(self, self->widget, &g->pyr_f_mult_detail, &g->pyr_f_view_detail, "pyr_f_mult_detail", _("visualize local contrast mask"), DT_PYR_MASK_DETAIL);
  pyr_create_slider_with_mask_button(self, self->widget, &g->pyr_f_mult_medium, &g->pyr_f_view_medium, "pyr_f_mult_medium", _("visualize broad contrast mask"), DT_PYR_MASK_MEDIUM);
  pyr_create_slider_with_mask_button(self, self->widget, &g->pyr_f_mult_broad, &g->pyr_f_view_broad, "pyr_f_mult_broad", _("visualize extended contrast mask"), DT_PYR_MASK_BROAD);

  // --- Expert Mode ---
  // Set target to expert_box
  self->widget = g->expert_box;

  // Create GUI for each scale
  for(int s = 0; s < N_SCALES; s++)
  {
    exp_create_scale_section(self, g, g->expert_box, s);
  }

  // Masking section (shared parameters)
  gtk_widget_set_margin_top(dt_ui_section_label_new(C_("section", "masking")), DT_PIXEL_APPLY_DPI(10));
  dt_gui_box_add(g->expert_box, dt_ui_section_label_new(C_("section", "masking")));

  g->exp_details = dt_bauhaus_combobox_from_params(self, "exp_details");
  dt_gui_box_add(g->expert_box, g->exp_details);
  g->exp_iterations = dt_bauhaus_slider_from_params(self, "exp_iterations");
  dt_gui_box_add(g->expert_box, g->exp_iterations);

  // Restore main widget
  self->widget = root;

  // Connect signals for pipe events
  DT_CONTROL_SIGNAL_HANDLE(DT_SIGNAL_DEVELOP_PREVIEW_PIPE_FINISHED, _develop_preview_pipe_finished_callback);
  DT_CONTROL_SIGNAL_HANDLE(DT_SIGNAL_DEVELOP_UI_PIPE_FINISHED, _develop_ui_pipe_finished_callback);
  DT_CONTROL_SIGNAL_HANDLE(DT_SIGNAL_DEVELOP_HISTORY_CHANGE, _develop_ui_pipe_started_callback);
}


void gui_cleanup(dt_iop_module_t *self)
{
  dt_iop_pyramidal_contrast_gui_data_t *g = self->gui_data;

  dt_free_align(g->exp_thumb_preview_buf_pixel);
  exp_free_smoothed_buffers(g->exp_thumb_preview_buf_smoothed);
  dt_free_align(g->exp_full_preview_buf_pixel);
  exp_free_smoothed_buffers(g->exp_full_preview_buf_smoothed);

  dt_free_align(g->pyr_thumb_preview_buf_pixel);
  dt_free_align(g->pyr_thumb_preview_buf_smoothed_broad);
  dt_free_align(g->pyr_thumb_preview_buf_smoothed_medium);
  dt_free_align(g->pyr_thumb_preview_buf_smoothed);
  dt_free_align(g->pyr_thumb_preview_buf_smoothed_fine);
  dt_free_align(g->pyr_thumb_preview_buf_smoothed_micro);
  dt_free_align(g->pyr_full_preview_buf_pixel);
  dt_free_align(g->pyr_full_preview_buf_smoothed_broad);
  dt_free_align(g->pyr_full_preview_buf_smoothed_medium);
  dt_free_align(g->pyr_full_preview_buf_smoothed);
  dt_free_align(g->pyr_full_preview_buf_smoothed_fine);
  dt_free_align(g->pyr_full_preview_buf_smoothed_micro);
}


// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on