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

#ifdef HAVE_SLANG_GPU

// lmmse_demosaic() on the Slang device, see data/kernels/slang/demosaic_lmmse.slang.
// PPG fills the 4 pixel border as on the CPU. The six buffers cover the
// image plus that border; the color planes and the median corrections reuse
// the ones whose contents are no longer read, as the CPU tile buffers do.
static gboolean process_lmmse_slang(dt_dev_pixelpipe_iop_t *piece,
                                    dt_slang_gpu_image_t *dev_in,
                                    dt_slang_gpu_image_t *dev_out,
                                    const int width,
                                    const int height,
                                    const uint32_t filters,
                                    const dt_iop_demosaic_lmmse_t mode)
{
  const int pw = width + 8;
  const int ph = height + 8;
  const size_t bsize = sizeof(float) * pw * ph;
  const float scaler = dt_iop_get_processed_maximum(piece);
  const float revscaler = 1.0f / scaler;
  const int green_border = 7;
  const int redblue_border = 4;

  float h0 = 1.0f;
  float h1 = expf(-1.0f / 8.0f);
  float h2 = expf(-4.0f / 8.0f);
  float h3 = expf(-9.0f / 8.0f);
  float h4 = expf(-16.0f / 8.0f);
  const float hs = h0 + 2.0f * (h1 + h2 + h3 + h4);
  h0 /= hs;
  h1 /= hs;
  h2 /= hs;
  h3 /= hs;
  h4 /= hs;

  const int medians = (mode < DT_LMMSE_REFINE_2) ? mode : 3;
  const int refine = (mode > DT_LMMSE_REFINE_2) ? mode - 2 : 0;

  enum { NBUF = 6 };
  // the module's memory estimate doesn't know these, so a large image falls
  // back to the CPU here rather than running out of device memory
  size_t available = 0;
  int max_dim = 0;
  dt_slang_gpu_memory_info(&available, &max_dim);
  if(NBUF * bsize > available)
  {
    dt_print(DT_DEBUG_SLANG, "[demosaic] LMMSE needs %zu MB of device memory, %zu MB available",
             NBUF * bsize / DT_MEGA, available / DT_MEGA);
    return FALSE;
  }

  dt_slang_gpu_image_t *dev_tmp = dt_slang_gpu_image_new(width, height, 4, NULL);
  gboolean ok = dev_tmp
    && !dt_slang_gpu_run_2d_args("demosaic_ppg", "border_interpolate", width, height,
                                 SLIMG_DEV(dev_in), SLIMG_DEV(dev_tmp), SLARG(width), SLARG(height), SLARG(filters))
    && !dt_slang_gpu_run_2d_args("demosaic_ppg", "ppg_demosaic_green", width, height,
                                 SLIMG_DEV(dev_in), SLIMG_DEV(dev_tmp), SLARG(width), SLARG(height),
                                 SLARG(filters), SLARG(green_border))
    && !dt_slang_gpu_run_2d_args("demosaic_ppg", "ppg_demosaic_redblue", width, height,
                                 SLIMG_DEV(dev_tmp), SLIMG_DEV(dev_out), SLARG(width), SLARG(height),
                                 SLARG(filters), SLARG(redblue_border));
  dt_slang_gpu_image_free(dev_tmp);

  dt_slang_gpu_buffer_t *b[NBUF] = { NULL };
  for(int k = 0; k < NBUF && ok; k++)
    ok = (b[k] = dt_slang_gpu_buffer_new(bsize, NULL)) != NULL;

  dt_slang_gpu_buffer_t *hdiff = b[0], *vdiff = b[1], *hlp = b[2], *vlp = b[3], *interp = b[4], *cfa = b[5];
  dt_slang_gpu_buffer_t *red = b[0], *green = b[1], *blue = b[2], *corr_r = b[3], *corr_b = b[4];

  ok = ok
    && !dt_slang_gpu_run_2d_args("demosaic_lmmse", "lmmse_init", pw, ph,
                                 SLIMG_DEV(dev_in), SLBUF_DEV(cfa),
                                 SLARG(width), SLARG(height), SLARG(pw), SLARG(ph), SLARG(revscaler))
    && !dt_slang_gpu_run_2d_args("demosaic_lmmse", "lmmse_diff", pw, ph,
                                 SLBUF_DEV(cfa), SLBUF_DEV(hdiff), SLBUF_DEV(vdiff),
                                 SLARG(pw), SLARG(ph), SLARG(filters))
    && !dt_slang_gpu_run_2d_args("demosaic_lmmse", "lmmse_lowpass", pw, ph,
                                 SLBUF_DEV(hdiff), SLBUF_DEV(vdiff), SLBUF_DEV(hlp), SLBUF_DEV(vlp),
                                 SLARG(pw), SLARG(ph), SLARG(h0), SLARG(h1), SLARG(h2), SLARG(h3), SLARG(h4))
    && !dt_slang_gpu_run_2d_args("demosaic_lmmse", "lmmse_interp", pw, ph,
                                 SLBUF_DEV(hdiff), SLBUF_DEV(vdiff), SLBUF_DEV(hlp), SLBUF_DEV(vlp),
                                 SLBUF_DEV(interp), SLARG(pw), SLARG(ph), SLARG(filters))
    && !dt_slang_gpu_run_2d_args("demosaic_lmmse", "lmmse_planes", pw, ph,
                                 SLBUF_DEV(cfa), SLBUF_DEV(interp), SLBUF_DEV(red), SLBUF_DEV(green),
                                 SLBUF_DEV(blue), SLARG(width), SLARG(height), SLARG(pw), SLARG(ph),
                                 SLARG(filters))
    && !dt_slang_gpu_run_2d_args("demosaic_lmmse", "lmmse_rb_at_g", pw, ph,
                                 SLBUF_DEV(red), SLBUF_DEV(green), SLBUF_DEV(blue),
                                 SLARG(pw), SLARG(ph), SLARG(filters))
    && !dt_slang_gpu_run_2d_args("demosaic_lmmse", "lmmse_rb_at_rb", pw, ph,
                                 SLBUF_DEV(red), SLBUF_DEV(green), SLBUF_DEV(blue),
                                 SLARG(pw), SLARG(ph), SLARG(filters));

  for(int pass = 0; pass < medians && ok; pass++)
    ok = !dt_slang_gpu_run_2d_args("demosaic_lmmse", "lmmse_median", pw, ph,
                                   SLBUF_DEV(red), SLBUF_DEV(green), SLBUF_DEV(blue),
                                   SLBUF_DEV(corr_r), SLBUF_DEV(corr_b), SLARG(pw), SLARG(ph))
      && !dt_slang_gpu_run_2d_args("demosaic_lmmse", "lmmse_median_apply", pw, ph,
                                   SLBUF_DEV(red), SLBUF_DEV(green), SLBUF_DEV(blue),
                                   SLBUF_DEV(corr_r), SLBUF_DEV(corr_b), SLARG(pw), SLARG(ph), SLARG(filters));

  ok = ok
    && !dt_slang_gpu_run_2d_args("demosaic_lmmse", "lmmse_fill", pw, ph,
                                 SLBUF_DEV(cfa), SLBUF_DEV(red), SLBUF_DEV(green), SLBUF_DEV(blue),
                                 SLARG(pw), SLARG(ph), SLARG(filters));

  for(int step = 0; step < refine && ok; step++)
    ok = !dt_slang_gpu_run_2d_args("demosaic_lmmse", "lmmse_refine_g", pw, ph,
                                   SLBUF_DEV(red), SLBUF_DEV(green), SLBUF_DEV(blue),
                                   SLARG(pw), SLARG(ph), SLARG(filters))
      && !dt_slang_gpu_run_2d_args("demosaic_lmmse", "lmmse_refine_rb_at_g", pw, ph,
                                   SLBUF_DEV(red), SLBUF_DEV(green), SLBUF_DEV(blue),
                                   SLARG(pw), SLARG(ph), SLARG(filters))
      && !dt_slang_gpu_run_2d_args("demosaic_lmmse", "lmmse_refine_rb_at_rb", pw, ph,
                                   SLBUF_DEV(red), SLBUF_DEV(green), SLBUF_DEV(blue),
                                   SLARG(pw), SLARG(ph), SLARG(filters));

  ok = ok
    && !dt_slang_gpu_run_2d_args("demosaic_lmmse", "lmmse_output", width, height,
                                 SLBUF_DEV(red), SLBUF_DEV(green), SLBUF_DEV(blue), SLIMG_DEV(dev_out),
                                 SLARG(width), SLARG(height), SLARG(pw), SLARG(scaler));

  for(int k = 0; k < NBUF; k++) dt_slang_gpu_buffer_free(b[k]);
  return ok;
}

#endif

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
