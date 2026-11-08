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

// amaze_demosaic() on the Slang device, see data/kernels/slang/demosaic_amaze.slang.
// The buffers cover the image plus the 16 pixel border of the CPU tiles.
// Twelve of them are live at the peak; later passes reuse the ones whose
// contents are no longer read, as the CPU code does with its tile buffers.
// The passes that update in place on the CPU run one 16x16 work group per
// CPU tile of 160 pixels, 128 apart.
static gboolean process_amaze_slang(dt_dev_pixelpipe_iop_t *piece,
                                    dt_slang_gpu_image_t *dev_in,
                                    dt_slang_gpu_image_t *dev_out,
                                    const int width,
                                    const int height,
                                    const uint32_t filters)
{
  const int pw = width + 32;
  const int ph = height + 32;
  const size_t bsize = sizeof(float) * pw * ph;
  const float clip_pt = dt_iop_get_processed_minimum(piece);
  const float clip_pt8 = 0.8f * clip_pt;
  const int tw = 16 * ((pw - 16 + 127) / 128);
  const int th = 16 * ((ph - 16 + 127) / 128);
  const int m8 = 8, m10 = 10;

  enum { NBUF = 12 };
  // the module's memory estimate doesn't know these, so a large image falls
  // back to the CPU here rather than running out of device memory
  size_t available = 0;
  int max_dim = 0;
  dt_slang_gpu_memory_info(&available, &max_dim);
  if(NBUF * bsize > available)
  {
    dt_print(DT_DEBUG_SLANG, "[demosaic] AMaZE needs %zu MB of device memory, %zu MB available",
             NBUF * bsize / DT_MEGA, available / DT_MEGA);
    return FALSE;
  }

  dt_slang_gpu_buffer_t *b[NBUF] = { NULL };
  gboolean ok = TRUE;
  for(int k = 0; k < NBUF && ok; k++)
    ok = (b[k] = dt_slang_gpu_buffer_new(bsize, NULL)) != NULL;

  dt_slang_gpu_buffer_t *cfa = b[0];
  // up to the hvwt pass
  dt_slang_gpu_buffer_t *delhvsqsum = b[1], *vcd = b[2], *hcd = b[3], *vcdalt = b[4], *hcdalt = b[5];
  dt_slang_gpu_buffer_t *dgintv = b[6], *dginth = b[7], *vcd2 = b[8], *hcd2 = b[9];
  dt_slang_gpu_buffer_t *hvwt = b[10], *nyquist = b[11];
  // after it, in the buffers that are free by then
  dt_slang_gpu_buffer_t *nyquist2 = b[2], *hvwt2 = b[3], *dgrb = b[4], *green = b[5];
  dt_slang_gpu_buffer_t *dgrb2h = b[6], *dgrb2v = b[7];
  dt_slang_gpu_buffer_t *rbm = b[8], *rbp = b[1], *pmwt = b[11], *pmwt2 = b[10], *rbint = b[2];
  dt_slang_gpu_buffer_t *dgr = b[6], *dgb = b[7];

  ok = ok
    && !dt_slang_gpu_run_2d_args("demosaic_amaze", "amaze_init", pw, ph,
                                 SLIMG_DEV(dev_in), SLBUF_DEV(cfa),
                                 SLARG(width), SLARG(height), SLARG(pw), SLARG(ph))
    && !dt_slang_gpu_run_2d_args("demosaic_amaze", "amaze_grad", pw, ph,
                                 SLBUF_DEV(cfa), SLBUF_DEV(delhvsqsum), SLARG(pw), SLARG(ph))
    && !dt_slang_gpu_run_2d_args("demosaic_amaze", "amaze_cd", pw, ph,
                                 SLBUF_DEV(cfa), SLBUF_DEV(vcd), SLBUF_DEV(hcd), SLBUF_DEV(vcdalt),
                                 SLBUF_DEV(hcdalt), SLBUF_DEV(dgintv), SLBUF_DEV(dginth),
                                 SLARG(pw), SLARG(ph), SLARG(filters), SLARG(clip_pt8))
    && !dt_slang_gpu_run_2d_args("demosaic_amaze", "amaze_cd_bound", tw, th,
                                 SLBUF_DEV(cfa), SLBUF_DEV(vcd), SLBUF_DEV(hcd), SLBUF_DEV(vcdalt),
                                 SLBUF_DEV(hcdalt), SLBUF_DEV(vcd2), SLBUF_DEV(hcd2),
                                 SLARG(pw), SLARG(ph), SLARG(filters), SLARG(clip_pt))
    && !dt_slang_gpu_run_2d_args("demosaic_amaze", "amaze_hvwt", pw, ph,
                                 SLBUF_DEV(cfa), SLBUF_DEV(vcd2), SLBUF_DEV(hcd2), SLBUF_DEV(dgintv),
                                 SLBUF_DEV(dginth), SLBUF_DEV(delhvsqsum),
                                 SLBUF_DEV(hvwt), SLBUF_DEV(nyquist),
                                 SLARG(pw), SLARG(ph), SLARG(filters))
    && !dt_slang_gpu_run_2d_args("demosaic_amaze", "amaze_nyquist2", pw, ph,
                                 SLBUF_DEV(nyquist), SLBUF_DEV(nyquist2), SLARG(pw), SLARG(ph), SLARG(filters))
    && !dt_slang_gpu_run_2d_args("demosaic_amaze", "amaze_area", pw, ph,
                                 SLBUF_DEV(cfa), SLBUF_DEV(nyquist2), SLBUF_DEV(hvwt),
                                 SLARG(pw), SLARG(ph), SLARG(filters))
    && !dt_slang_gpu_run_2d_args("demosaic_amaze", "amaze_vote", tw, th,
                                 SLBUF_DEV(hvwt), SLBUF_DEV(hvwt2), SLARG(pw), SLARG(ph), SLARG(filters), SLARG(m8))
    && !dt_slang_gpu_run_2d_args("demosaic_amaze", "amaze_green", pw, ph,
                                 SLBUF_DEV(cfa), SLBUF_DEV(vcd2), SLBUF_DEV(hcd2), SLBUF_DEV(hvwt2),
                                 SLBUF_DEV(nyquist2), SLBUF_DEV(dgrb), SLBUF_DEV(green),
                                 SLBUF_DEV(dgrb2h), SLBUF_DEV(dgrb2v),
                                 SLARG(pw), SLARG(ph), SLARG(filters))
    && !dt_slang_gpu_run_2d_args("demosaic_amaze", "amaze_nyquist_refine", pw, ph,
                                 SLBUF_DEV(cfa), SLBUF_DEV(vcd2), SLBUF_DEV(hcd2), SLBUF_DEV(nyquist2),
                                 SLBUF_DEV(dgrb2h), SLBUF_DEV(dgrb2v), SLBUF_DEV(dgrb), SLBUF_DEV(green),
                                 SLARG(pw), SLARG(ph), SLARG(filters))
    && !dt_slang_gpu_run_2d_args("demosaic_amaze", "amaze_diag", pw, ph,
                                 SLBUF_DEV(cfa), SLBUF_DEV(rbm), SLBUF_DEV(rbp), SLBUF_DEV(pmwt),
                                 SLARG(pw), SLARG(ph), SLARG(filters), SLARG(clip_pt))
    && !dt_slang_gpu_run_2d_args("demosaic_amaze", "amaze_vote", tw, th,
                                 SLBUF_DEV(pmwt), SLBUF_DEV(pmwt2), SLARG(pw), SLARG(ph), SLARG(filters), SLARG(m10))
    && !dt_slang_gpu_run_2d_args("demosaic_amaze", "amaze_rbint", pw, ph,
                                 SLBUF_DEV(cfa), SLBUF_DEV(rbm), SLBUF_DEV(rbp), SLBUF_DEV(pmwt2),
                                 SLBUF_DEV(rbint), SLARG(pw), SLARG(ph), SLARG(filters))
    && !dt_slang_gpu_run_2d_args("demosaic_amaze", "amaze_green_diag", pw, ph,
                                 SLBUF_DEV(cfa), SLBUF_DEV(hvwt2), SLBUF_DEV(pmwt2), SLBUF_DEV(rbint),
                                 SLBUF_DEV(dgrb), SLBUF_DEV(green),
                                 SLARG(pw), SLARG(ph), SLARG(filters), SLARG(clip_pt))
    && !dt_slang_gpu_run_2d_args("demosaic_amaze", "amaze_chroma", pw, ph,
                                 SLBUF_DEV(dgrb), SLBUF_DEV(dgr), SLBUF_DEV(dgb),
                                 SLARG(pw), SLARG(ph), SLARG(filters))
    && !dt_slang_gpu_run_2d_args("demosaic_amaze", "amaze_output", width, height,
                                 SLBUF_DEV(cfa), SLBUF_DEV(green), SLBUF_DEV(hvwt2), SLBUF_DEV(dgr),
                                 SLBUF_DEV(dgb), SLIMG_DEV(dev_out),
                                 SLARG(width), SLARG(height), SLARG(pw), SLARG(filters));

  for(int k = 0; k < NBUF; k++) dt_slang_gpu_buffer_free(b[k]);
  return ok;
}

#endif

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
