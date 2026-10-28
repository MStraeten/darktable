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

// Slang port of the kernels in ../basic.cl that the default scene-referred
// pipe runs. The others follow with their modules.

import color_conversion;
import rgb_norms;

// Metal drops buffers and images a kernel does not read, which would shift
// every later binding (see src/common/slang_gpu_metal.m). A kernel that ignores
// one of its OpenCL arguments reads it behind a condition that never holds.
void _keep(StructuredBuffer<float> b, int never, ImageOut out)
{
  if(never < 0) out.write(int2(0), float4(b[0]));
}

/* kernel for the exposure plugin. should work transparently with float4 and float image2d. */
[shader("compute")]
[numthreads(16, 16, 1)]
void exposure(uint3 gid : SV_DispatchThreadID,
              uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
              uniform float black, uniform float scale)
{
  const int x = gid.x;
  const int y = gid.y;

  if(x >= width || y >= height) return;
  float4 pixel = in.Areadpixel(x, y);
  pixel.xyz = ((pixel - black) * scale).xyz;
  out.write(int2(x, y), pixel);
}

float lerp_lookup_unbounded0(ImageIn lut, float x, StructuredBuffer<float> a, int c)
{
  // in case the tone curve is marked as linear, return the fast
  // path to linear unbounded (does not clip x at 1)
  if(a[3 * c] >= 0.0f)
  {
    if(x < 1.0f)
    {
      const float ft = clamp(x * float(0xffff), 0.0f, float(0xffff));
      const int t = ft < 0xfffe ? int(ft) : 0xfffe;
      const float f = ft - t;
      const int2 p1 = int2((t & 0xff), (t >> 8));
      const int2 p2 = int2(((t + 1) & 0xff), ((t + 1) >> 8));
      const float l1 = lut.read(p1).x;
      const float l2 = lut.read(p2).x;
      return l1 * (1.0f - f) + l2 * f;
    }
    else return a[3 * c + 1] * dtcl_pow(x * a[3 * c], a[3 * c + 2]);
  }
  else return x;
}

float4 _lookup_hsm(float4 hsv, int hue_div, int sat_div, int val_div, StructuredBuffer<float> hsm)
{
  const float h = (hsv.x - floor(hsv.x)) * float(hue_div);
  const float s = clipf(hsv.y) * float(sat_div-1);
  const float v = clipf(hsv.z) * float(val_div-1);
  const int hi[2] = { min(int(h), hue_div-1), (min(int(h), hue_div-1)+1) % hue_div };
  const int si[2] = { min(int(s), sat_div-1), min(int(s)+1, sat_div-1) };
  const int vi[2] = { min(int(v), val_div-1), min(int(v)+1, val_div-1) };
  const float hf = clipf(h - float(hi[0]));
  const float sf = clipf(s - float(si[0]));
  const float vf = clipf(v - float(vi[0]));

  float4 correction = float4(0.0f);
  for(int z = 0; z < 2; z++)
    for(int y = 0; y < 2; y++)
      for(int x = 0; x < 2; x++)
      {
        // DNG stores saturation first, then hue, then value, with three floats per cell
        const int index = 3 * ((vi[z] * hue_div + hi[y]) * sat_div + si[x]);
        const float weight = (z != 0 ? vf : 1.0f - vf)
                           * (y != 0 ? hf : 1.0f - hf)
                           * (x != 0 ? sf : 1.0f - sf);
        correction.x += weight * hsm[index + 0];
        correction.y += weight * hsm[index + 1];
        correction.z += weight * hsm[index + 2];
      }
  return correction;
}

float _srgb_to_linear(float x)
{
  return x <= 0.04045f ? x / 12.92f : dtcl_pow((x + 0.055f) / 1.055f, 2.4f);
}
float _linear_to_srgb(float x)
{
  return x <= 0.0031308f ? 12.92f * x : 1.055f * dtcl_pow(x, 1.0f / 2.4f) - 0.055f;
}

// colorin_unbound and colorin_clipping differ only in how camera RGB becomes XYZ
float4 _colorin(ImageIn in, int x, int y, StructuredBuffer<float> cmat, StructuredBuffer<float> lmat,
                ImageIn lutr, ImageIn lutg, ImageIn lutb, int blue_mapping, StructuredBuffer<float> a,
                StructuredBuffer<float> corr, float scale, int hue_div, int sat_div, int val_div,
                StructuredBuffer<float> dev_hsm, int encoding, int use_hsm, int hdr_transfer, bool clipping)
{
  const float4 corval = float4(corr[0], corr[1], corr[2], corr[3]);
  float4 pixel = corval * in.readpixel(x, y);
  const float alpha = pixel.w;

  float3 cam;
  if(hdr_transfer != 0)
  {
    cam = pixel.xyz;
    dt_hdr_decode(cam, hdr_transfer);
  }
  else
  {
    cam[0] = lerp_lookup_unbounded0(lutr, pixel.x, a, 0);
    cam[1] = lerp_lookup_unbounded0(lutg, pixel.y, a, 1);
    cam[2] = lerp_lookup_unbounded0(lutb, pixel.z, a, 2);
  }

  if(blue_mapping != 0)
  {
    const float YY = cam[0] + cam[1] + cam[2];
    if(YY > 0.0f)
    {
      // manual gamut mapping. these values cause trouble when converting back from Lab to sRGB:
      const float zz = cam[2] / YY;
      // lower amount and higher bound_z make the effect smaller.
      // the effect is weakened the darker input values are, saturating at bound_Y
      const float bound_z = 0.5f, bound_Y = 0.8f;
      const float amount = 0.11f;
      if (zz > bound_z)
      {
        const float t = (zz - bound_z) / (1.0f - bound_z) * min(1.0f, YY / bound_Y);
        cam[1] += t * amount;
        cam[2] -= t * amount;
      }
    }
  }

  float XYZ[3];
  if(clipping)
  {
    // convert camera to RGB using the first color matrix
    float RGB[3];
    for(int j=0;j<3;j++)
    {
      RGB[j] = 0.0f;
      for(int i=0;i<3;i++) RGB[j] += cmat[3*j+i] * cam[i];
    }

    // clamp at this stage
    for(int i=0; i<3; i++) RGB[i] = clipf(RGB[i]);

    // convert clipped RGB to XYZ
    for(int j=0;j<3;j++)
    {
      XYZ[j] = 0.0f;
      for(int i=0;i<3;i++) XYZ[j] += lmat[3*j+i] * RGB[i];
    }
  }
  else
  {
    // now convert camera to XYZ using the color matrix
    for(int j=0;j<3;j++)
    {
      XYZ[j] = 0.0f;
      for(int i=0;i<3;i++) XYZ[j] += cmat[3*j+i] * cam[i];
    }
  }

  float4 xyz = float4(XYZ[0], XYZ[1], XYZ[2], 0.0f);
  pixel.xyz = XYZ_to_Lab(xyz).xyz;
  if(use_hsm != 0)
  {
    float4 prophoto_rgb = Lab_to_prophotorgb(pixel) / scale;
    float4 hsv = RGB_2_HSV(prophoto_rgb);
    // Encode valueV for table lookup if needed
    float v_encoded = hsv.z;
    if(encoding == 1) // sRGB
      v_encoded = _linear_to_srgb(clipf(v_encoded));

    // Use encoded value for table lookup (need to temporarily set hsv[2])
    float4 hsv_for_lookup = hsv;
    hsv_for_lookup.z = v_encoded;
    float4 correction = _lookup_hsm(hsv_for_lookup, hue_div, sat_div, val_div, dev_hsm);

    hsv.x += correction.x / 360.0f;
    hsv.x -= floor(hsv.x); // stay within the 360° hue circle
    hsv.y = clipf(hsv.y * correction.y);

    float v_corrected = v_encoded * correction.z;
    if(encoding == 1)  // sRGB
      v_corrected = _srgb_to_linear(v_corrected);
    hsv.z = clipf(v_corrected);
    prophoto_rgb = HSV_2_RGB(hsv) * scale;
    pixel = prophotorgb_to_Lab(prophoto_rgb);
  }
  pixel.w = alpha;
  return pixel;
}

/* kernel for the plugin colorin: unbound processing */
[shader("compute")]
[numthreads(16, 16, 1)]
void colorin_unbound(uint3 gid : SV_DispatchThreadID,
                     uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
                     uniform StructuredBuffer<float> cmat, uniform StructuredBuffer<float> lmat,
                     uniform ImageIn lutr, uniform ImageIn lutg, uniform ImageIn lutb,
                     uniform int blue_mapping, uniform StructuredBuffer<float> a,
                     uniform StructuredBuffer<float> corr, uniform float scale,
                     uniform int hue_div, uniform int sat_div, uniform int val_div,
                     uniform StructuredBuffer<float> dev_hsm, uniform int encoding, uniform int use_hsm,
                     uniform int hdr_transfer)
{
  const int x = gid.x;
  const int y = gid.y;

  if(x >= width || y >= height) return;

  _keep(lmat, width, out);
  out.write(int2(x, y), _colorin(in, x, y, cmat, lmat, lutr, lutg, lutb, blue_mapping, a, corr, scale,
                                 hue_div, sat_div, val_div, dev_hsm, encoding, use_hsm, hdr_transfer, false));
}

/* kernel for the plugin colorin: with clipping */
[shader("compute")]
[numthreads(16, 16, 1)]
void colorin_clipping(uint3 gid : SV_DispatchThreadID,
                      uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
                      uniform StructuredBuffer<float> cmat, uniform StructuredBuffer<float> lmat,
                      uniform ImageIn lutr, uniform ImageIn lutg, uniform ImageIn lutb,
                      uniform int blue_mapping, uniform StructuredBuffer<float> a,
                      uniform StructuredBuffer<float> corr, uniform float scale,
                      uniform int hue_div, uniform int sat_div, uniform int val_div,
                      uniform StructuredBuffer<float> dev_hsm, uniform int encoding, uniform int use_hsm,
                      uniform int hdr_transfer)
{
  const int x = gid.x;
  const int y = gid.y;

  if(x >= width || y >= height) return;

  out.write(int2(x, y), _colorin(in, x, y, cmat, lmat, lutr, lutg, lutb, blue_mapping, a, corr, scale,
                                 hue_div, sat_div, val_div, dev_hsm, encoding, use_hsm, hdr_transfer, true));
}

/* kernel for the plugin colorout, fast matrix + shaper path only */
[shader("compute")]
[numthreads(16, 16, 1)]
void colorout(uint3 gid : SV_DispatchThreadID,
              uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
              uniform StructuredBuffer<float> mat,
              uniform ImageIn lutr, uniform ImageIn lutg, uniform ImageIn lutb,
              uniform StructuredBuffer<float> a, uniform int hdr_transfer)
{
  const int x = gid.x;
  const int y = gid.y;

  if(x >= width || y >= height) return;

  float4 pixel = in.Areadpixel(x, y);
  const float4 xyz = Lab_to_XYZ(pixel);
  const float XYZ[3] = { xyz.x, xyz.y, xyz.z };
  float3 rgb;
  for(int i=0;i<3;i++)
  {
    rgb[i] = 0.0f;
    for(int j=0;j<3;j++) rgb[i] += mat[3*i+j]*XYZ[j];
  }
  dt_hdr_encode(rgb, hdr_transfer);
  pixel.x = lerp_lookup_unbounded0(lutr, rgb[0], a, 0);
  pixel.y = lerp_lookup_unbounded0(lutg, rgb[1], a, 1);
  pixel.z = lerp_lookup_unbounded0(lutb, rgb[2], a, 2);
  out.write(int2(x, y), pixel);
}

// the raw stage: white balance and highlight reconstruction

// like _keep(), for the xtrans pattern of kernels that run on Bayer data
void _keep_xtrans(StructuredBuffer<uint> xtrans, int never, ImageOut out)
{
  if(never < 0) out.write(int2(0), float4(float(xtrans[0])));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void whitebalance_1f(uint3 gid : SV_DispatchThreadID,
                     uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
                     uniform StructuredBuffer<float> coeffs, uniform uint filters,
                     uniform StructuredBuffer<uint> xtrans)
{
  const int x = gid.x;
  const int y = gid.y;
  _keep_xtrans(xtrans, width, out);
  if(x >= width || y >= height) return;
  const float pixel = in.readsingle(x, y);
  out.write(int2(x, y), float4(pixel * coeffs[FC(y, x, filters)]));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void whitebalance_1f_xtrans(uint3 gid : SV_DispatchThreadID,
                            uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
                            uniform StructuredBuffer<float> coeffs, uniform uint filters,
                            uniform StructuredBuffer<uint> xtrans)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;
  const float pixel = in.readsingle(x, y);
  out.write(int2(x, y), float4(pixel * coeffs[FCxtrans(y, x, xtrans)]));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void whitebalance_4f(uint3 gid : SV_DispatchThreadID,
                     uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
                     uniform StructuredBuffer<float> coeffs, uniform uint filters,
                     uniform StructuredBuffer<uint> xtrans)
{
  const int x = gid.x;
  const int y = gid.y;
  _keep_xtrans(xtrans, width, out);
  if(x >= width || y >= height) return;
  const float4 pixel = in.Areadpixel(x, y);
  out.write(int2(x, y), float4(pixel.x * coeffs[0], pixel.y * coeffs[1], pixel.z * coeffs[2], pixel.w));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void highlights_1f_clip(uint3 gid : SV_DispatchThreadID,
                        uniform ImageIn in, uniform ImageOut out,
                        uniform int iwidth, uniform int iheight,
                        uniform int owidth, uniform int oheight,
                        uniform float4 clip, uniform int dx, uniform int dy,
                        uniform int filters, uniform StructuredBuffer<uint> xtrans)
{
  const int x = gid.x;
  const int y = gid.y;

  if(x >= owidth || y >= oheight) return;

  const int irow = y + dy;
  const int icol = x + dx;
  float pixel = 0.0f;
  const float clips[4] = { clip.x, clip.y, clip.z, clip.w };
  if((icol >= 0) && (irow >= 0) && (irow < iheight) && (icol < iwidth))
  {
    const int color = fcol(irow, icol, uint(filters), xtrans);
    pixel = in.Areadsingle(icol, irow);
    pixel = min(clips[color], pixel);
  }
  out.write(int2(x, y), float4(pixel));
}

// the masks of the opposed method are byte arrays, as in OpenCL, packed four
// to a word. Threads share words, so the masks start zeroed and bits are or-ed in.
int _mask_get(RWStructuredBuffer<uint> mask, int i)
{
  return int((mask[i >> 2] >> uint((i & 3) * 8)) & 0xffu);
}

void _mask_set(RWStructuredBuffer<uint> mask, int i, int val)
{
  if(val != 0) InterlockedOr(mask[i >> 2], uint(val & 0xff) << uint((i & 3) * 8));
}

// OpenCL leaves the masks as allocated; here they are zeroed first, as the
// kernels below only set bits. The words are laid out in rows of width.
[shader("compute")]
[numthreads(16, 16, 1)]
void highlights_clearmask(uint3 gid : SV_DispatchThreadID,
                          uniform RWStructuredBuffer<uint> mask, uniform int width, uniform int words)
{
  const int i = gid.y * width + gid.x;
  if(gid.x < width && i < words) mask[i] = 0u;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void highlights_initmask(uint3 gid : SV_DispatchThreadID,
                         uniform ImageIn in, uniform RWStructuredBuffer<uint> inmask,
                         uniform int msize, uniform int mwidth, uniform int mheight,
                         uniform int iwidth, uniform int iheight,
                         uniform uint filters, uniform StructuredBuffer<uint> xtrans,
                         uniform StructuredBuffer<float> clips)
{
  const int mcol = gid.x;
  const int mrow = gid.y;

  if(mcol >= mwidth || mrow >= mheight)
    return;

  const int mdx = mrow * mwidth + mcol;
  int mbuff[3] = { 0, 0, 0 };
  const int irow = 3*mrow;
  const int icol = 3*mcol;

  for(int y = irow; y < irow + 3; y++)
  {
    for(int x = icol; x < icol + 3; x++)
    {
      if(y < iheight && x < iwidth)
      {
        const int color = fcol(y, x, filters, xtrans);
        const float val = in.readsingle(x, y);
        mbuff[color] += (val >= clips[color]) ? 1 : 0;
      }
    }
  }

  for(int c = 0; c < 3; c++)
    _mask_set(inmask, c*msize + mdx, mbuff[c] != 0 ? 1 : 0);
}

int _mask_dilated(RWStructuredBuffer<uint> m, int i, int w1)
{
  if(_mask_get(m, i) != 0)
    return 1;

  if((_mask_get(m, i-w1-1) | _mask_get(m, i-w1) | _mask_get(m, i-w1+1) | _mask_get(m, i-1) | _mask_get(m, i+1)
      | _mask_get(m, i+w1-1) | _mask_get(m, i+w1) | _mask_get(m, i+w1+1)) != 0)
    return 1;

  const int w2 = 2*w1;
  const int w3 = 3*w1;
  return (_mask_get(m, i-w3-2) | _mask_get(m, i-w3-1) | _mask_get(m, i-w3)   | _mask_get(m, i-w3+1) | _mask_get(m, i-w3+2) |
          _mask_get(m, i-w2-3) | _mask_get(m, i-w2-2) | _mask_get(m, i-w2-1) | _mask_get(m, i-w2)   | _mask_get(m, i-w2+1) | _mask_get(m, i-w2+2) | _mask_get(m, i-w2+3) |
          _mask_get(m, i-w1-3) | _mask_get(m, i-w1-2) | _mask_get(m, i-w1+2) | _mask_get(m, i-w1+3) |
          _mask_get(m, i-3)    | _mask_get(m, i-2)    | _mask_get(m, i+2)     | _mask_get(m, i+3)     |
          _mask_get(m, i+w1-3) | _mask_get(m, i+w1-2) | _mask_get(m, i+w1+2) | _mask_get(m, i+w1+3) |
          _mask_get(m, i+w2-3) | _mask_get(m, i+w2-2) | _mask_get(m, i+w2-1) | _mask_get(m, i+w2)   | _mask_get(m, i+w2+1) | _mask_get(m, i+w2+2) | _mask_get(m, i+w2+3) |
          _mask_get(m, i+w3-2) | _mask_get(m, i+w3-1) | _mask_get(m, i+w3)   | _mask_get(m, i+w3+1) | _mask_get(m, i+w3+2)) != 0 ? 1 : 0;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void highlights_dilatemask(uint3 gid : SV_DispatchThreadID,
                           uniform RWStructuredBuffer<uint> in, uniform RWStructuredBuffer<uint> out,
                           uniform int msize, uniform int mwidth, uniform int mheight)
{
  const int col = gid.x;
  const int row = gid.y;

  if(col >= mwidth || row >= mheight)
    return;

  const int mx = row * mwidth + col;
  const bool safe = col >= 3 && row >= 3 && col < mwidth - 3 && row < mheight - 3;

  for(int c = 0; c < 3; c++)
  {
    const int i = mx + c * msize;
    _mask_set(out, i, safe ? _mask_dilated(in, i, mwidth) : _mask_get(in, i));
  }
}

float _calc_refavg(ImageIn in, StructuredBuffer<uint> xtrans, uint filters,
                   int row, int col, int maxrow, int maxcol, StructuredBuffer<float> correction)
{
  float mean[3] = { 0.0f, 0.0f, 0.0f };
  float cnt[3]  = { 0.0f, 0.0f, 0.0f };

  for(int dy = max(0, row - 1); dy < min(maxrow - 1, row + 2); dy++)
  {
    for(int dx = max(0, col - 1); dx < min(maxcol - 1, col + 2); dx++)
    {
      const float val = max(0.0f, in.Areadsingle(dx, dy));
      const int c = fcol(dy, dx, filters, xtrans);
      mean[c] += val;
      cnt[c] += 1.0f;
    }
  }

  for(int c = 0; c < 3; c++)
    mean[c] = (cnt[c] > 0.0f) ? dt_cbrt((correction[c] * mean[c]) / cnt[c]) : 0.0f;

  const float croot_refavg[3] = { 0.5f * (mean[1] + mean[2]), 0.5f * (mean[0] + mean[2]), 0.5f * (mean[0] + mean[1]) };
  const int color = fcol(row, col, filters, xtrans);
  return fcube(croot_refavg[color]);
}

// one row per work item, dispatched as height x 1
[shader("compute")]
[numthreads(16, 16, 1)]
void highlights_chroma(uint3 gid : SV_DispatchThreadID,
                       uniform ImageIn in, uniform RWStructuredBuffer<uint> mask, uniform RWStructuredBuffer<float> accu,
                       uniform int width, uniform int height, uniform int msize, uniform int mwidth,
                       uniform uint filters, uniform StructuredBuffer<uint> xtrans,
                       uniform StructuredBuffer<float> clips, uniform StructuredBuffer<float> correction)
{
  const int row = gid.y * 16 + gid.x;
  if(row >= height) return;

  float sum[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
  float cnt[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

  float clipped = 0.0f;
  for(int col = 0; col < width; col++)
  {
    const int color = fcol(row, col, filters, xtrans);
    const float inval = in.Areadsingle(col, row);
    const int px = color * msize + (row/3) * mwidth + col/3;
    const bool masked = _mask_get(mask, px) != 0;
    if(masked && (inval > 0.2f*clips[color]) && (inval < clips[color]))
    {
      const float ref = _calc_refavg(in, xtrans, filters, row, col, height, width, correction);
      sum[color] += inval - ref;
      cnt[color] += 1.0f;
    }
    if(masked) clipped += 1.0f;
  }

  for(int c = 0; c < 3; c++)
  {
    accu[row*8 + 2*c] = sum[c];
    accu[row*8 + 2*c +1] = cnt[c];
  }
  accu[row*8 + 6] = clipped;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void highlights_opposed(uint3 gid : SV_DispatchThreadID,
                        uniform ImageIn in, uniform ImageOut out,
                        uniform int owidth, uniform int oheight, uniform int iwidth, uniform int iheight,
                        uniform int dx, uniform int dy,
                        uniform uint filters, uniform StructuredBuffer<uint> xtrans,
                        uniform StructuredBuffer<float> clips, uniform StructuredBuffer<float> chroma,
                        uniform StructuredBuffer<float> correction)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= owidth || y >= oheight) return;

  const int irow = y + dy;
  const int icol = x + dx;
  float val = 0.0f;

  if((icol >= 0) && (icol < iwidth) && (irow >= 0) && (irow < iheight))
  {
    val = in.Areadsingle(icol, irow);
    const int color = fcol(irow, icol, filters, xtrans);
    if(val >= clips[color])
    {
      const float ref = _calc_refavg(in, xtrans, filters, irow, icol, iheight, iwidth, correction);
      val = max(val, ref + chroma[color]);
    }
  }
  out.write(int2(x, y), float4(val));
}

// interpolation_resample in ../basic.cl splits each output pixel's vertical
// taps across a work group and sums them in local memory. Here one work item
// does all taps of its pixel, so the sums differ in the last bits.
[shader("compute")]
[numthreads(16, 16, 1)]
void interpolation_resample(uint3 gid : SV_DispatchThreadID,
                            uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
                            uniform StructuredBuffer<int> hmeta, uniform StructuredBuffer<int> vmeta,
                            uniform StructuredBuffer<int> hlength, uniform StructuredBuffer<int> vlength,
                            uniform StructuredBuffer<int> hindex, uniform StructuredBuffer<int> vindex,
                            uniform StructuredBuffer<float> hkernel, uniform StructuredBuffer<float> vkernel,
                            uniform int htaps, uniform int vtaps)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const int hl = hlength[hmeta[x * 3]];
  const int hkidx = hmeta[x * 3 + 1];
  const int hiidx = hmeta[x * 3 + 2];
  const int vl = vlength[vmeta[y * 3]];
  const int vkidx = vmeta[y * 3 + 1];
  const int viidx = vmeta[y * 3 + 2];

  float4 pixel = float4(0.0f);
  for(int iy = 0; iy < vl; iy++)
  {
    const int yy = vindex[viidx + iy];
    float4 hpixel = float4(0.0f);
    for(int ix = 0; ix < hl; ix++)
      hpixel += in.readpixel(hindex[hiidx + ix], yy) * hkernel[hkidx + ix];
    pixel += hpixel * vkernel[vkidx + iy];
  }
  out.write(int2(x, y), pixel);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void interpolation_copy(uint3 gid : SV_DispatchThreadID,
                        uniform ImageIn dev_in, uniform ImageOut dev_out,
                        uniform int owidth, uniform int oheight, uniform int iwidth, uniform int iheight,
                        uniform int dx, uniform int dy)
{
  const int ocol = gid.x;
  const int orow = gid.y;
  if(ocol >= owidth || orow >= oheight) return;

  float4 pix = float4(0.0f);
  const int irow = orow + dy;
  const int icol = ocol + dx;
  if(irow < iheight && irow >= 0 && icol < iwidth && icol >= 0)
    pix = dev_in.Areadpixel(icol, irow);
  dev_out.write(int2(ocol, orow), pix);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void flip(uint3 gid : SV_DispatchThreadID,
          uniform ImageIn in, uniform ImageOut out,
          uniform int width, uniform int height, uniform int owidth, uniform int oheight,
          uniform int orientation)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  // ORIENTATION_FLIP_X = 2
  int ox = (orientation & 2) != 0 ? width - x - 1 : x;

  // ORIENTATION_FLIP_Y = 1
  int oy = (orientation & 1) != 0 ? height - y - 1 : y;

  // ORIENTATION_SWAP_XY = 4
  if((orientation & 4) != 0)
  {
    const int tmp = ox;
    ox = oy;
    oy = tmp;
  }

  if(ox >= 0 && oy >= 0 && ox < owidth && oy < oheight)
  {
    const float4 pixel = in.readpixel(x, y);
    out.write(int2(ox, oy), pixel);
  }
}

int BL(int row, int col)
{
  return ((row & 1) << 1) + (col & 1);
}

// fma(a, b, c) for the rawprepare division, whose result must match the CPU
// bit for bit: highlight reconstruction tells clipped pixels by comparing
// with 1.0. Vulkan allows fma to round twice, as a multiply and an add, and
// lavapipe does, so there the product is made exact with Dekker's algorithm.
// Metal's fma is fused.
float fma_exact(float a, float b, float c)
{
  __target_switch
  {
  case metal:
    return fma(a, b, c);
  default:
    precise const float p = a * b;
    precise const float ca = 4097.0f * a;
    precise const float ah = ca - (ca - a);
    precise const float al = a - ah;
    precise const float cb = 4097.0f * b;
    precise const float bh = cb - (cb - b);
    precise const float bl = b - bh;
    precise const float e = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
    precise const float s = p + c;
    precise const float bs = s - p;
    precise const float es = (p - (s - bs)) + (c - bs);
    precise const float r = s + (es + e);
    return r;
  }
}

// rawprepare_1f reads a uint16 image, which the Slang path never makes: the
// pipe converts uint16 input to float with u16_to_float, so this kernel
// serves both
[shader("compute")]
[numthreads(16, 16, 1)]
void rawprepare_1f_unnormalized(uint3 gid : SV_DispatchThreadID,
                                uniform ImageIn in, uniform ImageOut out,
                                uniform int width, uniform int height,
                                uniform int cx, uniform int cy,
                                uniform StructuredBuffer<float> sub, uniform StructuredBuffer<float> div,
                                uniform StructuredBuffer<float> rdiv,
                                uniform int rx, uniform int ry)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float pix = in.read(int2(x + cx, y + cy)).x;
  const int id = BL(ry + cy + y, rx + cx + x);
  const float n = pix - sub[id];
  const float q = n * rdiv[id];
  const float rem = fma_exact(-q, div[id], n);
  const float oval = fma_exact(rem, rdiv[id], q);

  out.write(int2(x, y), float4(oval));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void rawprepare_4f(uint3 gid : SV_DispatchThreadID,
                   uniform ImageIn in, uniform ImageOut out,
                   uniform int width, uniform int height,
                   uniform int cx, uniform int cy,
                   uniform StructuredBuffer<float> sub, uniform StructuredBuffer<float> div,
                   uniform StructuredBuffer<float> rdiv,
                   uniform int rx, uniform int ry)
{
  const int x = gid.x;
  const int y = gid.y;
  // rx and ry are unused, as in OpenCL, but Metal drops parameters nothing reads
  if(width < 0) out.write(int2(0), float4(float(rx + ry)));
  if(x >= width || y >= height) return;

  const float4 sub4 = float4(sub[0], sub[1], sub[2], sub[3]);
  const float4 div4 = float4(div[0], div[1], div[2], div[3]);
  const float4 rdiv4 = float4(rdiv[0], rdiv[1], rdiv[2], rdiv[3]);
  const float4 pix = in.readpixel(x + cx, y + cy);
  const float4 n = pix - sub4;
  const float4 q = n * rdiv4;
  float4 oval = float4(0.0f);
  for(int c = 0; c < 3; c++)
  {
    const float rem = fma_exact(-q[c], div4[c], n[c]);
    oval[c] = fma_exact(rem, rdiv4[c], q[c]);
  }
  out.write(int2(x, y), oval);
}

// uint16 raw data, two values to a uint, to a 1-channel float image. There is
// no OpenCL counterpart: OpenCL reads the uint16 image directly.
[shader("compute")]
[numthreads(16, 16, 1)]
void u16_to_float(uint3 gid : SV_DispatchThreadID,
                  uniform StructuredBuffer<uint> in, uniform ImageOut out,
                  uniform int width, uniform int height)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const uint k = uint(y) * uint(width) + uint(x);
  const uint pair = in[k >> 1];
  const uint v = (k & 1) != 0 ? pair >> 16 : pair & 0xffff;
  out.write(int2(x, y), float4(float(v)));
}

// ---------------------------------------------------------------------------
// lens correction

float interpolation_func_bicubic(float t)
{
  t = abs(t);
  return (t >= 2.0f) ? 0.0f : ((t > 1.0f) ? (0.5f * (t * (-t * t + 5.0f * t - 8.0f) + 4.0f)) : (0.5f * (t * (3.0f * t * t - 5.0f * t) + 2.0f)));
}

static const float DT_LANCZOS_EPSILON = 1e-9f;

// keep in sync with the copy in src/common/math.h
float sinf_fast(float t)
{
  const float a = 4.0f / (M_PI_F * M_PI_F);
  const float p = 0.225f;

  t = a * t * (M_PI_F - abs(t));

  return p * (t * abs(t) - t) + t;
}

float interpolation_func_lanczos(float width, float t)
{
  // a value for sinf(pi.t) in [-pi pi] for which the value will be correct
  const int a = int(t);
  const float r = t - float(a);

  // the correct sign for sinf(pi.r)
  const float sign = asfloat(uint((a & 1) << 31) | 0x3f800000u);

  return (DT_LANCZOS_EPSILON + width * sign * sinf_fast(M_PI_F * r) * sinf_fast(M_PI_F * t / width))
         / (DT_LANCZOS_EPSILON + M_PI_F * M_PI_F * t * t);
}

// the taps of the warp interpolators, see compute_upsampling_taps() in ../basic.cl.
// Each is computed four at a time, so 3 wide filters fill all eight.
static const int MAX_KERNEL_REQ = 8;

float maketaps_bilinear(out float taps[MAX_KERNEL_REQ], const int num_taps, const float first_tap, const float interval)
{
  taps = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
  float4 vt = first_tap + float4(0.0f, 1.0f, 2.0f, 3.0f) * interval;
  const int runs = (num_taps + 3) / 4;
  for(int i = 0; i < runs; i++)
  {
    for(int c = 0; c < 4; c++)
      taps[4 * i + c] = 1.0f - (vt[c] < 0.0f ? -vt[c] : vt[c]);
    vt += 4.0f * interval;
  }
  return 1.0f; // the kernel norm is 1 by construction
}

float maketaps_bicubic(out float taps[MAX_KERNEL_REQ], const int num_taps, const float first_tap, const float interval)
{
  taps = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
  float4 vt = first_tap + float4(0.0f, 1.0f, 2.0f, 3.0f) * interval;
  const int runs = (num_taps + 3) / 4;
  for(int i = 0; i < runs; i++)
  {
    for(int c = 0; c < 4; c++)
    {
      const float vt_abs = vt[c] < 0.0f ? -vt[c] : vt[c];
      const float t2 = vt[c] * vt[c];
      const float t5 = 5.0f * vt_abs;
      const float r12 = (vt_abs * (t5 - 8.0f - t2) + 4.0f) * 0.5f; // the value for 1 < t < 2
      const float r01 = ((3.0f * t2 - t5) * vt_abs + 2.0f) * 0.5f;
      taps[4 * i + c] = vt_abs <= 1.0f ? r01 : r12;
    }
    vt += 4.0f * interval;
  }
  return 1.0f; // the kernel norm is 1 by construction
}

float vector_sin(const float arg)
{
  const float a = 4.0f / (M_PI_F * M_PI_F);
  const float abs_arg = (arg < 0.0f) ? -arg : arg;
  const float scaled = a * arg * (M_PI_F - abs_arg);
  const float abs_scaled = (scaled < 0.0f) ? -scaled : scaled;
  return scaled * (0.225f * (abs_scaled - 1.0f) + 1.0f);
}

float maketaps_lanczos(out float taps[MAX_KERNEL_REQ], const int num_taps, const float width, const float first_tap, const float interval)
{
  taps = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
  float4 vt = first_tap + float4(0.0f, 1.0f, 2.0f, 3.0f) * interval;
  const int runs = (num_taps + 3) / 4;
  for(int i = 0; i < runs; i++)
  {
    for(int c = 0; c < 4; c++)
    {
      const int a = int(vt[c]);
      const float r = vt[c] - float(a);
      const float sign = (a & 1) != 0 ? -1.0f : 1.0f;
      const float sine1 = vector_sin(M_PI_F * r);
      const float sine2 = vector_sin(M_PI_F * vt[c] / width);
      const float num = (width * sign * sine1 * sine2) + 1e-9f;
      const float denom = (M_PI_F * M_PI_F * vt[c] * vt[c]) + 1e-9f;
      taps[4 * i + c] = num / denom;
    }
    vt += 4.0f * interval;
  }
  float norm = 0.0f;
  for(int i = 0; i < num_taps; i++)
    norm += taps[i];
  return norm;
}

float compute_upsampling_taps(const int itor_mode, const int itor_width, out float taps[MAX_KERNEL_REQ], const float tt)
{
  const int f = int(floor(tt)) - itor_width + 1;
  const float t = tt - float(f);

  if(itor_mode == 1)
    return maketaps_bicubic(taps, 2 * itor_width, t, -1.0f);
  else if(itor_mode == 2)
    return maketaps_lanczos(taps, 2 * itor_width, float(itor_width), t, -1.0f);
  else
    return maketaps_bilinear(taps, 2 * itor_width, t, -1.0f);
}

float interpolation_compute_single(ImageIn in, const int itor_mode, const int itor_width,
                                   const float x, const float y, const int width, const int height,
                                   const int channel)
{
  float kernelh[MAX_KERNEL_REQ];
  float kernelv[MAX_KERNEL_REQ];

  int ix = int(x);
  int iy = int(y);

  float s = 0.0f;
  if(ix >= 0 && iy >= 0 && ix < width && iy < height)
  {
    const float normh = compute_upsampling_taps(itor_mode, itor_width, kernelh, x);
    const float normv = compute_upsampling_taps(itor_mode, itor_width, kernelv, y);

    iy -= itor_width - 1;
    ix -= itor_width - 1;

    for(int i = 0; i < 2 * itor_width; i++)
    {
      float h = 0.0f;
      for(int j = 0; j < 2 * itor_width; j++)
        h += kernelh[j] * in.Areadpixel(clip_mirror(ix + j, width - 1), clip_mirror(iy + i, height - 1))[channel];
      s += kernelv[i] * h;
    }
    s = s / (normh * normv);
  }
  return s;
}

// the OpenCL kernels check the coordinates of all three channels for NaN, and
// let a pixel through only when all of its colors are finite
bool _lens_coords_valid(StructuredBuffer<float> pi, const int k)
{
  bool valid = true;
  for(int i = 0; i < 6; i++) valid = valid && isfinite(pi[k + i]);
  return valid;
}

float4 _lens_bilinear(ImageIn in, const float rx, const float ry, const int iwidth, const int iheight)
{
  float4 px = float4(0.0f);
  if(rx >= -1.0f && ry >= -1.0f && rx < float(iwidth) && ry < float(iheight))
  {
    const float fx = floor(rx);
    const float fy = floor(ry);
    const float dx = rx - fx;
    const float dy = ry - fy;

    const int x0 = clip_mirror(int(fx), iwidth - 1);
    const int x1 = clip_mirror(int(fx) + 1, iwidth - 1);
    const int y0 = clip_mirror(int(fy), iheight - 1);
    const int y1 = clip_mirror(int(fy) + 1, iheight - 1);

    px = (1.0f - dy) * ((1.0f - dx) * in.Areadpixel(x0, y0) + dx * in.Areadpixel(x1, y0))
         + dy * ((1.0f - dx) * in.Areadpixel(x0, y1) + dx * in.Areadpixel(x1, y1));
  }
  return px;
}

// the weighted sum of the kwidth wide filter around (rx, ry), in all channels.
// lanczos is 0 for bicubic. For the blue channel the bicubic OpenCL kernel
// clamps the tap positions before it weighs them, which clamp_taps follows.
float4 _lens_filter(ImageIn in, const float rx, const float ry, const int kwidth, const int lanczos,
                    const bool clamp_taps, const int iwidth, const int iheight)
{
  const int tx = int(floor(rx));
  const int ty = int(floor(ry));

  float4 sum = float4(0.0f);
  float weight = 0.0f;
  for(int jj = 1 - kwidth; jj <= kwidth; jj++)
    for(int ii = 1 - kwidth; ii <= kwidth; ii++)
    {
      int i = tx + ii;
      int j = ty + jj;
      if(clamp_taps)
      {
        i = clamp(i, 0, iwidth - 1);
        j = clamp(j, 0, iheight - 1);
      }

      const float wx = lanczos != 0 ? interpolation_func_lanczos(float(lanczos), float(i) - rx)
                                    : interpolation_func_bicubic(float(i) - rx);
      const float wy = lanczos != 0 ? interpolation_func_lanczos(float(lanczos), float(j) - ry)
                                    : interpolation_func_bicubic(float(j) - ry);
      const float w = wx * wy;

      sum += in.Areadpixel(clip_mirror(i, iwidth - 1), clip_mirror(j, iheight - 1)) * w;
      weight += w;
    }
  return sum / weight;
}

// lens_distort_bilinear, _bicubic, _lanczos2 and _lanczos3 in ../basic.cl:
// mode 0 is bilinear, otherwise kwidth and lanczos select the filter
void _lens_distort(const uint3 gid, ImageIn in, ImageOut out, const int width, const int height,
                   const int iwidth, const int iheight, const int roi_in_x, const int roi_in_y,
                   StructuredBuffer<float> pi, const int do_nan_checks,
                   const int bilinear, const int kwidth, const int lanczos)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const int k = y * 2 * 3 * width + 2 * 3 * x;

  if(do_nan_checks != 0 && !_lens_coords_valid(pi, k))
  {
    out.write(int2(x, y), float4(0.0f));
    return;
  }

  float4 pixel = float4(0.0f);
  for(int c = 0; c < 3; c++)
  {
    const float rx = pi[k + 2 * c] - float(roi_in_x);
    const float ry = pi[k + 2 * c + 1] - float(roi_in_y);
    const float4 px = bilinear != 0 ? _lens_bilinear(in, rx, ry, iwidth, iheight)
                                    : _lens_filter(in, rx, ry, kwidth, lanczos, c == 2 && lanczos == 0, iwidth, iheight);
    // green carries the mask along
    if(c == 0) pixel.x = px.x;
    else if(c == 1) pixel.yw = px.yw;
    else pixel.z = px.z;
  }

  pixel = all(isfinite(pixel.xyz)) ? pixel : float4(0.0f);
  out.write(int2(x, y), pixel);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void lens_distort_bilinear(uint3 gid : SV_DispatchThreadID,
                           uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
                           uniform int iwidth, uniform int iheight, uniform int roi_in_x, uniform int roi_in_y,
                           uniform StructuredBuffer<float> pi, uniform int do_nan_checks)
{
  _lens_distort(gid, in, out, width, height, iwidth, iheight, roi_in_x, roi_in_y, pi, do_nan_checks, 1, 1, 0);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void lens_distort_bicubic(uint3 gid : SV_DispatchThreadID,
                          uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
                          uniform int iwidth, uniform int iheight, uniform int roi_in_x, uniform int roi_in_y,
                          uniform StructuredBuffer<float> pi, uniform int do_nan_checks)
{
  _lens_distort(gid, in, out, width, height, iwidth, iheight, roi_in_x, roi_in_y, pi, do_nan_checks, 0, 2, 0);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void lens_distort_lanczos2(uint3 gid : SV_DispatchThreadID,
                           uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
                           uniform int iwidth, uniform int iheight, uniform int roi_in_x, uniform int roi_in_y,
                           uniform StructuredBuffer<float> pi, uniform int do_nan_checks)
{
  _lens_distort(gid, in, out, width, height, iwidth, iheight, roi_in_x, roi_in_y, pi, do_nan_checks, 0, 2, 2);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void lens_distort_lanczos3(uint3 gid : SV_DispatchThreadID,
                           uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
                           uniform int iwidth, uniform int iheight, uniform int roi_in_x, uniform int roi_in_y,
                           uniform StructuredBuffer<float> pi, uniform int do_nan_checks)
{
  _lens_distort(gid, in, out, width, height, iwidth, iheight, roi_in_x, roi_in_y, pi, do_nan_checks, 0, 3, 3);
}

float _calc_vignette_spline(const float radius, StructuredBuffer<float> spline, const int splinesize)
{
  if(radius >= 1.0f) return spline[splinesize - 1];

  const float r = radius * float(splinesize - 1 - 1);
  const int i = int(r);
  const float frac = r - float(i);

  const float p0 = spline[i];
  return p0 + (spline[i + 1] - p0) * frac;
}

// yi starts at yoff, for the rows of md_lens_correction's cor_rgb
float _interpolate_linear_spline(StructuredBuffer<float> xi, StructuredBuffer<float> yi, const int yoff,
                                 const int ni, const float x)
{
  if(x < xi[0]) return yi[yoff];
  for(int i = 1; i < ni; i++)
  {
    if(x >= xi[i - 1] && x <= xi[i])
    {
      const float dydx = (yi[yoff + i] - yi[yoff + i - 1]) / (xi[i] - xi[i - 1]);
      return yi[yoff + i - 1] + (x - xi[i - 1]) * dydx;
    }
  }
  return yi[yoff + ni - 1];
}

[shader("compute")]
[numthreads(16, 16, 1)]
void md_vignette(uint3 gid : SV_DispatchThreadID,
                 uniform ImageIn in, uniform ImageOut out,
                 uniform StructuredBuffer<float> knots_vig, uniform StructuredBuffer<float> vig,
                 uniform int width, uniform int height, uniform float w2, uniform float h2, uniform float r,
                 uniform int roix, uniform int roiy, uniform int knots)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float cx = (float(roix + x) - w2);
  const float cy = (float(roiy + y) - h2);
  const float spline = _interpolate_linear_spline(knots_vig, vig, 0, knots, r * dt_fast_hypot(cx, cy));

  float4 pixel = in.Areadpixel(x, y);
  const float mask = pixel.w;
  pixel /= max(1e-4f, spline);
  pixel.w = mask;
  out.write(int2(x, y), pixel);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void lens_man_vignette(uint3 gid : SV_DispatchThreadID,
                       uniform ImageIn in, uniform ImageOut out, uniform StructuredBuffer<float> spline,
                       uniform int width, uniform int height, uniform float w2, uniform float h2,
                       uniform int roix, uniform int roiy, uniform float inv_maxr, uniform float intensity,
                       uniform int splinesize, uniform int vigmask)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float dx = (float(roix + x) - w2);
  const float dy = (float(roiy + y) - h2);
  const float radius = dt_fast_hypot(dx, dy) * inv_maxr;
  const float val = max(0.0f, intensity * _calc_vignette_spline(radius, spline, splinesize));

  float4 pixel = in.Areadpixel(x, y);
  const float mask = pixel.w;
  pixel *= (1.0f + val);
  pixel.w = vigmask != 0 ? val : mask;
  out.write(int2(x, y), pixel);
}

// pi holds four floats per pixel
[shader("compute")]
[numthreads(16, 16, 1)]
void lens_vignette(uint3 gid : SV_DispatchThreadID,
                   uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
                   uniform StructuredBuffer<float> pi)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  float4 pixel = in.Areadpixel(x, y);
  const int k = 4 * (y * width + x);
  const float3 scale = 2.0f * float3(pi[k], pi[k + 1], pi[k + 2]);
  pixel.xyz *= scale;
  out.write(int2(x, y), pixel);
}

static const int MAXKNOTS = 16;

[shader("compute")]
[numthreads(16, 16, 1)]
void md_lens_correction(uint3 gid : SV_DispatchThreadID,
                        uniform ImageIn in, uniform ImageOut out,
                        uniform StructuredBuffer<float> knots_dist, uniform StructuredBuffer<float> cor_rgb,
                        uniform int owidth, uniform int oheight, uniform int iwidth, uniform int iheight,
                        uniform float w2, uniform float h2, uniform float r, uniform float scale,
                        uniform int roix, uniform int roiy, uniform int roox, uniform int rooy,
                        uniform int knots, uniform int itor_mode, uniform int itor_width, uniform int pass_mode)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= owidth || y >= oheight) return;

  const float cx = (float(roox + x) - w2) / scale;
  const float cy = (float(rooy + y) - h2) / scale;
  const float radius = r * dt_fast_hypot(cx, cy);
  const float limw = float(iwidth) - 1.0f;
  const float limh = float(iheight) - 1.0f;
  float4 pixel;
  for(int c = 0; c < 4; c++)
  {
    const int plane = (c == 3 || pass_mode != 0) ? 1 : c;
    const float dr = _interpolate_linear_spline(knots_dist, cor_rgb, plane * MAXKNOTS, knots, radius);
    const float xs = clamp(dr * cx + w2 - roix, 0.0f, limw);
    const float ys = clamp(dr * cy + h2 - roiy, 0.0f, limh);
    pixel[c] = interpolation_compute_single(in, itor_mode, itor_width, xs, ys, iwidth, iheight, c);
  }
  out.write(int2(x, y), pixel);
}

// the darkroom's clipping indicators

static const int DT_CLIPPING_PREVIEW_GAMUT = 0;
static const int DT_CLIPPING_PREVIEW_ANYRGB = 1;
static const int DT_CLIPPING_PREVIEW_LUMINANCE = 2;
static const int DT_CLIPPING_PREVIEW_SATURATION = 3;

bool _overexposed_saturated(float4 pixel_tmp, float luminance, float upper)
{
  const float4 diff = pixel_tmp - float4(luminance);
  const float4 saturation = sqrt(diff * diff / (float4(luminance * luminance) + pixel_tmp * pixel_tmp));
  return saturation.x > upper || saturation.y > upper || saturation.z > upper
      || pixel_tmp.x >= upper || pixel_tmp.y >= upper || pixel_tmp.z >= upper;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void overexposed(uint3 gid : SV_DispatchThreadID,
                 uniform ImageIn in, uniform ImageOut out, uniform ImageIn tmp,
                 uniform int width, uniform int height, uniform float lower, uniform float upper,
                 uniform float lower_color[4], uniform float upper_color[4],
                 uniform StructuredBuffer<ProfileInfo> profile_info, uniform ImageIn lut,
                 uniform int use_work_profile, uniform int mode)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  float4 pixel = in.readpixel(x, y);
  const float4 pixel_tmp = tmp.readpixel(x, y);
  const float3 upper_rgb = float3(upper_color[0], upper_color[1], upper_color[2]);
  const float3 lower_rgb = float3(lower_color[0], lower_color[1], lower_color[2]);

  if(mode == DT_CLIPPING_PREVIEW_ANYRGB)
  {
    if(pixel_tmp.x >= upper || pixel_tmp.y >= upper || pixel_tmp.z >= upper)
      pixel.xyz = upper_rgb;
    else if(pixel_tmp.x <= lower && pixel_tmp.y <= lower && pixel_tmp.z <= lower)
      pixel.xyz = lower_rgb;
  }
  else if(use_work_profile != 0
          && (mode == DT_CLIPPING_PREVIEW_GAMUT || mode == DT_CLIPPING_PREVIEW_LUMINANCE
              || mode == DT_CLIPPING_PREVIEW_SATURATION))
  {
    const ProfileInfo info = profile_info[0];
    const float luminance = get_rgb_matrix_luminance(pixel, info, info.matrix_in, lut);

    if(mode == DT_CLIPPING_PREVIEW_GAMUT)
    {
      if(luminance >= upper)
        pixel.xyz = upper_rgb;
      else if(luminance <= lower)
        pixel.xyz = lower_rgb;
      else if(_overexposed_saturated(pixel_tmp, luminance, upper))
        pixel.xyz = upper_rgb;
      else if(pixel_tmp.x <= lower && pixel_tmp.y <= lower && pixel_tmp.z <= lower)
        pixel.xyz = lower_rgb;
    }
    else if(mode == DT_CLIPPING_PREVIEW_LUMINANCE)
    {
      if(luminance >= upper)
        pixel.xyz = upper_rgb;
      else if(luminance <= lower)
        pixel.xyz = lower_rgb;
    }
    else if(luminance < upper && luminance > lower)
    {
      if(_overexposed_saturated(pixel_tmp, luminance, upper))
        pixel.xyz = upper_rgb;
      else if(pixel_tmp.x <= lower && pixel_tmp.y <= lower && pixel_tmp.z <= lower)
        pixel.xyz = lower_rgb;
    }
  }

  out.write(int2(x, y), pixel);
}

// the raw overexposed kernels only write the pixels they mark, over a copy
// of the input. The raw data is uploaded as a buffer of uint16 pairs.
uint _raw_u16(StructuredBuffer<uint> raw, int i)
{
  return (raw[i >> 1] >> uint((i & 1) * 16)) & 0xffffu;
}

// the raw pixel behind output pixel (x, y), and its CFA color, or -1 when the
// pixel is outside the raw or below the threshold
int _rawoverexposed_color(int x, int y, int width, StructuredBuffer<float> pi,
                          StructuredBuffer<uint> raw, int raw_width, int raw_height,
                          uint filters, StructuredBuffer<uint> xtrans, StructuredBuffer<uint> threshold)
{
  const int ppi = y * 2 * width + 2 * x;
  const int raw_x = int(pi[ppi]);
  const int raw_y = int(pi[ppi + 1]);
  if(raw_x < 0 || raw_y < 0 || raw_x >= raw_width || raw_y >= raw_height) return -1;

  const uint raw_pixel = _raw_u16(raw, raw_y * raw_width + raw_x);
  const int c = fcol(raw_y, raw_x, filters, xtrans);
  return raw_pixel < threshold[c] ? -1 : c;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void rawoverexposed_mark_cfa(uint3 gid : SV_DispatchThreadID,
                             uniform ImageIn in, uniform ImageOut out, uniform StructuredBuffer<float> pi,
                             uniform int width, uniform int height,
                             uniform StructuredBuffer<uint> raw, uniform int raw_width, uniform int raw_height,
                             uniform uint filters, uniform StructuredBuffer<uint> xtrans,
                             uniform StructuredBuffer<uint> threshold, uniform StructuredBuffer<float4> colors)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const int c = _rawoverexposed_color(x, y, width, pi, raw, raw_width, raw_height, filters, xtrans, threshold);
  if(c < 0) return;

  float4 pixel = max(0.0f, in.Areadpixel(x, y));
  pixel.xyz = colors[c & 3].xyz;
  out.write(int2(x, y), pixel);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void rawoverexposed_mark_solid(uint3 gid : SV_DispatchThreadID,
                               uniform ImageIn in, uniform ImageOut out, uniform StructuredBuffer<float> pi,
                               uniform int width, uniform int height,
                               uniform StructuredBuffer<uint> raw, uniform int raw_width, uniform int raw_height,
                               uniform uint filters, uniform StructuredBuffer<uint> xtrans,
                               uniform StructuredBuffer<uint> threshold, uniform float solid_color[4])
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const int c = _rawoverexposed_color(x, y, width, pi, raw, raw_width, raw_height, filters, xtrans, threshold);
  if(c < 0) return;

  float4 pixel = max(0.0f, in.Areadpixel(x, y));
  pixel.xyz = float3(solid_color[0], solid_color[1], solid_color[2]);
  out.write(int2(x, y), pixel);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void rawoverexposed_falsecolor(uint3 gid : SV_DispatchThreadID,
                               uniform ImageIn in, uniform ImageOut out, uniform StructuredBuffer<float> pi,
                               uniform int width, uniform int height,
                               uniform StructuredBuffer<uint> raw, uniform int raw_width, uniform int raw_height,
                               uniform uint filters, uniform StructuredBuffer<uint> xtrans,
                               uniform StructuredBuffer<uint> threshold)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const int c = _rawoverexposed_color(x, y, width, pi, raw, raw_width, raw_height, filters, xtrans, threshold);
  if(c < 0) return;

  float4 pixel = max(0.0f, in.Areadpixel(x, y));
  if(c == 2)      pixel.z = 0.0f;
  else if(c == 1) pixel.y = 0.0f;
  else if(c == 0) pixel.x = 0.0f;
  out.write(int2(x, y), pixel);
}

float _lookup_unbounded_buf(ImageIn lut, float x, StructuredBuffer<float> coeffs)
{
  const float a[3] = { coeffs[0], coeffs[1], coeffs[2] };
  return lookup_unbounded(lut, x, a);
}

// the curve of an a or b channel with extrapolation on both sides: a[0..2]
// for the right one, a[3..5] for the left one with an inverted x axis
float _lookup_unbounded_twosided(ImageIn lut, float x, StructuredBuffer<float> coeffs, int o)
{
  // in case the tone curve is marked as linear, return the fast
  // path to linear unbounded (does not clip x at 1)
  if(coeffs[o] >= 0.0f)
  {
    const float ar = 1.0f / coeffs[o];
    const float al = 1.0f - 1.0f / coeffs[o + 3];
    if(x < ar && x >= al)
    {
      const int xi = clamp(int(x * 65536.0f), 0, 0xffff);
      return lut.read(int2(xi & 0xff, xi >> 8)).x;
    }
    else
    {
      const float xx = (x >= ar) ? x : 1.0f - x;
      const int s = (x >= ar) ? o : o + 3;
      return coeffs[s + 1] * dtcl_pow(xx * coeffs[s], coeffs[s + 2]);
    }
  }
  else return x;
}

/* kernel for the tonecurve plugin. */
[shader("compute")]
[numthreads(16, 16, 1)]
void tonecurve(uint3 gid : SV_DispatchThreadID,
               uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
               uniform ImageIn table_L, uniform ImageIn table_a, uniform ImageIn table_b,
               uniform int autoscale_ab, uniform int unbound_ab, uniform StructuredBuffer<float> coeffs_L,
               uniform StructuredBuffer<float> coeffs_ab, uniform float low_approximation,
               uniform int preserve_colors, uniform StructuredBuffer<ProfileInfo> profile_info,
               uniform ImageIn lut)
{
  const int x = gid.x;
  const int y = gid.y;

  if(x >= width || y >= height) return;

  float4 pixel = in.read(int2(x, y));
  const float L_in = pixel.x / 100.0f;
  // use lut or extrapolation:
  const float L = _lookup_unbounded_buf(table_L, L_in, coeffs_L);

  if(autoscale_ab == 0)
  {
    const float a_in = (pixel.y + 128.0f) / 256.0f;
    const float b_in = (pixel.z + 128.0f) / 256.0f;

    if(unbound_ab == 0)
    {
      pixel.y = lookup(table_a, a_in);
      pixel.z = lookup(table_b, b_in);
    }
    else
    {
      // use lut or two-sided extrapolation
      pixel.y = _lookup_unbounded_twosided(table_a, a_in, coeffs_ab, 0);
      pixel.z = _lookup_unbounded_twosided(table_b, b_in, coeffs_ab, 6);
    }
    pixel.x = L;
  }
  else if(autoscale_ab == 1)
  {
    if(L_in > 0.01f)
    {
      pixel.y *= L / pixel.x;
      pixel.z *= L / pixel.x;
    }
    else
    {
      pixel.y *= low_approximation;
      pixel.z *= low_approximation;
    }
    pixel.x = L;
  }
  else if(autoscale_ab == 2)
  {
    float4 xyz = Lab_to_XYZ(pixel);
    xyz.x = _lookup_unbounded_buf(table_L, xyz.x, coeffs_L);
    xyz.y = _lookup_unbounded_buf(table_L, xyz.y, coeffs_L);
    xyz.z = _lookup_unbounded_buf(table_L, xyz.z, coeffs_L);
    pixel.xyz = XYZ_to_Lab(xyz).xyz;
  }
  else if(autoscale_ab == 3)
  {
    float4 rgb = Lab_to_prophotorgb(pixel);

    if(preserve_colors == DT_RGB_NORM_NONE)
    {
      rgb.x = _lookup_unbounded_buf(table_L, rgb.x, coeffs_L);
      rgb.y = _lookup_unbounded_buf(table_L, rgb.y, coeffs_L);
      rgb.z = _lookup_unbounded_buf(table_L, rgb.z, coeffs_L);
    }
    else
    {
      float ratio = 1.f;
      const float lum = dt_rgb_norm(rgb, preserve_colors, 1, profile_info[0], lut);
      if(lum > 0.f)
      {
        const float curve_lum = _lookup_unbounded_buf(table_L, lum, coeffs_L);
        ratio = curve_lum / lum;
      }
      rgb.xyz *= ratio;
    }
    pixel.xyz = prophotorgb_to_Lab(rgb).xyz;
  }

  out.write(int2(x, y), pixel);
}

// ---------------------------------------------------------------------------
// ashift

// ashift_bilinear, _bicubic, _lanczos2 and _lanczos3 in ../basic.cl: kwidth
// 0 is bilinear, otherwise lanczos selects the filter, 0 for bicubic
void _ashift(const uint3 gid, ImageIn in, ImageOut out, const int width, const int height,
             const int iwidth, const int iheight, const int2 roi_in, const int2 roi_out,
             const float in_scale, const float out_scale, const float2 clip,
             StructuredBuffer<float> homograph, const int kwidth, const int lanczos)
{
  const int x = gid.x;
  const int y = gid.y;

  if(x >= width || y >= height) return;

  float pin[3], pout[3];

  // convert output pixel coordinates to original image coordinates
  pout[0] = roi_out.x + x + clip.x;
  pout[1] = roi_out.y + y + clip.y;
  pout[0] /= out_scale;
  pout[1] /= out_scale;
  pout[2] = 1.0f;

  // apply homograph
  for(int i = 0; i < 3; i++)
  {
    pin[i] = 0.0f;
    for(int j = 0; j < 3; j++) pin[i] += homograph[3 * i + j] * pout[j];
  }

  // convert to input pixel coordinates
  pin[0] /= pin[2];
  pin[1] /= pin[2];
  pin[0] *= in_scale;
  pin[1] *= in_scale;
  pin[0] -= roi_in.x;
  pin[1] -= roi_in.y;
  const float rx = pin[0];
  const float ry = pin[1];

  float4 pixel = float4(0.0f);
  if(kwidth == 0)
  {
    // done by hand, as hardware bilinear interpolation can't mirror
    if(rx >= -1.0f && ry >= -1.0f && rx < float(iwidth) && ry < float(iheight))
    {
      const float fx = floor(rx);
      const float fy = floor(ry);
      const float dx = rx - fx;
      const float dy = ry - fy;

      const int x0 = clip_mirror(int(fx), iwidth - 1);
      const int x1 = clip_mirror(int(fx) + 1, iwidth - 1);
      const int y0 = clip_mirror(int(fy), iheight - 1);
      const int y1 = clip_mirror(int(fy) + 1, iheight - 1);

      pixel = (1.0f - dy) * ((1.0f - dx) * in.Areadpixel(x0, y0) + dx * in.Areadpixel(x1, y0))
              + dy * ((1.0f - dx) * in.Areadpixel(x0, y1) + dx * in.Areadpixel(x1, y1));
    }
  }
  else
  {
    const int tx = int(floor(rx));
    const int ty = int(floor(ry));

    float weight = 0.0f;
    for(int jj = 1 - kwidth; jj <= kwidth; jj++)
      for(int ii = 1 - kwidth; ii <= kwidth; ii++)
      {
        const int i = tx + ii;
        const int j = ty + jj;

        const float wx = lanczos != 0 ? interpolation_func_lanczos(float(lanczos), float(i) - rx)
                                      : interpolation_func_bicubic(float(i) - rx);
        const float wy = lanczos != 0 ? interpolation_func_lanczos(float(lanczos), float(j) - ry)
                                      : interpolation_func_bicubic(float(j) - ry);
        const float w = wx * wy;

        pixel += in.Areadpixel(clip_mirror(i, iwidth - 1), clip_mirror(j, iheight - 1)) * w;
        weight += w;
      }

    pixel = (tx >= -1 && ty >= -1 && tx < iwidth && ty < iheight) ? pixel / weight : float4(0.0f);
  }
  out.write(int2(x, y), pixel);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void ashift_bilinear(uint3 gid : SV_DispatchThreadID,
                     uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
                     uniform int iwidth, uniform int iheight, uniform int2 roi_in, uniform int2 roi_out,
                     uniform float in_scale, uniform float out_scale, uniform float2 clip,
                     uniform StructuredBuffer<float> homograph)
{
  _ashift(gid, in, out, width, height, iwidth, iheight, roi_in, roi_out, in_scale, out_scale, clip,
          homograph, 0, 0);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void ashift_bicubic(uint3 gid : SV_DispatchThreadID,
                    uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
                    uniform int iwidth, uniform int iheight, uniform int2 roi_in, uniform int2 roi_out,
                    uniform float in_scale, uniform float out_scale, uniform float2 clip,
                    uniform StructuredBuffer<float> homograph)
{
  _ashift(gid, in, out, width, height, iwidth, iheight, roi_in, roi_out, in_scale, out_scale, clip,
          homograph, 2, 0);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void ashift_lanczos2(uint3 gid : SV_DispatchThreadID,
                     uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
                     uniform int iwidth, uniform int iheight, uniform int2 roi_in, uniform int2 roi_out,
                     uniform float in_scale, uniform float out_scale, uniform float2 clip,
                     uniform StructuredBuffer<float> homograph)
{
  _ashift(gid, in, out, width, height, iwidth, iheight, roi_in, roi_out, in_scale, out_scale, clip,
          homograph, 2, 2);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void ashift_lanczos3(uint3 gid : SV_DispatchThreadID,
                     uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
                     uniform int iwidth, uniform int iheight, uniform int2 roi_in, uniform int2 roi_out,
                     uniform float in_scale, uniform float out_scale, uniform float2 clip,
                     uniform StructuredBuffer<float> homograph)
{
  _ashift(gid, in, out, width, height, iwidth, iheight, roi_in, roi_out, in_scale, out_scale, clip,
          homograph, 3, 3);
}

// the Lab and display-referred modules that have no scene-referred successor
// in the default pipe

[shader("compute")]
[numthreads(16, 16, 1)]
void colisa(uint3 gid : SV_DispatchThreadID,
            uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
            uniform float saturation, uniform ImageIn ctable, uniform StructuredBuffer<float> ca,
            uniform ImageIn ltable, uniform StructuredBuffer<float> la)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float4 i = in.readpixel(x, y);
  float4 o;
  o.x = _lookup_unbounded_buf(ctable, i.x / 100.0f, ca);
  o.x = _lookup_unbounded_buf(ltable, o.x / 100.0f, la);
  o.y = i.y * saturation;
  o.z = i.z * saturation;
  o.w = i.w;
  out.write(int2(x, y), o);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void colorcorrection(uint3 gid : SV_DispatchThreadID,
                     uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
                     uniform float saturation, uniform float a_scale, uniform float a_base,
                     uniform float b_scale, uniform float b_base)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  float4 pixel = in.readpixel(x, y);
  pixel.y = saturation * (pixel.y + pixel.x * a_scale + a_base);
  pixel.z = saturation * (pixel.z + pixel.x * b_scale + b_base);
  out.write(int2(x, y), pixel);
}

// dt_iop_colorzones_channel_t
static const int DT_IOP_COLORZONES_L = 0;
static const int DT_IOP_COLORZONES_C = 1;

[shader("compute")]
[numthreads(16, 16, 1)]
void colorzones(uint3 gid : SV_DispatchThreadID,
                uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
                uniform int channel, uniform ImageIn table_L, uniform ImageIn table_C, uniform ImageIn table_h)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  float4 pixel = in.readpixel(x, y);
  const float normalize_C = 1.f / (128.0f * M_SQRT2_F);
  float4 LCh = Lab_2_LCH(pixel);

  float select = channel == DT_IOP_COLORZONES_L ? LCh.x * 0.01f
               : channel == DT_IOP_COLORZONES_C ? LCh.y * normalize_C
               : LCh.z;
  select = clipf(select);

  LCh.x *= dtcl_pow(2.0f, 4.0f * (lookup(table_L, select) - .5f));
  LCh.y *= 2.f * lookup(table_C, select);
  LCh.z += lookup(table_h, select) - .5f;

  pixel.xyz = LCH_2_Lab(LCh).xyz;
  out.write(int2(x, y), pixel);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void colorzones_v3(uint3 gid : SV_DispatchThreadID,
                   uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
                   uniform int channel, uniform ImageIn table_L, uniform ImageIn table_a, uniform ImageIn table_b)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  float4 pixel = in.readpixel(x, y);

  const float a = pixel.y;
  const float b = pixel.z;
  const float h = fmod(atan2(b, a) + DT_2PI_F, DT_2PI_F) / DT_2PI_F;
  const float C = dt_fast_hypot(b, a);

  float select = h;
  float blend = 0.0f;
  if(channel == DT_IOP_COLORZONES_L)
    select = min(1.0f, pixel.x / 100.0f);
  else if(channel == DT_IOP_COLORZONES_C)
    select = min(1.0f, C / 128.0f);
  else
    blend = fsquare(1.0f - C / 128.0f);

  const float Lm = (blend * 0.5f + (1.0f - blend) * lookup(table_L, select)) - 0.5f;
  const float hm = (blend * 0.5f + (1.0f - blend) * lookup(table_b, select)) - 0.5f;
  const float Cm = 2.0f * lookup(table_a, select);
  const float L = pixel.x * dtcl_pow(2.0f, 4.0f * Lm);

  pixel.x = L;
  pixel.y = dtcl_cos(DT_2PI_F * (h + hm)) * Cm * C;
  pixel.z = dtcl_sin(DT_2PI_F * (h + hm)) * Cm * C;
  out.write(int2(x, y), pixel);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void invert_1f(uint3 gid : SV_DispatchThreadID,
               uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
               uniform StructuredBuffer<float> color, uniform uint filters, uniform int rx, uniform int ry)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float pixel = in.readsingle(x, y);
  out.write(int2(x, y), float4(clipf(color[FC(ry + y, rx + x, filters)] - pixel), 0.0f, 0.0f, 0.0f));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void invert_4f(uint3 gid : SV_DispatchThreadID,
               uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
               uniform StructuredBuffer<float> color)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  float4 pixel = in.readpixel(x, y);
  pixel.xyz = clamp(float3(color[0], color[1], color[2]) - pixel.xyz, 0.0f, 1.0f);
  out.write(int2(x, y), pixel);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void levels(uint3 gid : SV_DispatchThreadID,
            uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
            uniform ImageIn lut, uniform float in_low, uniform float in_high, uniform float in_inv_gamma)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  float4 pixel = in.Areadpixel(x, y);
  const float L = pixel.x;
  const float L_in = pixel.x / 100.0f;
  const float percentage = (L_in - in_low) / (in_high - in_low);

  if(L_in <= in_low)
    pixel.x = 0.0f;
  else if(L_in >= in_high)
    pixel.x = 100.0f * dtcl_pow(percentage, in_inv_gamma);
  else
    pixel.x = lookup(lut, percentage);

  const float f = L_in > 0.01f ? pixel.x / L : pixel.x;
  pixel.y *= f;
  pixel.z *= f;
  out.write(int2(x, y), pixel);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void lowlight(uint3 gid : SV_DispatchThreadID,
              uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
              uniform float4 XYZ_sw, uniform ImageIn lut)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float c = 0.5f;
  const float threshold = 0.01f;

  float4 pixel = in.readpixel(x, y);
  float4 XYZ = Lab_to_XYZ(pixel);

  // scotopic luminance, with a floor on red that avoids "snow" in dark noisy areas
  float V = XYZ.y * (1.33f * (1.0f + (XYZ.y + XYZ.z) / (XYZ.x > threshold ? XYZ.x : threshold)) - 1.68f);

  // scale using empiric coefficient and fit inside limits
  V = clipf(c * V);

  // blending coefficient from curve
  const float w = lookup(lut, pixel.x / 100.0f);

  XYZ = w * XYZ + (1.0f - w) * V * XYZ_sw;
  out.write(int2(x, y), XYZ_to_Lab(XYZ));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void profilegamma(uint3 gid : SV_DispatchThreadID,
                  uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
                  uniform ImageIn table, uniform StructuredBuffer<float> ta)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float4 i = in.readpixel(x, y);
  float4 o;
  o.x = _lookup_unbounded_buf(table, i.x, ta);
  o.y = _lookup_unbounded_buf(table, i.y, ta);
  o.z = _lookup_unbounded_buf(table, i.z, ta);
  o.w = i.w;
  out.write(int2(x, y), o);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void profilegamma_log(uint3 gid : SV_DispatchThreadID,
                      uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
                      uniform float dynamic_range, uniform float shadows_range, uniform float grey)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  float4 i = in.readpixel(x, y);
  const float4 noise = float4(1.0f / 65536.0f);

  float4 o = select(i < noise, noise, i / grey);
  o = (dtcl_log2(o) - shadows_range) / dynamic_range;
  o = select(o < noise, noise, o);
  i.xyz = o.xyz;
  out.write(int2(x, y), i);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void zonesystem(uint3 gid : SV_DispatchThreadID,
                uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
                uniform int size, uniform StructuredBuffer<float> zonemap_offset,
                uniform StructuredBuffer<float> zonemap_scale)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  float4 pixel = in.readpixel(x, y);

  const float rzscale = float(size - 1) / 100.0f;
  const int rz = clamp(int(pixel.x * rzscale), 0, size - 2);
  const float zs = ((rz > 0) ? (zonemap_offset[rz] / pixel.x) : 0.0f) + zonemap_scale[rz];

  pixel.xyz *= zs;
  out.write(int2(x, y), pixel);
}

// the OpenCL path fills the border color, the frame and its inside one after
// the other and then copies the image in: here every pixel takes the last of
// them that covers it
[shader("compute")]
[numthreads(16, 16, 1)]
void borders(uint3 gid : SV_DispatchThreadID,
             uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
             uniform int in_x, uniform int in_y, uniform int in_width, uniform int in_height,
             uniform int frame, uniform int4 frame_out, uniform int4 frame_in,
             uniform float4 color, uniform float4 frame_color)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  // frame_out and frame_in are top left x and y, and bottom right x and y
  float4 pixel = color;
  if(frame != 0 && x >= frame_out.x && y >= frame_out.y && x < frame_out.z && y < frame_out.w)
    pixel = frame_color;
  if(frame != 0 && x >= frame_in.x && y >= frame_in.y && x < frame_in.z && y < frame_in.w)
    pixel = color;
  if(x >= in_x && y >= in_y && x < in_x + in_width && y < in_y + in_height)
    pixel = in.Areadpixel(x - in_x, y - in_y);
  out.write(int2(x, y), pixel);
}

float _monochrome_envelope(const float L)
{
  const float x = clipf(L / 100.0f);
  const float beta = 0.6f;
  if(x < beta)
  {
    const float tmp = abs(x / beta - 1.0f);
    return 1.0f - tmp * tmp;
  }
  else
  {
    const float tmp1 = (1.0f - x) / (1.0f - beta);
    const float tmp2 = tmp1 * tmp1;
    const float tmp3 = tmp2 * tmp1;
    return 3.0f * tmp2 - 2.0f * tmp3;
  }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void monochrome_filter(uint3 gid : SV_DispatchThreadID,
                       uniform ImageIn in, uniform ImageOut out, uniform int width, uniform int height,
                       uniform float a, uniform float b, uniform float size)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  float4 pixel = in.Areadpixel(x, y);
  pixel.x = 100.0f * dt_fast_expf(-clipf((fsquare(pixel.y - a) + fsquare(pixel.z - b)) / (2.0f * size)));
  out.write(int2(x, y), pixel);
}

// the OpenCL kernel also takes the a, b and size of the filter, which it
// never uses
[shader("compute")]
[numthreads(16, 16, 1)]
void monochrome(uint3 gid : SV_DispatchThreadID,
                uniform ImageIn in, uniform ImageIn base, uniform ImageOut out,
                uniform int width, uniform int height, uniform float highlights)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  float4 pixel = in.Areadpixel(x, y);
  const float4 basep = base.Areadpixel(x, y);
  const float tt = _monochrome_envelope(pixel.x);
  const float t = tt + (1.0f - tt) * (1.0f - highlights);
  pixel.x = lerp(pixel.x, pixel.x * basep.x / 100.0f, t);
  pixel.y = pixel.z = 0.0f;
  out.write(int2(x, y), pixel);
}
