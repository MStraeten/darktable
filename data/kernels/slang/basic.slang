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
  const int row = gid.x;
  if(row >= height || gid.y != 0) return;

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
