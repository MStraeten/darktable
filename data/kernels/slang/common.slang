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

// Slang module form of ../common.h. Kernels `import common;` instead of
// including the header, so nothing here depends on the C preprocessor.

module common;

public static const float NORM_MIN = 1.52587890625e-05f; // norm can't be < to 2^(-16)

public static const float M_PI_F = 3.14159265358979323846f;
public static const float M_LN2f = 0.69314718055994530942f;
public static const float M_PI_2f = 1.57079632679489661923f;
public static const float M_PI_4f = 0.78539816339744830962f;
public static const float M_SQRT2_F = 1.41421356237309504880f;
public static const float DT_2PI_F = 6.28318530717958647693f;

public static const int LUT_ELEM = 512; // gamut LUT number of elements

public static const int RED = 0;
public static const int GREEN = 1;
public static const int BLUE = 2;
public static const int ALPHA = 3;

// the dtcl_ wrappers keep the kernel sources close to the .cl ones. The OpenCL
// fast path maps them to native_*, which has no portable counterpart: Vulkan and
// Metal builtins already have implementation-defined precision.
public float dtcl_sin(float a) { return sin(a); }
public float dtcl_cos(float a) { return cos(a); }
public float dtcl_sqrt(float a) { return sqrt(a); }
public float dtcl_pow(float a, float b) { return pow(a, b); }
public float dtcl_exp(float a) { return exp(a); }
public float dtcl_log(float a) { return log(a); }
public float dtcl_log2(float a) { return log2(a); }
public float dtcl_exp2(float a) { return exp2(a); }
public float dtcl_exp10(float a) { return exp10(a); }
public float dtcl_log10(float a) { return log10(a); }

public vector<float, N> dtcl_sin<let N : int>(vector<float, N> a) { return sin(a); }
public vector<float, N> dtcl_cos<let N : int>(vector<float, N> a) { return cos(a); }
public vector<float, N> dtcl_sqrt<let N : int>(vector<float, N> a) { return sqrt(a); }
public vector<float, N> dtcl_pow<let N : int>(vector<float, N> a, vector<float, N> b) { return pow(a, b); }
public vector<float, N> dtcl_pow<let N : int>(vector<float, N> a, float b) { return pow(a, vector<float, N>(b)); }
public vector<float, N> dtcl_exp<let N : int>(vector<float, N> a) { return exp(a); }
public vector<float, N> dtcl_log<let N : int>(vector<float, N> a) { return log(a); }
public vector<float, N> dtcl_log2<let N : int>(vector<float, N> a) { return log2(a); }
public vector<float, N> dtcl_exp2<let N : int>(vector<float, N> a) { return exp2(a); }
public vector<float, N> dtcl_exp10<let N : int>(vector<float, N> a) { return exp10(a); }
public vector<float, N> dtcl_log10<let N : int>(vector<float, N> a) { return log10(a); }

// OpenCL builtins that Slang lacks
public float dt_hypot(float x, float y)
{
  // scale by the larger magnitude so that the squares cannot overflow
  const float ax = abs(x), ay = abs(y);
  const float m = max(ax, ay);
  if(m == 0.0f || isinf(m)) return m;
  const float rx = ax / m, ry = ay / m;
  return m * sqrt(rx * rx + ry * ry);
}

// pow() alone is a few ulp off; one Newton step brings it close to the
// OpenCL cbrt(), which matters in XYZ_to_Lab where a and b are differences
// of nearly equal cube roots
public float dt_cbrt(float x)
{
  const float ax = abs(x);
  float y = pow(ax, 1.0f / 3.0f);
  y = y > 0.0f && !isinf(y) ? y - (y * y * y - ax) / (3.0f * y * y) : y;
  return copysign(y, x);
}
public vector<float, N> dt_cbrt<let N : int>(vector<float, N> x)
{
  vector<float, N> r;
  for(int k = 0; k < N; k++) r[k] = dt_cbrt(x[k]);
  return r;
}

public float dt_fast_hypot(float x, float y) { return dt_hypot(x, y); }

// a + b and a - b that the driver may not rearrange. slangc ignores precise
// on the SPIR-V target, and without the NoContraction decoration lavapipe
// simplifies the compensation term of Kahan_sum, (t2 - m) - t1, to 0. Metal
// keeps plain operations exact only because the libraries are built without
// fast math, see CMakeLists.txt
float _exact_add(float a, float b)
{
  __target_switch
  {
  case spirv:
    return spirv_asm {
      %r:$$float = OpFAdd $a $b;
      OpDecorate %r NoContraction;
      result:$$float = OpCopyObject %r;
    };
  default:
    return a + b;
  }
}

float _exact_sub(float a, float b)
{
  __target_switch
  {
  case spirv:
    return spirv_asm {
      %r:$$float = OpFSub $a $b;
      OpDecorate %r NoContraction;
      result:$$float = OpCopyObject %r;
    };
  default:
    return a - b;
  }
}

// Kahan summation algorithm
public void Kahan_sum(inout float m, inout float c, float add)
{
  const float t1 = _exact_sub(add, c);
  const float t2 = _exact_add(m, t1);
  c = _exact_sub(_exact_sub(t2, m), t1);
  m = t2;
}

public float scharr_gradient(StructuredBuffer<float> in, int k, int w)
{
  const float gx = 47.0f / 255.0f * (in[k-w-1] - in[k-w+1] + in[k+w-1] - in[k+w+1])
                + 162.0f / 255.0f * (in[k-1]   - in[k+1]);
  const float gy = 47.0f / 255.0f * (in[k-w-1] - in[k+w-1] + in[k-w+1] - in[k+w+1])
                + 162.0f / 255.0f * (in[k-w]   - in[k+w]);
  return dt_fast_hypot(gx, gy);
}

// Newton-Raphson refined division, see fp32_div() in ../common.h
public float fp32_div(float a, float b)
{
  float y = rcp(b);
  const float e = fma(-b, y, 1.0f);
  y = fma(y, e, y);
  const float q = a * y;
  const float r = fma(-b, q, a);
  return fma(r, y, q);
}

public int FC(int row, int col, uint filters)
{
  return int(filters >> ((((row) << 1 & 14) + ((col) & 1)) << 1) & 3);
}

// the xtrans pattern is the host's 6x6 byte array, read as 9 words since
// byte-sized buffer elements are not portable
public int FCxtrans(int row, int col, StructuredBuffer<uint> xtrans)
{
  // the +600 ensures a non-negative array index as in CPU code
  const int i = ((row + 600) % 6) * 6 + (col + 600) % 6;
  return int((xtrans[i >> 2] >> uint((i & 3) * 8)) & 0xffu);
}

public int fcol(int row, int col, uint filters, StructuredBuffer<uint> xtrans)
{
  return (filters == 9) ? FCxtrans(row, col, xtrans) : FC(row, col, filters);
}

public void atomic_add_f(RWStructuredBuffer<float> val, int index, float delta)
{
  InterlockedAdd(val[index], delta);
}

public float fast_mexp2f(float x)
{
  const float i1 = float(0x3f800000u); // 2^0
  const float i2 = float(0x3f000000u); // 2^-1
  const float k0 = i1 + x * (i2 - i1);
  return asfloat((k0 >= float(0x800000u)) ? uint(k0) : 0u);
}

// we use this exp approximation to maintain full identity with cpu path
public float dt_fast_expf(float x)
{
  // meant for the range [-100.0f, 0.0f]. largest error ~ -0.06 at 0.0f.
  // will get _a_lot_ worse for x > 0.0f (9000 at 10.0f)..
  const int i1 = 0x3f800000;
  // e^x, the comment would be 2^x
  const int i2 = 0x402DF854;
  const int k0 = int(i1 + x * (i2 - i1));
  return asfloat(k0 > 0 ? k0 : 0);
}

public float fsquare(float a) { return a * a; }
public float fcube(float a) { return a * a * a; }
public float clipf(float a) { return clamp(a, 0.0f, 1.0f); }
public float4 clip4(float4 a) { return clamp(a, float4(0.0f), float4(1.0f)); }
public float fmax3(float4 o) { return max(max(o.x, o.y), o.z); }

public int clip_mirror(int i, int max)
{
  if(i < 0)
    i = -i;
  else if(i > max)
    i = max - (i - max);
  return i;
}

// Images. OpenCL reads go through sampler objects; Vulkan storage images and
// Metal read access have none, so each addressing mode in ../common.h is a
// method here. Keeping every kernel behind these two types also means a later
// move from 2D images to flat buffers changes this file only.
public struct ImageIn
{
  Texture2D<float4> tex;

  public int2 size()
  {
    uint w, h;
    tex.GetDimensions(w, h);
    return int2(int(w), int(h));
  }

  // sampleri: CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST
  public float4 read(int2 p) { return tex.Load(int3(clamp(p, int2(0), size() - 1), 0)); }

  // samplerA: CLK_ADDRESS_NONE, only for positions already bounds checked
  public float4 readA(int2 p) { return tex.Load(int3(p, 0)); }

  // samplerc: CLK_ADDRESS_CLAMP, zero outside the image
  public float4 readC(int2 p)
  {
    return all(p >= int2(0) && p < size()) ? tex.Load(int3(p, 0)) : float4(0.0f);
  }

  // the common.h helpers, by their old names
  public float readsingle(int col, int row) { return read(int2(col, row)).x; }
  public float Areadsingle(int col, int row) { return readA(int2(col, row)).x; }
  public float4 readpixel(int col, int row) { return read(int2(col, row)); }
  public float4 Areadpixel(int col, int row) { return readA(int2(col, row)); }
  public float4 Creadpixel(int col, int row) { return readC(int2(col, row)); }
  public float readalpha(int col, int row) { return clipf(read(int2(col, row)).w); }
}

public struct ImageOut
{
  RWTexture2D<float4> tex;

  public void write(int2 p, float4 v) { tex[p] = v; }
}

// expm1 and log1p are OpenCL builtins Slang lacks. These are Kahan's
// formulations, which cancel the rounding of 1 + x and keep the precision
// near zero that the callers need
public float dt_log1p(float x)
{
  const float u = 1.0f + x;
  return u == 1.0f ? x : log(u) * x / (u - 1.0f);
}

public float dt_expm1(float x)
{
  const float u = exp(x);
  if(u == 1.0f) return x;
  const float um1 = u - 1.0f;
  return um1 == -1.0f ? -1.0f : um1 * x / log(u);
}
