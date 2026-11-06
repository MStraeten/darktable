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

// Slang port of the 5D permutohedral lattice of src/iop/Permutohedral.h, as
// surface blur (src/iop/bilateral.cc) uses it, plus that module's direct
// stamp for small radii. Neither has OpenCL code.
//
// The CPU fills one hash table per thread and merges them. Here all pixels
// insert into one open-addressing table: slots holds two words per slot, the
// fingerprint of the key with the insert round in its low 3 bits, and the
// index of the key's entry. A slot is claimed by a compare-and-swap of the
// first word, so only 32-bit atomics are needed. The winner writes the key
// after the swap, so a work item that meets a slot with its own fingerprint,
// claimed in this round, can't compare keys yet and takes it as its own. The
// splat then counts the vertices it can't find, and the host runs another
// round for those. Keeping the round in the swapped word, rather than in a
// separate array, is what stops a racing work item from comparing against a
// key that is still unwritten and inserting a duplicate.
//
// Keys are the CPU's 5 shorts, kept with the same 16-bit wrap-around, and the
// bucket is the CPU's hash. The table layout doesn't change the output; only
// the order of the float sums does.

import common;

static const int D = 5;

// the prm buffer: 1/sigma for x, y, r, g, b, the lattice's scale factors,
// then the roi origin in the image
static const int PRM_SCALE = 5;
static const int PRM_ORIGIN = 10;

int _sext16(const int v)
{
  return (v << 16) >> 16;
}

uint3 _pack(const int k[D])
{
  return uint3((uint(k[0]) & 0xffffu) | (uint(k[1]) << 16),
               (uint(k[2]) & 0xffffu) | (uint(k[3]) << 16),
               uint(k[4]) & 0xffffu);
}

// Key::setHash()
uint _hash(const int k[D])
{
  uint h = 0u;
  for(int i = 0; i < D; i++)
  {
    h += uint(k[i]);
    h *= 2531011u;
  }
  return h;
}

uint _fmix(uint h)
{
  h ^= h >> 16;
  h *= 0x85ebca6bu;
  h ^= h >> 13;
  h *= 0xc2b2ae35u;
  h ^= h >> 16;
  return h;
}

// an independent hash: Key::setHash() is linear in the coordinates, so
// different vertices often share it, and it can't tell them apart
uint _fingerprint(const int k[D])
{
  uint h = 0x9747b28cu;
  for(int i = 0; i < D; i++)
    h = _fmix(h ^ (uint(k[i]) & 0xffffu)) + 0x9e3779b9u * uint(i + 1);
  h &= ~7u;
  return h == 0u ? 8u : h;
}

// PermutohedralLattice::splat() without the table: the D+1 vertices of the
// simplex that holds the pixel, and their barycentric weights
void _simplex(const int x, const int y, const float4 p, StructuredBuffer<float> prm,
              out int keys[(D + 1) * D], out float weight[D + 1])
{
  float position[D];
  position[0] = (float(x) + prm[PRM_ORIGIN]) * prm[0];
  position[1] = (float(y) + prm[PRM_ORIGIN + 1]) * prm[1];
  position[2] = p.x * prm[2];
  position[3] = p.y * prm[3];
  position[4] = p.z * prm[4];

  float elevated[D + 1];
  int greedy[D + 1];
  int rank[D + 1];
  float barycentric[D + 2];

  elevated[D] = -D * position[D - 1] * prm[PRM_SCALE + D - 1];
  for(int i = D - 1; i > 0; i--)
    elevated[i] = (elevated[i + 1] - i * position[i - 1] * prm[PRM_SCALE + i - 1]
                   + (i + 2) * position[i] * prm[PRM_SCALE + i]);
  elevated[0] = elevated[1] + 2 * position[0] * prm[PRM_SCALE];

  const float scale = 1.0f / (D + 1);
  for(int i = 0; i <= D; i++)
  {
    const float v = elevated[i] * scale;
    const float up = ceil(v) * (D + 1);
    const float down = floor(v) * (D + 1);
    greedy[i] = (up - elevated[i] < elevated[i] - down) ? int(up) : int(down);
  }

  int sum = 0;
  for(int i = 0; i <= D; i++) sum += greedy[i];
  sum /= D + 1;

  for(int i = 0; i <= D; i++) rank[i] = 0;
  for(int i = 0; i < D; i++)
    for(int j = i + 1; j <= D; j++)
      if(elevated[i] - greedy[i] < elevated[j] - greedy[j])
        rank[i]++;
      else
        rank[j]++;

  if(sum > 0)
  {
    for(int i = 0; i <= D; i++)
    {
      if(rank[i] >= D + 1 - sum)
      {
        greedy[i] -= D + 1;
        rank[i] += sum - (D + 1);
      }
      else
        rank[i] += sum;
    }
  }
  else if(sum < 0)
  {
    for(int i = 0; i <= D; i++)
    {
      if(rank[i] < -sum)
      {
        greedy[i] += D + 1;
        rank[i] += (D + 1) + sum;
      }
      else
        rank[i] += sum;
    }
  }

  for(int i = 0; i < D + 2; i++) barycentric[i] = 0.0f;
  for(int i = 0; i <= D; i++)
  {
    barycentric[D - rank[i]] += (elevated[i] - greedy[i]) * scale;
    barycentric[D + 1 - rank[i]] -= (elevated[i] - greedy[i]) * scale;
  }
  barycentric[0] += 1.0f + barycentric[D + 1];

  for(int r = 0; r <= D; r++)
  {
    // the canonical simplex of the PermutohedralLattice constructor
    for(int i = 0; i < D; i++)
      keys[r * D + i] = _sext16(greedy[i] + (rank[i] <= D - r ? r : r - (D + 1)));
    weight[r] = barycentric[r];
  }
}

void _key(const int keys[(D + 1) * D], const int r, out int k[D])
{
  for(int i = 0; i < D; i++) k[i] = keys[r * D + i];
}

// HashTablePermutohedral::lookupOffset() with create = false, from a kernel
// that runs after every insert has finished
int _find(const int k[D], RWStructuredBuffer<uint> slots, RWStructuredBuffer<uint> keys,
          const uint capacity, const uint entries)
{
  const uint3 pk = _pack(k);
  const uint fp = _fingerprint(k);
  uint b = _hash(k) % capacity;
  for(uint n = 0; n < capacity; n++)
  {
    const uint w = slots[2 * b];
    if(w == 0u) return -1;
    if((w & ~7u) == fp)
    {
      const uint e = slots[2 * b + 1];
      if(e < entries && keys[3 * e] == pk.x && keys[3 * e + 1] == pk.y && keys[3 * e + 2] == pk.z)
        return int(e);
    }
    b = (b + 1 == capacity) ? 0u : b + 1;
  }
  return -1;
}

void _metal_atomic_addf(RWStructuredBuffer<uint> buf, const uint i, const float d)
{
  __intrinsic_asm "atomic_fetch_add_explicit((device atomic_float *)($0 + $1), $2, memory_order_relaxed)";
}

// Vulkan needs an extension for float atomics, so the add is done by
// compare-and-swap there. Metal has them, and its only compare-and-swap is
// the weak one, which Slang uses without its result: a spurious failure
// would look like success and lose the add.
void _atomic_addf(RWStructuredBuffer<uint> buf, const uint i, const float d)
{
  __target_switch
  {
  case metal:
    _metal_atomic_addf(buf, i, d);
  default:
    uint old = buf[i];
    for(;;)
    {
      uint orig;
      InterlockedCompareExchange(buf[i], old, asuint(asfloat(old) + d), orig);
      if(orig == old) break;
      old = orig;
    }
  }
}

static const uint CNT_ENTRIES = 0;
static const uint CNT_OVERFLOW = 1;
static const uint CNT_MISSING = 2;

void _insert(const int k[D], RWStructuredBuffer<uint> slots, RWStructuredBuffer<uint> keys,
             RWStructuredBuffer<uint> counters, const uint capacity, const uint max_entries,
             const uint round)
{
  const uint3 pk = _pack(k);
  const uint fp = _fingerprint(k);
  const uint want = fp | round;
  uint b = _hash(k) % capacity;
  // a retry of the same slot after a spurious failure also counts here
  for(uint n = 0; n < 2 * capacity; n++)
  {
    uint orig;
    InterlockedCompareExchange(slots[2 * b], 0u, want, orig);
    if(orig == 0u)
    {
      // a weak compare-and-swap (Metal) reports a spurious failure as a
      // success, so check that the slot really holds this work item's word.
      // Another work item with the same key may have won it meanwhile; both
      // then write the same key, and the slot keeps one of the two entries.
      InterlockedOr(slots[2 * b], 0u, orig);
      if(orig == 0u) continue;
      if(orig == want)
      {
        uint e;
        InterlockedAdd(counters[CNT_ENTRIES], 1u, e);
        if(e < max_entries)
        {
          keys[3 * e] = pk.x;
          keys[3 * e + 1] = pk.y;
          keys[3 * e + 2] = pk.z;
          slots[2 * b + 1] = e;
        }
        else
          counters[CNT_OVERFLOW] = 1u;
        return;
      }
    }
    if((orig & ~7u) == fp)
    {
      // claimed in this round: its key may not be written yet, see the top
      if((orig & 7u) == round) return;
      const uint e = slots[2 * b + 1];
      if(e < max_entries && keys[3 * e] == pk.x && keys[3 * e + 1] == pk.y && keys[3 * e + 2] == pk.z)
        return;
    }
    b = (b + 1 == capacity) ? 0u : b + 1;
  }
  counters[CNT_OVERFLOW] = 1u;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void permutohedral_insert(uint3 gid : SV_DispatchThreadID,
                          uniform ImageIn in, uniform RWStructuredBuffer<uint> slots,
                          uniform RWStructuredBuffer<uint> keys, uniform RWStructuredBuffer<uint> counters,
                          uniform StructuredBuffer<float> prm, uniform int width, uniform int height,
                          uniform uint capacity, uniform uint max_entries, uniform uint round)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  int vk[(D + 1) * D];
  float w[D + 1];
  _simplex(x, y, in.readA(int2(x, y)), prm, vk, w);
  for(int r = 0; r <= D; r++)
  {
    int k[D];
    _key(vk, r, k);
    _insert(k, slots, keys, counters, capacity, max_entries, round);
  }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void permutohedral_splat(uint3 gid : SV_DispatchThreadID,
                         uniform ImageIn in, uniform RWStructuredBuffer<uint> slots,
                         uniform RWStructuredBuffer<uint> keys, uniform RWStructuredBuffer<uint> counters,
                         uniform RWStructuredBuffer<uint> values, uniform StructuredBuffer<float> prm,
                         uniform int width, uniform int height, uniform uint capacity, uniform uint entries)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float4 p = in.readA(int2(x, y));
  const float4 val = float4(p.x, p.y, p.z, 1.0f);
  int vk[(D + 1) * D];
  float w[D + 1];
  _simplex(x, y, p, prm, vk, w);
  for(int r = 0; r <= D; r++)
  {
    int k[D];
    _key(vk, r, k);
    const int e = _find(k, slots, keys, capacity, entries);
    if(e < 0)
    {
      InterlockedAdd(counters[CNT_MISSING], 1u);
      continue;
    }
    _atomic_addf(values, 4 * e, w[r] * val.x);
    _atomic_addf(values, 4 * e + 1, w[r] * val.y);
    _atomic_addf(values, 4 * e + 2, w[r] * val.z);
    _atomic_addf(values, 4 * e + 3, w[r] * val.w);
  }
}

float4 _value(RWStructuredBuffer<uint> values, const int e)
{
  if(e < 0) return float4(0.0f);
  return float4(asfloat(values[4 * e]), asfloat(values[4 * e + 1]),
                asfloat(values[4 * e + 2]), asfloat(values[4 * e + 3]));
}

// one pass of PermutohedralLattice::blur() along lattice axis j. The entries
// are laid out row_width wide, which keeps the dispatch within the work
// group count limits for tens of millions of entries.
[shader("compute")]
[numthreads(16, 16, 1)]
void permutohedral_blur(uint3 gid : SV_DispatchThreadID,
                        uniform RWStructuredBuffer<uint> slots, uniform RWStructuredBuffer<uint> keys,
                        uniform RWStructuredBuffer<uint> src, uniform RWStructuredBuffer<uint> dst,
                        uniform uint capacity, uniform uint entries, uniform int row_width, uniform int j)
{
  const uint e = gid.y * uint(row_width) + gid.x;
  if(gid.x >= uint(row_width) || e >= entries) return;

  int k[D];
  k[0] = _sext16(int(keys[3 * e] & 0xffffu));
  k[1] = int(keys[3 * e]) >> 16;
  k[2] = _sext16(int(keys[3 * e + 1] & 0xffffu));
  k[3] = int(keys[3 * e + 1]) >> 16;
  k[4] = _sext16(int(keys[3 * e + 2] & 0xffffu));

  // Key(origin, dim, direction)
  int n1[D];
  int n2[D];
  for(int i = 0; i < D; i++)
  {
    n1[i] = _sext16(k[i] + 1);
    n2[i] = _sext16(k[i] - 1);
  }
  if(j < D)
  {
    n1[j] = _sext16(k[j] - D);
    n2[j] = _sext16(k[j] + D);
  }

  const float4 vm = _value(src, _find(n1, slots, keys, capacity, entries));
  const float4 vp = _value(src, _find(n2, slots, keys, capacity, entries));
  const float4 v = _value(src, int(e));
  const float4 mixed = 0.25f * vm + 0.5f * v + 0.25f * vp;
  dst[4 * e] = asuint(mixed.x);
  dst[4 * e + 1] = asuint(mixed.y);
  dst[4 * e + 2] = asuint(mixed.z);
  dst[4 * e + 3] = asuint(mixed.w);
}

// PermutohedralLattice::slice() and the normalization of process(). The
// simplex is computed again instead of stored, which saves the 52 bytes
// per pixel of the CPU's replay array.
[shader("compute")]
[numthreads(16, 16, 1)]
void permutohedral_slice(uint3 gid : SV_DispatchThreadID,
                         uniform ImageIn in, uniform ImageOut out, uniform RWStructuredBuffer<uint> slots,
                         uniform RWStructuredBuffer<uint> keys, uniform RWStructuredBuffer<uint> values,
                         uniform StructuredBuffer<float> prm, uniform int width, uniform int height,
                         uniform uint capacity, uniform uint entries)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  int vk[(D + 1) * D];
  float w[D + 1];
  _simplex(x, y, in.readA(int2(x, y)), prm, vk, w);
  float4 col = float4(0.0f);
  for(int r = 0; r <= D; r++)
  {
    int k[D];
    _key(vk, r, k);
    const int e = _find(k, slots, keys, capacity, entries);
    if(e >= 0) col += w[r] * _value(values, e);
  }
  out.write(int2(x, y), float4(col.xyz / col.w, 1.0f));
}

// the direct stamp of process() for radii up to MAX_DIRECT_STAMP_RADIUS:
// weights holds the normalized spatial gaussian, (2 rad + 1)^2 values. The
// border of rad pixels is copied, as on the CPU.
[shader("compute")]
[numthreads(16, 16, 1)]
void bilateral_direct(uint3 gid : SV_DispatchThreadID,
                      uniform ImageIn in, uniform ImageOut out, uniform StructuredBuffer<float> weights,
                      uniform int width, uniform int height, uniform int rad, uniform float4 isig2col)
{
  const int x = gid.x;
  const int y = gid.y;
  if(x >= width || y >= height) return;

  const float4 pixel = in.readA(int2(x, y));
  if(y < rad || y >= height - rad || x < rad || x >= width - rad)
  {
    out.write(int2(x, y), pixel);
    return;
  }

  const int wd = 2 * rad + 1;
  float sumw = 0.0f;
  float4 res = float4(0.0f);
  for(int l = -rad; l <= rad; l++)
  {
    for(int k = -rad; k <= rad; k++)
    {
      const float4 inp = in.readA(int2(x + k, y + l));
      const float4 chandiff = (pixel - inp) * (pixel - inp) * isig2col;
      const float diff = chandiff.x + chandiff.y + chandiff.z;
      const float pix_weight = weights[(l + rad) * wd + k + rad] * exp(-diff);
      res += inp * pix_weight;
      sumw += pix_weight;
    }
  }
  out.write(int2(x, y), res / sumw);
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
