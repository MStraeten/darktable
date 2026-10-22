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

// Slang module form of src/common/hdr-transfer.h, which the C pipeline and the
// OpenCL kernels share through preprocessor switches. Keep the two in sync.

module hdr_transfer;

public static const int DT_HDR_NONE = 0;
public static const int DT_HDR_PQ = 1;
public static const int DT_HDR_HLG_REC2020 = 2;
public static const int DT_HDR_HLG_P3 = 3;

// BT.2408 reference white; BT.2100 nominal HLG display, zero black
static const float DT_HDR_REFERENCE_WHITE_NITS = 203.0f;
static const float DT_HDR_PQ_PEAK_NITS = 10000.0f;
static const float DT_HDR_HLG_PEAK_NITS = 1000.0f;
static const float DT_HDR_HLG_SYSTEM_GAMMA = 1.2f;
static const float DT_HDR_PQ_M1 = 2610.0f / 16384.0f;
static const float DT_HDR_PQ_M2 = 2523.0f / 32.0f;
static const float DT_HDR_PQ_C1 = 3424.0f / 4096.0f;
static const float DT_HDR_PQ_C2 = 2413.0f / 128.0f;
static const float DT_HDR_PQ_C3 = 2392.0f / 128.0f;
static const float DT_HDR_HLG_A = 0.17883277f;
static const float DT_HDR_HLG_B = 0.28466892f;
static const float DT_HDR_HLG_C = 0.55991073f;

public float dt_hdr_pq_decode(float signal)
{
  const float e = min(max(signal, 0.0f), 1.0f);
  const float p = pow(e, 1.0f / DT_HDR_PQ_M2);
  const float n = max(p - DT_HDR_PQ_C1, 0.0f);
  const float d = DT_HDR_PQ_C2 - DT_HDR_PQ_C3 * p;
  return (DT_HDR_PQ_PEAK_NITS / DT_HDR_REFERENCE_WHITE_NITS)
    * pow(n / d, 1.0f / DT_HDR_PQ_M1);
}

public float dt_hdr_pq_encode(float linear)
{
  const float l = min(max(linear, 0.0f)
    * (DT_HDR_REFERENCE_WHITE_NITS / DT_HDR_PQ_PEAK_NITS), 1.0f);
  const float p = pow(l, DT_HDR_PQ_M1);
  return pow((DT_HDR_PQ_C1 + DT_HDR_PQ_C2 * p)
             / (1.0f + DT_HDR_PQ_C3 * p), DT_HDR_PQ_M2);
}

public float dt_hdr_hlg_decode(float signal)
{
  const float e = min(max(signal, 0.0f), 1.0f);
  return e <= 0.5f ? e * e / 3.0f
    : (exp((e - DT_HDR_HLG_C) / DT_HDR_HLG_A) + DT_HDR_HLG_B) / 12.0f;
}

public float dt_hdr_hlg_encode(float scene)
{
  const float e = max(scene, 0.0f);
  return min(e <= 1.0f / 12.0f ? sqrt(3.0f * e)
    : DT_HDR_HLG_A * log(12.0f * e - DT_HDR_HLG_B) + DT_HDR_HLG_C, 1.0f);
}

public float dt_hdr_hlg_luminance(float3 rgb, int transfer)
{
  // D65 luminance coefficients, before chromatic adaptation to D50
  return transfer == DT_HDR_HLG_P3
    ? 0.22897456f * rgb[0] + 0.69173852f * rgb[1] + 0.07928691f * rgb[2]
    : 0.2627f * rgb[0] + 0.6780f * rgb[1] + 0.0593f * rgb[2];
}

public void dt_hdr_decode(inout float3 rgb, int transfer)
{
  if(transfer == DT_HDR_PQ)
  {
    for(int c = 0; c < 3; c++) rgb[c] = dt_hdr_pq_decode(rgb[c]);
  }
  else if(transfer != DT_HDR_NONE)
  {
    for(int c = 0; c < 3; c++) rgb[c] = dt_hdr_hlg_decode(rgb[c]);
    const float y = dt_hdr_hlg_luminance(rgb, transfer);
    const float scale = (DT_HDR_HLG_PEAK_NITS / DT_HDR_REFERENCE_WHITE_NITS)
      * pow(y, DT_HDR_HLG_SYSTEM_GAMMA - 1.0f);
    for(int c = 0; c < 3; c++) rgb[c] *= scale;
  }
}

public void dt_hdr_encode(inout float3 rgb, int transfer)
{
  if(transfer == DT_HDR_PQ)
  {
    for(int c = 0; c < 3; c++) rgb[c] = dt_hdr_pq_encode(rgb[c]);
  }
  else if(transfer != DT_HDR_NONE)
  {
    for(int c = 0; c < 3; c++) rgb[c] = max(rgb[c], 0.0f);
    const float y = dt_hdr_hlg_luminance(rgb, transfer);
    const float scale = y > 0.0f
      ? pow(DT_HDR_REFERENCE_WHITE_NITS / DT_HDR_HLG_PEAK_NITS,
            1.0f / DT_HDR_HLG_SYSTEM_GAMMA)
        * pow(y, 1.0f / DT_HDR_HLG_SYSTEM_GAMMA - 1.0f)
      : 0.0f;
    for(int c = 0; c < 3; c++) rgb[c] = dt_hdr_hlg_encode(rgb[c] * scale);
  }
}
