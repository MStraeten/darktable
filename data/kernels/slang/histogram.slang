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

// the module histograms of dt_histogram_helper() in src/common/histogram.c,
// counted on the device. OpenCL has no such kernel: it reads the whole image
// back and counts on the host. Here only the bins come back.

import color_conversion;

static const int HIST_RGB = 0;
static const int HIST_RGB_COMPENSATED = 1;
static const int HIST_LAB = 2;
static const int HIST_LCH = 3;

// dt_ioppr_compensate_middle_grey(): the lightness of a gray of value x
float _compensate(float x, ProfileInfo info, ImageIn lut)
{
  const float4 xyz = rgb_matrix_to_xyz(float4(x, x, x, 0.0f), info, info.matrix_in, lut);
  return XYZ_to_Lab(xyz).x * 0.01f;
}

// one work item per pixel of the histogram's roi; bins holds 4 counts per
// bin, of which the first three are used, as on the CPU
[shader("compute")]
[numthreads(16, 16, 1)]
void histogram_collect(uint3 gid : SV_DispatchThreadID,
                       uniform ImageIn in, uniform RWStructuredBuffer<uint> bins,
                       uniform int4 crop, uniform int bins_count, uniform int mode,
                       uniform StructuredBuffer<ProfileInfo> profile_info, uniform ImageIn lut)
{
  const int x = crop.x + int(gid.x);
  const int y = crop.y + int(gid.y);
  if(x >= crop.z || y >= crop.w) return;

  const float4 p = in.Areadpixel(x, y);
  const float max_bin = float(bins_count - 1);
  float3 b;
  if(mode == HIST_RGB)
    b = max_bin * p.xyz;
  else if(mode == HIST_RGB_COMPENSATED)
  {
    const ProfileInfo info = profile_info[0];
    b = max_bin * float3(_compensate(p.x, info, lut), _compensate(p.y, info, lut), _compensate(p.z, info, lut));
  }
  else if(mode == HIST_LAB)
    b = float3(max_bin / 100.0f, max_bin / 256.0f, max_bin / 256.0f) * (p.xyz + float3(0.0f, 128.0f, 128.0f));
  else
  {
    // dt_Lab_2_LCH()
    const float h = atan2(p.z, p.y);
    const float H = h > 0.0f ? h / DT_2PI_F : 1.0f - abs(h) / DT_2PI_F;
    const float C = sqrt(p.y * p.y + p.z * p.z);
    b = float3(max_bin / 100.0f, max_bin / (128.0f * 1.41421356237309504880f), max_bin) * float3(p.x, C, H);
  }

  const uint3 bin = uint3(clamp(b, float3(0.0f), float3(max_bin)));
  InterlockedAdd(bins[bin.x * 4], 1u);
  InterlockedAdd(bins[bin.y * 4 + 1], 1u);
  InterlockedAdd(bins[bin.z * 4 + 2], 1u);
}
