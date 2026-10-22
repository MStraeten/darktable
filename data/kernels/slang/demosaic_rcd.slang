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

// Slang port of the RCD kernels in ../demosaic_rcd.cl. The OpenCL kernels pick
// one of the rgb0/rgb1/rgb2 buffers through a pointer, which SPIR-V's logical
// addressing does not allow, so here the color indexes them through a branch.

import common;

static const float eps = 1e-5f;   // Tolerance to avoid dividing by zero
static const float epssq = 1e-10f;

/** Correct index into "half-width" buffers if the buffer has an odd width
    is not a simple idx-full / 2!
*/
int _half_idx(int row, int col, int w)
{
  return row * ((w + 1) / 2) + col / 2;
}

float _rgb_get(RWStructuredBuffer<float> rgb0, RWStructuredBuffer<float> rgb1, RWStructuredBuffer<float> rgb2,
               int c, int idx)
{
  return c == RED ? rgb0[idx] : c == GREEN ? rgb1[idx] : rgb2[idx];
}

void _rgb_set(RWStructuredBuffer<float> rgb0, RWStructuredBuffer<float> rgb1, RWStructuredBuffer<float> rgb2,
              int c, int idx, float val)
{
  if(c == RED) rgb0[idx] = val;
  else if(c == GREEN) rgb1[idx] = val;
  else rgb2[idx] = val;
}

// Populate cfa and rgb data by normalized input
[shader("compute")]
[numthreads(16, 16, 1)]
void rcd_populate(uint3 gid : SV_DispatchThreadID,
                  uniform ImageIn in, uniform RWStructuredBuffer<float> cfa, uniform RWStructuredBuffer<float> rgb0,
                  uniform RWStructuredBuffer<float> rgb1, uniform RWStructuredBuffer<float> rgb2,
                  uniform int w, uniform int height, uniform uint filters, uniform float scale)
{
  const int col = gid.x;
  const int row = gid.y;
  if(col >= w || row >= height) return;
  const float val = scale * max(0.0f, in.Areadsingle(col, row));
  const int c0 = FC(row, 0, filters);
  const int c1 = FC(row, 1, filters);
  const int idx = row * w + col;
  cfa[idx] = val;
  _rgb_set(rgb0, rgb1, rgb2, c0 == GREEN ? GREEN : c0 == BLUE ? BLUE : RED, idx, val);
  _rgb_set(rgb0, rgb1, rgb2, c1 == GREEN ? GREEN : c1 == BLUE ? BLUE : RED, idx, val);
}

// Write back-normalized data in rgb channels to output
[shader("compute")]
[numthreads(16, 16, 1)]
void rcd_write_output(uint3 gid : SV_DispatchThreadID,
                      uniform ImageOut out, uniform RWStructuredBuffer<float> rgb0,
                      uniform RWStructuredBuffer<float> rgb1, uniform RWStructuredBuffer<float> rgb2,
                      uniform int w, uniform int height, uniform float scale, uniform int border)
{
  const int col = gid.x;
  const int row = gid.y;
  if(!(col >= border && col < w - border && row >= border && row < height - border)) return;
  const int idx = row * w + col;

  out.write(int2(col, row), max(float4(0.0f), float4(scale * rgb0[idx], scale * rgb1[idx], scale * rgb2[idx], 0.0f)));
}

float _cfa(RWStructuredBuffer<float> cfa, int col, int row, int w, int height)
{
  return cfa[clamp(row, 0, height - 1) * w + clamp(col, 0, w - 1)];
}

// the OpenCL kernel reads the tile and its 4 pixel border, clamped to the image, into local memory
float _vdiff(RWStructuredBuffer<float> cfa, int col, int row, int w, int height)
{
  return fsquare(_cfa(cfa, col, row - 3, w, height) - _cfa(cfa, col, row - 1, w, height)
                 - _cfa(cfa, col, row + 1, w, height) + _cfa(cfa, col, row + 3, w, height)
                 - 3.0f * (_cfa(cfa, col, row - 2, w, height) + _cfa(cfa, col, row + 2, w, height))
                 + 6.0f * _cfa(cfa, col, row, w, height));
}

float _hdiff(RWStructuredBuffer<float> cfa, int col, int row, int w, int height)
{
  return fsquare(_cfa(cfa, col - 3, row, w, height) - _cfa(cfa, col - 1, row, w, height)
                 - _cfa(cfa, col + 1, row, w, height) + _cfa(cfa, col + 3, row, w, height)
                 - 3.0f * (_cfa(cfa, col - 2, row, w, height) + _cfa(cfa, col + 2, row, w, height))
                 + 6.0f * _cfa(cfa, col, row, w, height));
}

// Step 1.1 + 1.2: derive the directional discrimination from the squared
// vertical and horizontal high pass filters, without materializing those
[shader("compute")]
[numthreads(16, 16, 1)]
void rcd_step_1(uint3 gid : SV_DispatchThreadID,
                uniform RWStructuredBuffer<float> cfa, uniform RWStructuredBuffer<float> VH_dir,
                uniform int w, uniform int height)
{
  const int col = 2 + gid.x;
  const int row = 2 + gid.y;
  if((row > height - 3) || (col > w - 3)) return;
  const int idx = row * w + col;

  const float V_Stat = max(epssq, _vdiff(cfa, col, row - 1, w, height)
                                + _vdiff(cfa, col, row, w, height)
                                + _vdiff(cfa, col, row + 1, w, height));
  const float H_Stat = max(epssq, _hdiff(cfa, col - 1, row, w, height)
                                + _hdiff(cfa, col, row, w, height)
                                + _hdiff(cfa, col + 1, row, w, height));
  VH_dir[idx] = (row < 4 || row > height - 5 || col < 4 || col > w - 5)
                ? 0.0f : fp32_div(V_Stat, V_Stat + H_Stat);
}

// Step 2: Low pass filter incorporating green, red and blue local samples from the raw data
[shader("compute")]
[numthreads(16, 16, 1)]
void rcd_step_2(uint3 gid : SV_DispatchThreadID,
                uniform RWStructuredBuffer<float> lpf, uniform RWStructuredBuffer<float> cfa,
                uniform int w, uniform int height, uniform uint filters)
{
  const int row = 2 + gid.y;
  const int col = 2 + (FC(row, 0, filters) & 1) + 2 * int(gid.x);
  if((col > w - 3) || (row > height - 3)) return;
  const int idx = row * w + col;

  lpf[_half_idx(row, col, w)] = cfa[idx]
     + 0.5f * (cfa[idx - w    ] + cfa[idx + w    ] + cfa[idx     - 1] + cfa[idx     + 1])
    + 0.25f * (cfa[idx - w - 1] + cfa[idx - w + 1] + cfa[idx + w - 1] + cfa[idx + w + 1]);
}

// Step 3: Populate the green channel at blue and red CFA positions
[shader("compute")]
[numthreads(16, 16, 1)]
void rcd_step_3(uint3 gid : SV_DispatchThreadID,
                uniform RWStructuredBuffer<float> lpf, uniform RWStructuredBuffer<float> cfa,
                uniform RWStructuredBuffer<float> rgb1, uniform RWStructuredBuffer<float> VH_Dir,
                uniform int w, uniform int height, uniform uint filters)
{
  const int row = 4 + gid.y;
  const int col = 4 + (FC(row, 0, filters) & 1) + 2 * int(gid.x);
  if((col > w - 5) || (row > height - 5)) return;
  const int idx = row * w + col;
  const int lidx = _half_idx(row, col, w);
  const int hw2 = 2 * ((w + 1) / 2); // two rows in the half-width buffer
  const int w2 = 2 * w;
  const int w3 = 3 * w;
  const int w4 = 4 * w;

  // Refined vertical and horizontal local discrimination
  const float VH_Central_Value   = VH_Dir[idx];
  const float VH_Neighbourhood_Value = 0.25f * (VH_Dir[idx - w - 1] + VH_Dir[idx - w + 1] + VH_Dir[idx + w - 1] + VH_Dir[idx + w + 1]);
  const float VH_Disc = (abs( 0.5f - VH_Central_Value ) < abs(0.5f - VH_Neighbourhood_Value)) ? VH_Neighbourhood_Value : VH_Central_Value;

  const float cfai = cfa[idx];
  // Cardinal gradients
  const float N_Grad = eps + abs(cfa[idx - w] - cfa[idx + w]) + abs(cfai - cfa[idx - w2]) + abs(cfa[idx - w] - cfa[idx - w3]) + abs(cfa[idx - w2] - cfa[idx - w4]);
  const float S_Grad = eps + abs(cfa[idx + w] - cfa[idx - w]) + abs(cfai - cfa[idx + w2]) + abs(cfa[idx + w] - cfa[idx + w3]) + abs(cfa[idx + w2] - cfa[idx + w4]);
  const float W_Grad = eps + abs(cfa[idx - 1] - cfa[idx + 1]) + abs(cfai - cfa[idx -  2]) + abs(cfa[idx - 1] - cfa[idx -  3]) + abs(cfa[idx -  2] - cfa[idx -  4]);
  const float E_Grad = eps + abs(cfa[idx + 1] - cfa[idx - 1]) + abs(cfai - cfa[idx +  2]) + abs(cfa[idx + 1] - cfa[idx +  3]) + abs(cfa[idx +  2] - cfa[idx +  4]);

  const float lfpi = lpf[lidx];
  // Cardinal pixel estimations
  const float N_Est = fp32_div(cfa[idx - w] * (lfpi + lfpi), eps + lfpi + lpf[lidx - hw2]);
  const float S_Est = fp32_div(cfa[idx + w] * (lfpi + lfpi), eps + lfpi + lpf[lidx + hw2]);
  const float W_Est = fp32_div(cfa[idx - 1] * (lfpi + lfpi), eps + lfpi + lpf[lidx - 1]);
  const float E_Est = fp32_div(cfa[idx + 1] * (lfpi + lfpi), eps + lfpi + lpf[lidx + 1]);

  // Vertical and horizontal estimations
  const float V_Est = fp32_div(S_Grad * N_Est + N_Grad * S_Est, N_Grad + S_Grad);
  const float H_Est = fp32_div(W_Grad * E_Est + E_Grad * W_Est, E_Grad + W_Grad);

  // G@B and G@R interpolation
  rgb1[idx] = lerp(V_Est, H_Est, clipf(VH_Disc));
}

// Step 4.0: Calculate the square of the P/Q diagonals color difference high pass filter
[shader("compute")]
[numthreads(16, 16, 1)]
void rcd_step_4_0(uint3 gid : SV_DispatchThreadID,
                  uniform RWStructuredBuffer<float> cfa, uniform RWStructuredBuffer<float> p_diff,
                  uniform RWStructuredBuffer<float> q_diff, uniform int w, uniform int height, uniform uint filters)
{
  const int row = 3 + gid.y;
  const int col = 3 + 2 * int(gid.x);
  if((col > w - 4) || (row > height - 4)) return;
  const int idx = row * w + col;
  const int idx2 = _half_idx(row, col, w);
  const int w2 = 2 * w;
  const int w3 = 3 * w;

  p_diff[idx2] = fsquare((cfa[idx - w3 - 3] - cfa[idx - w - 1] - cfa[idx + w + 1] + cfa[idx + w3 + 3]) - 3.0f * (cfa[idx - w2 - 2] + cfa[idx + w2 + 2]) + 6.0f * cfa[idx]);
  q_diff[idx2] = fsquare((cfa[idx - w3 + 3] - cfa[idx - w + 1] - cfa[idx + w - 1] + cfa[idx + w3 - 3]) - 3.0f * (cfa[idx - w2 + 2] + cfa[idx + w2 - 2]) + 6.0f * cfa[idx]);
}

// Step 4.1: Calculate P/Q diagonals local discrimination strength
[shader("compute")]
[numthreads(16, 16, 1)]
void rcd_step_4_1(uint3 gid : SV_DispatchThreadID,
                  uniform RWStructuredBuffer<float> PQ_dir, uniform RWStructuredBuffer<float> p_diff,
                  uniform RWStructuredBuffer<float> q_diff, uniform int w, uniform int height, uniform uint filters)
{
  const int row = 2 + gid.y;
  const int col = 2 + (FC(row, 0, filters) & 1) + 2 * int(gid.x);
  if(row < 4 || (col > w - 5) || (row > height - 5)) return;
  const int idx2 = _half_idx(row, col, w);
  const int idx3 = _half_idx(row - 1, col - 1, w);
  const int idx4 = _half_idx(row + 1, col - 1, w);

  const float P_Stat = max(epssq, p_diff[idx3]     + p_diff[idx2] + p_diff[idx4 + 1]);
  const float Q_Stat = max(epssq, q_diff[idx3 + 1] + q_diff[idx2] + q_diff[idx4]);
  PQ_dir[idx2] = fp32_div(P_Stat, P_Stat + Q_Stat);
}

// Step 4.2: Populate the red and blue channels at blue and red CFA positions
[shader("compute")]
[numthreads(16, 16, 1)]
void rcd_step_4_2(uint3 gid : SV_DispatchThreadID,
                  uniform RWStructuredBuffer<float> PQ_dir, uniform RWStructuredBuffer<float> rgb0,
                  uniform RWStructuredBuffer<float> rgb1, uniform RWStructuredBuffer<float> rgb2,
                  uniform int w, uniform int height, uniform uint filters)
{
  const int row = 4 + gid.y;
  const int col = 4 + (FC(row, 0, filters) & 1) + 2 * int(gid.x);
  if((col > w - 5) || (row > height - 5)) return;

  const int color = 2 - FC(row, col, filters);
  const int c = color == GREEN ? GREEN : color == BLUE ? BLUE : RED;

  const int idx = row * w + col;
  const int pqidx = _half_idx(row, col, w);
  const int pqidx2 = _half_idx(row - 1, col - 1, w);
  const int pqidx3 = _half_idx(row + 1, col - 1, w);
  const int w2 = 2 * w;
  const int w3 = 3 * w;

  const float PQ_Central_Value   = PQ_dir[pqidx];
  const float PQ_Neighbourhood_Value = 0.25f * (PQ_dir[pqidx2] + PQ_dir[pqidx2 + 1] + PQ_dir[pqidx3] + PQ_dir[pqidx3 + 1]);
  const float PQ_Disc = (abs(0.5f - PQ_Central_Value) < abs(0.5f - PQ_Neighbourhood_Value)) ? PQ_Neighbourhood_Value : PQ_Central_Value;

  const float c_nw = _rgb_get(rgb0, rgb1, rgb2, c, idx - w - 1);
  const float c_ne = _rgb_get(rgb0, rgb1, rgb2, c, idx - w + 1);
  const float c_sw = _rgb_get(rgb0, rgb1, rgb2, c, idx + w - 1);
  const float c_se = _rgb_get(rgb0, rgb1, rgb2, c, idx + w + 1);

  const float NW_Grad = eps + abs(c_nw - c_se) + abs(c_nw - _rgb_get(rgb0, rgb1, rgb2, c, idx - w3 - 3)) + abs(rgb1[idx] - rgb1[idx - w2 - 2]);
  const float NE_Grad = eps + abs(c_ne - c_sw) + abs(c_ne - _rgb_get(rgb0, rgb1, rgb2, c, idx - w3 + 3)) + abs(rgb1[idx] - rgb1[idx - w2 + 2]);
  const float SW_Grad = eps + abs(c_ne - c_sw) + abs(c_sw - _rgb_get(rgb0, rgb1, rgb2, c, idx + w3 - 3)) + abs(rgb1[idx] - rgb1[idx + w2 - 2]);
  const float SE_Grad = eps + abs(c_nw - c_se) + abs(c_se - _rgb_get(rgb0, rgb1, rgb2, c, idx + w3 + 3)) + abs(rgb1[idx] - rgb1[idx + w2 + 2]);

  const float NW_Est = c_nw - rgb1[idx - w - 1];
  const float NE_Est = c_ne - rgb1[idx - w + 1];
  const float SW_Est = c_sw - rgb1[idx + w - 1];
  const float SE_Est = c_se - rgb1[idx + w + 1];

  const float P_Est = fp32_div(NW_Grad * SE_Est + SE_Grad * NW_Est, NW_Grad + SE_Grad);
  const float Q_Est = fp32_div(NE_Grad * SW_Est + SW_Grad * NE_Est, NE_Grad + SW_Grad);

  _rgb_set(rgb0, rgb1, rgb2, c, idx, rgb1[idx] + lerp(P_Est, Q_Est, clipf(PQ_Disc)));
}

// Step 4.3: Populate the red and blue channels at green CFA positions
[shader("compute")]
[numthreads(16, 16, 1)]
void rcd_step_4_3(uint3 gid : SV_DispatchThreadID,
                  uniform RWStructuredBuffer<float> VH_dir, uniform RWStructuredBuffer<float> rgb0,
                  uniform RWStructuredBuffer<float> rgb1, uniform RWStructuredBuffer<float> rgb2,
                  uniform int w, uniform int height, uniform uint filters)
{
  const int row = 4 + gid.y;
  const int col = 4 + (FC(row, 1, filters) & 1) + 2 * int(gid.x);
  if((col > w - 5) || (row > height - 5)) return;

  const int idx = row * w + col;
  const int w2 = 2 * w;
  const int w3 = 3 * w;

  // Refined vertical and horizontal local discrimination
  const float VH_Central_Value   = VH_dir[idx];
  const float VH_Neighbourhood_Value = 0.25f * (VH_dir[idx - w - 1] + VH_dir[idx - w + 1] + VH_dir[idx + w - 1] + VH_dir[idx + w + 1]);
  const float VH_Disc = (abs(0.5f - VH_Central_Value) < abs(0.5f - VH_Neighbourhood_Value)) ? VH_Neighbourhood_Value : VH_Central_Value;

  const float rgbi1 = rgb1[idx];
  const float N1 = eps + abs(rgbi1 - rgb1[idx - w2]);
  const float S1 = eps + abs(rgbi1 - rgb1[idx + w2]);
  const float W1 = eps + abs(rgbi1 - rgb1[idx -  2]);
  const float E1 = eps + abs(rgbi1 - rgb1[idx +  2]);

  const float rgb1mw1 = rgb1[idx - w];
  const float rgb1pw1 = rgb1[idx + w];
  const float rgb1m1 =  rgb1[idx - 1];
  const float rgb1p1 =  rgb1[idx + 1];

  for(int c = 0; c <= 2; c += 2)
  {
    const float c_n = _rgb_get(rgb0, rgb1, rgb2, c, idx - w);
    const float c_s = _rgb_get(rgb0, rgb1, rgb2, c, idx + w);
    const float c_w = _rgb_get(rgb0, rgb1, rgb2, c, idx - 1);
    const float c_e = _rgb_get(rgb0, rgb1, rgb2, c, idx + 1);

    const float SNabs = abs(c_n - c_s);
    const float EWabs = abs(c_w - c_e);

    // Cardinal gradients
    const float N_Grad = N1 + SNabs + abs(c_n - _rgb_get(rgb0, rgb1, rgb2, c, idx - w3));
    const float S_Grad = S1 + SNabs + abs(c_s - _rgb_get(rgb0, rgb1, rgb2, c, idx + w3));
    const float W_Grad = W1 + EWabs + abs(c_w - _rgb_get(rgb0, rgb1, rgb2, c, idx -  3));
    const float E_Grad = E1 + EWabs + abs(c_e - _rgb_get(rgb0, rgb1, rgb2, c, idx +  3));

    // Cardinal colour differences
    const float N_Est = c_n - rgb1mw1;
    const float S_Est = c_s - rgb1pw1;
    const float W_Est = c_w - rgb1m1;
    const float E_Est = c_e - rgb1p1;

    // Vertical and horizontal estimations
    const float V_Est = fp32_div(N_Grad * S_Est + S_Grad * N_Est, N_Grad + S_Grad);
    const float H_Est = fp32_div(E_Grad * W_Est + W_Grad * E_Est, E_Grad + W_Grad);

    // R@G and B@G interpolation
    _rgb_set(rgb0, rgb1, rgb2, c, idx, rgb1[idx] + lerp(V_Est, H_Est, clipf(VH_Disc)));
  }
}
