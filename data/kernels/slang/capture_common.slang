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

// the 9x9 convolution shared by capture.slang and gaussian.slang

module capture_common;

// the 9x9 sum of the capture kernels and the 1-channel gaussian_kernel_9x9:
// kern[5 * |dy| + |dx|] weighs the pixel at (dx, dy). A small kernel only
// reaches 2 pixels out. Away from the borders the sum keeps the order of the
// OpenCL expression, so that the results match it
public float sum_9x9(StructuredBuffer<float> in, StructuredBuffer<float> kern, int kb,
                     int col, int row, int w1, int height, bool small)
{
  const int i = row * w1 + col;
  const int w2 = 2 * w1;
  const int w3 = 3 * w1;
  const int w4 = 4 * w1;
  const int bd = small ? 2 : 4;

  float val = 0.0f;
  if(col >= bd && row >= bd && (col < w1 - bd) && (row < height - bd))
  {
    if(small)
    {
      val =
          kern[kb + 5+2] * (in[i-w2-1]  + in[i-w2+1]  + in[i-w1-2]  + in[i-w1+2] + in[i+w1-2] + in[i+w1+2] + in[i+w2-1] + in[i+w2+1]) +
          kern[kb +   2] * (in[i-w2  ]  + in[i   -2]  + in[i    +2]  + in[i+w2  ]) +
          kern[kb + 5+1] * (in[i-w1-1]  + in[i-w1+1]  + in[i+w1-1]  + in[i+w1+1]) +
          kern[kb +   1] * (in[i-w1  ]  + in[i   -1]  + in[i    +1]  + in[i+w1  ]) +
          kern[kb +   0] * (in[i]);
    }
    else
    {
      val =
          kern[kb + 10+4] * (in[i-w4-2]  + in[i-w4+2]  + in[i-w2-4]  + in[i-w2+4] + in[i+w2-4] + in[i+w2+4] + in[i+w4-2] + in[i+w4+2]) +
          kern[kb + 5 +4] * (in[i-w4-1]  + in[i-w4+1]  + in[i-w1-4]  + in[i-w1+4] + in[i+w1-4] + in[i+w1+4] + in[i+w4-1] + in[i+w4+1]) +
          kern[kb + 4]    * (in[i-w4  ]  + in[i   -4]  + in[i    +4]  + in[i+w4  ]) +
          kern[kb + 15+3] * (in[i-w3-3]  + in[i-w3+3]  + in[i+w3-3]  + in[i+w3+3]) +
          kern[kb + 10+3] * (in[i-w3-2]  + in[i-w3+2]  + in[i-w2-3]  + in[i-w2+3] + in[i+w2-3] + in[i+w2+3] + in[i+w3-2] + in[i+w3+2]) +
          kern[kb + 5+3]  * (in[i-w3-1]  + in[i-w3+1]  + in[i-w1-3]  + in[i-w1+3] + in[i+w1-3] + in[i+w1+3] + in[i+w3-1] + in[i+w3+1]) +
          kern[kb +   3]  * (in[i-w3  ]  + in[i   -3]  + in[i    +3]  + in[i+w3  ]) +
          kern[kb + 10+2] * (in[i-w2-2]  + in[i-w2+2]  + in[i+w2-2]  + in[i+w2+2]) +
          kern[kb + 5+2]  * (in[i-w2-1]  + in[i-w2+1]  + in[i-w1-2]  + in[i-w1+2] + in[i+w1-2] + in[i+w1+2] + in[i+w2-1] + in[i+w2+1]) +
          kern[kb +   2]  * (in[i-w2  ]  + in[i   -2]  + in[i    +2]  + in[i+w2  ]) +
          kern[kb + 5+1]  * (in[i-w1-1]  + in[i-w1+1]  + in[i+w1-1]  + in[i+w1+1]) +
          kern[kb +   1]  * (in[i-w1  ]  + in[i   -1]  + in[i    +1]  + in[i+w1  ]) +
          kern[kb +   0]  * (in[i]);
    }
  }
  else
  {
    for(int ir = -bd; ir <= bd; ir++)
    {
      const int irow = row + ir;
      if(irow >= 0 && irow < height)
      {
        for(int ic = -bd; ic <= bd; ic++)
        {
          const int icol = col + ic;
          if(icol >= 0 && icol < w1)
            val += kern[kb + 5 * abs(ir) + abs(ic)] * in[irow * w1 + icol];
        }
      }
    }
  }
  return val;
}
