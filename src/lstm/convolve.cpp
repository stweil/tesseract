///////////////////////////////////////////////////////////////////////
// File:        convolve.cpp
// Description: Convolutional layer that stacks the inputs over its rectangle
//              and pulls in random data to fill out-of-input inputs.
//              Output is therefore same size as its input, but deeper.
// Author:      Ray Smith
//
// (C) Copyright 2014, Google Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
///////////////////////////////////////////////////////////////////////

#ifdef HAVE_CONFIG_H
#  include "config_auto.h"
#endif

#include "convolve.h"

#include <cstdint>

#include "networkscratch.h"
#include "serialis.h"
#include "tprintf.h"

namespace tesseract {

Convolve::Convolve(const std::string &name, int ni, int half_x, int half_y)
    : Network(NT_CONVOLVE, name, ni, ni * (2 * half_x + 1) * (2 * half_y + 1))
    , half_x_(half_x)
    , half_y_(half_y) {}

// Writes to the given file. Returns false in case of error.
bool Convolve::Serialize(TFile *fp) const {
  return Network::Serialize(fp) && fp->Serialize(&half_x_) && fp->Serialize(&half_y_);
}

// Reads from the given file. Returns false in case of error.
bool Convolve::DeSerialize(TFile *fp) {
  if (!fp->DeSerialize(&half_x_)) {
    return false;
  }
  if (!fp->DeSerialize(&half_y_)) {
    return false;
  }
  if (half_x_ < 0 || half_y_ < 0 || ni_ <= 0) {
    tprintf("Error: invalid Convolve parameters: ni=%d half_x=%d half_y=%d\n", ni_, half_x_,
            half_y_);
    return false;
  }
  int64_t kx = 2LL * half_x_ + 1;
  int64_t ky = 2LL * half_y_ + 1;
  // Stepwise overflow check: ni_ * kx * ky must fit in int.
  if (kx > INT_MAX / ky) {
    tprintf("Error: Convolve output-channel count overflows: ni=%d half_x=%d half_y=%d\n", ni_,
            half_x_, half_y_);
    return false;
  }
  int64_t kxky = kx * ky;
  if (static_cast<int64_t>(ni_) > INT_MAX / kxky) {
    tprintf("Error: Convolve output-channel count overflows: ni=%d half_x=%d half_y=%d\n", ni_,
            half_x_, half_y_);
    return false;
  }
  no_ = static_cast<int>(static_cast<int64_t>(ni_) * kxky);
  return true;
}

// Runs forward propagation of activations on the input line.
// See NetworkCpp for a detailed discussion of the arguments.
void Convolve::Forward(bool debug, const NetworkIO &input,
                       const TransposedArray * /*input_transpose*/,
                       NetworkScratch * /*scratch*/, NetworkIO *output) {
  output->Resize(input, no_);
  int y_scale = 2 * half_y_ + 1;
  const StrideMap &stride_map = output->stride_map();
  // The timestep of a neighbor in the same image differs by 1 for each step
  // in x and by the width of the map for each step in y, so the neighbors
  // are found without a new index for each of them.
  const int t_step_y = stride_map.Size(FD_WIDTH);
  StrideMap::Index dest_index(stride_map);
  int batch = -1;
  int max_x = 0;
  int max_y = 0;
  do {
    if (dest_index.index(FD_BATCH) != batch) {
      // The size of the image can be different for each batch.
      batch = dest_index.index(FD_BATCH);
      max_x = dest_index.MaxIndexOfDim(FD_WIDTH);
      max_y = dest_index.MaxIndexOfDim(FD_HEIGHT);
    }
    // Stack x_scale groups of y_scale * ni_ inputs together.
    int t = dest_index.t();
    int dest_x = dest_index.index(FD_WIDTH);
    int dest_y = dest_index.index(FD_HEIGHT);
    int out_ix = 0;
    for (int x = -half_x_; x <= half_x_; ++x, out_ix += y_scale * ni_) {
      int src_x = dest_x + x;
      if (src_x < 0 || src_x > max_x) {
        // This x is outside the image.
        output->Randomize(t, out_ix, y_scale * ni_, randomizer_);
      } else {
        int out_iy = out_ix;
        for (int y = -half_y_; y <= half_y_; ++y, out_iy += ni_) {
          int src_y = dest_y + y;
          if (src_y < 0 || src_y > max_y) {
            // This y is outside the image.
            output->Randomize(t, out_iy, ni_, randomizer_);
          } else {
            output->CopyTimeStepGeneral(t, out_iy, ni_, input, t + x + y * t_step_y, 0);
          }
        }
      }
    }
  } while (dest_index.Increment());
#ifndef GRAPHICS_DISABLED
  if (debug) {
    DisplayForward(*output);
  }
#endif
}

// Runs backward propagation of errors on the deltas line.
// See NetworkCpp for a detailed discussion of the arguments.
bool Convolve::Backward(bool /*debug*/, const NetworkIO &fwd_deltas, NetworkScratch *scratch,
                        NetworkIO *back_deltas) {
  back_deltas->Resize(fwd_deltas, ni_);
  NetworkScratch::IO delta_sum;
  delta_sum.ResizeFloat(fwd_deltas, ni_, scratch);
  delta_sum->Zero();
  int y_scale = 2 * half_y_ + 1;
  const StrideMap &stride_map = fwd_deltas.stride_map();
  // See Forward for the timesteps of the neighbors.
  const int t_step_y = stride_map.Size(FD_WIDTH);
  StrideMap::Index src_index(stride_map);
  int batch = -1;
  int max_x = 0;
  int max_y = 0;
  do {
    if (src_index.index(FD_BATCH) != batch) {
      batch = src_index.index(FD_BATCH);
      max_x = src_index.MaxIndexOfDim(FD_WIDTH);
      max_y = src_index.MaxIndexOfDim(FD_HEIGHT);
    }
    // Stack x_scale groups of y_scale * ni_ inputs together.
    int t = src_index.t();
    int src_x = src_index.index(FD_WIDTH);
    int src_y = src_index.index(FD_HEIGHT);
    int out_ix = 0;
    for (int x = -half_x_; x <= half_x_; ++x, out_ix += y_scale * ni_) {
      int dest_x = src_x + x;
      if (dest_x >= 0 && dest_x <= max_x) {
        int out_iy = out_ix;
        for (int y = -half_y_; y <= half_y_; ++y, out_iy += ni_) {
          int dest_y = src_y + y;
          if (dest_y >= 0 && dest_y <= max_y) {
            fwd_deltas.AddTimeStepPart(t, out_iy, ni_, delta_sum->f(t + x + y * t_step_y));
          }
        }
      }
    }
  } while (src_index.Increment());
  back_deltas->CopyAll(*delta_sum);
  return true;
}

} // namespace tesseract.
