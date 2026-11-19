///////////////////////////////////////////////////////////////////////
// File:        dropout.cpp
// Description: Dropout layer.
// Author:      Stefan Weil
//
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

#include "dropout.h"
#include "networkio.h"
#include "serialis.h"
#include "tprintf.h"

#include <cstdint>   // for uint32_t
#include <cstring>   // for memset
#include <stdexcept> // for std::invalid_argument

namespace tesseract {

Dropout::Dropout(const std::string &name, int ni, float dropout_rate, unsigned dropout_dim)
    : Network(NT_DROPOUT, name, ni, ni), dropout_rate_(dropout_rate), dropout_dim_(dropout_dim) {
  if (!IsValid(dropout_rate, dropout_dim)) {
    throw std::invalid_argument(
        "Invalid dropout layer. The rate must be in [0, 1), the dimension 0, 1 or 2.");
  }
}

void Dropout::DebugWeights() {
  tprintf("Dropout layer '%s': rate=%g, dimension=%u\n", name_.c_str(),
          static_cast<double>(dropout_rate_), dropout_dim_);
}

// Writes to the given file. Returns false in case of error.
bool Dropout::Serialize(TFile *fp) const {
  uint32_t dropout_dim = dropout_dim_;
  return Network::Serialize(fp) && fp->Serialize(&dropout_rate_) && fp->Serialize(&dropout_dim);
}

// Reads from the given file. Returns false in case of error.
bool Dropout::DeSerialize(TFile *fp) {
  float dropout_rate;
  uint32_t dropout_dim;
  if (!fp->DeSerialize(&dropout_rate) || !fp->DeSerialize(&dropout_dim)) {
    return false;
  }
  if (!IsValid(dropout_rate, dropout_dim)) {
    // Reject a corrupt model.
    return false;
  }
  dropout_rate_ = dropout_rate;
  dropout_dim_ = dropout_dim;
  no_ = ni_;
  return true;
}

// Runs forward propagation of activations on the input line.
// See Network for a detailed discussion of the arguments.
void Dropout::Forward([[maybe_unused]] bool debug, const NetworkIO &input,
                      [[maybe_unused]] const TransposedArray *input_transpose,
                      [[maybe_unused]] NetworkScratch *scratch, NetworkIO *output) {
  // The output has the same shape as the input (ni_ == no_).
  output->Resize(input, no_);

  const int width = input.Width();
  const int num_features = input.NumFeatures();

  if (IsTraining() && dropout_rate_ > 0.0f) {
    // Inverted dropout: scale the retained activations by 1 / keep_prob, so
    // the recognition can pass its input through unchanged.
    const float keep_prob = 1.0f - dropout_rate_;
    const float scale = 1.0f / keep_prob;

    if (dropout_dim_ == 2) {
      // Feature dropout: one mask value per feature for all timesteps.
      dropout_mask_.resize(num_features);
      for (int i = 0; i < num_features; ++i) {
        float r = randomizer_->UnsignedRand(1.0f);
        dropout_mask_[i] = (r < keep_prob) ? 1 : 0;
      }
      for (int t = 0; t < width; ++t) {
        const float *in = input.f(t);
        float *out = output->f(t);
        for (int i = 0; i < num_features; ++i) {
          out[i] = dropout_mask_[i] ? in[i] * scale : 0.0f;
        }
      }
    } else if (dropout_dim_ == 1) {
      // Temporal dropout: one mask value per timestep for all features.
      dropout_mask_.resize(width);
      for (int t = 0; t < width; ++t) {
        float r = randomizer_->UnsignedRand(1.0f);
        dropout_mask_[t] = (r < keep_prob) ? 1 : 0;
      }
      for (int t = 0; t < width; ++t) {
        const float *in = input.f(t);
        float *out = output->f(t);
        if (dropout_mask_[t]) {
          for (int i = 0; i < num_features; ++i) {
            out[i] = in[i] * scale;
          }
        } else {
          memset(out, 0, sizeof(float) * num_features);
        }
      }
    } else {
      // Dropout of single elements (dimension 0, the default).
      dropout_mask_.resize(static_cast<size_t>(width) * num_features);
      for (int t = 0; t < width; ++t) {
        const float *in = input.f(t);
        float *out = output->f(t);
        char *mask = dropout_mask_.data() + static_cast<size_t>(t) * num_features;
        for (int i = 0; i < num_features; ++i) {
          // UnsignedRand(1.0f) returns a value in [0, 1].
          float r = randomizer_->UnsignedRand(1.0f);
          mask[i] = (r < keep_prob) ? 1 : 0;
          out[i] = mask[i] ? in[i] * scale : 0.0f;
        }
      }
    }
  } else {
    // Recognition (or a dropout rate of 0): pass the input through unchanged.
    // CopyTimeStepFrom is mode-aware, so this also works for a network which
    // was converted to integer mode.
    for (int t = 0; t < width; ++t) {
      output->CopyTimeStepFrom(t, input, t);
    }
  }

#ifndef GRAPHICS_DISABLED
  if (debug) {
    DisplayForward(*output);
  }
#endif
}

// Runs backward propagation of errors on the deltas line.
// See Network for a detailed discussion of the arguments.
bool Dropout::Backward([[maybe_unused]] bool debug, const NetworkIO &fwd_deltas,
                       [[maybe_unused]] NetworkScratch *scratch, NetworkIO *back_deltas) {
  back_deltas->Resize(fwd_deltas, ni_);

  const int width = fwd_deltas.Width();
  const int num_features = fwd_deltas.NumFeatures();

  if (IsTraining() && dropout_rate_ > 0.0f) {
    // Use the mask of the forward pass again.
    const float scale = 1.0f / (1.0f - dropout_rate_);

    if (dropout_dim_ == 2) {
      for (int t = 0; t < width; ++t) {
        const float *in = fwd_deltas.f(t);
        float *out = back_deltas->f(t);
        for (int i = 0; i < num_features; ++i) {
          out[i] = dropout_mask_[i] ? in[i] * scale : 0.0f;
        }
      }
    } else if (dropout_dim_ == 1) {
      for (int t = 0; t < width; ++t) {
        const float *in = fwd_deltas.f(t);
        float *out = back_deltas->f(t);
        if (dropout_mask_[t]) {
          for (int i = 0; i < num_features; ++i) {
            out[i] = in[i] * scale;
          }
        } else {
          memset(out, 0, sizeof(float) * num_features);
        }
      }
    } else {
      for (int t = 0; t < width; ++t) {
        const float *in = fwd_deltas.f(t);
        float *out = back_deltas->f(t);
        const char *mask = dropout_mask_.data() + static_cast<size_t>(t) * num_features;
        for (int i = 0; i < num_features; ++i) {
          out[i] = mask[i] ? in[i] * scale : 0.0f;
        }
      }
    }
  } else {
    // Recognition: pass the deltas through unchanged.
    for (int t = 0; t < width; ++t) {
      back_deltas->CopyTimeStepFrom(t, fwd_deltas, t);
    }
  }

#ifndef GRAPHICS_DISABLED
  if (debug) {
    DisplayBackward(*back_deltas);
  }
#endif
  return needs_to_backprop_;
}

} // namespace tesseract.
