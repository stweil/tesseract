///////////////////////////////////////////////////////////////////////
// File:        intsimdmatrixvnni.cpp
// Description: matrix-vector product for 8-bit data on AVX512-VNNI.
// Author:      Stefan Weil
//
// (C) Copyright 2026, Stefan Weil
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

#include "intsimdmatrix.h"

#if !defined(__AVX512VNNI__)
#  if defined(__i686__) || defined(__x86_64__)
#    error Implementation only for AVX512-VNNI capable architectures
#  endif
#else
#  include <immintrin.h>
#  include <climits>

namespace tesseract {

// AVX512-VNNI _mm512_dpbusd_epi32() computes 16 dot products of four 8-bit
// values each, adding them to the corresponding 32-bit accumulators. For each
// 32-bit element it multiplies a SIGNED int8 (operand a) with an UNSIGNED
// uint8 (operand b). This is the integer analogue of the ARM SDOT instruction
// and does in one instruction what the AVX2 kernel needs two for (maddubs +
// madd), so it is roughly twice as fast on the int8 multiply.
//
// Weights and inputs are stored as signed 8-bit values, but dpbusd requires
// the weight operand to be unsigned. We therefore store each weight byte w as
// (w + 128) mod 256 (equivalently w XOR 0x80), which as an unsigned integer
// equals w + 128. Then
//
//     u * (w + 128) = u * w + 128 * u
//
// so every output accumulator gains a constant 128 * sum_k u[k] that is
// independent of the output. We subtract that single value from each output
// at the end, which keeps the integer result bit-exact with the generic C
// implementation.
//
// One dpbusd instruction handles 16 outputs at once (16 int32 accumulators in
// a single 512-bit register), so the weight descriptor uses 16 outputs per
// register; the weight layout produced by IntSimdMatrix::Init is, per block of
// 16 outputs, num_in/4 groups of 16 outputs x 4 inputs (output-major),
// followed by 16 bias bytes.

constexpr int kNumOutputsPerRegister = 16;
constexpr int kMaxOutputRegisters = 1;
constexpr int kNumInputsPerRegister = 4;
constexpr int kNumInputsPerGroup = 4;

// Computes part of matrix.vector v = Wu for a block of
// kNumOutputsPerRegister (16) consecutive outputs.
static void PartialVNNI(const int8_t *wi, const TFloat *scales, const int8_t *u,
                        int num_in, TFloat *v) {
  const int groups =
      IntSimdMatrix::Roundup(num_in, kNumInputsPerGroup) / kNumInputsPerGroup;
  const __m512i sign_offset = _mm512_set1_epi8(static_cast<int8_t>(128));
  __m512i acc = _mm512_setzero_si512();
  for (int g = 0; g < groups; ++g) {
    // a: the four input values u[g*4 .. g*4+3] (signed), replicated to all 16
    // outputs (every output sees the same four inputs of this group).
    const int32_t uquad =
        *reinterpret_cast<const int32_t *>(u + g * kNumInputsPerGroup);
    const __m512i a = _mm512_set1_epi32(uquad);
    // b: the 16 outputs' four weights each (64 bytes), offset by 128 to make
    // them unsigned for dpbusd.
    __m512i b = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(wi));
    b = _mm512_xor_si512(b, sign_offset);
    acc = _mm512_dpbusd_epi32(acc, a, b);
    wi += kNumOutputsPerRegister * kNumInputsPerGroup;
  }
  // dpbusd treated the weights as (w + 128), so each output accumulator
  // contains the true dot product plus 128 * sum_k u[k]. Remove the constant.
  int32_t u_sum = 0;
  for (int k = 0; k < num_in; ++k) {
    u_sum += u[k];
  }
  const __m512i correction = _mm512_set1_epi32(128 * u_sum);
  __m512i result = _mm512_sub_epi32(acc, correction);

  // Add the bias (16 bytes, one per output) times 127 (INT8_MAX).
  const __m512i bias = _mm512_cvtepi8_epi32(
      _mm_loadu_si128(reinterpret_cast<const __m128i *>(wi)));
  result = _mm512_add_epi32(
      result, _mm512_mullo_epi32(bias, _mm512_set1_epi32(INT8_MAX)));

  int32_t tmp[16];
  _mm512_storeu_si512(reinterpret_cast<__m512i *>(tmp), result);
  for (int i = 0; i < kNumOutputsPerRegister; ++i) {
    v[i] = static_cast<TFloat>(tmp[i]) * scales[i];
  }
}

static void matrixDotVector(int dim1, int dim2, const int8_t *wi,
                            const TFloat *scales, const int8_t *u, TFloat *v) {
  const int num_in = dim2 - 1;
  const int rounded_num_out =
      IntSimdMatrix::Roundup(dim1, kNumOutputsPerRegister);
  const int w_step = (IntSimdMatrix::Roundup(num_in, kNumInputsPerGroup) + 1) *
                     kNumOutputsPerRegister;
  for (int output = 0; output < rounded_num_out;
       output += kNumOutputsPerRegister, wi += w_step,
           scales += kNumOutputsPerRegister, v += kNumOutputsPerRegister) {
    PartialVNNI(wi, scales, u, num_in, v);
  }
}

const IntSimdMatrix IntSimdMatrix::intSimdMatrixAVX512VNNI = {
    // Function.
    matrixDotVector,
    // Number of 32 bit outputs held in each register.
    kNumOutputsPerRegister,
    // Maximum number of registers that we will use to hold outputs.
    kMaxOutputRegisters,
    // Number of 8 bit inputs in the inputs register.
    kNumInputsPerRegister,
    // Number of inputs in each weight group.
    kNumInputsPerGroup};

} // namespace tesseract.

#endif
