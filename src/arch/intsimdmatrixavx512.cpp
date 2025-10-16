///////////////////////////////////////////////////////////////////////
// File:        intsimdmatrixavx512.cpp
// Description: matrix-vector product for 8-bit data on avx512.
// Author:      <Your Name>
//
// (C) Copyright 2025, <Your Organization>
// Licensed under the Apache License, Version 2.0
///////////////////////////////////////////////////////////////////////

#include "intsimdmatrix.h"

#if !defined(__AVX512F__)
#  if defined(__i686__) || defined(__x86_64__)
#    error Implementation only for AVX512 capable architectures
#  endif
#else
#  include <immintrin.h>
#  include <algorithm>
#  include <cstdint>
#  include <vector>

namespace tesseract {

// Number of outputs held in each register. 16 x 32 bit ints.
constexpr int kNumOutputsPerRegister = 16;
// Maximum number of registers that we will use.
constexpr int kMaxOutputRegisters = 8;
// Number of inputs in the inputs register.
constexpr int kNumInputsPerRegister = 64;
// Number of inputs in each weight group.
constexpr int kNumInputsPerGroup = 4;
constexpr int kNumInputGroups = kNumInputsPerRegister / kNumInputsPerGroup;

// Computes one set of 4x16 products of inputs and weights, adding to result.
static inline void MultiplyGroup(const __m512i &rep_input, const __m512i &ones, const int8_t *&wi,
                                 __m512i &weights, __m512i &reps, __m512i &result) {
  weights = _mm512_loadu_si512(reinterpret_cast<const void *>(wi));
  wi += kNumInputsPerRegister;
  reps = _mm512_sign_epi8(rep_input, weights);
  weights = _mm512_sign_epi8(weights, weights);
  weights = _mm512_maddubs_epi16(weights, reps);
  weights = _mm512_madd_epi16(weights, ones);
  result = _mm512_add_epi32(result, weights);
}

// Load 128 bits into the bottom of a 512bit register.
static inline __m128i load64_to_128(const int8_t *wi_) {
  const auto *wi = reinterpret_cast<const int64_t *>(wi_);
  return _mm_set_epi64x(0, wi[0]);
}

#if defined(FAST_FLOAT)

static inline void ExtractResults16(__m512i result, const int8_t *wi,
                                   const float *scales, float *v) {
  __m128i w128 = load64_to_128(wi);
  __m512i w512 = _mm512_cvtepi8_epi32(w128); // 16x32bit
  __m512i bias_scale = _mm512_set1_epi32(127);
  __m512 scale012... = _mm512_loadu_ps(scales);
  w512 = _mm512_mullo_epi32(w512, bias_scale);
  result = _mm512_add_epi32(result, w512);
  __m512 res = _mm512_cvtepi32_ps(result);
  res = _mm512_mul_ps(res, scale012...);
  _mm512_storeu_ps(v, res);
}

// Similar ExtractResults32, PartialMatrixDotVector128, etc.
// You may want to unroll loops for N=128, N=64, N=32, N=16...
// Follow the AVX2 logic and extend vector width.

static void matrixDotVector(int dim1, int dim2, const int8_t *wi, const float *scales,
                            const int8_t *u, float *v) {
  const int num_out = dim1;
  const int num_in = dim2 - 1;
  const int rounded_num_in = IntSimdMatrix::Roundup(num_in, kNumInputsPerGroup);
  const int rounded_num_out = IntSimdMatrix::Roundup(num_out, kNumOutputsPerRegister);
  int group_size = kNumOutputsPerRegister * kMaxOutputRegisters;
  int output = 0;
  int w_step = (rounded_num_in + 1) * group_size;

  for (; output + group_size <= rounded_num_out; output += group_size) {
    // PartialMatrixDotVector128(wi, scales, u, rounded_num_in, v);
    wi += w_step;
    scales += group_size;
    v += group_size;
  }
  // Continue for N=64, N=32, N=16 ...
}

#else
// Double-precision code as in AVX2, but with __m512d and 16 results.
#endif

const IntSimdMatrix IntSimdMatrix::intSimdMatrixAVX512 = {
    matrixDotVector,
    kNumOutputsPerRegister,
    kMaxOutputRegisters,
    kNumInputsPerRegister,
    kNumInputsPerGroup
};

} // namespace tesseract

#endif
