///////////////////////////////////////////////////////////////////////
// File:        intsimdmatrixavx512.cpp
// Description: matrix-vector product for 8-bit data on avx512.
// Author:      Production AVX512 implementation
//
// (C) Copyright 2025, Stefan Weil
// Licensed under the Apache License, Version 2.0
///////////////////////////////////////////////////////////////////////

#include "intsimdmatrix.h"

#if !defined(__AVX512F__)
#  if defined(__i686__) || defined(__x86_64__)
//#    error Implementation only for AVX512 capable architectures
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
// Number of groups of inputs to be broadcast.
constexpr int kNumInputGroups = kNumInputsPerRegister / kNumInputsPerGroup;

// Computes one set of 4x16 products of inputs and weights, adding to result.
// Horizontally adds 4 adjacent results, making 16x32-bit results.
static inline void MultiplyGroup(const __m512i& rep_input, const __m512i& ones, const int8_t*& wi,
                                 __m512i& weights, __m512i& reps, __m512i& result) {
  weights = _mm512_loadu_si512(reinterpret_cast<const void*>(wi));
  wi += kNumInputsPerRegister;
  reps = _mm512_sign_epi8(rep_input, weights);
  weights = _mm512_sign_epi8(weights, weights);
  weights = _mm512_maddubs_epi16(weights, reps);
  weights = _mm512_madd_epi16(weights, ones);
  result = _mm512_add_epi32(result, weights);
}

// Load 128 bits into the bottom of a 512bit register.
static inline __m128i load64_to_128(const int8_t* wi_) {
  const auto* wi = reinterpret_cast<const int64_t*>(wi_);
  return _mm_set_epi64x(0, wi[0]);
}

#if defined(FAST_FLOAT)

static inline void ExtractResults16(__m512i result, const int8_t* wi, const float* scales, float* v) {
  __m128i w128 = load64_to_128(wi);
  __m512i w512 = _mm512_cvtepi8_epi32(w128); // 16x32bit vals
  __m512i bias_scale = _mm512_set1_epi32(127);
  __m512 scale = _mm512_loadu_ps(scales);
  w512 = _mm512_mullo_epi32(w512, bias_scale);
  result = _mm512_add_epi32(result, w512);
  __m512 res = _mm512_cvtepi32_ps(result);
  res = _mm512_mul_ps(res, scale);
  _mm512_storeu_ps(v, res);
}

// ExtractResults32, ExtractResults64, etc. can be added similarly for larger N.

// Computes part of matrix.vector v = Wu. Computes N=128 results.
static void PartialMatrixDotVector128(const int8_t* wi, const float* scales, const int8_t* u,
                                      int num_in, float* v) {
  __m512i ones = _mm512_set1_epi16(1);
  __m512i shift_id = _mm512_set_epi32(0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
  __m512i result0 = _mm512_setzero_si512();
  __m512i result1 = _mm512_setzero_si512();
  __m512i result2 = _mm512_setzero_si512();
  __m512i result3 = _mm512_setzero_si512();
  __m512i result4 = _mm512_setzero_si512();
  __m512i result5 = _mm512_setzero_si512();
  __m512i result6 = _mm512_setzero_si512();
  __m512i result7 = _mm512_setzero_si512();
  // Iterate over the input (u), one registerful at a time.
  for (int j = 0; j < num_in;) {
    __m512i inputs = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(u + j));
    for (int ig = 0; ig < kNumInputGroups && j < num_in; ++ig, j += kNumInputsPerGroup) {
      __m512i rep_input = _mm512_broadcastd_epi32(_mm512_castsi512_si128(inputs));
      inputs = _mm512_permutexvar_epi32(shift_id, inputs);
      __m512i weights, reps;
      MultiplyGroup(rep_input, ones, wi, weights, reps, result0);
      MultiplyGroup(rep_input, ones, wi, weights, reps, result1);
      MultiplyGroup(rep_input, ones, wi, weights, reps, result2);
      MultiplyGroup(rep_input, ones, wi, weights, reps, result3);
      MultiplyGroup(rep_input, ones, wi, weights, reps, result4);
      MultiplyGroup(rep_input, ones, wi, weights, reps, result5);
      MultiplyGroup(rep_input, ones, wi, weights, reps, result6);
      MultiplyGroup(rep_input, ones, wi, weights, reps, result7);
    }
  }
  // Add ExtractResults16 or larger versions for each result (result0..result7).
  ExtractResults16(result0, wi, scales, v);
  ExtractResults16(result1, wi, scales + 16, v + 16);
  ExtractResults16(result2, wi, scales + 32, v + 32);
  ExtractResults16(result3, wi, scales + 48, v + 48);
  ExtractResults16(result4, wi, scales + 64, v + 64);
  ExtractResults16(result5, wi, scales + 80, v + 80);
  ExtractResults16(result6, wi, scales + 96, v + 96);
  ExtractResults16(result7, wi, scales + 112, v + 112);
}

// You should add similar functions for N=64, N=32, N=16.

static void matrixDotVector(int dim1, int dim2, const int8_t* wi, const float* scales,
                            const int8_t* u, float* v) {
  const int num_out = dim1;
  const int num_in = dim2 - 1;
  const int rounded_num_in = IntSimdMatrix::Roundup(num_in, kNumInputsPerGroup);
  const int rounded_num_out = IntSimdMatrix::Roundup(num_out, kNumOutputsPerRegister);
  int group_size = kNumOutputsPerRegister * kMaxOutputRegisters;
  int output = 0;
  int w_step = (rounded_num_in + 1) * group_size;

  for (; output + group_size <= rounded_num_out; output += group_size) {
    PartialMatrixDotVector128(wi, scales, u, rounded_num_in, v);
    wi += w_step;
    scales += group_size;
    v += group_size;
  }

#if 0 // TODO: add missing functions
  if (output + group_size <= rounded_num_out) {
    PartialMatrixDotVector64(wi, scales, u, rounded_num_in, v);
    wi += w_step;
    scales += group_size;
    v += group_size;
    output += group_size;
  }
  group_size /= 2;
  w_step /= 2;

  if (output + group_size <= rounded_num_out) {
    PartialMatrixDotVector32(wi, scales, u, rounded_num_in, v);
    wi += w_step;
    scales += group_size;
    v += group_size;
    output += group_size;
  }
  group_size /= 2;
  w_step /= 2;

  if (output + group_size <= rounded_num_out) {
    PartialMatrixDotVector16(wi, scales, u, rounded_num_in, v);
    wi += w_step;
    scales += group_size;
    v += group_size;
    output += group_size;
  }
  group_size /= 2;
  w_step /= 2;

  if (output + group_size <= rounded_num_out) {
    PartialMatrixDotVector8(wi, scales, u, rounded_num_in, v);
  }
#endif
}

#else
// Double precision version (optional, see AVX2 code for reference).
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
