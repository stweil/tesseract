///////////////////////////////////////////////////////////////////////
// File:        intsimdmatrixavx512.cpp
// Description: matrix-vector product for 8-bit data on avx512.
// Author:      Ray Smith, AVX512 adaptation
//
// (C) Copyright 2025, Google Inc.
// Licensed under the Apache License, Version 2.0
///////////////////////////////////////////////////////////////////////

#include "intsimdmatrix.h"

#if !defined(__AVX512F__) || !defined(__AVX512BW__)
#  if defined(__i686__) || defined(__x86_64__)
#    error Implementation only for AVX512 capable architectures
#  endif
#else
#  include <immintrin.h>
#  include <algorithm>
#  include <cstdint>

namespace tesseract {

// Number of outputs held in each register. 16 x 32 bit ints.
constexpr int kNumOutputsPerRegister = 16;
constexpr int kMaxOutputRegisters = 8;
constexpr int kNumInputsPerRegister = 64;
constexpr int kNumInputsPerGroup = 4;
constexpr int kNumInputGroups = kNumInputsPerRegister / kNumInputsPerGroup;

// Efficient AVX512 sign normalization for 8-bit data
static inline __m512i sign_epi8_avx512(__m512i a, __m512i b) {
    __mmask64 neg_mask = _mm512_cmp_epi8_mask(b, _mm512_setzero_si512(), _MM_CMPINT_LT);
    __mmask64 zero_mask = _mm512_cmp_epi8_mask(b, _mm512_setzero_si512(), _MM_CMPINT_EQ);
    __m512i neg_a = _mm512_sub_epi8(_mm512_setzero_si512(), a); // -a
    __m512i result = _mm512_mask_mov_epi8(a, neg_mask, neg_a); // -a where b < 0, else a
    result = _mm512_mask_mov_epi8(result, zero_mask, _mm512_setzero_si512()); // 0 where b == 0
    return result;
}

// Computes one set of 4x16 products of inputs and weights, adding to result.
// Sign normalization with AVX512 masks.
static inline void MultiplyGroup(const __m512i& rep_input, const __m512i& ones, const int8_t*& wi,
                                 __m512i& weights, __m512i& reps, __m512i& result) {
  weights = _mm512_loadu_si512(reinterpret_cast<const void*>(wi));
  wi += kNumInputsPerRegister;
  reps = sign_epi8_avx512(rep_input, weights);
  weights = sign_epi8_avx512(weights, weights);
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
  __m128i w128 = load64_to_128(wi); // 8x8bit vals in bottom of 128bit reg
  __m512i w512 = _mm512_cvtepi8_epi32(w128); // 16x32bit vals in 512bit reg
  __m512i bias_scale = _mm512_set1_epi32(127);
  __m512 scale = _mm512_loadu_ps(scales);
  w512 = _mm512_mullo_epi32(w512, bias_scale); // 16x32 <bias * 127>
  result = _mm512_add_epi32(result, w512);     // result += bias * 127
  __m512 res = _mm512_cvtepi32_ps(result);
  res = _mm512_mul_ps(res, scale);
  _mm512_storeu_ps(v, res);
}

static void ExtractResults128(__m512i result0, __m512i result1, __m512i result2, __m512i result3,
                              __m512i result4, __m512i result5, __m512i result6, __m512i result7,
                              const int8_t*& wi, const float*& scales, float*& v) {
  for (int i = 0; i < 8; ++i) {
    ExtractResults16((&result0)[i], wi, scales, v);
    wi += 16;
    scales += 16;
    v += 16;
  }
}

static void PartialMatrixDotVector128(const int8_t* wi, const float* scales, const int8_t* u,
                                      int num_in, float* v) {
  __m512i ones = _mm512_set1_epi16(1);
  __m512i shift_id = _mm512_set_epi32(0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
  __m512i result[8];
  for (int i = 0; i < 8; ++i) result[i] = _mm512_setzero_si512();
  for (int j = 0; j < num_in;) {
    __m512i inputs = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(u + j));
    for (int ig = 0; ig < kNumInputGroups && j < num_in; ++ig, j += kNumInputsPerGroup) {
      __m512i rep_input = _mm512_broadcastd_epi32(_mm512_castsi512_si128(inputs));
      inputs = _mm512_permutexvar_epi32(shift_id, inputs);
      for (int r = 0; r < 8; ++r) {
        __m512i weights, reps;
        MultiplyGroup(rep_input, ones, wi, weights, reps, result[r]);
      }
    }
  }
  ExtractResults128(result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7], wi, scales, v);
}

static void PartialMatrixDotVector64(const int8_t* wi, const float* scales, const int8_t* u,
                                     int num_in, float* v) {
  __m512i ones = _mm512_set1_epi16(1);
  __m512i shift_id = _mm512_set_epi32(0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
  __m512i result[4];
  for (int i = 0; i < 4; ++i) result[i] = _mm512_setzero_si512();
  for (int j = 0; j < num_in;) {
    __m512i inputs = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(u + j));
    for (int ig = 0; ig < kNumInputGroups && j < num_in; ++ig, j += kNumInputsPerGroup) {
      __m512i rep_input = _mm512_broadcastd_epi32(_mm512_castsi512_si128(inputs));
      inputs = _mm512_permutexvar_epi32(shift_id, inputs);
      for (int r = 0; r < 4; ++r) {
        __m512i weights, reps;
        MultiplyGroup(rep_input, ones, wi, weights, reps, result[r]);
      }
    }
  }
  for (int i = 0; i < 4; ++i) {
    ExtractResults16(result[i], wi, scales, v);
    wi += 16;
    scales += 16;
    v += 16;
  }
}

static void PartialMatrixDotVector32(const int8_t* wi, const float* scales, const int8_t* u,
                                     int num_in, float* v) {
  __m512i ones = _mm512_set1_epi16(1);
  __m512i shift_id = _mm512_set_epi32(0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
  __m512i result[2];
  for (int i = 0; i < 2; ++i) result[i] = _mm512_setzero_si512();
  for (int j = 0; j < num_in;) {
    __m512i inputs = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(u + j));
    for (int ig = 0; ig < kNumInputGroups && j < num_in; ++ig, j += kNumInputsPerGroup) {
      __m512i rep_input = _mm512_broadcastd_epi32(_mm512_castsi512_si128(inputs));
      inputs = _mm512_permutexvar_epi32(shift_id, inputs);
      for (int r = 0; r < 2; ++r) {
        __m512i weights, reps;
        MultiplyGroup(rep_input, ones, wi, weights, reps, result[r]);
      }
    }
  }
  for (int i = 0; i < 2; ++i) {
    ExtractResults16(result[i], wi, scales, v);
    wi += 16;
    scales += 16;
    v += 16;
  }
}

static void PartialMatrixDotVector16(const int8_t* wi, const float* scales, const int8_t* u,
                                     int num_in, float* v) {
  __m512i ones = _mm512_set1_epi16(1);
  __m512i shift_id = _mm512_set_epi32(0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
  __m512i result = _mm512_setzero_si512();
  for (int j = 0; j < num_in;) {
    __m512i inputs = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(u + j));
    for (int ig = 0; ig < kNumInputGroups && j < num_in; ++ig, j += kNumInputsPerGroup) {
      __m512i rep_input = _mm512_broadcastd_epi32(_mm512_castsi512_si128(inputs));
      inputs = _mm512_permutexvar_epi32(shift_id, inputs);
      __m512i weights, reps;
      MultiplyGroup(rep_input, ones, wi, weights, reps, result);
    }
  }
  ExtractResults16(result, wi, scales, v);
}

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
  group_size /= 2;
  w_step /= 2;

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

  if (output + group_size <= rounded_num_out) {
    PartialMatrixDotVector16(wi, scales, u, rounded_num_in, v);
  }
}
#else
// Double precision version: implement as needed, similar to above but using __m512d and doubles.
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
