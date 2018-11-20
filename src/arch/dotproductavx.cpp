///////////////////////////////////////////////////////////////////////
// File:        dotproductavx.cpp
// Description: Architecture-specific dot-product function.
// Author:      Ray Smith
// Created:     Wed Jul 22 10:48:05 PDT 2015
//
// (C) Copyright 2015, Google Inc.
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

#if !defined(__AVX__)
// Implementation for non-avx archs.

#include "dotproductavx.h"
#include <cstdio>
#include <cstdlib>

namespace tesseract {
double DotProductAVX(const double* u, const double* v, int n) {
  fprintf(stderr, "DotProductAVX can't be used on Android\n");
  abort();
}
}  // namespace tesseract

#else  // !defined(__AVX__)
// Implementation for avx capable archs.
#include <immintrin.h>
#include <cstdint>
#include "dotproductavx.h"
#include <stdio.h>
#include <math.h>
#include <omp.h>

namespace tesseract {

// Computes and returns the dot product of the n-vectors u and v.
// Uses Intel AVX intrinsics to access the SIMD instruction set.
double DotProductAVX(const double* u, const double* v, int n) {
  int max_offset = n - 4;
  int offset = 0;
  // Accumulate a set of 4 sums in sum, by loading pairs of 4 values from u and
  // v, and multiplying them together in parallel.
  __m256d sum = _mm256_setzero_pd();
  if (offset <= max_offset) {
    offset = 4;
    // Aligned load is reputedly faster but requires 32 byte aligned input.
    if ((reinterpret_cast<uintptr_t>(u) & 31) == 0 &&
        (reinterpret_cast<uintptr_t>(v) & 31) == 0) {
      // Use aligned load.
      __m256d floats1 = _mm256_load_pd(u);
      __m256d floats2 = _mm256_load_pd(v);
      // Multiply.
      sum = _mm256_mul_pd(floats1, floats2);
      while (offset <= max_offset) {
        floats1 = _mm256_load_pd(u + offset);
        floats2 = _mm256_load_pd(v + offset);
        offset += 4;
        __m256d product = _mm256_mul_pd(floats1, floats2);
        sum = _mm256_add_pd(sum, product);
      }
    } else {
      // Use unaligned load.
      __m256d floats1 = _mm256_loadu_pd(u);
      __m256d floats2 = _mm256_loadu_pd(v);
      // Multiply.
      sum = _mm256_mul_pd(floats1, floats2);
      while (offset <= max_offset) {
        floats1 = _mm256_loadu_pd(u + offset);
        floats2 = _mm256_loadu_pd(v + offset);
        offset += 4;
        __m256d product = _mm256_mul_pd(floats1, floats2);
        sum = _mm256_add_pd(sum, product);
      }
    }
  }
  // Add the 4 product sums together horizontally. Not so easy as with sse, as
  // there is no add across the upper/lower 128 bit boundary, so permute to
  // move the upper 128 bits to lower in another register.
  __m256d sum2 = _mm256_permute2f128_pd(sum, sum, 1);
  sum = _mm256_hadd_pd(sum, sum2);
  sum = _mm256_hadd_pd(sum, sum);
  double result;
  // _mm256_extract_f64 doesn't exist, but resist the temptation to use an sse
  // instruction, as that introduces a 70 cycle delay. All this casting is to
  // fool the intrinsics into thinking we are extracting the bottom int64.
  auto cast_sum = _mm256_castpd_si256(sum);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
  *(reinterpret_cast<int64_t*>(&result)) =
#if defined(_WIN32) || defined(__i386__)
      // This is a very simple workaround that is activated
      // for all platforms that do not have _mm256_extract_epi64.
      // _mm256_extract_epi64(X, Y) == ((uint64_t*)&X)[Y]
      ((uint64_t*)&cast_sum)[0]
#else
      _mm256_extract_epi64(cast_sum, 0)
#endif
      ;
#pragma GCC diagnostic pop
  while (offset < n) {
    result += u[offset] * v[offset];
    ++offset;
  }
  return result;
}

float DotProductAVXFloat(const float* u, const float* v, int n) {
  if (n == 0) return 0.0f;
  __m256 sum1, c1, sum2, c2, sum3, c3, sum4, c4;
  sum1 = _mm256_set1_ps(0.0);
  sum2 = _mm256_set1_ps(0.0);
  sum3 = _mm256_set1_ps(0.0);
  sum4 = _mm256_set1_ps(0.0);
  c1 = _mm256_set1_ps(0.0);
  c2 = _mm256_set1_ps(0.0);
  c3 = _mm256_set1_ps(0.0);
  c4 = _mm256_set1_ps(0.0);
  int i, rem;
  rem = n % 32;
  __m256 prod1, y1, t1, a1, b1;
  __m256 prod2, y2, t2, a2, b2;
  __m256 prod3, y3, t3, a3, b3;
  __m256 prod4, y4, t4, a4, b4;
  /* use four way unrolling */
  for (i = 0; i < n - rem; i += 32) {
    /* load 4x4 floats into four vector registers */
    a1 = _mm256_loadu_ps(&u[i]);
    a2 = _mm256_loadu_ps(&u[i + 8]);
    a3 = _mm256_loadu_ps(&u[i + 16]);
    a4 = _mm256_loadu_ps(&u[i] + 24);
    /* load 4x4 floats into four vector registers */
    b1 = _mm256_loadu_ps(&v[i]);
    b2 = _mm256_loadu_ps(&v[i + 8]);
    b3 = _mm256_loadu_ps(&v[i + 16]);
    b4 = _mm256_loadu_ps(&v[i] + 24);
    /* multiply components */
    prod1 = _mm256_mul_ps(a1, b1);
    prod2 = _mm256_mul_ps(a2, b2);
    prod3 = _mm256_mul_ps(a3, b3);
    prod4 = _mm256_mul_ps(a4, b4);
    y1 = _mm256_sub_ps(prod1, c1);
    y2 = _mm256_sub_ps(prod2, c2);
    y3 = _mm256_sub_ps(prod3, c3);
    y4 = _mm256_sub_ps(prod4, c4);
    t1 = _mm256_add_ps(sum1, y1);
    t2 = _mm256_add_ps(sum2, y2);
    t3 = _mm256_add_ps(sum3, y3);
    t4 = _mm256_add_ps(sum4, y4);
    c1 = _mm256_sub_ps(_mm256_sub_ps(t1, sum1), y1);
    c2 = _mm256_sub_ps(_mm256_sub_ps(t2, sum2), y2);
    c3 = _mm256_sub_ps(_mm256_sub_ps(t3, sum3), y3);
    c4 = _mm256_sub_ps(_mm256_sub_ps(t4, sum4), y4);
    sum1 = t1;
    sum2 = t2;
    sum3 = t3;
    sum4 = t4;
  }
  /* reduce four simd vectors to one simd vector using Kahan */
  c1 = _mm256_sub_ps(c1, c2);
  c3 = _mm256_sub_ps(c3, c4);
  y1 = _mm256_sub_ps(sum2, c1);
  y3 = _mm256_sub_ps(sum4, c3);
  t1 = _mm256_add_ps(sum1, y1);
  t3 = _mm256_add_ps(sum3, y3);
  c1 = _mm256_sub_ps(_mm256_sub_ps(t1, sum1), y1);
  c3 = _mm256_sub_ps(_mm256_sub_ps(t3, sum3), y3);
  sum1 = t1;
  sum3 = t3;
  c1 = _mm256_sub_ps(c1, c3);
  y1 = _mm256_sub_ps(sum3, c1);
  t1 = _mm256_add_ps(sum1, y1);
  c1 = _mm256_sub_ps(_mm256_sub_ps(t1, sum1), y1);
  sum1 = t1;
  /* store results of vector register onto stack,
   * horizontal reduction in register using AVX hadd
   * won't give us much of a benefit here. */
  float tmp[8];
  float c_tmp[8];
  _mm256_store_ps(&tmp[0], sum1);
  _mm256_store_ps(&c_tmp[0], c1);
  float sum = 0.0;
  float c = c_tmp[0] + c_tmp[1] + c_tmp[2] + c_tmp[3] + c_tmp[4] + c_tmp[5] +
            c_tmp[6] + c_tmp[7];
  /* perform scalar Kahan sum of partial sums */
#  pragma novector
  for (i = 0; i < 8; ++i) {
    float y = tmp[i] - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }
  /* perform scalar Kahan sum of loop remainer */
#  pragma novector
  for (i = n - rem; i < n; ++i) {
    float prod = u[i] * v[i];
    float y = prod - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }
  return sum;
}

}  // namespace tesseract.

#endif  // ANDROID_BUILD
