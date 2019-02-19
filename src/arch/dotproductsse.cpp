///////////////////////////////////////////////////////////////////////
// File:        dotproductsse.cpp
// Description: Architecture-specific dot-product function.
// Author:      Ray Smith
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

#if !defined(__SSE4_1__)
#error Implementation only for SSE 4.1 capable architectures
#endif

#include <emmintrin.h>
#include <smmintrin.h>
#include <cstdint>
#include "dotproductsse.h"

// TODO:
// Use _mm_dp_pd, _mm_dp_ps?

namespace tesseract {

// Computes and returns the dot product of the n-vectors u and v.
// Uses Intel SSE intrinsics to access the SIMD instruction set.
// See https://software.intel.com/sites/landingpage/IntrinsicsGuide/.

// Implementation using double.
double DotProductSSE(const double* u, const double* v, int n) {
  const unsigned quot = n / 4;
  const unsigned rem = n % 4;
  __m128d t0 = _mm_setzero_pd();
  __m128d t1 = _mm_setzero_pd();
  for (unsigned k = 0; k < quot; k++) {
    __m128d f0 = _mm_loadu_pd(u);
    __m128d f1 = _mm_loadu_pd(v);
    f0 = _mm_mul_pd(f0, f1);
    t0 = _mm_add_pd(t0, f0);
    u += 2;
    v += 2;
    __m128d f2 = _mm_loadu_pd(u);
    __m128d f3 = _mm_loadu_pd(v);
    f2 = _mm_mul_pd(f2, f3);
    t1 = _mm_add_pd(t1, f2);
    u += 2;
    v += 2;
  }
  t0 = _mm_hadd_pd(t0, t1);
  double tmp[2];
  _mm_store_pd(tmp, t0);
  double result = tmp[0] + tmp[1];
  for (unsigned k = 0; k < rem; k++) {
    result += *u++ * *v++;
  }
  return result;
}

// Implementation using float.
float DotProductSSE(const float* u, const float* v, int n) {
  const unsigned quot = n / 8;
  const unsigned rem = n % 8;
  __m128 t0 = _mm_setzero_ps();
  __m128 t1 = _mm_setzero_ps();
  for (unsigned k = 0; k < quot; k++) {
    __m128 f0 = _mm_loadu_ps(u);
    __m128 f1 = _mm_loadu_ps(v);
    f0 = _mm_mul_ps(f0, f1);
    t0 = _mm_add_ps(t0, f0);
    u += 4;
    v += 4;
    __m128 f2 = _mm_loadu_ps(u);
    __m128 f3 = _mm_loadu_ps(v);
    f2 = _mm_mul_ps(f2, f3);
    t1 = _mm_add_ps(t1, f2);
    u += 4;
    v += 4;
  }
  t0 = _mm_hadd_ps(t0, t1);
  float tmp[4];
  _mm_store_ps(tmp, t0);
  float result = tmp[0] + tmp[1] + tmp[2] + tmp[3];
  for (unsigned k = 0; k < rem; k++) {
    result += *u++ * *v++;
  }
  return result;
}

}  // namespace tesseract.
