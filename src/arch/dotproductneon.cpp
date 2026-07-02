///////////////////////////////////////////////////////////////////////
// File:        dotproductneon.cpp
// Description: Dot product function for ARM NEON.
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

#if defined(__ARM_NEON)

#include <arm_neon.h>
#include "dotproduct.h"

namespace tesseract {

// Documentation:
// https://developer.arm.com/architectures/instruction-sets/intrinsics/

#if defined(FAST_FLOAT) && defined(__ARM_ARCH_ISA_A64)

float DotProductNEON(const float *u, const float *v, int n) {
  float32x4_t result0 = vdupq_n_f32(0.0f);
  float32x4_t result1 = vdupq_n_f32(0.0f);
  float32x4_t result2 = vdupq_n_f32(0.0f);
  float32x4_t result3 = vdupq_n_f32(0.0f);
  while (n >= 16) {
    float32x4_t u0 = vld1q_f32(u);
    float32x4_t v0 = vld1q_f32(v);
    float32x4_t u4 = vld1q_f32(u + 4);
    float32x4_t v4 = vld1q_f32(v + 4);
    float32x4_t u8 = vld1q_f32(u + 8);
    float32x4_t v8 = vld1q_f32(v + 8);
    float32x4_t u12 = vld1q_f32(u + 12);
    float32x4_t v12 = vld1q_f32(v + 12);
    result0 = vfmaq_f32(result0, u0, v0);
    result1 = vfmaq_f32(result1, u4, v4);
    result2 = vfmaq_f32(result2, u8, v8);
    result3 = vfmaq_f32(result3, u12, v12);
    u += 16;
    v += 16;
    n -= 16;
  }
  float total = vaddvq_f32(result0) + vaddvq_f32(result1);
  total += vaddvq_f32(result2) + vaddvq_f32(result3);
  while (n >= 4) {
    float32x4_t u0 = vld1q_f32(u);
    float32x4_t v0 = vld1q_f32(v);
    total += vaddvq_f32(vmulq_f32(u0, v0));
    u += 4;
    v += 4;
    n -= 4;
  }
  while (n > 0) {
    total += *u++ * *v++;
    n--;
  }
  return total;
}

#else

// Computes and returns the dot product of the two n-vectors u and v.
TFloat DotProductNEON(const TFloat *u, const TFloat *v, int n) {
  TFloat total = 0;
#if defined(OPENMP_SIMD) || defined(_OPENMP)
#pragma omp simd reduction(+:total)
#endif
  for (int k = 0; k < n; k++) {
    total += u[k] * v[k];
  }
  return total;
}

#endif

} // namespace tesseract

#endif /* __ARM_NEON */
