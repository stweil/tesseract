///////////////////////////////////////////////////////////////////////
// File:        dotproductsve.cpp
// Description: Dot product function for ARM SVE.
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

#if defined(__ARM_FEATURE_SVE)

#include <arm_sve.h>
#include "dotproduct.h"

namespace tesseract {

#if defined(FAST_FLOAT)

float DotProductSVE(const float *u, const float *v, int n) {
  svfloat32_t sum0 = svdup_f32(0.0f);
  svfloat32_t sum1 = svdup_f32(0.0f);
  int i = 0;
  while (i + svcntw() * 2 <= n) {
    svfloat32_t u0 = svld1_f32(svptrue_b32(), u + i);
    svfloat32_t v0 = svld1_f32(svptrue_b32(), v + i);
    sum0 = svmla_f32_z(svptrue_b32(), sum0, u0, v0);
    i += svcntw();
    svfloat32_t u1 = svld1_f32(svptrue_b32(), u + i);
    svfloat32_t v1 = svld1_f32(svptrue_b32(), v + i);
    sum1 = svmla_f32_z(svptrue_b32(), sum1, u1, v1);
    i += svcntw();
  }
  while (i < n) {
    int remaining = n - i;
    svbool_t pg = svwhilelt_b32(0, remaining);
    svfloat32_t u0 = svld1_f32(pg, u + i);
    svfloat32_t v0 = svld1_f32(pg, v + i);
    sum0 = svmla_f32_z(pg, sum0, u0, v0);
    break;
  }
  sum0 = svadd_f32_z(svptrue_b32(), sum0, sum1);
  return svaddv_f32(svptrue_b32(), sum0);
}

#else

double DotProductSVE(const double *u, const double *v, int n) {
  svfloat64_t sum0 = svdup_f64(0.0);
  svfloat64_t sum1 = svdup_f64(0.0);
  int i = 0;
  while (i + svcntd() * 2 <= n) {
    svfloat64_t u0 = svld1_f64(svptrue_b64(), u + i);
    svfloat64_t v0 = svld1_f64(svptrue_b64(), v + i);
    sum0 = svmla_f64_z(svptrue_b64(), sum0, u0, v0);
    i += svcntd();
    svfloat64_t u1 = svld1_f64(svptrue_b64(), u + i);
    svfloat64_t v1 = svld1_f64(svptrue_b64(), v + i);
    sum1 = svmla_f64_z(svptrue_b64(), sum1, u1, v1);
    i += svcntd();
  }
  while (i < n) {
    int remaining = n - i;
    svbool_t pg = svwhilelt_b64(0, remaining);
    svfloat64_t u0 = svld1_f64(pg, u + i);
    svfloat64_t v0 = svld1_f64(pg, v + i);
    sum0 = svmla_f64_z(pg, sum0, u0, v0);
    break;
  }
  sum0 = svadd_f64_z(svptrue_b64(), sum0, sum1);
  return svaddv_f64(svptrue_b64(), sum0);
}

#endif

} // namespace tesseract

#endif /* __ARM_FEATURE_SVE */