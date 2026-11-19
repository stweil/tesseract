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
#include <cstddef> // for size_t
#include <cstdint> // for int64_t
#include "dotproduct.h"

namespace tesseract {

// Returns the number of bits of the native SVE vector length of the CPU.
int SVENativeVectorLengthInBits() {
  return static_cast<int>(svcntw()) * 32;
}

#if defined(FAST_FLOAT)

// All loops below handle whole vectors as long as possible. The remaining
// elements are handled with a predicate which selects the valid elements,
// and with a merging multiply-add which keeps the sums of the other elements,
// so no scalar code is needed for them.
// The indices of the inner loops are 64 bit integers. With them the compiler
// can use the index register of the SVE load instructions directly. GCC 13
// needs an additional instruction for each load with 32 bit indices.

float DotProductSVE(const float *u, const float *v, int n) {
  const int64_t step = static_cast<int64_t>(svcntw());
  const svbool_t all = svptrue_b32();
  svfloat32_t sum0 = svdup_f32(0.0f);
  svfloat32_t sum1 = sum0, sum2 = sum0, sum3 = sum0;
  const int64_t size = n;
  int64_t i = 0;
  for (; i + 4 * step <= size; i += 4 * step) {
    sum0 = svmla_f32_x(all, sum0, svld1_f32(all, u + i), svld1_f32(all, v + i));
    sum1 = svmla_f32_x(all, sum1, svld1_f32(all, u + i + step),
                       svld1_f32(all, v + i + step));
    sum2 = svmla_f32_x(all, sum2, svld1_f32(all, u + i + 2 * step),
                       svld1_f32(all, v + i + 2 * step));
    sum3 = svmla_f32_x(all, sum3, svld1_f32(all, u + i + 3 * step),
                       svld1_f32(all, v + i + 3 * step));
  }
  for (; i + step <= size; i += step) {
    sum0 = svmla_f32_x(all, sum0, svld1_f32(all, u + i), svld1_f32(all, v + i));
  }
  if (i < size) {
    const svbool_t pg = svwhilelt_b32(i, size);
    sum1 = svmla_f32_m(pg, sum1, svld1_f32(pg, u + i), svld1_f32(pg, v + i));
  }
  sum0 = svadd_f32_x(all, sum0, sum1);
  sum2 = svadd_f32_x(all, sum2, sum3);
  return svaddv_f32(all, svadd_f32_x(all, sum0, sum2));
}

// Computes v[i] = w[i] . u for num_rows rows of w with n used elements each.
// Four rows are handled per pass, so u is loaded only once for them and
// eight independent sums are calculated at the same time.
void RowDotProductsSVE(const float *w, int w_stride, int num_rows,
                       const float *u, int n, float *v) {
  const int64_t step = static_cast<int64_t>(svcntw());
  const svbool_t all = svptrue_b32();
  const int64_t size = n;
  int i = 0;
  for (; i + 4 <= num_rows; i += 4) {
    const float *w0 = w + static_cast<size_t>(i) * w_stride;
    const float *w1 = w0 + w_stride;
    const float *w2 = w1 + w_stride;
    const float *w3 = w2 + w_stride;
    svfloat32_t a0 = svdup_f32(0.0f);
    svfloat32_t a1 = a0, a2 = a0, a3 = a0;
    svfloat32_t b0 = a0, b1 = a0, b2 = a0, b3 = a0;
    int64_t j = 0;
    for (; j + 2 * step <= size; j += 2 * step) {
      const svfloat32_t x = svld1_f32(all, u + j);
      const svfloat32_t y = svld1_f32(all, u + j + step);
      a0 = svmla_f32_x(all, a0, svld1_f32(all, w0 + j), x);
      b0 = svmla_f32_x(all, b0, svld1_f32(all, w0 + j + step), y);
      a1 = svmla_f32_x(all, a1, svld1_f32(all, w1 + j), x);
      b1 = svmla_f32_x(all, b1, svld1_f32(all, w1 + j + step), y);
      a2 = svmla_f32_x(all, a2, svld1_f32(all, w2 + j), x);
      b2 = svmla_f32_x(all, b2, svld1_f32(all, w2 + j + step), y);
      a3 = svmla_f32_x(all, a3, svld1_f32(all, w3 + j), x);
      b3 = svmla_f32_x(all, b3, svld1_f32(all, w3 + j + step), y);
    }
    if (j + step <= size) {
      const svfloat32_t x = svld1_f32(all, u + j);
      a0 = svmla_f32_x(all, a0, svld1_f32(all, w0 + j), x);
      a1 = svmla_f32_x(all, a1, svld1_f32(all, w1 + j), x);
      a2 = svmla_f32_x(all, a2, svld1_f32(all, w2 + j), x);
      a3 = svmla_f32_x(all, a3, svld1_f32(all, w3 + j), x);
      j += step;
    }
    if (j < size) {
      const svbool_t pg = svwhilelt_b32(j, size);
      const svfloat32_t x = svld1_f32(pg, u + j);
      b0 = svmla_f32_m(pg, b0, svld1_f32(pg, w0 + j), x);
      b1 = svmla_f32_m(pg, b1, svld1_f32(pg, w1 + j), x);
      b2 = svmla_f32_m(pg, b2, svld1_f32(pg, w2 + j), x);
      b3 = svmla_f32_m(pg, b3, svld1_f32(pg, w3 + j), x);
    }
    v[i] = svaddv_f32(all, svadd_f32_x(all, a0, b0));
    v[i + 1] = svaddv_f32(all, svadd_f32_x(all, a1, b1));
    v[i + 2] = svaddv_f32(all, svadd_f32_x(all, a2, b2));
    v[i + 3] = svaddv_f32(all, svadd_f32_x(all, a3, b3));
  }
  for (; i < num_rows; i++) {
    v[i] = DotProductSVE(w + static_cast<size_t>(i) * w_stride, u, n);
  }
}

// Computes out[i * out_stride + j] = u[i] . v[j] for the num_u rows of u and
// the num_v rows of v, which all have n elements and are stored without gaps.
// Four rows of u and four rows of v are handled per pass, so each of them is
// loaded only once for 16 sums which are calculated at the same time.
// The accumulators are named variables, because arrays of vectors without a
// fixed size are not allowed.
void CrossDotProductsSVE(const float *u, int num_u, const float *v, int num_v,
                         int n, float *out, int out_stride) {
  const int64_t step = static_cast<int64_t>(svcntw());
  const svbool_t all = svptrue_b32();
  const int64_t size = n;
  int i = 0;
  for (; i + 4 <= num_u; i += 4) {
    const float *u0 = u + static_cast<size_t>(i) * n;
    const float *u1 = u0 + n;
    const float *u2 = u1 + n;
    const float *u3 = u2 + n;
    float *out0 = out + static_cast<size_t>(i) * out_stride;
    float *out1 = out0 + out_stride;
    float *out2 = out1 + out_stride;
    float *out3 = out2 + out_stride;
    int j = 0;
    for (; j + 4 <= num_v; j += 4) {
      const float *v0 = v + static_cast<size_t>(j) * n;
      const float *v1 = v0 + n;
      const float *v2 = v1 + n;
      const float *v3 = v2 + n;
      svfloat32_t a0 = svdup_f32(0.0f);
      svfloat32_t a1 = a0, a2 = a0, a3 = a0;
      svfloat32_t b0 = a0, b1 = a0, b2 = a0, b3 = a0;
      svfloat32_t c0 = a0, c1 = a0, c2 = a0, c3 = a0;
      svfloat32_t d0 = a0, d1 = a0, d2 = a0, d3 = a0;
      int64_t k = 0;
      for (; k + step <= size; k += step) {
        const svfloat32_t y0 = svld1_f32(all, v0 + k);
        const svfloat32_t y1 = svld1_f32(all, v1 + k);
        const svfloat32_t y2 = svld1_f32(all, v2 + k);
        const svfloat32_t y3 = svld1_f32(all, v3 + k);
        svfloat32_t x = svld1_f32(all, u0 + k);
        a0 = svmla_f32_x(all, a0, x, y0);
        a1 = svmla_f32_x(all, a1, x, y1);
        a2 = svmla_f32_x(all, a2, x, y2);
        a3 = svmla_f32_x(all, a3, x, y3);
        x = svld1_f32(all, u1 + k);
        b0 = svmla_f32_x(all, b0, x, y0);
        b1 = svmla_f32_x(all, b1, x, y1);
        b2 = svmla_f32_x(all, b2, x, y2);
        b3 = svmla_f32_x(all, b3, x, y3);
        x = svld1_f32(all, u2 + k);
        c0 = svmla_f32_x(all, c0, x, y0);
        c1 = svmla_f32_x(all, c1, x, y1);
        c2 = svmla_f32_x(all, c2, x, y2);
        c3 = svmla_f32_x(all, c3, x, y3);
        x = svld1_f32(all, u3 + k);
        d0 = svmla_f32_x(all, d0, x, y0);
        d1 = svmla_f32_x(all, d1, x, y1);
        d2 = svmla_f32_x(all, d2, x, y2);
        d3 = svmla_f32_x(all, d3, x, y3);
      }
      if (k < size) {
        const svbool_t pg = svwhilelt_b32(k, size);
        const svfloat32_t y0 = svld1_f32(pg, v0 + k);
        const svfloat32_t y1 = svld1_f32(pg, v1 + k);
        const svfloat32_t y2 = svld1_f32(pg, v2 + k);
        const svfloat32_t y3 = svld1_f32(pg, v3 + k);
        svfloat32_t x = svld1_f32(pg, u0 + k);
        a0 = svmla_f32_m(pg, a0, x, y0);
        a1 = svmla_f32_m(pg, a1, x, y1);
        a2 = svmla_f32_m(pg, a2, x, y2);
        a3 = svmla_f32_m(pg, a3, x, y3);
        x = svld1_f32(pg, u1 + k);
        b0 = svmla_f32_m(pg, b0, x, y0);
        b1 = svmla_f32_m(pg, b1, x, y1);
        b2 = svmla_f32_m(pg, b2, x, y2);
        b3 = svmla_f32_m(pg, b3, x, y3);
        x = svld1_f32(pg, u2 + k);
        c0 = svmla_f32_m(pg, c0, x, y0);
        c1 = svmla_f32_m(pg, c1, x, y1);
        c2 = svmla_f32_m(pg, c2, x, y2);
        c3 = svmla_f32_m(pg, c3, x, y3);
        x = svld1_f32(pg, u3 + k);
        d0 = svmla_f32_m(pg, d0, x, y0);
        d1 = svmla_f32_m(pg, d1, x, y1);
        d2 = svmla_f32_m(pg, d2, x, y2);
        d3 = svmla_f32_m(pg, d3, x, y3);
      }
      out0[j] = svaddv_f32(all, a0);
      out0[j + 1] = svaddv_f32(all, a1);
      out0[j + 2] = svaddv_f32(all, a2);
      out0[j + 3] = svaddv_f32(all, a3);
      out1[j] = svaddv_f32(all, b0);
      out1[j + 1] = svaddv_f32(all, b1);
      out1[j + 2] = svaddv_f32(all, b2);
      out1[j + 3] = svaddv_f32(all, b3);
      out2[j] = svaddv_f32(all, c0);
      out2[j + 1] = svaddv_f32(all, c1);
      out2[j + 2] = svaddv_f32(all, c2);
      out2[j + 3] = svaddv_f32(all, c3);
      out3[j] = svaddv_f32(all, d0);
      out3[j + 1] = svaddv_f32(all, d1);
      out3[j + 2] = svaddv_f32(all, d2);
      out3[j + 3] = svaddv_f32(all, d3);
    }
    for (; j < num_v; j++) {
      const float *vj = v + static_cast<size_t>(j) * n;
      out0[j] = DotProductSVE(u0, vj, n);
      out1[j] = DotProductSVE(u1, vj, n);
      out2[j] = DotProductSVE(u2, vj, n);
      out3[j] = DotProductSVE(u3, vj, n);
    }
  }
  // The remaining rows of u are single vectors for the rows of v.
  for (; i < num_u; i++) {
    RowDotProductsSVE(v, n, num_v, u + static_cast<size_t>(i) * n, n,
                      out + static_cast<size_t>(i) * out_stride);
  }
}

#else

double DotProductSVE(const double *u, const double *v, int n) {
  const int step = static_cast<int>(svcntd());
  const svbool_t all = svptrue_b64();
  svfloat64_t sum0 = svdup_f64(0.0);
  svfloat64_t sum1 = svdup_f64(0.0);
  int i = 0;
  for (; i + 2 * step <= n; i += 2 * step) {
    sum0 = svmla_f64_x(all, sum0, svld1_f64(all, u + i), svld1_f64(all, v + i));
    sum1 = svmla_f64_x(all, sum1, svld1_f64(all, u + i + step),
                       svld1_f64(all, v + i + step));
  }
  // Fewer than two vectors remain. The predicate selects the valid elements
  // and the merging multiply-add keeps the sums of the other elements.
  for (; i < n; i += step) {
    const svbool_t pg = svwhilelt_b64(i, n);
    sum0 = svmla_f64_m(pg, sum0, svld1_f64(pg, u + i), svld1_f64(pg, v + i));
  }
  return svaddv_f64(all, svadd_f64_x(all, sum0, sum1));
}

#endif

} // namespace tesseract

#endif /* __ARM_FEATURE_SVE */