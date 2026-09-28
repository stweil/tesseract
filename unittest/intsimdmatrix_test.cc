///////////////////////////////////////////////////////////////////////
// File:        intsimdmatrix_test.cc
// Author:      rays@google.com (Ray Smith)
//
// Copyright 2017 Google Inc. All Rights Reserved.
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
#include <gtest/gtest.h>
#include <gtest/internal/gtest-port.h>
#include <chrono>
#include <iomanip> // for std::setw and std::left
#include <iostream>
#include <memory>
#include <vector>
#include "include_gunit.h"
#include "matrix.h"
#include "simddetect.h"

namespace tesseract {

class IntSimdMatrixTest : public ::testing::Test {
protected:
  void SetUp() override {
    std::locale::global(std::locale(""));
  }

  // Makes a random weights matrix of the given size.
  GENERIC_2D_ARRAY<int8_t> InitRandom(int no, int ni) {
    GENERIC_2D_ARRAY<int8_t> a(no, ni, 0);
    for (int i = 0; i < no; ++i) {
      for (int j = 0; j < ni; ++j) {
        a(i, j) = static_cast<int8_t>(random_.SignedRand(INT8_MAX));
      }
    }
    return a;
  }
  // Makes a random input vector of the given size, with rounding up.
  std::vector<int8_t> RandomVector(int size, const IntSimdMatrix &matrix) {
    int rounded_size = matrix.RoundInputs(size);
    std::vector<int8_t> v(rounded_size, 0);
    for (int i = 0; i < size; ++i) {
      v[i] = static_cast<int8_t>(random_.SignedRand(INT8_MAX));
    }
    return v;
  }
  // Makes a random scales vector of the given size.
  std::vector<TFloat> RandomScales(int size) {
    std::vector<TFloat> v(size);
    for (int i = 0; i < size; ++i) {
      v[i] = (1.0 + random_.SignedRand(1.0)) / INT8_MAX;
    }
    return v;
  }
  // Tests a range of sizes and compares the results against the generic version.
  void ExpectEqualResults(const IntSimdMatrix &matrix) {
    TFloat total = 0.0;
    for (int num_out = 1; num_out < 130; ++num_out) {
      for (int num_in = 1; num_in < 130; ++num_in) {
        GENERIC_2D_ARRAY<int8_t> w = InitRandom(num_out, num_in + 1);
        std::vector<int8_t> u = RandomVector(num_in, matrix);
        std::vector<TFloat> scales = RandomScales(num_out);
        // The kernel under test may round the output to a different size than
        // the globally selected one, so size the result buffer for it.
        int ro = matrix.matrixDotVectorFunction != nullptr
                     ? matrix.RoundOutputs(num_out)
                     : num_out;
        std::vector<TFloat> base_result(num_out);
        IntSimdMatrix::MatrixDotVector(w, scales, u.data(), base_result.data());
        std::vector<TFloat> test_result(ro);
        std::vector<int8_t> shaped_wi;
        int32_t rounded_num_out;
        matrix.Init(w, shaped_wi, rounded_num_out);
        scales.resize(rounded_num_out);
        if (matrix.matrixDotVectorFunction) {
          matrix.matrixDotVectorFunction(w.dim1(), w.dim2(), &shaped_wi[0], &scales[0], &u[0],
                                         &test_result[0]);
        } else {
          IntSimdMatrix::MatrixDotVector(w, scales, u.data(), test_result.data());
        }
        for (int i = 0; i < num_out; ++i) {
          EXPECT_FLOAT_EQ(base_result[i], test_result[i]) << "i=" << i;
          total += base_result[i];
        }
      }
    }
    // Compare sum of all results with expected value.
#ifdef FAST_FLOAT
    EXPECT_FLOAT_EQ(total, -423236.53f);
#else
    EXPECT_FLOAT_EQ(total, -423243.392011);
#endif
  }

  // Measures the throughput of a matrix kernel and reports it in GFLOPS.
  // When matrix.matrixDotVectorFunction is null (the generic C kernel), the
  // base class implementation IntSimdMatrix::MatrixDotVector is timed instead.
  static void MeasurePerformance(const IntSimdMatrix &matrix, const char *name,
                                 int num_out, int num_in, int iterations) {
    TRand random;
    GENERIC_2D_ARRAY<int8_t> w = [&]() {
      GENERIC_2D_ARRAY<int8_t> a(num_out, num_in + 1, 0);
      for (int i = 0; i < num_out; ++i) {
        for (int j = 0; j < num_in + 1; ++j) {
          a(i, j) = static_cast<int8_t>(random.SignedRand(INT8_MAX));
        }
      }
      return a;
    }();
    std::vector<int8_t> u(matrix.RoundInputs(num_in), 0);
    for (int i = 0; i < num_in; ++i) {
      u[i] = static_cast<int8_t>(random.SignedRand(INT8_MAX));
    }
    std::vector<TFloat> scales(num_out);
    for (int i = 0; i < num_out; ++i) {
      scales[i] = (1.0 + random.SignedRand(1.0)) / INT8_MAX;
    }
    const bool have_fn = matrix.matrixDotVectorFunction != nullptr;
    std::vector<int8_t> shaped_wi;
    int rounded_num_out;
    std::vector<TFloat> result;
    if (have_fn) {
      matrix.Init(w, shaped_wi, rounded_num_out);
      scales.resize(rounded_num_out);
      result.resize(rounded_num_out);
    } else {
      result.resize(num_out);
    }
    auto run = [&]() {
      if (have_fn) {
        matrix.matrixDotVectorFunction(num_out, num_in, &shaped_wi[0], &scales[0], &u[0],
                                       &result[0]);
      } else {
        IntSimdMatrix::MatrixDotVector(w, scales, u.data(), result.data());
      }
    };
    // Warmup.
    run();
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
      run();
    }
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    // Two floating point operations per (output, input) pair.
    double gflops = (2.0 * num_out * num_in * iterations) / elapsed / 1e9;
    std::cout << "  " << std::setw(18) << std::left << name << ": " << elapsed << "s, "
              << gflops << " GFLOPS" << std::endl;
  }

  TRand random_;
};

// Test the C++ implementation without SIMD.
TEST_F(IntSimdMatrixTest, C) {
  static const IntSimdMatrix matrix = {nullptr, 1, 1, 1, 1};
  ExpectEqualResults(matrix);
}

// Tests that the SSE implementation gets the same result as the vanilla.
TEST_F(IntSimdMatrixTest, SSE) {
#if defined(HAVE_SSE4_1)
  if (!SIMDDetect::IsSSEAvailable()) {
    GTEST_LOG_(INFO) << "No SSE found! Not tested!";
    GTEST_SKIP();
  }
  ExpectEqualResults(IntSimdMatrix::intSimdMatrixSSE);
#else
  GTEST_LOG_(INFO) << "SSE unsupported! Not tested!";
  GTEST_SKIP();
#endif
}

// Tests that the AVX2 implementation gets the same result as the vanilla.
TEST_F(IntSimdMatrixTest, AVX2) {
#if defined(HAVE_AVX2)
  if (!SIMDDetect::IsAVX2Available()) {
    GTEST_LOG_(INFO) << "No AVX2 found! Not tested!";
    GTEST_SKIP();
  }
  ExpectEqualResults(IntSimdMatrix::intSimdMatrixAVX2);
#else
  GTEST_LOG_(INFO) << "AVX2 unsupported! Not tested!";
  GTEST_SKIP();
#endif
}

// Tests that the AVX512-VNNI implementation gets the same result as the
// vanilla. Skipped when the CPU does not implement VNNI.
TEST_F(IntSimdMatrixTest, AVX512VNNI) {
#if defined(HAVE_AVX512VNNI)
  if (!SIMDDetect::IsAVX512VNNIAvailable()) {
    GTEST_LOG_(INFO) << "No AVX512-VNNI found! Not tested!";
    GTEST_SKIP();
  }
  ExpectEqualResults(IntSimdMatrix::intSimdMatrixAVX512VNNI);
#else
  GTEST_LOG_(INFO) << "AVX512-VNNI unsupported! Not tested!";
  GTEST_SKIP();
#endif
}

// Tests that the NEON implementation gets the same result as the vanilla.
TEST_F(IntSimdMatrixTest, NEON) {
#if defined(HAVE_NEON)
  if (!SIMDDetect::IsNEONAvailable()) {
    GTEST_LOG_(INFO) << "No NEON found! Not tested!";
    GTEST_SKIP();
  }
  ExpectEqualResults(IntSimdMatrix::intSimdMatrixNEON);
#else
  GTEST_LOG_(INFO) << "NEON unsupported! Not tested!";
  GTEST_SKIP();
#endif
}

// Tests that the NEON dotprod (SDOT) implementation gets the same result as
// the vanilla. Skipped when the CPU does not implement the optional ARMv8.2-A
// dotprod instruction.
TEST_F(IntSimdMatrixTest, NEON_DotProd) {
#if defined(__aarch64__) || defined(__ARM_FEATURE_DOTPROD)
  if (!SIMDDetect::IsNEONAvailable()) {
    GTEST_LOG_(INFO) << "No NEON found! Not tested!";
    GTEST_SKIP();
  }
  if (!SIMDDetect::IsDotProdAvailable()) {
    GTEST_LOG_(INFO) << "No dotprod (SDOT) found! Not tested!";
    GTEST_SKIP();
  }
  ExpectEqualResults(IntSimdMatrix::intSimdMatrixNEONDotProd);
#else
  GTEST_LOG_(INFO) << "dotprod (SDOT) unsupported! Not tested!";
  GTEST_SKIP();
#endif
}

// Performance benchmark - runs and reports GFLOPS for the available int8
// matrix kernels, like DotProductTest.Performance.
TEST_F(IntSimdMatrixTest, Performance) {
  std::cout << "IntSimdMatrix Performance:" << std::endl;

  const int num_out = 128;
  const int num_in = 1024;
  const int iterations = 100;

  // Generic C++ implementation (null matrixDotVectorFunction).
  static const IntSimdMatrix c_matrix = {nullptr, 1, 1, 1, 1};
  MeasurePerformance(c_matrix, "C", num_out, num_in, iterations);

  if (IntSimdMatrix::intSimdMatrix != nullptr) {
    MeasurePerformance(*IntSimdMatrix::intSimdMatrix, "Default", num_out,
                       num_in, iterations);
  }
#if defined(HAVE_SSE4_1)
  if (SIMDDetect::IsSSEAvailable()) {
    MeasurePerformance(IntSimdMatrix::intSimdMatrixSSE, "SSE", num_out, num_in,
                       iterations);
  }
#endif
#if defined(HAVE_AVX2)
  if (SIMDDetect::IsAVX2Available()) {
    MeasurePerformance(IntSimdMatrix::intSimdMatrixAVX2, "AVX2", num_out,
                       num_in, iterations);
  }
#endif
#if defined(HAVE_AVX512VNNI)
  if (SIMDDetect::IsAVX512VNNIAvailable()) {
    MeasurePerformance(IntSimdMatrix::intSimdMatrixAVX512VNNI, "AVX512VNNI",
                       num_out, num_in, iterations);
  }
#endif
#if defined(HAVE_NEON)
  if (SIMDDetect::IsNEONAvailable()) {
    MeasurePerformance(IntSimdMatrix::intSimdMatrixNEON, "NEON", num_out,
                       num_in, iterations);
  }
#endif
#if defined(__aarch64__) || defined(__ARM_FEATURE_DOTPROD)
  if (SIMDDetect::IsNEONAvailable() && SIMDDetect::IsDotProdAvailable()) {
    MeasurePerformance(IntSimdMatrix::intSimdMatrixNEONDotProd, "NEON_DotProd",
                       num_out, num_in, iterations);
  }
#endif

  // Ensure the test doesn't fail due to performance variations.
  SUCCEED();
}

} // namespace tesseract
