///////////////////////////////////////////////////////////////////////
// File:        dotproduct_test.cc
// Author:      Stefan Weil
//
// Copyright 2026 Stefan Weil
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

#include "dotproduct.h"
#include <chrono>
#include <gtest/gtest.h>
#include <gtest/internal/gtest-port.h>
#include <iostream>
#include <iomanip> // for std::setw and std::left
#include <string>
#include <vector>
#include "include_gunit.h"
#include "simddetect.h"

namespace tesseract {

class DotProductTest : public ::testing::Test {
protected:
  void SetUp() override {
    std::locale::global(std::locale(""));
  }

  static TFloat DotProductGeneric(const TFloat *u, const TFloat *v, int n) {
    TFloat total = 0;
    for (int k = 0; k < n; ++k) {
      total += u[k] * v[k];
    }
    return total;
  }

  static void InitRandom(std::vector<TFloat> &vec) {
    TRand random;
    for (auto &v : vec) {
      v = static_cast<TFloat>(random.SignedRand(1.0));
    }
  }

  void ExpectEqualResults(DotProductFunction func, const char *name) {
    std::vector<TFloat> u(500), v(500);
    InitRandom(u);
    InitRandom(v);
    TFloat expected = DotProductGeneric(u.data(), v.data(), 500);
    TFloat result = func(u.data(), v.data(), 500);
    EXPECT_FLOAT_EQ(expected, result) << name;
  }

  static void MeasurePerformance(DotProductFunction func, const char *name,
                                 int n, int iterations) {
    std::vector<TFloat> u(n), v(n);
    InitRandom(u);
    InitRandom(v);

    // Warmup
    func(u.data(), v.data(), n);

    // Measure
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
      func(u.data(), v.data(), n);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(end - start).count();
    double gflops = (n * iterations) / elapsed / 1e9;

    std::cout << "  " << std::setw(10) << name << ": " << elapsed << "s, "
              << gflops << " GFLOPS" << std::endl;
  }

  TRand random_;
};

// Test correctness of the current default implementation.
TEST_F(DotProductTest, Correctness) {
  std::vector<TFloat> u(1000), v(1000);
  InitRandom(u);
  InitRandom(v);
  TFloat expected = DotProductGeneric(u.data(), v.data(), 1000);
  TFloat result = DotProduct(u.data(), v.data(), 1000);
  // SIMD implementations may have different accumulation order due to
  // different instruction sequences (e.g., vfma vs separate mul/add).
  // The relative error is typically < 1e-6 for float32.
  EXPECT_NEAR(expected, result, std::abs(expected) * 1e-5)
      << "Default DotProduct";
}

// Test Neon implementation.
TEST_F(DotProductTest, NEON) {
#if defined(HAVE_NEON) || defined(__aarch64__)
  if (!SIMDDetect::IsNEONAvailable()) {
    GTEST_LOG_(INFO) << "No NEON found! Not tested!";
    GTEST_SKIP();
  }
  ExpectEqualResults(DotProductNEON, "DotProductNEON");
#else
  GTEST_LOG_(INFO) << "NEON unsupported! Not tested!";
  GTEST_SKIP();
#endif
}

// Test SSE implementation.
TEST_F(DotProductTest, SSE) {
#if defined(HAVE_SSE4_1)
  if (!SIMDDetect::IsSSEAvailable()) {
    GTEST_LOG_(INFO) << "No SSE found! Not tested!";
    GTEST_SKIP();
  }
  ExpectEqualResults(DotProductSSE, "DotProductSSE");
#else
  GTEST_LOG_(INFO) << "SSE unsupported! Not tested!";
  GTEST_SKIP();
#endif
}

// Test AVX implementation.
TEST_F(DotProductTest, AVX) {
#if defined(HAVE_AVX)
  if (!SIMDDetect::IsAVXAvailable()) {
    GTEST_LOG_(INFO) << "No AVX found! Not tested!";
    GTEST_SKIP();
  }
  ExpectEqualResults(DotProductAVX, "DotProductAVX");
#else
  GTEST_LOG_(INFO) << "AVX unsupported! Not tested!";
  GTEST_SKIP();
#endif
}

// Test FMA implementation.
TEST_F(DotProductTest, FMA) {
#if defined(HAVE_FMA)
  if (!SIMDDetect::IsFMAAvailable()) {
    GTEST_LOG_(INFO) << "No FMA found! Not tested!";
    GTEST_SKIP();
  }
  ExpectEqualResults(DotProductFMA, "DotProductFMA");
#else
  GTEST_LOG_(INFO) << "FMA unsupported! Not tested!";
  GTEST_SKIP();
#endif
}

// Test SVE implementation.
TEST_F(DotProductTest, SVE) {
#if defined(__ARM_FEATURE_SVE)
  if (!SIMDDetect::IsSVEAvailable()) {
    GTEST_LOG_(INFO) << "No SVE found! Not tested!";
    GTEST_SKIP();
  }
  ExpectEqualResults(DotProductSVE, "DotProductSVE");
#else
  GTEST_LOG_(INFO) << "SVE unsupported! Not tested!";
  GTEST_SKIP();
#endif
}

// Performance benchmark - runs and reports GFLOPS for available implementations.
TEST_F(DotProductTest, Performance) {
  std::cout << "DotProduct Performance:" << std::endl;

  const int n = 1000000;
  const int iterations = 1000;

  // Generic baseline
  MeasurePerformance(DotProductGeneric, "Generic", n, iterations);

  // Default implementation
  MeasurePerformance(DotProduct, "Default", n, iterations);

#if defined(HAVE_NEON) || defined(__aarch64__)
  if (SIMDDetect::IsNEONAvailable()) {
    MeasurePerformance(DotProductNEON, "NEON", n, iterations);
  }
#endif

#if defined(HAVE_SSE4_1)
  if (SIMDDetect::IsSSEAvailable()) {
    MeasurePerformance(DotProductSSE, "SSE", n, iterations);
  }
#endif

#if defined(HAVE_AVX)
  if (SIMDDetect::IsAVXAvailable()) {
    MeasurePerformance(DotProductAVX, "AVX", n, iterations);
  }
#endif

#if defined(HAVE_FMA)
  if (SIMDDetect::IsFMAAvailable()) {
    MeasurePerformance(DotProductFMA, "FMA", n, iterations);
  }
#endif

#if defined(__ARM_FEATURE_SVE)
  if (SIMDDetect::IsSVEAvailable()) {
    MeasurePerformance(DotProductSVE, "SVE", n, iterations);
  }
#endif

#if defined(HAVE_FRAMEWORK_ACCELERATE)
    MeasurePerformance(DotProductAccelerate, "Accelerate", n, iterations);
#endif

  // Ensure the test doesn't fail due to performance variations.
  SUCCEED();
}

} // namespace tesseract
