///////////////////////////////////////////////////////////////////////
// File:        dotproduct.h
// Description: Native dot product function.
//
// (C) Copyright 2018, Google Inc.
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

namespace tesseract {

// Computes and returns the dot product of the two n-vectors u and v.
double DotProductNative(const double* u, const double* v, int n) {
  double t0 = 0.0;
  double t1 = 0.0;
  double t2 = 0.0;
  double t3 = 0.0;
  const unsigned quot = n / 4;
  const unsigned rem = n % 4;
  for (unsigned k = 0; k < quot; ++k) {
    t0 += *u++ * *v++;
    t1 += *u++ * *v++;
    t2 += *u++ * *v++;
    t3 += *u++ * *v++;
  }
  t0 += t1;
  t2 += t3;
  t0 += t2;
  for (unsigned k = 0; k < rem; ++k) {
    t0 += *u++ * *v++;
  }
  return t0;
}

// Computes and returns the dot product of the two n-vectors u and v.
float DotProductNative(const float* u, const float* v, int n) {
  float t0 = 0.0f;
  float t1 = 0.0f;
  float t2 = 0.0f;
  float t3 = 0.0f;
  const unsigned quot = n / 4;
  const unsigned rem = n % 4;
  for (unsigned k = 0; k < quot; ++k) {
    t0 += *u++ * *v++;
    t1 += *u++ * *v++;
    t2 += *u++ * *v++;
    t3 += *u++ * *v++;
  }
  t0 += t1;
  t2 += t3;
  t0 += t2;
  for (unsigned k = 0; k < rem; ++k) {
    t0 += *u++ * *v++;
  }
  return t0;
}

}  // namespace tesseract
