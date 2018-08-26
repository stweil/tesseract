// (C) Copyright 2017, Google Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//~ #include <vector>

#include "params_model.h"

#include "include_gunit.h"

namespace {

// Test some basic I/O of params model files (automated learning of language
// model weights).
class ParamsModelTest : public testing::Test {
 protected:
  string TestDataNameToPath(const string& name) const {
    return file::JoinPath(FLAGS_test_srcdir, "testdata/" + name);
  }
  string OutputNameToPath(const string& name) const {
    return file::JoinPath(FLAGS_test_tmpdir, name);
  }
  // Test that we are able to load a params model, save it, reload it,
  // and verify that the re-serialized version is the same as the original.
  void TestParamsModelRoundTrip(const string& params_model_filename) const {
    tesseract::ParamsModel orig_model;
    tesseract::ParamsModel duplicate_model;
    string orig_file = TestDataNameToPath(params_model_filename);
    string out_file = OutputNameToPath(params_model_filename);

    EXPECT_TRUE(orig_model.LoadFromFile("eng", orig_file.c_str()));
    EXPECT_TRUE(orig_model.SaveToFile(out_file.c_str()));

    EXPECT_TRUE(duplicate_model.LoadFromFile("eng", out_file.c_str()));
    EXPECT_TRUE(orig_model.Equivalent(duplicate_model));
  }
};

TEST_F(ParamsModelTest, TestEngParamsModelIO) {
  TestParamsModelRoundTrip("eng.params_model");
}

}  // namespace
