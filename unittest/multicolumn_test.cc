// (C) Copyright 2026, Tesseract contributors.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "include_gunit.h"

#include <tesseract/baseapi.h>
#include <tesseract/renderer.h>

#include "image.h" // for Image

#include <memory>
#include <string>

namespace tesseract {

// Test the automatic multi-column (newspaper) recognition path
// (textord_auto_columns). It detects the vertical gutters between columns,
// recognises each column independently and merges the per-column results back
// into a single page. When no gutter is found it must fall back to the normal
// single-page path and produce exactly the same output.
//
// These tests double as the AddressSanitizer/UndefinedBehaviorSanitizer
// coverage for that path: Recognizing a multi-column page moves BLOCK /
// BLOCK_RES / ROW / WERD / C_BLOB nodes between BLOCK_LIST and PAGE_RES
// containers and restores the full-page thresholded image, which is exactly
// the kind of intrusive-list / pix ownership handling the sanitizer build
// wants to exercise.
//
// 8071_093.3B.tif is a genuine two-column newspaper page that the gutter
// detector triggers on (it splits into two columns), which drives the full
// split + recognise + merge path. phototest.tif is a single-column page that
// yields no gutter, which drives the safe no-split fallback.
class FriendlyTessBaseAPI : public tesseract::TessBaseAPI {
 public:
  // Expose the (protected) number of columns the automatic multi-column path
  // used for the last recognised page, without adding a public API method.
  int auto_columns_used() const {
    return auto_columns_;
  }
};

class MultiColumnTest : public testing::Test {
 protected:
  static std::string TestDataNameToPath(const std::string &name) {
    return file::JoinPath(TESTING_DIR, name);
  }
  static std::string TessdataPath() {
    return TESSDATA_DIR;
  }

  void SetUp() override {
    REQUIRE_TRAINEDDATA(TESSDATA_DIR, "eng");
    if (api_.Init(TessdataPath().c_str(), "eng", tesseract::OEM_LSTM_ONLY) == -1) {
      // eng.traineddata not found.
      GTEST_SKIP();
    }
    api_.SetPageSegMode(tesseract::PSM_AUTO);
  }

  // Recognise the named test image with textord_auto_columns set to |enable|.
  // On success (assertion failures abort the test), the UTF-8 text and the
  // number of columns the automatic path used (0 => it fell back to the normal
  // single-page path) are stored in last_text_ and last_columns_.
  void Recognise(const std::string &image, const std::string &enable) {
    Image pix = pixRead(TestDataNameToPath(image).c_str());
    ASSERT_TRUE(pix) << "Could not read " << image;
    ASSERT_TRUE(api_.SetVariable("textord_auto_columns", enable.c_str()));
    api_.SetInputName(image.c_str());
    api_.SetImage(pix);
    pix.destroy();  // SetImage clones, so the caller's copy is no longer needed.
    ASSERT_EQ(api_.Recognize(nullptr), 0) << "Recognize failed on " << image;
    last_columns_ = api_.auto_columns_used();
    char *utf8 = api_.GetUTF8Text();
    ASSERT_TRUE(utf8 != nullptr);
    last_text_ = utf8;
    delete[] utf8;
  }

  FriendlyTessBaseAPI api_;
  std::string last_text_;
  int last_columns_ = 0;
};

// A genuine multi-column page must be split into more than one column, and
// the merged result must contain recognisable text and well-formed layout.
TEST_F(MultiColumnTest, SplitsMultiColumnPage) {
  Recognise("8071_093.3B.tif", "1");
  // The detector must have found a gutter, so the automatic path -- not the
  // fallback -- ran, and it split the page into at least two columns.
  EXPECT_GE(last_columns_, 2) << "Expected the automatic path to split 8071 into "
                                 "columns, but it reported "
                              << last_columns_;

  // Merged text must be non-trivial.
  EXPECT_GE(last_text_.size(), 50) << "Multi-column page produced too little text";

  // hOCR rendering walks every merged line/word box (whose coordinates were
  // shifted into full-page space). It must succeed and emit word boxes.
  ASSERT_TRUE(api_.SetVariable("textord_auto_columns", "1"));
  Image pix = pixRead(TestDataNameToPath("8071_093.3B.tif").c_str());
  ASSERT_TRUE(pix);
  api_.SetInputName("8071_093.3B.tif");
  api_.SetImage(pix);
  pix.destroy();
  ASSERT_EQ(api_.Recognize(nullptr), 0);
  char *hocr = api_.GetHOCRText(0);
  ASSERT_TRUE(hocr != nullptr);
  const std::string hocr_str(hocr);
  EXPECT_NE(hocr_str.find("class='ocr_page'"), std::string::npos)
      << "hOCR output is missing the page element";
  EXPECT_NE(hocr_str.find("ocrx_word"), std::string::npos)
      << "hOCR output is missing word boxes";
  delete[] hocr;
}

// On a single-column page the automatic path must not split at all and must
// fall back to the normal path, yielding output identical to the flag-off run.
TEST_F(MultiColumnTest, NoSplitOnSingleColumnPage) {
  Recognise("phototest.tif", "0");
  const std::string off_text = last_text_;
  const int off_columns = last_columns_;

  Recognise("phototest.tif", "1");

  // No gutter => the automatic path fell back to the normal single-page path.
  EXPECT_EQ(off_columns, 0) << "Expected no column split with the flag off";
  EXPECT_EQ(last_columns_, 0) << "Expected no column split on single-column page";

  // The output must be byte-for-byte identical to the disabled run.
  EXPECT_EQ(off_text, last_text_) << "Flag-on output differs from flag-off output";
}

} // namespace tesseract
