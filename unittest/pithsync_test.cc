// (C) Copyright 2026, Google Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <tesseract/baseapi.h>
#include <tesseract/resultiterator.h>

#include "include_gunit.h"

#include <leptonica/allheaders.h>

#include <cstdint>  // for INT16_MAX
#include <string>
#include <vector>

namespace tesseract {

// Regression test for an int16_t overflow in the pitch synchronization stage
// (check_pitch_sync3 / FPCUTPT::assign in src/textord/pithsync.cpp) that
// caused an out-of-bounds read/write and a crash (DoS) during layout analysis.
//
// The pitch-sync dynamic program walks a coordinate `x` across the whole row
// and, in its tail loop, advances past the right edge by one extra pitch:
// `while (x < right_edge + pitch)`. The row is bounded by the int16_t limits of
// the layout stage, but `right_edge + pitch` is evaluated in a wider type, so
// the tail can advance `x` above INT16_MAX (32767). The cut-point constructors
// (FPCUTPT::setup/assign/assign_cheap) used to accept the position as an
// int16_t, so once `x` reached 32768 it was truncated to -32768, turning
// `cutpts[x - array_origin]` into a large negative index and performing an
// out-of-bounds read/write on the heap-allocated cut-point vector.
//
// A text line whose last occupied column sits close to INT16_MAX with a pitch
// of about 13 or more reaches this state through the normal public layout path:
// TessBaseAPI::AnalyseLayout() -> FindLines() -> Tesseract::SegmentPage() ->
// Textord::TextordPage() -> make_words() -> compute_fixed_pitch() ->
// check_pitch_sync3() (the default, pitsync_linear_version=6).
//
// This test drives that public path with a 32767-pixel-wide image (one row of
// hollow glyph outlines at pitch 13, last glyph ending at x=32756) built in
// memory. The layout path does not require a traineddata model, so the test
// always runs. Under an AddressSanitizer build the pre-fix code aborts with an
// out-of-bounds access inside FPCUTPT::assign; the fixed code completes and
// yields a non-null layout iterator.
class PitchSyncRegressionTest : public testing::Test {};

// Build the PBM (P4) image buffer used to reproduce the overflow, matching the
// layout that drives the pitch-sync tail past INT16_MAX: a single row of
// hollow, 9x20 glyphs at a fixed pitch of 13 starting at x=14 on a 32767-wide
// image. The last glyph's right edge lands at x=32756.
static std::string MakePitchRowPbm(int width, int height, int left, int top, int glyph_width,
                                   int glyph_height, int pitch) {
  std::string pbm = "P4\n";
  pbm += std::to_string(width);
  pbm += ' ';
  pbm += std::to_string(height);
  pbm += '\n';

  const int row_bytes = (width + 7) / 8;
  const int bottom = top + glyph_height - 1;
  for (int y = 0; y < height; ++y) {
    std::vector<unsigned char> row(row_bytes, 0);
    if (top <= y && y <= bottom) {
      const bool horizontal = (y == top) || (y == bottom);
      for (int l = left; l <= width - glyph_width; l += pitch) {
        for (int x = l; x < l + glyph_width; ++x) {
          if (horizontal || x == l || x == l + glyph_width - 1) {
            row[x >> 3] |= 0x80 >> (x & 7);
          }
        }
      }
    }
    pbm.append(reinterpret_cast<const char *>(row.data()), row.size());
  }
  return pbm;
}

TEST_F(PitchSyncRegressionTest, WideRowTailIsInBounds) {
  // 32767 is just under INT16_MAX so the image passes the layout-stage width
  // limit, but right_edge (32756) + pitch (13) = 32769 pushes the tail past
  // INT16_MAX, where the pre-fix code truncated the coordinate to a negative
  // index into the cut-point vector.
  const int kWidth = 32767;
  const int kHeight = 60;
  const std::string pbm = MakePitchRowPbm(kWidth, kHeight, /*left=*/14, /*top=*/20,
                                          /*glyph_width=*/9, /*glyph_height=*/20, /*pitch=*/13);

  Pix *image = pixReadMem(reinterpret_cast<const l_uint8 *>(pbm.data()), pbm.size());
  ASSERT_NE(image, nullptr) << "Failed to read in-memory test image.";

  tesseract::TessBaseAPI api;
  api.SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
  api.SetImage(image);

  // This call is where the pre-fix overflow crashed. It must complete and
  // produce a layout (the row is real content, so it should be found).
  tesseract::PageIterator *layout = api.AnalyseLayout();
  EXPECT_NE(layout, nullptr);
  delete layout;

  pixDestroy(&image);
}

} // namespace tesseract
