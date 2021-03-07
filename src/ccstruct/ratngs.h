/**********************************************************************
 * File:        ratngs.h  (Formerly ratings.h)
 * Description: Definition of the WERD_CHOICE and BLOB_CHOICE classes.
 * Author:      Ray Smith
 *
 * (C) Copyright 1992, Hewlett-Packard Ltd.
 ** Licensed under the Apache License, Version 2.0 (the "License");
 ** you may not use this file except in compliance with the License.
 ** You may obtain a copy of the License at
 ** http://www.apache.org/licenses/LICENSE-2.0
 ** Unless required by applicable law or agreed to in writing, software
 ** distributed under the License is distributed on an "AS IS" BASIS,
 ** WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 ** See the License for the specific language governing permissions and
 ** limitations under the License.
 *
 **********************************************************************/

#ifndef           RATNGS_H
#define           RATNGS_H

#ifdef HAVE_CONFIG_H
#include "config_auto.h" // DISABLED_LEGACY_ENGINE
#endif

#include "clst.h"
#include "elst.h"
#ifndef DISABLED_LEGACY_ENGINE
#include "fontinfo.h"
#endif  // undef DISABLED_LEGACY_ENGINE
#include "matrix.h"
#include "unicharset.h"
#include "werd.h"

#include "genericvector.h"
#include <tesseract/unichar.h>

#include <cassert>
#include <cfloat>      // for FLT_MAX

namespace tesseract {

class MATRIX;
struct TBLOB;
struct TWERD;

// Enum to describe the source of a BLOB_CHOICE to make it possible to determine
// whether a blob has been classified by inspecting the BLOB_CHOICEs.
enum BlobChoiceClassifier {
  BCC_STATIC_CLASSIFIER,   // From the char_norm classifier.
  BCC_ADAPTED_CLASSIFIER,  // From the adaptive classifier.
  BCC_SPECKLE_CLASSIFIER,  // Backup for failed classification.
  BCC_AMBIG,               // Generated by ambiguity detection.
  BCC_FAKE,                // From some other process.
};

class BLOB_CHOICE: public ELIST_LINK
{
  public:
    BLOB_CHOICE() {
      unichar_id_ = UNICHAR_SPACE;
      fontinfo_id_ = -1;
      fontinfo_id2_ = -1;
      rating_ = 10.0;
      certainty_ = -1.0;
      script_id_ = -1;
      min_xheight_ = 0.0f;
      max_xheight_ = 0.0f;
      yshift_ = 0.0f;
      classifier_ = BCC_FAKE;
    }
    BLOB_CHOICE(UNICHAR_ID src_unichar_id,  // character id
                float src_rating,          // rating
                float src_cert,            // certainty
                int script_id,             // script
                float min_xheight,         // min xheight in image pixel units
                float max_xheight,         // max xheight allowed by this char
                float yshift,           // the larger of y shift (top or bottom)
                BlobChoiceClassifier c);   // adapted match or other
    BLOB_CHOICE(const BLOB_CHOICE &other);
    ~BLOB_CHOICE() = default;

    UNICHAR_ID unichar_id() const {
      return unichar_id_;
    }
    float rating() const {
      return rating_;
    }
    float certainty() const {
      return certainty_;
    }
    int16_t fontinfo_id() const {
      return fontinfo_id_;
    }
    int16_t fontinfo_id2() const {
      return fontinfo_id2_;
    }
  #ifndef DISABLED_LEGACY_ENGINE
    const std::vector<ScoredFont>& fonts() const {
      return fonts_;
    }
    void set_fonts(const std::vector<ScoredFont>& fonts) {
      fonts_ = fonts;
      int score1 = 0, score2 = 0;
      fontinfo_id_ = -1;
      fontinfo_id2_ = -1;
      for (auto&font : fonts_) {
        if (font.score > score1) {
          score2 = score1;
          fontinfo_id2_ = fontinfo_id_;
          score1 = font.score;
          fontinfo_id_ = font.fontinfo_id;
        } else if (font.score > score2) {
          score2 = font.score;
          fontinfo_id2_ = font.fontinfo_id;
        }
      }
    }
  #endif  // ndef DISABLED_LEGACY_ENGINE
    int script_id() const {
      return script_id_;
    }
    const MATRIX_COORD& matrix_cell() {
      return matrix_cell_;
    }
    float min_xheight() const {
      return min_xheight_;
    }
    float max_xheight() const {
      return max_xheight_;
    }
    float yshift() const {
      return yshift_;
    }
    BlobChoiceClassifier classifier() const {
      return classifier_;
    }
    bool IsAdapted() const {
      return classifier_ == BCC_ADAPTED_CLASSIFIER;
    }
    bool IsClassified() const {
      return classifier_ == BCC_STATIC_CLASSIFIER ||
             classifier_ == BCC_ADAPTED_CLASSIFIER ||
             classifier_ == BCC_SPECKLE_CLASSIFIER;
    }

    void set_unichar_id(UNICHAR_ID newunichar_id) {
      unichar_id_ = newunichar_id;
    }
    void set_rating(float newrat) {
      rating_ = newrat;
    }
    void set_certainty(float newrat) {
      certainty_ = newrat;
    }
    void set_script(int newscript_id) {
      script_id_ = newscript_id;
    }
    void set_matrix_cell(int col, int row) {
      matrix_cell_.col = col;
      matrix_cell_.row = row;
    }
    void set_classifier(BlobChoiceClassifier classifier) {
      classifier_ = classifier;
    }
    static BLOB_CHOICE* deep_copy(const BLOB_CHOICE* src) {
      auto* choice = new BLOB_CHOICE;
      *choice = *src;
      return choice;
    }
    // Returns true if *this and other agree on the baseline and x-height
    // to within some tolerance based on a given estimate of the x-height.
    bool PosAndSizeAgree(const BLOB_CHOICE& other, float x_height,
                         bool debug) const;

    void print(const UNICHARSET *unicharset) const {
      tprintf("r%.2f c%.2f x[%g,%g]: %d %s",
              rating_, certainty_,
              min_xheight_, max_xheight_, unichar_id_,
              (unicharset == nullptr) ? "" :
              unicharset->debug_str(unichar_id_).c_str());
    }
    void print_full() const {
      print(nullptr);
      tprintf(" script=%d, font1=%d, font2=%d, yshift=%g, classifier=%d\n",
              script_id_, fontinfo_id_, fontinfo_id2_, yshift_, classifier_);
    }
    // Sort function for sorting BLOB_CHOICEs in increasing order of rating.
    static int SortByRating(const void *p1, const void *p2) {
      const BLOB_CHOICE *bc1 = *static_cast<const BLOB_CHOICE *const *>(p1);
      const BLOB_CHOICE *bc2 = *static_cast<const BLOB_CHOICE *const *>(p2);
      return (bc1->rating_ < bc2->rating_) ? -1 : 1;
    }

 private:
  // Copy assignment operator.
  BLOB_CHOICE& operator=(const BLOB_CHOICE& other);

  UNICHAR_ID unichar_id_;          // unichar id
#ifndef DISABLED_LEGACY_ENGINE
  // Fonts and scores. Allowed to be empty.
  std::vector<ScoredFont> fonts_;
#endif  // ndef DISABLED_LEGACY_ENGINE
  int16_t fontinfo_id_;              // char font information
  int16_t fontinfo_id2_;             // 2nd choice font information
  // Rating is the classifier distance weighted by the length of the outline
  // in the blob. In terms of probability, classifier distance is -klog p such
  // that the resulting distance is in the range [0, 1] and then
  // rating = w (-k log p) where w is the weight for the length of the outline.
  // Sums of ratings may be compared meaningfully for words of different
  // segmentation.
  float rating_;                  // size related
  // Certainty is a number in [-20, 0] indicating the classifier certainty
  // of the choice. In terms of probability, certainty = 20 (k log p) where
  // k is defined as above to normalize -klog p to the range [0, 1].
  float certainty_;               // absolute
  int script_id_;
  // Holds the position of this choice in the ratings matrix.
  // Used to location position in the matrix during path backtracking.
  MATRIX_COORD matrix_cell_;
  // X-height range (in image pixels) that this classification supports.
  float min_xheight_;
  float max_xheight_;
  // yshift_ - The vertical distance (in image pixels) the character is
  //           shifted (up or down) from an acceptable y position.
  float yshift_;
  BlobChoiceClassifier classifier_;  // What generated *this.
};

// Make BLOB_CHOICE listable.
ELISTIZEH(BLOB_CHOICE)

// Return the BLOB_CHOICE in bc_list matching a given unichar_id,
// or nullptr if there is no match.
BLOB_CHOICE *FindMatchingChoice(UNICHAR_ID char_id, BLOB_CHOICE_LIST *bc_list);

// Permuter codes used in WERD_CHOICEs.
enum PermuterType {
  NO_PERM,            // 0
  PUNC_PERM,          // 1
  TOP_CHOICE_PERM,    // 2
  LOWER_CASE_PERM,    // 3
  UPPER_CASE_PERM,    // 4
  NGRAM_PERM,         // 5
  NUMBER_PERM,        // 6
  USER_PATTERN_PERM,  // 7
  SYSTEM_DAWG_PERM,   // 8
  DOC_DAWG_PERM,      // 9
  USER_DAWG_PERM,     // 10
  FREQ_DAWG_PERM,     // 11
  COMPOUND_PERM,      // 12

  NUM_PERMUTER_TYPES
};

// ScriptPos tells whether a character is subscript, superscript or normal.
enum ScriptPos {
  SP_NORMAL,
  SP_SUBSCRIPT,
  SP_SUPERSCRIPT,
  SP_DROPCAP
};

const char *ScriptPosToString(ScriptPos script_pos);

class TESS_API WERD_CHOICE : public ELIST_LINK {
 public:
  static const float kBadRating;
  static const char *permuter_name(uint8_t permuter);

  WERD_CHOICE(const UNICHARSET *unicharset)
    : unicharset_(unicharset) { this->init(8); }
  WERD_CHOICE(const UNICHARSET *unicharset, int reserved)
    : unicharset_(unicharset) { this->init(reserved); }
  WERD_CHOICE(const char *src_string,
              const char *src_lengths,
              float src_rating,
              float src_certainty,
              uint8_t src_permuter,
              const UNICHARSET &unicharset)
    : unicharset_(&unicharset) {
    this->init(src_string, src_lengths, src_rating,
               src_certainty, src_permuter);
  }
  WERD_CHOICE(const char *src_string, const UNICHARSET &unicharset);
  WERD_CHOICE(const WERD_CHOICE &word)
      : ELIST_LINK(word), unicharset_(word.unicharset_) {
    this->init(word.length());
    this->operator=(word);
  }
  ~WERD_CHOICE();

  const UNICHARSET *unicharset() const {
    return unicharset_;
  }
  inline int length() const {
    return length_;
  }
  float adjust_factor() const {
    return adjust_factor_;
  }
  void set_adjust_factor(float factor) {
    adjust_factor_ = factor;
  }
  inline const UNICHAR_ID *unichar_ids() const {
    return unichar_ids_;
  }
  inline UNICHAR_ID unichar_id(int index) const {
    assert(index < length_);
    return unichar_ids_[index];
  }
  inline int state(int index) const {
    return state_[index];
  }
  ScriptPos BlobPosition(int index) const {
    if (index < 0 || index >= length_)
      return SP_NORMAL;
    return script_pos_[index];
  }
  inline float rating() const {
    return rating_;
  }
  inline float certainty() const {
    return certainty_;
  }
  inline float certainty(int index) const {
    return certainties_[index];
  }
  inline float min_x_height() const {
    return min_x_height_;
  }
  inline float max_x_height() const {
    return max_x_height_;
  }
  inline void set_x_heights(float min_height, float max_height) {
    min_x_height_ = min_height;
    max_x_height_ = max_height;
  }
  inline uint8_t permuter() const {
    return permuter_;
  }
  const char *permuter_name() const;
  // Returns the BLOB_CHOICE_LIST corresponding to the given index in the word,
  // taken from the appropriate cell in the ratings MATRIX.
  // Borrowed pointer, so do not delete.
  BLOB_CHOICE_LIST* blob_choices(int index, MATRIX* ratings) const;

  // Returns the MATRIX_COORD corresponding to the location in the ratings
  // MATRIX for the given index into the word.
  MATRIX_COORD MatrixCoord(int index) const;

  inline void set_unichar_id(UNICHAR_ID unichar_id, int index) {
    assert(index < length_);
    unichar_ids_[index] = unichar_id;
  }
  bool dangerous_ambig_found() const {
    return dangerous_ambig_found_;
  }
  void set_dangerous_ambig_found_(bool value) {
    dangerous_ambig_found_ = value;
  }
  inline void set_rating(float new_val) {
    rating_ = new_val;
  }
  inline void set_certainty(float new_val) {
    certainty_ = new_val;
  }
  inline void set_permuter(uint8_t perm) {
    permuter_ = perm;
  }
  // Note: this function should only be used if all the fields
  // are populated manually with set_* functions (rather than
  // (copy)constructors and append_* functions).
  inline void set_length(int len) {
    ASSERT_HOST(reserved_ >= len);
    length_ = len;
  }

  /// Make more space in unichar_id_ and fragment_lengths_ arrays.
  inline void double_the_size() {
    if (reserved_ > 0) {
      unichar_ids_ = GenericVector<UNICHAR_ID>::double_the_size_memcpy(
          reserved_, unichar_ids_);
      script_pos_ = GenericVector<ScriptPos>::double_the_size_memcpy(
          reserved_, script_pos_);
      state_ = GenericVector<int>::double_the_size_memcpy(
          reserved_, state_);
      certainties_ = GenericVector<float>::double_the_size_memcpy(
          reserved_, certainties_);
      reserved_ *= 2;
    } else {
      unichar_ids_ = new UNICHAR_ID[1];
      script_pos_ = new ScriptPos[1];
      state_ = new int[1];
      certainties_ = new float[1];
      reserved_ = 1;
    }
  }

  /// Initializes WERD_CHOICE - reserves length slots in unichar_ids_ and
  /// fragment_length_ arrays. Sets other values to default (blank) values.
  inline void init(int reserved) {
    reserved_ = reserved;
    if (reserved > 0) {
      unichar_ids_ = new UNICHAR_ID[reserved];
      script_pos_ = new ScriptPos[reserved];
      state_ = new int[reserved];
      certainties_ = new float[reserved];
    } else {
      unichar_ids_ = nullptr;
      script_pos_ = nullptr;
      state_ = nullptr;
      certainties_ = nullptr;
    }
    length_ = 0;
    adjust_factor_ = 1.0f;
    rating_ = 0.0;
    certainty_ = FLT_MAX;
    min_x_height_ = 0.0f;
    max_x_height_ = FLT_MAX;
    permuter_ = NO_PERM;
    unichars_in_script_order_ = false;  // Tesseract is strict left-to-right.
    dangerous_ambig_found_ = false;
  }

  /// Helper function to build a WERD_CHOICE from the given string,
  /// fragment lengths, rating, certainty and permuter.
  /// The function assumes that src_string is not nullptr.
  /// src_lengths argument could be nullptr, in which case the unichars
  /// in src_string are assumed to all be of length 1.
  void init(const char *src_string, const char *src_lengths,
            float src_rating, float src_certainty,
            uint8_t src_permuter);

  /// Set the fields in this choice to be default (bad) values.
  inline void make_bad() {
    length_ = 0;
    rating_ = kBadRating;
    certainty_ = -FLT_MAX;
  }

  /// This function assumes that there is enough space reserved
  /// in the WERD_CHOICE for adding another unichar.
  /// This is an efficient alternative to append_unichar_id().
  inline void append_unichar_id_space_allocated(
      UNICHAR_ID unichar_id, int blob_count,
      float rating, float certainty) {
    assert(reserved_ > length_);
    length_++;
    this->set_unichar_id(unichar_id, blob_count,
                         rating, certainty, length_-1);
  }

  void append_unichar_id(UNICHAR_ID unichar_id, int blob_count,
                         float rating, float certainty);

  inline void set_unichar_id(UNICHAR_ID unichar_id, int blob_count,
                             float rating, float certainty, int index) {
    assert(index < length_);
    unichar_ids_[index] = unichar_id;
    state_[index] = blob_count;
    certainties_[index] = certainty;
    script_pos_[index] = SP_NORMAL;
    rating_ += rating;
    if (certainty < certainty_) {
      certainty_ = certainty;
    }
  }
  // Sets the entries for the given index from the BLOB_CHOICE, assuming
  // unit fragment lengths, but setting the state for this index to blob_count.
  void set_blob_choice(int index, int blob_count,
                       const BLOB_CHOICE* blob_choice);

  bool contains_unichar_id(UNICHAR_ID unichar_id) const;
  void remove_unichar_ids(int index, int num);
  inline void remove_last_unichar_id() { --length_; }
  inline void remove_unichar_id(int index) {
    this->remove_unichar_ids(index, 1);
  }
  bool has_rtl_unichar_id() const;
  void reverse_and_mirror_unichar_ids();

  // Returns the half-open interval of unichar_id indices [start, end) which
  // enclose the core portion of this word -- the part after stripping
  // punctuation from the left and right.
  void punct_stripped(int *start_core, int *end_core) const;

  // Returns the indices [start, end) containing the core of the word, stripped
  // of any superscript digits on either side. (i.e., the non-footnote part
  // of the word). There is no guarantee that the output range is non-empty.
  void GetNonSuperscriptSpan(int *start, int *end) const;

  // Return a copy of this WERD_CHOICE with the choices [start, end).
  // The result is useful only for checking against a dictionary.
  WERD_CHOICE shallow_copy(int start, int end) const;

  void string_and_lengths(STRING *word_str, STRING *word_lengths_str) const;
  const STRING debug_string() const {
    STRING word_str;
    for (int i = 0; i < length_; ++i) {
      word_str += unicharset_->debug_str(unichar_ids_[i]);
      word_str += " ";
    }
    return word_str;
  }
  // Returns true if any unichar_id in the word is a non-space-delimited char.
  bool ContainsAnyNonSpaceDelimited() const {
    for (int i = 0; i < length_; ++i) {
      if (!unicharset_->IsSpaceDelimited(unichar_ids_[i])) return true;
    }
    return false;
  }
  // Returns true if the word is all spaces.
  bool IsAllSpaces() const {
    for (int i = 0; i < length_; ++i) {
      if (unichar_ids_[i] != UNICHAR_SPACE) return false;
    }
    return true;
  }

  // Call this to override the default (strict left to right graphemes)
  // with the fact that some engine produces a "reading order" set of
  // Graphemes for each word.
  bool set_unichars_in_script_order(bool in_script_order) {
    return unichars_in_script_order_ = in_script_order;
  }

  bool unichars_in_script_order() const {
    return unichars_in_script_order_;
  }

  // Returns a UTF-8 string equivalent to the current choice
  // of UNICHAR IDs.
  STRING &unichar_string() {
      this->string_and_lengths(&unichar_string_, &unichar_lengths_);
      return unichar_string_;
  }

  // Returns a UTF-8 string equivalent to the current choice
  // of UNICHAR IDs.
  const STRING &unichar_string() const {
    this->string_and_lengths(&unichar_string_, &unichar_lengths_);
    return unichar_string_;
  }

  // Returns the lengths, one byte each, representing the number of bytes
  // required in the unichar_string for each UNICHAR_ID.
  const STRING &unichar_lengths() const {
    this->string_and_lengths(&unichar_string_, &unichar_lengths_);
    return unichar_lengths_;
  }

  // Sets up the script_pos_ member using the blobs_list to get the bln
  // bounding boxes, *this to get the unichars, and this->unicharset
  // to get the target positions. If small_caps is true, sub/super are not
  // considered, but dropcaps are.
  // NOTE: blobs_list should be the chopped_word blobs. (Fully segemented.)
  void SetScriptPositions(bool small_caps, TWERD* word, int debug = 0);
  // Sets the script_pos_ member from some source positions with a given length.
  void SetScriptPositions(const ScriptPos* positions, int length);
  // Sets all the script_pos_ positions to the given position.
  void SetAllScriptPositions(ScriptPos position);

  static ScriptPos ScriptPositionOf(bool print_debug,
                                               const UNICHARSET& unicharset,
                                               const TBOX& blob_box,
                                               UNICHAR_ID unichar_id);

  // Returns the "dominant" script ID for the word.  By "dominant", the script
  // must account for at least half the characters.  Otherwise, it returns 0.
  // Note that for Japanese, Hiragana and Katakana are simply treated as Han.
  int GetTopScriptID() const;

  // Fixes the state_ for a chop at the given blob_posiiton.
  void UpdateStateForSplit(int blob_position);

  // Returns the sum of all the state elements, being the total number of blobs.
  int TotalOfStates() const;

  void print() const { this->print(""); }
  void print(const char *msg) const;
  // Prints the segmentation state with an introductory message.
  void print_state(const char *msg) const;

  // Displays the segmentation state of *this (if not the same as the last
  // one displayed) and waits for a click in the window.
  void DisplaySegmentation(TWERD* word);

  WERD_CHOICE& operator+= (     // concatanate
    const WERD_CHOICE & second);// second on first

  WERD_CHOICE& operator= (const WERD_CHOICE& source);

 private:
  const UNICHARSET *unicharset_;
  // TODO(rays) Perhaps replace the multiple arrays with an array of structs?
  // unichar_ids_ is an array of classifier "results" that make up a word.
  // For each unichar_ids_[i], script_pos_[i] has the sub/super/normal position
  // of each unichar_id.
  // state_[i] indicates the number of blobs in WERD_RES::chopped_word that
  // were put together to make the classification results in the ith position
  // in unichar_ids_, and certainties_[i] is the certainty of the choice that
  // was used in this word.
  // == Change from before ==
  // Previously there was fragment_lengths_ that allowed a word to be
  // artificially composed of multiple fragment results. Since the new
  // segmentation search doesn't do fragments, treatment of fragments has
  // been moved to a lower level, augmenting the ratings matrix with the
  // combined fragments, and allowing the language-model/segmentation-search
  // to deal with only the combined unichar_ids.
  UNICHAR_ID *unichar_ids_;  // unichar ids that represent the text of the word
  ScriptPos* script_pos_;  // Normal/Sub/Superscript of each unichar.
  int* state_;               // Number of blobs in each unichar.
  float* certainties_;       // Certainty of each unichar.
  int reserved_;             // size of the above arrays
  int length_;               // word length
  // Factor that was used to adjust the rating.
  float adjust_factor_;
  // Rating is the sum of the ratings of the individual blobs in the word.
  float rating_;             // size related
  // certainty is the min (worst) certainty of the individual blobs in the word.
  float certainty_;          // absolute
  // xheight computed from the result, or 0 if inconsistent.
  float min_x_height_;
  float max_x_height_;
  uint8_t permuter_;           // permuter code

  // Normally, the ratings_ matrix represents the recognition results in order
  // from left-to-right.  However, some engines (say Cube) may return
  // recognition results in the order of the script's major reading direction
  // (for Arabic, that is right-to-left).
  bool unichars_in_script_order_;
  // True if NoDangerousAmbig found an ambiguity.
  bool dangerous_ambig_found_;

  // The following variables are populated and passed by reference any
  // time unichar_string() or unichar_lengths() are called.
  mutable STRING unichar_string_;
  mutable STRING unichar_lengths_;
};

// Make WERD_CHOICE listable.
ELISTIZEH(WERD_CHOICE)
using BLOB_CHOICE_LIST_VECTOR = GenericVector<BLOB_CHOICE_LIST *>;

// Utilities for comparing WERD_CHOICEs

bool EqualIgnoringCaseAndTerminalPunct(const WERD_CHOICE &word1,
                                       const WERD_CHOICE &word2);

// Utilities for debug printing.
void print_ratings_list(
    const char *msg,                      // intro message
    BLOB_CHOICE_LIST *ratings,            // list of results
    const UNICHARSET &current_unicharset  // unicharset that can be used
                                          // for id-to-unichar conversion
    );

} // namespace tesseract

#endif
