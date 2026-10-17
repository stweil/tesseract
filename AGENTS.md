# Agents

This document describes how AI agents should interact with this codebase.

Tesseract is an OCR engine: `libtesseract` and the `tesseract` CLI, plus the
training tools built from `src/training/`.

## Build (local)

- CMake out-of-tree builds. `build-dropout/` is the working build directory;
  the other `build*/` dirs have stale Homebrew glib paths and their binaries
  fail at runtime. Reuse `build-dropout/`, or delete a stale dir and
  reconfigure before reusing it.
- Build a target: `cd build-dropout && make -j8 lstm_test lstmtraining`
  (binaries land in `build-dropout/bin/`).
- Autotools is also supported (`./autogen.sh && ./configure && make`); CI
  builds Linux with autotools and Windows with CMake.

## Tests

- Unit tests are googletest, one binary per `unittest/*_test.cc`:
  `cd build-dropout && ./bin/lstm_test`, or
  `./bin/lstm_test --gtest_filter="LSTMTrainerTest.SomeTest"` for one test.
  The full `lstm_test` suite takes several minutes.
- Never run two `lstm_test` instances in the same build directory at once:
  all tests write to the shared `./tmp` directory (`FLAGS_test_tmpdir`) and
  concurrent runs corrupt each other's models and checkpoints.
- Model-based tests (`lstm_test`, `lstm_recoder_test`, ...) need traineddata
  from the *sibling* directories `../tessdata` (models) and `../langdata_lstm`
  (both outside this repo). The in-tree `tessdata/` holds config files only,
  no models; without the sibling dirs these tests are skipped silently.
- LSTM training test data: `test/testdata/` (e.g. `eng.Arial.exp0.lstmf`).

## Code layout (where things actually live)

- `src/lstm/` — network layers (LSTM, FullyConnected, NetworkIO), `recodebeam`.
- `src/training/unicharset/lstmtrainer.{h,cpp}` — `LSTMTrainer`: the
  recognition/training loop, error tracking, checkpointing, sub-trainer
  stall retry.
- `src/training/lstmtraining.cpp` — the `lstmtraining` CLI.
- `src/wordrec/` — word recognition; `src/dict/` — DAWGs and language model.
- `unittest/` — unit test sources (see also `unittest/CMakeLists.txt`).

## lstmtraining behavior (non-obvious)

- Main loop: train `kNumPagesPerBatch` = 100 samples, then call
  `MaintainCheckpoints` once.
- Checkpoints: a rolling `<model>_checkpoint` (crash-safe, previous one
  rotated to `.bak`), written at most every `--checkpoint_every` samples
  (default 100), plus one additional full `<model>_epoch<N>.checkpoint` per
  epoch (always on, one epoch = one pass over all training lines).
- `--net_mode` defaults to 192 = Adam (128) | layer-specific learning rates
  (64); layer-specific LRs make the learning-rate search triggered on a
  stall much more expensive.
- Stall retry: after `min_stall_iterations` (default 10000, settable with
  `--min_stall_iterations`) without improvement, a sub-trainer with reduced
  learning rates is started and races the main trainer; `--no_subtrainer`
  disables the mechanism.
- Training is deterministic for identical input and settings
  (`LSTMTrainerTest.DeterminismTest`).

## CI notes

- The unittest workflow's clang job sets
  `ASAN_OPTIONS=alloc_dealloc_mismatch=0`: Ubuntu 22.04's packaged libc++
  misreports any thrown `logic_error` as alloc-dealloc-mismatch
  (llvm/llvm-project#59432). Do not remove that option when debugging new
  sanitizer failures, and check whether the gcc job also fails first.
- `pull_request` triggers match source paths only (`**.cpp`, `**.h`,
  `**Makefile.am`, `configure.ac`, `unittest/**`); workflow- or CMake-only
  changes do not run CI.

## Conventions

- C++ style follows `.clang-format` (Google-based, right-aligned pointers).
- New functionality gets a unit test in `unittest/` that can run under
  `make check` with ASan/UBSan (see the CI notes above).
