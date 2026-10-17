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

// Generating the training data:
// If the format of the lstmf (ImageData) file changes, the training data will
// have to be regenerated as follows:
//
// Use --xsize 800 for text2image to be similar to original training data.
//
// tesstrain.py --fonts_dir /usr/share/fonts --lang eng \
// --linedata_only   --noextract_font_properties --langdata_dir ../langdata_lstm \
// --tessdata_dir ../tessdata --output_dir ~/tesseract/test/testdata \
// --fontlist "Arial" --maxpages 10
//

#include "lstm_test.h"

namespace tesseract {

// Tests that some simple networks can learn Arial and meet accuracy targets.
TEST_F(LSTMTrainerTest, BasicTest) {
  // A Convolver sliding window classifier without LSTM.
  SetupTrainer(
      "[1,32,0,1 Ct5,5,16 Mp4,4 Ct1,1,16 Ct3,3,128 Mp4,1 Ct1,1,64 S2,1 "
      "Ct1,1,64O1c1]",
      "no-lstm", "eng/eng.unicharset", "eng.Arial.exp0.lstmf", false, false, 2e-4, false, "eng");
  double non_lstm_err = TrainIterations(kTrainerIterations * 4);
  EXPECT_LT(non_lstm_err, 98);
  LOG(INFO) << "********** Expected  < 98 ************\n";

  // A basic single-layer, single direction LSTM.
  SetupTrainerEng("[1,1,0,32 Lfx100 O1c1]", "1D-lstm", false, false);
  double lstm_uni_err = TrainIterations(kTrainerIterations * 2);
  EXPECT_LT(lstm_uni_err, 86);
  LOG(INFO) << "********** Expected  < 86 ************\n";
  // Beats the convolver. (Although it does have a lot more weights, it still
  // iterates faster.)
  EXPECT_LT(lstm_uni_err, non_lstm_err);
}

// Color learns almost as fast as normalized grey/2D.
TEST_F(LSTMTrainerTest, ColorTest) {
  // A basic single-layer, single direction LSTM.
  SetupTrainerEng("[1,32,0,3 S4,2 L2xy16 Ct1,1,16 S8,1 Lbx100 O1c1]", "2D-color-lstm", true, true);
  double lstm_uni_err = TrainIterations(kTrainerIterations);
  EXPECT_LT(lstm_uni_err, 85);
  //  EXPECT_GT(lstm_uni_err, 66);
  LOG(INFO) << "********** Expected  < 85 ************\n";
}

TEST_F(LSTMTrainerTest, BidiTest) {
  // A basic single-layer, bi-di 1d LSTM.
  SetupTrainerEng("[1,1,0,32 Lbx100 O1c1]", "bidi-lstm", false, false);
  double lstm_bi_err = TrainIterations(kTrainerIterations);
  EXPECT_LT(lstm_bi_err, 75);
  LOG(INFO) << "********** Expected   < 75 ************\n";
  // Int mode training is dead, so convert the trained network to int and check
  // that its error rate is close to the float version.
  TestIntMode(kTrainerIterations);
}

// Tests that a 2d-2-layer network learns correctly.
// It takes a lot of iterations to get there.
TEST_F(LSTMTrainerTest, Test2D) {
  // A 2-layer LSTM with a 2-D feature-extracting LSTM on the bottom.
  SetupTrainerEng("[1,32,0,1 S4,2 L2xy16 Ct1,1,16 S8,1 Lbx100 O1c1]", "2-D-2-layer-lstm", false,
                  false);
  double lstm_2d_err = TrainIterations(kTrainerIterations * 3 / 2);
  EXPECT_LT(lstm_2d_err, 98);
  //  EXPECT_GT(lstm_2d_err, 90);
  LOG(INFO) << "********** Expected  < 98 ************\n";
  // Int mode training is dead, so convert the trained network to int and check
  // that its error rate is close to the float version.
  TestIntMode(kTrainerIterations);
}

// Tests that a 2d-2-layer network with Adam does *a lot* better than
// without it.
TEST_F(LSTMTrainerTest, TestAdam) {
  // A 2-layer LSTM with a 2-D feature-extracting LSTM on the bottom.
  SetupTrainerEng("[1,32,0,1 S4,2 L2xy16 Ct1,1,16 S8,1 Lbx100 O1c1]", "2-D-2-layer-lstm", false,
                  true);
  double lstm_2d_err = TrainIterations(kTrainerIterations);
  EXPECT_LT(lstm_2d_err, 70);
  LOG(INFO) << "********** Expected   < 70 ************\n";
  TestIntMode(kTrainerIterations);
}

// Trivial test of training speed on a fairly complex network.
TEST_F(LSTMTrainerTest, SpeedTest) {
  SetupTrainerEng(
      "[1,30,0,1 Ct5,5,16 Mp2,2 L2xy24 Ct1,1,48 Mp5,1 Ct1,1,32 S3,1 Lbx64 "
      "O1c1]",
      "2-D-2-layer-lstm", false, true);
  TrainIterations(kTrainerIterations);
  LOG(INFO) << "********** *** ************\n";
}

// Tests that two identical networks trained the same get the same results.
// Also tests that the same happens with a serialize/deserialize in the middle.
TEST_F(LSTMTrainerTest, DeterminismTest) {
  SetupTrainerEng("[1,32,0,1 S4,2 L2xy16 Ct1,1,16 S8,1 Lbx100 O1c1]", "2-D-2-layer-lstm", false,
                  false);
  double lstm_2d_err_a = TrainIterations(kTrainerIterations);
  double act_error_a = trainer_->ActivationError();
  double char_error_a = trainer_->CharError();
  std::vector<char> trainer_a_data;
  EXPECT_TRUE(trainer_->SaveTrainingDump(NO_BEST_TRAINER, *trainer_, &trainer_a_data));
  SetupTrainerEng("[1,32,0,1 S4,2 L2xy16 Ct1,1,16 S8,1 Lbx100 O1c1]", "2-D-2-layer-lstm", false,
                  false);
  double lstm_2d_err_b = TrainIterations(kTrainerIterations);
  double act_error_b = trainer_->ActivationError();
  double char_error_b = trainer_->CharError();
  EXPECT_FLOAT_EQ(lstm_2d_err_a, lstm_2d_err_b);
  EXPECT_FLOAT_EQ(act_error_a, act_error_b);
  EXPECT_FLOAT_EQ(char_error_a, char_error_b);
  // Now train some more iterations.
  lstm_2d_err_b = TrainIterations(kTrainerIterations / 3);
  act_error_b = trainer_->ActivationError();
  char_error_b = trainer_->CharError();
  // Unpack into a new trainer and train that some more too.
  SetupTrainerEng("[1,32,0,1 S4,2 L2xy16 Ct1,1,16 S8,1 Lbx100 O1c1]", "2-D-2-layer-lstm", false,
                  false);
  EXPECT_TRUE(trainer_->ReadTrainingDump(trainer_a_data, *trainer_));
  lstm_2d_err_a = TrainIterations(kTrainerIterations / 3);
  act_error_a = trainer_->ActivationError();
  char_error_a = trainer_->CharError();
  EXPECT_FLOAT_EQ(lstm_2d_err_a, lstm_2d_err_b);
  EXPECT_FLOAT_EQ(act_error_a, act_error_b);
  EXPECT_FLOAT_EQ(char_error_a, char_error_b);
  LOG(INFO) << "********** *** ************\n";
}

// The baseline network against which to test the built-in softmax.
TEST_F(LSTMTrainerTest, SoftmaxBaselineTest) {
  // A basic single-layer, single direction LSTM.
  SetupTrainerEng("[1,1,0,32 Lfx96 O1c1]", "1D-lstm", false, true);
  double lstm_uni_err = TrainIterations(kTrainerIterations * 2);
  EXPECT_LT(lstm_uni_err, 60);
  //  EXPECT_GT(lstm_uni_err, 48);
  LOG(INFO) << "********** Expected  < 60 ************\n";
  // Check that it works in int mode too.
  TestIntMode(kTrainerIterations);
  // If we run TestIntMode again, it tests that int_mode networks can
  // serialize and deserialize correctly.
  double delta = TestIntMode(kTrainerIterations);
  // The two tests (both of int mode this time) should be almost identical.
  LOG(INFO) << "Delta in Int mode error rates = " << delta << "\n";
  EXPECT_LT(delta, 0.01);
}

// Tests that the built-in softmax does better than the external one,
// which has an error rate slightly less than 55%, as tested by
// SoftmaxBaselineTest.
TEST_F(LSTMTrainerTest, SoftmaxTest) {
  // LSTM with a built-in softmax can beat the external softmax.
  SetupTrainerEng("[1,1,0,32 LS96]", "Lstm-+-softmax", false, true);
  double lstm_sm_err = TrainIterations(kTrainerIterations * 2);
  EXPECT_LT(lstm_sm_err, 49.0);
  LOG(INFO) << "********** Expected  < 49 ************\n";
  // Check that it works in int mode too.
  TestIntMode(kTrainerIterations);
}

// Tests that the built-in encoded softmax does better than the external one.
// It takes a lot of iterations to get there.
TEST_F(LSTMTrainerTest, EncodedSoftmaxTest) {
  // LSTM with a built-in encoded softmax can beat the external softmax.
  SetupTrainerEng("[1,1,0,32 LE96]", "Lstm-+-softmax", false, true);
  double lstm_sm_err = TrainIterations(kTrainerIterations * 2);
  EXPECT_LT(lstm_sm_err, 62.0);
  LOG(INFO) << "********** Expected   < 62 ************\n";
  // Check that it works in int mode too.
  TestIntMode(kTrainerIterations);
}

// Tests that layer access methods work correctly.
TEST_F(LSTMTrainerTest, TestLayerAccess) {
  // A 2-layer LSTM with a Squashed feature-extracting LSTM on the bottom.
  SetupTrainerEng("[1,32,0,1 Ct5,5,16 Mp2,2 Lfys32 Lbx128 O1c1]", "SQU-lstm", false, false);
  // Number of layers.
  const size_t kNumLayers = 8;
  // Expected layer names.
  const char *kLayerIds[kNumLayers] = {":0", ":1:0", ":1:1", ":2", ":3:0", ":4:0", ":4:1:0", ":5"};
  const char *kLayerNames[kNumLayers] = {"Input",  "Convolve",  "ConvNL", "Maxpool",
                                         "Lfys32", "Lbx128LTR", "Lbx128", "Output"};
  // Expected number of weights.
  const int kNumWeights[kNumLayers] = {0,
                                       0,
                                       16 * (25 + 1),
                                       0,
                                       32 * (4 * (32 + 16 + 1)),
                                       128 * (4 * (128 + 32 + 1)),
                                       128 * (4 * (128 + 32 + 1)),
                                       112 * (2 * 128 + 1)};

  auto layers = trainer_->EnumerateLayers();
  EXPECT_EQ(kNumLayers, layers.size());
  for (unsigned i = 0; i < kNumLayers && i < layers.size(); ++i) {
    EXPECT_STREQ(kLayerIds[i], layers[i].c_str());
    EXPECT_STREQ(kLayerNames[i], trainer_->GetLayer(layers[i])->name().c_str());
    EXPECT_EQ(kNumWeights[i], trainer_->GetLayer(layers[i])->num_weights());
  }
}

// Tests that MaintainCheckpoints writes an additional restorable training
// checkpoint at the end of each epoch, where one epoch is one full pass over
// all training samples.
TEST_F(LSTMTrainerTest, EpochCheckpointTest) {
  SetupTrainerEng("[1,1,0,32 Lfx100 O1c1]", "epoch-lstm", false, false);
  const std::string model_base = file::JoinPath(FLAGS_test_tmpdir, "epoch-lstm");
  const int epoch_size = trainer_->mutable_training_data()->TotalPages();
  LOG(INFO) << "epoch_size=" << epoch_size << "\n";
  CHECK_GT(epoch_size, 0);
  // Start clean so stale checkpoints from earlier runs don't confuse the test.
  for (int epoch = 1; epoch <= 3; ++epoch) {
    file::Delete((model_base + "_epoch" + std::to_string(epoch) + ".checkpoint").c_str());
  }
  trainer_->SetEpochSize(epoch_size);
  // Train for more than two full epochs, calling MaintainCheckpoints every
  // 100 samples, as lstmtraining does between batches.
  const int batch_size = 100;
  const int total_iterations = (5 * epoch_size) / 2;
  int iteration = trainer_->training_iteration();
  while (iteration < total_iterations) {
    const int target =
        iteration + batch_size < total_iterations ? iteration + batch_size
                                                  : total_iterations;
    while (iteration < target) {
      trainer_->TrainOnLine(trainer_.get(), false);
      iteration = trainer_->training_iteration();
    }
    std::stringstream log;
    trainer_->MaintainCheckpoints(nullptr, log);
    iteration = trainer_->training_iteration();
  }
  const int trained = trainer_->training_iteration();
  LOG(INFO) << "trained=" << trained << "\n";
  // There must be a checkpoint for every completed epoch.
  for (int epoch = 1; epoch * epoch_size <= trained; ++epoch) {
    const std::string name =
        model_base + "_epoch" + std::to_string(epoch) + ".checkpoint";
    LOG(INFO) << "Expecting " << name << "\n";
    CHECK(file::Readable(name));
    // The checkpoint must be restorable into a fresh trainer, and it must
    // reflect the state from somewhere within the corresponding epoch.
    LSTMTrainer restored;
    restored.InitCharSet(
        file::JoinPath(file::JoinPath(FLAGS_test_tmpdir, "eng"), "eng") + ".traineddata");
    EXPECT_TRUE(restored.TryLoadingCheckpoint(name.c_str(), nullptr));
    EXPECT_GE(restored.training_iteration(), epoch * epoch_size);
    EXPECT_LT(restored.training_iteration(), (epoch + 1) * epoch_size);
  }
}

// Tests that the rolling checkpoint is written crash-safely, so that an
// interrupted training can resume as if there had been no interruption.
// WriteCheckpoint atomically replaces the checkpoint and keeps the previous
// one in checkpoint.bak, so there is always a complete, loadable checkpoint
// even if the process dies while writing the next one.
TEST_F(LSTMTrainerTest, CrashSafeCheckpointTest) {
  SetupTrainerEng("[1,1,0,32 Lfx100 O1c1]", "crash-lstm", false, false);
  const std::string model_base = file::JoinPath(FLAGS_test_tmpdir, "crash-lstm");
  const std::string checkpoint = model_base + "_checkpoint";
  const std::string checkpoint_bak = checkpoint + ".bak";
  // Remove leftovers from previous test runs.
  file::Delete(checkpoint.c_str());
  file::Delete(checkpoint_bak.c_str());
  // Write the first checkpoint (no .bak exists yet).
  for (int i = 0; i < 250; ++i) {
    trainer_->TrainOnLine(trainer_.get(), false);
  }
  std::stringstream log;
  trainer_->MaintainCheckpoints(nullptr, log);
  CHECK(file::Readable(checkpoint));
  const int first_iteration = trainer_->training_iteration();
  // Write a second checkpoint. The first must now be preserved as .bak.
  for (int i = 0; i < 250; ++i) {
    trainer_->TrainOnLine(trainer_.get(), false);
  }
  trainer_->MaintainCheckpoints(nullptr, log);
  CHECK(file::Readable(checkpoint));
  CHECK(file::Readable(checkpoint_bak));
  // Both must be loadable, and the .bak must be the older state.
  auto load_state = [](const char *filename, int *iteration) {
    LSTMTrainer state;
    state.InitCharSet(file::JoinPath(file::JoinPath(FLAGS_test_tmpdir, "eng"), "eng") +
                      ".traineddata");
    if (!state.TryLoadingCheckpoint(filename, nullptr)) {
      return false;
    }
    *iteration = state.training_iteration();
    return true;
  };
  int new_iteration = 0;
  int old_iteration = 0;
  CHECK(load_state(checkpoint.c_str(), &new_iteration));
  CHECK(load_state(checkpoint_bak.c_str(), &old_iteration));
  EXPECT_GT(new_iteration, old_iteration);
  EXPECT_EQ(new_iteration, trainer_->training_iteration());
  EXPECT_EQ(old_iteration, first_iteration);
  // Simulate an interruption during a checkpoint write by truncating the
  // live checkpoint, and verify recovery: a fresh run restores from the
  // good .bak and can keep writing checkpoints.
  {
    std::string contents;
    CHECK(file::GetContents(checkpoint, &contents, file::Defaults()));
    file::WriteStringToFile(contents.substr(0, contents.size() / 2), checkpoint);
  }
  LSTMTrainer restored;
  restored.InitCharSet(
      file::JoinPath(file::JoinPath(FLAGS_test_tmpdir, "eng"), "eng") + ".traineddata");
  // The corrupt checkpoint cannot be loaded...
  EXPECT_FALSE(restored.TryLoadingCheckpoint(checkpoint.c_str(), nullptr));
  // ...but the backup can, and it restores the previous training state, so
  // the interrupted run resumes without losing more than one checkpoint
  // interval of progress.
  CHECK(load_state(checkpoint_bak.c_str(), &old_iteration));
  EXPECT_EQ(old_iteration, first_iteration);
}

// Tests that the interval between rolling checkpoint writes is configurable.
// With an interval larger than the batch size, checkpoints are written only
// once the interval has been reached (not on every batch), and the written
// checkpoint is still complete and loadable.
TEST_F(LSTMTrainerTest, CheckpointIntervalTest) {
  SetupTrainerEng("[1,1,0,32 Lfx100 O1c1]", "ckpt-interval-lstm", false, false);
  const std::string model_base = file::JoinPath(FLAGS_test_tmpdir, "ckpt-interval-lstm");
  const std::string checkpoint = model_base + "_checkpoint";
  // Remove leftovers from previous test runs.
  file::Delete(checkpoint.c_str());
  // Write a checkpoint at most every 300 samples. The main training loop
  // calls MaintainCheckpoints every 100 samples (one batch), so the
  // checkpoint must be written after the third batch only.
  trainer_->SetCheckpointInterval(300);
  EXPECT_EQ(300, trainer_->checkpoint_interval());
  for (int batch = 1; batch <= 4; ++batch) {
    for (int i = 0; i < 100; ++i) {
      trainer_->TrainOnLine(trainer_.get(), false);
    }
    std::stringstream log;
    trainer_->MaintainCheckpoints(nullptr, log);
    const bool written =
        log.str().find("wrote checkpoint") != std::string::npos;
    const bool expect_write = (batch == 3);
    LOG(INFO) << "batch=" << batch << " written=" << written << "\n";
    EXPECT_EQ(expect_write, written) << "batch " << batch;
  }
  EXPECT_EQ(400, trainer_->training_iteration());
  // The checkpoint on disk must be from the last write, i.e. after batch 3.
  EXPECT_TRUE(file::Readable(checkpoint));
  LSTMTrainer restored;
  restored.InitCharSet(
      file::JoinPath(file::JoinPath(FLAGS_test_tmpdir, "eng"), "eng") + ".traineddata");
  CHECK(restored.TryLoadingCheckpoint(checkpoint.c_str(), nullptr));
  EXPECT_EQ(300, restored.training_iteration());
}

// Tests that the sub-trainer retry (which tries different learning rates
// when training makes no progress) is triggered when enabled, and fully
// suppressed when disabled, under otherwise identical deterministic
// conditions.
TEST_F(LSTMTrainerTest, SubtrainerConfigTest) {
  // Trains the trainer in batches of 100 (calling MaintainCheckpoints after
  // each batch, as lstmtraining does) until the best error rate drops below
  // the start-up threshold (75%). This populates best_trainer_ and puts the
  // trainer past the start-up phase, which the stall trigger requires.
  // Returns the final training iteration.
  auto train_until_started = [](LSTMTrainer *trainer) {
    const int kMaxIterations = 2000;
    const double kMinStartedErrorRate = 75.0;
    int iteration = 0;
    while (iteration < kMaxIterations &&
           trainer->best_error_rate() >= kMinStartedErrorRate) {
      const int target = iteration + 100;
      while (iteration < target) {
        trainer->TrainOnLine(trainer, false);
        iteration = trainer->training_iteration();
      }
      std::stringstream batch_log;
      trainer->MaintainCheckpoints(nullptr, batch_log);
      iteration = trainer->training_iteration();
    }
    return iteration;
  };
  // A stall is declared when the rolling char error is more than
  // kSubTrainerMarginFraction (3/128 ~= 2.34%) above the best, the best is
  // below the start-up threshold, and the stall iteration has been reached.
  // Rather than rely on training dynamics to drift the rolling mean far
  // enough above the best (which is slow and not guaranteed), we reach a
  // stable best with real training and then force the rolling error to a
  // known value above the best with FillErrorBuffer. This makes the trigger
  // condition deterministic.
  const double kMarginFraction = 3.0 / 128;
  // Trains until start-up is over, then forces the rolling char error above
  // the best so the stall condition is met, and runs MaintainCheckpoints.
  // Returns whether a sub-trainer trial was started (per the log).
  auto train_and_try_stall =
      [this, train_until_started, kMarginFraction](LSTMTrainer *trainer) {
        train_until_started(trainer);
        CHECK_LT(trainer->best_error_rate(), 75.0);
        // Force the rolling char error to just above the margin over the
        // best so that error_rate > best_error_rate_ * (1 + margin).
        const double best = trainer->best_error_rate();
        const double forced_error = best * (1.0 + 2.0 * kMarginFraction);
        trainer->FillErrorBuffer(forced_error / 100.0, ET_CHAR_ERROR);
        CHECK_GT(trainer->CharError(), best * (1.0 + kMarginFraction));
        std::stringstream log;
        trainer->MaintainCheckpoints(nullptr, log);
        return log.str().find("Trial sub_trainer_") != std::string::npos;
      };
  const char *kSpec = "[1,1,0,32 Lfx100 O1c1]";
  const float kLearningRate = 1e-3;
  // Enabled: with sub-trainer trials on and no stall delay, the forced stall
  // must start a sub-trainer trial, which logs "Trial sub_trainer_".
  SetupTrainer(kSpec, "subtrainer-on", "eng/eng.unicharset", "eng.Arial.exp0.lstmf",
               false, false, kLearningRate, false, "eng");
  trainer_->SetMinStallIterations(0);
  const bool enabled_started = train_and_try_stall(trainer_.get());
  const int enabled_iteration = trainer_->training_iteration();
  LOG(INFO) << "enabled: best=" << trainer_->best_error_rate()
            << " sub_trainer_started=" << enabled_started << "\n";
  EXPECT_TRUE(enabled_started);
  // Disabled: the identical run must never start a sub-trainer.
  SetupTrainer(kSpec, "subtrainer-off", "eng/eng.unicharset", "eng.Arial.exp0.lstmf",
               false, false, kLearningRate, false, "eng");
  trainer_->SetMinStallIterations(0);
  trainer_->SetSubtrainerEnabled(false);
  const bool disabled_started = train_and_try_stall(trainer_.get());
  LOG(INFO) << "disabled: best=" << trainer_->best_error_rate()
            << " sub_trainer_started=" << disabled_started << "\n";
  EXPECT_FALSE(disabled_started);
  // The two runs are deterministic and must train to the same point.
  EXPECT_EQ(enabled_iteration, trainer_->training_iteration());
}

} // namespace tesseract.
