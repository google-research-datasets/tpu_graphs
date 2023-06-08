/* Copyright 2023 Google LLC. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "third_party/py/tpu_graphs/process_data/xla/tuning_data_iterator.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "testing/base/public/gunit.h"
#include "third_party/py/tpu_graphs/proto/tuning.proto.h"
#include "third_party/tensorflow/compiler/xla/service/hlo.proto.h"
#include "third_party/tensorflow/compiler/xla/xla_data.proto.h"
#include "third_party/tensorflow/tsl/lib/core/status_test_util.h"
#include "third_party/tensorflow/tsl/platform/env.h"
#include "third_party/tensorflow/tsl/platform/path.h"
#include "third_party/tensorflow/tsl/platform/platform.h"

#if defined(PLATFORM_GOOGLE) || defined(PLATFORM_GOOGLE_ANDROID)
#include "testing/base/public/gmock.h"  // IWYU pragma: export
#else
#include <gmock/gmock-actions.h>
#include <gmock/gmock-matchers.h>       // IWYU pragma: export
#include <gmock/gmock-more-matchers.h>  // IWYU pragma: export
#endif

namespace xla {
namespace ml_cost_model {
namespace {

using ::testing::EqualsProto;

void IteratorProducesAllSamplesInModuleTuningProto(
    const std::unique_ptr<TuningDataIterator>& data_iterator,
    const tpu_graphs::ModuleTuningData& proto_data) {
  ASSERT_TRUE(proto_data.has_module());
  ASSERT_GT(proto_data.runs_size(), 0);
  ASSERT_GT(proto_data.fingerprint(), 0);

  int64_t default_runtime = 0;
  std::vector<const tpu_graphs::ConfigProfile*> runs;
  for (const auto& config_profile : proto_data.runs()) {
    if (default_runtime == 0 && config_profile.is_default()) {
      default_runtime = config_profile.profile().compute_time_ns();
    }
    if (!config_profile.error()) {
      runs.push_back(&config_profile);
    }
  }
  ASSERT_GT(default_runtime, 0);
  CHECK_EQ(data_iterator->GetSampleCount(), runs.size());

  tpu_graphs::TileSizeConfig empty_tile_size;
  for (const auto* config_profile : runs) {
    TuningDataIterator::SampleStats stats = data_iterator->GetSampleStats();
    EXPECT_GT(stats.compute_time_ns, 0);
    EXPECT_EQ(data_iterator->GetHloModule().entry_computation_name(),
              proto_data.module().entry_computation_name());
    EXPECT_EQ(data_iterator->GetHloModule().ByteSizeLong(),
              proto_data.module().ByteSizeLong());
    EXPECT_EQ(stats.config_count, runs.size());
    EXPECT_TRUE(config_profile->has_profile());
    EXPECT_EQ(stats.compute_time_ns,
              config_profile->profile().compute_time_ns());
    EXPECT_EQ(stats.normalization_value, default_runtime);
    EXPECT_THAT(data_iterator->GetTileSizeConfig(),
                EqualsProto(empty_tile_size));
    EXPECT_NE(data_iterator->GetModuleUniqueId(), 0);
    data_iterator->NextSample();
  }
}

TEST(TuningDataIteratorTest, ModuleTuningDataNoFilter) {
  const std::string source_path = tsl::io::JoinPath(
      absl::GetFlag(FLAGS_test_srcdir),
      "google3/third_party/py/tpu_graphs/process_data/tensorflow_ops/testdata/"
      "module_tuning_data_layout.pb");
  tpu_graphs::ModuleTuningData proto_data;
  TF_ASSERT_OK(
      tsl::ReadBinaryProto(tsl::Env::Default(), source_path, &proto_data));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TuningDataIterator> data_iterator,
      CreateTuningDataIterator(kModuleTuning, source_path));  // no filter
  EXPECT_NE(data_iterator, nullptr);
  EXPECT_EQ(data_iterator->Name(), "ModuleTuningDataIterator");

  // First pass.
  {
    SCOPED_TRACE("First pass");
    IteratorProducesAllSamplesInModuleTuningProto(data_iterator, proto_data);
  }

  // Also work for more iteration passes.
  data_iterator->ResetIterator();
  {
    SCOPED_TRACE("Second pass");
    IteratorProducesAllSamplesInModuleTuningProto(data_iterator, proto_data);
  }
  data_iterator->Finalize();
}

void IteratorProducesAllSamplesInOpTuningProtos(
    const std::unique_ptr<TuningDataIterator>& data_iterator,
    const tpu_graphs::TuningData& tuning_data) {
  int count = 0;
  for (const auto& proto_data : tuning_data.modules()) {
    ASSERT_TRUE(proto_data.has_module());
    ASSERT_GT(proto_data.runs_size(), 0);
    ASSERT_GT(proto_data.fingerprint(), 0);

    int64_t default_runtime = 0;
    for (const auto& config_profile : proto_data.runs()) {
      if (config_profile.is_default()) {
        default_runtime = config_profile.profile().compute_time_ns();
        break;
      }
    }

    ASSERT_GT(default_runtime, 0);

    for (const auto& config_profile : proto_data.runs()) {
      if (config_profile.error()) continue;
      ++count;
      tpu_graphs::TileSizeConfig expected_config =
          config_profile.op_config().tile_size_config();
      TuningDataIterator::SampleStats stats = data_iterator->GetSampleStats();
      EXPECT_GT(stats.compute_time_ns, 0);
      EXPECT_EQ(data_iterator->GetHloModule().entry_computation_name(),
                proto_data.module().entry_computation_name());
      EXPECT_EQ(data_iterator->GetHloModule().ByteSizeLong(),
                proto_data.module().ByteSizeLong());
      EXPECT_GT(stats.config_count, 0);
      if (config_profile.has_profile()) {
        EXPECT_EQ(stats.compute_time_ns,
                  config_profile.profile().compute_time_ns());
      } else {
        EXPECT_EQ(stats.compute_time_ns, std::numeric_limits<int64_t>::max());
      }
      EXPECT_EQ(stats.normalization_value, default_runtime);
      EXPECT_THAT(data_iterator->GetTileSizeConfig(),
                  EqualsProto(expected_config));
      data_iterator->NextSample();
    }
  }

  EXPECT_EQ(data_iterator->GetSampleCount(), count);
}

TEST(TuningDataIteratorTest, OpTuningDataNoFilter) {
  const std::string source_path = tsl::io::JoinPath(
      absl::GetFlag(FLAGS_test_srcdir),
      "google3/third_party/py/tpu_graphs/process_data/tensorflow_ops/testdata/"
      "op_tuning_data.pb");

  tpu_graphs::TuningData proto_data;
  TF_ASSERT_OK(
      tsl::ReadBinaryProto(tsl::Env::Default(), source_path, &proto_data));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TuningDataIterator> data_iterator,
      CreateTuningDataIterator(kOpTuning, source_path));  // no filter
  ASSERT_NE(data_iterator, nullptr);
  EXPECT_EQ(data_iterator->Name(), "OpTuningDataIterator");

  // First pass.
  {
    SCOPED_TRACE("First pass");
    IteratorProducesAllSamplesInOpTuningProtos(data_iterator, proto_data);
  }

  // Also work for more iteration passes.
  data_iterator->ResetIterator();
  {
    SCOPED_TRACE("Second pass");
    IteratorProducesAllSamplesInOpTuningProtos(data_iterator, proto_data);
  }
  data_iterator->Finalize();
}

}  // namespace
}  // namespace ml_cost_model
}  // namespace xla
