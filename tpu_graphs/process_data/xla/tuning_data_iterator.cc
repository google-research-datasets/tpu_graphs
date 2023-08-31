/* Copyright 2023 The tpu_graphs Authors.

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

#include "tpu_graphs/process_data/xla/tuning_data_iterator.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "absl/random/random.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "tpu_graphs/proto/tuning.pb.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/tstring.h"

namespace xla {

namespace tf = ::tensorflow;

StatusOr<std::unique_ptr<TuningDataIterator>> CreateTuningDataIterator(
    TuningDataType type, const tf::tstring& source_path,
    const tf::tstring& proto_data, const TuningDataIterator::Options& options) {
  if (options.fingerprint_range && type != kOpTuning) {
    return tensorflow::errors::Unimplemented(absl::StrCat(
        "Fingerprint based filtering is not supported for iterator of type ",
        type));
  }

  std::unique_ptr<TuningDataIterator> tuning_data;
  switch (type) {
    case kModuleTuning:
      tuning_data = std::make_unique<ModuleTuningDataIterator>(options);
      break;
    case kOpTuning:
      tuning_data = std::make_unique<OpTuningDataIterator>(options);
      break;
    default:
      return tensorflow::errors::InvalidArgument("Unknown TuningDataType");
  }
  TF_RETURN_IF_ERROR(tuning_data->LoadAndPrepareData(source_path, proto_data));
  return tuning_data;
}

Status ModuleTuningDataIterator::LoadData(const tf::tstring& source_path,
                                          const tf::tstring& proto_data) {
  if (proto_data.empty()) {
    LOG(INFO) << "proto_data is empty. Reading from source_path: "
              << source_path;
    TF_RETURN_IF_ERROR(tf::ReadBinaryProto(tf::Env::Default(), source_path,
                                           &module_tuning_data_));
  } else {
    if (!module_tuning_data_.ParseFromArray(proto_data.data(),
                                            proto_data.size())) {
      return tf::errors::InvalidArgument(
          "Couldn't parse input protobuf as ModuleTuningData.");
    }
  }

  // Read in HLO module.
  hlo_module_proto_ = module_tuning_data_.mutable_module();
  CHECK_GT(module_tuning_data_.config_index_to_node_size(), 0);
  config_index_to_node_.reserve(
      module_tuning_data_.config_index_to_node_size());
  for (const auto& id : module_tuning_data_.config_index_to_node()) {
    config_index_to_node_.push_back(id);
  }

  config_profiles_.reserve(module_tuning_data_.runs_size());
  for (const auto& run : module_tuning_data_.runs()) {
    if (run.has_module_config() && !run.error() && run.has_profile()) {
      config_profiles_.push_back(&run);
      CHECK_GT(run.profile().compute_time_ns(), 0);
    }
  }

  // Set normalization value.
  const tpu_graphs::ConfigProfile* default_config = nullptr;
  for (const auto* config : config_profiles_) {
    if (config->is_default()) {
      default_config = config;
      break;
    }
  }

  if (default_config) {
    normalization_value_ = default_config->profile().compute_time_ns();
  } else {
    normalization_value_ = 0;
    config_profiles_.clear();
  }

  // Compute fingerprint according to source path. Same graph with different
  // paths won't have the same fingerprint.
  fingerprint_ = tf::Fingerprint64(source_path);

  // TODO: Support up-down sampling.
  // if (batch_size_) {
  //   ml_lib::NormalizeDistAcrossBins(/*num_bins=*/5, batch_size_,
  //                                   &config_profiles_);
  // }
  return OkStatus();
}

void ModuleTuningDataIterator::FilterOutSamplesWithoutNormalization() {
  // Do nothing because all samples have normalization value.
}

void ModuleTuningDataIterator::Subsample(const int take_every,
                                         const float sample_rate,
                                         const float samples_limit) {
  if (config_profiles_.empty()) {
    config_profiles_size_ = config_profiles_.size();
    return;
  }

  // TODO: Support up-down sampling.
  // float local_sample_rate = sample_rate;
  // if (sample_rate >= 1.0 && samples_limit > config_profiles_.size()) {
  //   // Upsample
  //   local_sample_rate = samples_limit / config_profiles_.size();
  // } else if (samples_limit > 0) {
  //   // Downsample
  //   local_sample_rate =
  //       std::min(sample_rate, samples_limit / config_profiles_.size());
  // }
  // ml_lib::UpDownSamples(&config_profiles_, take_every, local_sample_rate);
  config_profiles_size_ = config_profiles_.size();

  if (shuffle_) {
    const int64_t seed =
        absl::ToInt64Milliseconds(absl::Now() - absl::UnixEpoch());
    std::shuffle(config_profiles_.begin(), config_profiles_.end(),
                 std::default_random_engine(seed));
  }
}

const HloModuleProto& ModuleTuningDataIterator::GetHloModule() const {
  return *hlo_module_proto_;
}

const std::vector<int>& ModuleTuningDataIterator::GetConfigIndexToNode() const {
  return config_index_to_node_;
}

int64_t ModuleTuningDataIterator::GetModuleUniqueId() const {
  return fingerprint_;
}

TuningDataIterator::SampleStats ModuleTuningDataIterator::GetSampleStats()
    const {
  int64_t compute_time_ns;
  if (config_profiles_[idx_]->has_profile()) {
    CHECK_GT(config_profiles_[idx_]->profile().compute_time_ns(), 0);
    compute_time_ns = config_profiles_[idx_]->profile().compute_time_ns();
  } else {
    compute_time_ns = std::numeric_limits<int64_t>::max();
  }
  return {compute_time_ns, normalization_value_, config_profiles_size_};
}

const tpu_graphs::HloModuleConfig* ModuleTuningDataIterator::GetModuleConfig()
    const {
  CHECK(config_profiles_[idx_]->has_module_config()) << idx_;
  return &config_profiles_[idx_]->module_config();
}

namespace {
std::vector<std::vector<int>> FeaturizeLayoutConfig(
    const tpu_graphs::LayoutConfig& layout_config) {
  const constexpr int kLayoutMaxDims = 6;
  const constexpr int kLayoutMaxTensors = 3;
  constexpr int32_t kPaddingValue = -1;
  std::vector<std::vector<int>> config_values;
  for (auto& node : layout_config.nodes()) {
    std::vector<int> layout_dims(kLayoutMaxDims * kLayoutMaxTensors,
                                 kPaddingValue);
    int tensor_idx = 0;
    CHECK_LE(node.tensors_size(), kLayoutMaxTensors);
    for (auto& tensor : node.tensors()) {
      int dims_size = std::min(tensor.dims().size(), kLayoutMaxDims);
      for (int dim_idx = 0; dim_idx < dims_size; ++dim_idx) {
        layout_dims[tensor_idx * kLayoutMaxDims + dim_idx] =
            tensor.dims()[dim_idx];
      }
      ++tensor_idx;
    }
    config_values.push_back(layout_dims);
  }
  return config_values;
}
}  // namespace

std::vector<std::vector<int>> ModuleTuningDataIterator::GetModuleConfigValues()
    const {
  CHECK(GetModuleConfig()->has_layout_config());
  return FeaturizeLayoutConfig(GetModuleConfig()->layout_config());
}

Status OpTuningDataIterator::LoadData(const tf::tstring& source_path,
                                      const tf::tstring& proto_data) {
  if (proto_data.empty()) {
    LOG(INFO) << "proto_data is empty. Reading from source_path: "
              << source_path;
    TF_RETURN_IF_ERROR(
        tf::ReadBinaryProto(tf::Env::Default(), source_path, &tuning_data_));
  } else {
    if (!tuning_data_.ParseFromArray(proto_data.data(), proto_data.size())) {
      return tf::errors::InvalidArgument(
          "Couldn't parse input protobuf as ModuleTuningData.");
    }
  }

  for (auto& module_tuning_data : *tuning_data_.mutable_modules()) {
    if (module_tuning_data.runs_size() > 1) {
      TF_RETURN_IF_ERROR(LoadModuleTuningProto(&module_tuning_data));
    }
  }
  return OkStatus();
}

Status OpTuningDataIterator::LoadModuleTuningProto(
    tpu_graphs::ModuleTuningData* module_tuning_data) {
  // Read in HLO module.
  if (!module_tuning_data->has_module()) {
    return OkStatus();
  }

  if (fingerprint_range_) {
    if (module_tuning_data->fingerprint() < fingerprint_range_->first ||
        module_tuning_data->fingerprint() > fingerprint_range_->second) {
      return OkStatus();
    }
  }
  const HloModuleProto* hlo_module_proto = &module_tuning_data->module();

  // Skip modules that have empty tile size configs.
  for (const auto& config : module_tuning_data->runs()) {
    if (!config.op_config().has_tile_size_config()) {
      return OkStatus();
    }
  }

  op_modules_.push_back(hlo_module_proto);
  fingerprints_.push_back(module_tuning_data->fingerprint());

  std::vector<const tpu_graphs::ConfigProfile*> configs;
  configs.reserve(module_tuning_data->runs_size());
  for (const auto& config : module_tuning_data->runs()) {
    if (config.error() || config.profile().compute_time_ns() <= 1) {
      continue;
    }
    configs.push_back(&config);
  }
  config_data_.push_back(configs);
  size_ += configs.size();

  // Set normalization value.
  const tpu_graphs::ConfigProfile* default_config = nullptr;
  for (const auto* config : configs) {
    if (config->is_default()) {
      default_config = config;
      break;
    }
  }

  if (!default_config) {
    return tf::errors::InvalidArgument(
        absl::StrCat("There is no default config in ModuleTuningData."));
  }

  normalization_values_[hlo_module_proto->name()] =
      default_config->profile().compute_time_ns();
  if (normalization_values_[hlo_module_proto->name()] == 0) {
    return tf::errors::InvalidArgument(
        absl::StrCat("The runtime of default config for module ",
                     hlo_module_proto->name(), " in ModuleTuningData is 0."));
  }

  name_to_op_idx_[hlo_module_proto->name()] = op_modules_.size() - 1;

  return OkStatus();
}

void OpTuningDataIterator::FilterOutSamplesWithoutNormalization() {
  // Do nothing because all samples have normalization values.
}

void OpTuningDataIterator::Subsample(const int take_every,
                                     const float sample_rate,
                                     const float samples_limit) {
  absl::BitGen bitgen;
  for (int i = 0; i < config_data_.size(); ++i) {
    // If a kernel is not selected, we delete all its data samples.
    if (take_every > 1 && i % take_every != 0) {
      config_data_[i].clear();
    } else if (sample_rate < 1.0 && !absl::Bernoulli(bitgen, sample_rate)) {
      config_data_[i].clear();
    }
  }

  // TODO: Support up-down sampling.
  // const int64_t kLimitPerModule = 1000;  // work best empirically
  // // When samples_limit = -1, must include everything.
  // if (samples_limit >= 0 && kLimitPerModule >= 0) {
  //   for (auto& op_config_data : config_data_) {
  //     if (op_config_data.empty()) {
  //       continue;
  //     }
  //     const float rate =
  //         std::min<float>(1.0, 1.0 * kLimitPerModule /
  //         op_config_data.size());
  //     ml_lib::UpDownSamples(&op_config_data, /*take_every=*/1, rate);
  //   }
  // }

  size_ = 0;
  for (const auto& op_config_data : config_data_) {
    size_ += op_config_data.size();
  }
  ResetIterator();
}

void OpTuningDataIterator::ResetIterator() {
  op_idx_ = 0;
  config_idx_ = -1;
  NextSample();
}

void OpTuningDataIterator::NextSample() {
  ++config_idx_;
  while (op_idx_ < config_data_.size() &&
         config_idx_ == config_data_[op_idx_].size()) {
    config_idx_ = 0;
    ++op_idx_;
  }
}

void OpTuningDataIterator::Finalize() const {
  CHECK_EQ(op_idx_, config_data_.size());
  CHECK_EQ(config_idx_, 0);
}

const HloModuleProto& OpTuningDataIterator::GetHloModule() const {
  return *op_modules_[op_idx_];
}

int64_t OpTuningDataIterator::GetModuleUniqueId() const {
  return fingerprints_[op_idx_];
}

tpu_graphs::TileSizeConfig OpTuningDataIterator::GetTileSizeConfig() const {
  CHECK(config_data_[op_idx_][config_idx_]->has_op_config());
  CHECK(config_data_[op_idx_][config_idx_]->op_config().has_tile_size_config());
  return config_data_[op_idx_][config_idx_]->op_config().tile_size_config();
}

TuningDataIterator::SampleStats OpTuningDataIterator::GetSampleStats() const {
  const tpu_graphs::ConfigProfile* config_profile =
      config_data_[op_idx_][config_idx_];
  const int64_t compute_time_ns = config_profile->profile().compute_time_ns();
  CHECK_GT(compute_time_ns, 0);

  return {compute_time_ns,
          normalization_values_.at(op_modules_[op_idx_]->name()),
          static_cast<int64_t>(config_data_[op_idx_].size())};
}

}  // namespace xla
