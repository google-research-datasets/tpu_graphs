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

#ifndef THIRD_PARTY_PY_TPU_GRAPHS_PROCESS_DATA_XLA_TUNING_DATA_ITERATOR_H_
#define THIRD_PARTY_PY_TPU_GRAPHS_PROCESS_DATA_XLA_TUNING_DATA_ITERATOR_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tpu_graphs/proto/tuning.pb.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/tstring.h"

namespace xla {

namespace tf = ::tensorflow;

enum TuningDataType {
  kModuleTuning,  // Data from whole-module autotuner.
  kOpTuning       // Data from op-level autotuner (e.g. window).
};

// Returns TuningDataType from string.
inline TuningDataType GetTuningDataType(absl::string_view str) {
  static const absl::flat_hash_map<std::string, TuningDataType>* str_to_type =
      new absl::flat_hash_map<std::string, TuningDataType>(
          {{"module_tuning", kModuleTuning}, {"op_tuning", kOpTuning}});
  CHECK(str_to_type->contains(str))
      << "Cannot convert '" << str << "' to TuningDataType.";
  return str_to_type->find(str)->second;
}

// An interface class for iterating sampled data from XLA autotuning results and
// feeding them to HLO encoder.
class TuningDataIterator {
 public:
  struct Options {
    // Include samples with no normalization values.
    bool include_no_normalization = true;

    // Take one sample every `take_every' samples.
    int take_every = 1;

    // The maximum number of samples to include. -1 is no limit.
    float samples_limit = -1.0;

    // Include a sample with the given probability.
    float sample_rate = 1.0;

    // If batch_size is positive, a data iterator may up and down sample
    // to select a uniform number of samples from each bin.
    int batch_size = 0;

    // If true, a data iterator may shuffle data.
    bool shuffle = false;

    // If specified, then iterator should only return modules whose fingerprints
    // satisfy are in range
    // [fingerprint_range_.first, fingerprint_range_.second] (both inclusive).
    // Currently, only supported for OpTuningDataIterator.
    std::optional<std::pair<uint64_t, uint64_t>> fingerprint_range =
        std::nullopt;
  };

  virtual ~TuningDataIterator() {}
  explicit TuningDataIterator(const Options& options)
      : fingerprint_range_(options.fingerprint_range),
        batch_size_(options.batch_size),
        shuffle_(options.shuffle),
        take_every_(options.take_every),
        samples_limit_(options.samples_limit),
        sample_rate_(options.sample_rate),
        include_no_normalization_(options.include_no_normalization) {}

  // Loads protobuf from source_path and filters out some samples as follows:
  // * If `include_no_normalization_` is false, drop samples without
  // normalization values.
  // * If `take_every_ > 1`, take one out of `take_every_` samples.
  // * If `subsample_rate_ < 1.0`, randomly subsample with such probability.
  Status LoadAndPrepareData(const tf::tstring& source_path,
                            const tf::tstring& proto_data) {
    source_path_ = source_path;
    TF_RETURN_IF_ERROR(LoadData(source_path, proto_data));
    if (!include_no_normalization_) {
      FilterOutSamplesWithoutNormalization();
    }
    Subsample(take_every_, sample_rate_, samples_limit_);
    return OkStatus();
  }

  // Name of the class.
  virtual std::string Name() const = 0;

  // Returns the current number of HLO op modules being handled.
  virtual int64_t GetSampleCount() const = 0;

  // Sets the module iterator to the beginning.
  virtual void ResetIterator() = 0;

  // Iterates to the next op module.
  virtual void NextSample() = 0;

  // Checks that all the data has been traversed.
  virtual void Finalize() const = 0;

  // Returns the current op module.
  virtual const HloModuleProto& GetHloModule() const = 0;

  // Returns config index to node id map.
  virtual const std::vector<int>& GetConfigIndexToNode() const = 0;

  // Returns a unique id of the current HLO op (modulo window config).
  // This id is unique across data files.
  virtual int64_t GetModuleUniqueId() const = 0;

  struct SampleStats {
    // The actual runtime in nanoseconds.
    int64_t compute_time_ns;

    // The normalization value for runtime. 0 if no normalization value.
    int64_t normalization_value;

    // The number of configs in the data for the current HLO module.
    int64_t config_count;
  };

  // Returns SampleStats.
  virtual SampleStats GetSampleStats() const = 0;

  // Returns the tile size of the current op sample. If no tile size config,
  // returns an empty proto.
  virtual tpu_graphs::TileSizeConfig GetTileSizeConfig() const { return {}; }

  // Returns the module config of the current sample.
  virtual const tpu_graphs::HloModuleConfig* GetModuleConfig() const {
    return nullptr;
  }

  virtual std::vector<std::vector<int>> GetModuleConfigValues() const {
    return {{}};
  }

 protected:
  tf::tstring source_path_;
  const std::optional<std::pair<uint64_t, uint64_t>> fingerprint_range_;
  const int batch_size_;
  const bool shuffle_;
  const int take_every_;
  const float samples_limit_;
  const float sample_rate_;

 private:
  // Loads protobuf from source_path.
  virtual Status LoadData(const tf::tstring& source_path,
                          const tf::tstring& proto_data) = 0;

  // Drops samples without normalization values from an internal storage.
  virtual void FilterOutSamplesWithoutNormalization() = 0;

  // Subsamples data and saves them in an internal storage.
  virtual void Subsample(const int take_every, const float sample_rate,
                         const float samples_limit) = 0;

  const bool include_no_normalization_;
};

// Returns an appropriate TuningDataIterator for iterating `proto_data`.
// Users are responsible for keeping `proto_data` alive (no deallocation),
// during the life of a TuningDataIterator object.
// If `proto_data` is empty, this function will attempt to create a
// TuningDataIterator for data reading from `source_path`.
StatusOr<std::unique_ptr<TuningDataIterator>> CreateTuningDataIterator(
    TuningDataType type, const tf::tstring& source_path,
    const tf::tstring& proto_data, const TuningDataIterator::Options& options);

inline StatusOr<std::unique_ptr<TuningDataIterator>> CreateTuningDataIterator(
    TuningDataType type, const tf::tstring& source_path) {
  TuningDataIterator::Options default_options;
  return CreateTuningDataIterator(type, source_path, /*proto_data=*/"",
                                  default_options);
}

// A class for iterating new protobuf tuning::ModuleTuningData
// from whole-program autotuning results and feeding them to HLO encoder.
class ModuleTuningDataIterator : public TuningDataIterator {
 public:
  ~ModuleTuningDataIterator() override {}
  explicit ModuleTuningDataIterator(const TuningDataIterator::Options& options)
      : TuningDataIterator(options) {}

  std::string Name() const override { return "ModuleTuningDataIterator"; }
  int64_t GetSampleCount() const override { return config_profiles_.size(); }
  void ResetIterator() override { idx_ = 0; }
  void NextSample() override { ++idx_; }
  void Finalize() const override { CHECK(idx_ == GetSampleCount()); }

  Status LoadData(const tf::tstring& source_path,
                  const tf::tstring& proto_data) override;
  const HloModuleProto& GetHloModule() const override;
  const std::vector<int>& GetConfigIndexToNode() const override;
  int64_t GetModuleUniqueId() const override;
  SampleStats GetSampleStats() const override;
  const tpu_graphs::HloModuleConfig* GetModuleConfig() const override;
  std::vector<std::vector<int>> GetModuleConfigValues() const override;

 private:
  void FilterOutSamplesWithoutNormalization() override;
  void Subsample(const int take_every, const float sample_rate,
                 const float samples_limit) override;

  tpu_graphs::ModuleTuningData module_tuning_data_;
  int64_t idx_ = 0;
  std::vector<const tpu_graphs::ConfigProfile*> config_profiles_;
  int64_t config_profiles_size_;
  HloModuleProto* hlo_module_proto_;
  int64_t normalization_value_ = 0;
  uint64_t fingerprint_;
  std::vector<int> config_index_to_node_;
};

// A class for iterating over a collection of tuning::ModuleTuningData protos
// from whole-program autotuning results and feeding them to HLO encoder.
class OpTuningDataIterator : public TuningDataIterator {
 public:
  explicit OpTuningDataIterator(const TuningDataIterator::Options& options)
      : TuningDataIterator(options) {}

  std::string Name() const override { return "OpTuningDataIterator"; }
  int64_t GetSampleCount() const override { return size_; }
  void ResetIterator() override;
  void NextSample() override;
  void Finalize() const override;

  Status LoadData(const tf::tstring& source_path,
                  const tf::tstring& proto_data) override;
  const HloModuleProto& GetHloModule() const override;
  const std::vector<int>& GetConfigIndexToNode() const {
    LOG(FATAL) << "Not supported. This shouldn't be reached.";
  }
  int64_t GetModuleUniqueId() const override;
  SampleStats GetSampleStats() const override;
  tpu_graphs::TileSizeConfig GetTileSizeConfig() const override;

 private:
  void FilterOutSamplesWithoutNormalization() override;
  void Subsample(const int take_every, const float sample_rate,
                 const float samples_limit) override;

  Status LoadModuleTuningProto(
      tpu_graphs::ModuleTuningData* module_tuning_data);

  tpu_graphs::TuningData tuning_data_;
  int64_t op_idx_ = 0;
  int64_t config_idx_ = 0;
  int64_t size_ = 0;
  std::vector<const HloModuleProto*> op_modules_;
  std::vector<uint64_t> fingerprints_;
  absl::flat_hash_map<std::string, int64_t> normalization_values_;
  absl::flat_hash_map<std::string, int64_t> name_to_op_idx_;

  // config_data_[i][j] is a config data j for op module i.
  std::vector<std::vector<const tpu_graphs::ConfigProfile*>> config_data_;
};

}  // namespace xla

#endif  // THIRD_PARTY_PY_TPU_GRAPHS_PROCESS_DATA_XLA_TUNING_DATA_ITERATOR_H_
