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

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "tpu_graphs/process_data/xla/hlo_opcode.h"
#include "tpu_graphs/process_data/xla/tuning_data_iterator.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace ml_cost_model {
namespace {

namespace tf = ::tensorflow;

/**
 * Turns an HLO tuning proto into variants of opcodes, adj. matrices,
 * and extra features corresponding to each unique decomposed op run.
 *
 * In addition to outputs produced by HloEncoder, this op produces:
 * - compute_times_ns: M-length array
 * - compute_times_ns[i] is the real runtime of HLO module i.
 *
 * This op may do up/down sampling:
 * - If batch_size is positive, up and down sampling data to select a uniform
 *   number of samples from each bin.
 * - On top of the above sampling mechanism, if sample_rate < 1,
 *   down sampling data uniformly.
 * - If samples_limit is not -1, adjust sample_rate if
 *   sample_limit / sample_count < sample_rate.
 */
class EncodeHloConfigData : public tf::OpKernel {
 public:
  explicit EncodeHloConfigData(tf::OpKernelConstruction* context)
      : tf::OpKernel(context) {
    CHECK(HloOpcodeCount() <= std::numeric_limits<uint8_t>::max());
    OP_REQUIRES_OK(context, context->GetAttr("directed", &directed_));
    OP_REQUIRES_OK(context, context->GetAttr("take_every", &take_every_));
    OP_REQUIRES(context, take_every_ >= 1,
                tensorflow::errors::InvalidArgument(
                    "take_every must be in 1 or greater."));
    OP_REQUIRES_OK(context, context->GetAttr("include_no_normalization",
                                             &include_no_normalization_));
    OP_REQUIRES_OK(context, context->GetAttr("shuffle", &shuffle_));
    OP_REQUIRES_OK(context, context->GetAttr("task", &task_));
    OP_REQUIRES_OK(context, context->GetAttr("split", &split_));
  }

  void Compute(tf::OpKernelContext* context) override {
    // Grab the source path, if available
    const tf::tstring& proto_data = context->input(0).flat<tf::tstring>()(0);
    const tf::tstring& source_path = context->input(1).flat<tf::tstring>()(0);
    const tf::tstring& tuning_type_str =
        context->input(2).flat<tf::tstring>()(0);
    const float sample_rate = context->input(3).flat<float>()(0);
    const float samples_limit = context->input(4).flat<float>()(0);
    const int32_t batch_size = context->input(5).flat<int32_t>()(0);

    // Parse the input protobuf.
    TuningDataType tuning_type = GetTuningDataType(tuning_type_str);
    TuningDataIterator::Options options;
    options.include_no_normalization = include_no_normalization_;
    options.take_every = take_every_;
    options.samples_limit = samples_limit;
    options.sample_rate = sample_rate;
    options.fingerprint_range = SplitToFingerprintRange(split_);
    options.batch_size = batch_size;
    options.shuffle = shuffle_;
    StatusOr<std::unique_ptr<TuningDataIterator>> status_data =
        CreateTuningDataIterator(tuning_type, source_path, proto_data, options);
    OP_REQUIRES_OK(context, status_data.status());
    std::unique_ptr<TuningDataIterator> data = std::move(status_data).value();

    // If there are no samples, bail with empty output.
    const int64_t sample_count = data->GetSampleCount();
    if (sample_count == 0) {
      OP_REQUIRES_OK(context, ProduceEmptyOutputs(context));
      return;
    }

    const std::vector<int>& config_index_to_node = data->GetConfigIndexToNode();

    int64_t output_id = 0;
    tf::Tensor* normalization_values;
    tf::Tensor* compute_times;
    tf::Tensor* module_config_counts;
    tf::Tensor* module_ids;
    tf::Tensor* config_features_tensor;
    OP_REQUIRES_OK(context,
                   context->allocate_output(output_id++, {sample_count},
                                            &normalization_values));
    OP_REQUIRES_OK(context, context->allocate_output(
                                output_id++, {sample_count}, &compute_times));
    OP_REQUIRES_OK(context,
                   context->allocate_output(output_id++, {sample_count},
                                            &module_config_counts));
    OP_REQUIRES_OK(context, context->allocate_output(
                                output_id++, {sample_count}, &module_ids));
    OP_REQUIRES_OK(context, context->allocate_output(
                                output_id++,
                                {sample_count,
                                 static_cast<long>(config_index_to_node.size()),
                                 ml_lib::GetConfigFeatureCount(task_)},
                                &config_features_tensor));

    data->ResetIterator();
    for (int i = 0; i < sample_count; ++i) {
      const int64_t module_id = data->GetModuleUniqueId();
      TuningDataIterator::SampleStats stats = data->GetSampleStats();
      CHECK_GT(stats.compute_time_ns, 0);
      normalization_values->vec<int64_t>()(i) = stats.normalization_value;
      compute_times->vec<int64_t>()(i) = stats.compute_time_ns;
      module_config_counts->vec<int64_t>()(i) = stats.config_count;
      module_ids->vec<uint64_t>()(i) = module_id;
      std::vector<std::vector<int>> config_features =
          data->GetModuleConfigValues();
      CHECK_EQ(config_features.size(), config_index_to_node.size());
      for (int j = 0; j < config_features.size(); j++) {
        for (int k = 0; k < config_features[j].size(); k++) {
          config_features_tensor->tensor<float, 3>()(i, j, k) =
              static_cast<float>(config_features[j][k]);
        }
      }
      data->NextSample();
    }
    data->Finalize();
  }

 private:
  tf::Status ProduceEmptyOutputs(tf::OpKernelContext* context) {
    tf::Tensor* dummy_tensor;
    TF_RETURN_IF_ERROR(context->allocate_output(0, {0}, &dummy_tensor));
    TF_RETURN_IF_ERROR(context->allocate_output(1, {0}, &dummy_tensor));
    TF_RETURN_IF_ERROR(context->allocate_output(2, {0}, &dummy_tensor));
    TF_RETURN_IF_ERROR(context->allocate_output(3, {0}, &dummy_tensor));
    TF_RETURN_IF_ERROR(context->allocate_output(
        4, {0, 0, ml_lib::GetConfigFeatureCount(task_)}, &dummy_tensor));
    return OkStatus();
  }

  // Maps a split name to a range of fingerprints. Currently hardcoded to
  // emit ranges such that train:val:test :: 0.8:0.1:0.1
  // If no split is specified, returns a std::nullopt corresponding to the full
  // range.
  static std::optional<std::pair<uint64_t, uint64_t>> SplitToFingerprintRange(
      absl::string_view split) {
    if (split.empty()) {
      return std::nullopt;
    }

    uint64_t max = std::numeric_limits<uint64_t>::max();
    if (split == "train") {
      return std::make_pair(0, max * 0.8);
    }
    if (split == "val") {
      return std::make_pair(max * 0.8 + 1, max * 0.9);
    }
    if (split == "test") {
      return std::make_pair(max * 0.9 + 1, max);
    }
    LOG(FATAL) << "Unknown split value." << split
               << ". Expected one of {train, val, test}";
  }

  bool directed_;
  int take_every_;
  bool include_no_normalization_;
  bool shuffle_;
  std::string task_;
  std::string split_;
};

REGISTER_OP("EncodeHloConfigData")
    .Input("proto_data: string")
    .Input("source_path: string")
    .Input("tuning_data_type: string")
    .Input("sample_rate: float")
    .Input("samples_limit: float")
    .Input("batch_size: int32")
    .Output("normalization_values: int64")
    .Output("compute_times_ns: int64")
    .Output("module_config_counts: int64")
    .Output("module_ids: uint64")
    .Output("config_features: float")
    .Attr("task: string")
    .Attr("directed: bool = false")
    .Attr("take_every: int = 1")
    .Attr("include_no_normalization: bool = true")
    .Attr("shuffle: bool = true")
    .Attr("split: {'train', 'val', 'test', ''} = ''")
    .SetShapeFn([](tf::shape_inference::InferenceContext* c) {
      std::string task;
      CHECK(c->GetAttr("task", &task).ok());
      c->set_output(0, c->Vector(-1));
      c->set_output(1, c->Vector(-1));
      c->set_output(2, c->Vector(-1));
      c->set_output(3, c->Vector(-1));
      c->set_output(
          4, c->MakeShape({-1, -1, ml_lib::GetConfigFeatureCount(task)}));
      return OkStatus();
    });

REGISTER_KERNEL_BUILDER(Name("EncodeHloConfigData").Device(tf::DEVICE_CPU),
                        EncodeHloConfigData);

}  // namespace
}  // namespace ml_cost_model
}  // namespace xla
