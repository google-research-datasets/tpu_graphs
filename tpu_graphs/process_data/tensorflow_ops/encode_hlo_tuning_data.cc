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

#include "tpu_graphs/process_data/xla/featurizers.h"
#include "tpu_graphs/process_data/xla/hlo_encoder.h"
#include "tpu_graphs/process_data/xla/hlo_opcode.h"
#include "tpu_graphs/process_data/xla/tuning_data_iterator.h"
#include "tpu_graphs/proto/tuning.pb.h"
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
 */
class EncodeHloTuningData : public tf::OpKernel {
 public:
  explicit EncodeHloTuningData(tf::OpKernelConstruction* context)
      : tf::OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("directed", &directed_));
    OP_REQUIRES_OK(context, context->GetAttr("take_every", &take_every_));
    OP_REQUIRES(context, take_every_ >= 1,
                tensorflow::errors::InvalidArgument(
                    "take_every must be in 1 or greater."));
    OP_REQUIRES_OK(context, context->GetAttr("include_no_normalization",
                                             &include_no_normalization_));
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

    // Parse the input protobuf.
    TuningDataType tuning_type = GetTuningDataType(tuning_type_str);
    TuningDataIterator::Options options;
    options.include_no_normalization = include_no_normalization_;
    options.take_every = take_every_;
    options.samples_limit = samples_limit;
    options.sample_rate = sample_rate;
    options.fingerprint_range = SplitToFingerprintRange(split_);
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

    // Create the output builders
    ml_lib::SparseHloEncoder hlo_encoder(directed_, task_);

    // Count total nodes and edges in all computations in all decomposed op
    // graphs. We do this first pass to avoid the cost of reallocating
    // incorrectly-sized tensors later.
    ml_lib::HloEncoder::HloModuleStat stat;
    for (int i = 0; i < sample_count; ++i) {
      hlo_encoder.CollectStats(data->GetHloModule(), &stat);
      data->NextSample();
    }

    const int64_t kModuleFeatureCount = ml_lib::GetModuleFeatureCount(task_);
    StatusOr<int> num_outputs = hlo_encoder.CreateOutputBuilders(stat, context);
    int64_t output_id = num_outputs.value();
    OP_REQUIRES_OK(context, num_outputs.status());
    tf::Tensor* normalization_values;
    tf::Tensor* compute_times;
    tf::Tensor* module_config_counts;
    tf::Tensor* module_ids;
    tf::Tensor* module_features_tensor;
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
    OP_REQUIRES_OK(context,
                   context->allocate_output(output_id++,
                                            {sample_count, kModuleFeatureCount},
                                            &module_features_tensor));

    // Second pass over the data -- fill in the output tensors.
    data->ResetIterator();
    int64_t last_module_id = data->GetModuleUniqueId() - 1;
    const uint8_t include_features = ml_lib::GetIncludeFeatureBits(task_);
    for (int i = 0; i < sample_count; ++i) {
      const int64_t module_id = data->GetModuleUniqueId();
      TuningDataIterator::SampleStats stats = data->GetSampleStats();
      CHECK_GT(stats.compute_time_ns, 0);
      normalization_values->vec<int64_t>()(i) = stats.normalization_value;
      compute_times->vec<int64_t>()(i) = stats.compute_time_ns;
      module_config_counts->vec<int64_t>()(i) = stats.config_count;
      module_ids->vec<int64_t>()(i) = module_id;
      tpu_graphs::TileSizeConfig tile_size_config = data->GetTileSizeConfig();

      // Encode HLO graph.
      const HloModuleProto& hlo_proto = data->GetHloModule();
      OP_REQUIRES_OK(context, hlo_encoder.EncodeHloModule(hlo_proto));

      // Encode module features (not associated to individual nodes).
      if (module_id != last_module_id) {
        last_module_id = module_id;
      }
      std::vector<float> modules_features;
      ml_lib::FeaturizeTileSizeConfig(&modules_features, &tile_size_config,
                                      include_features);
      for (int j = 0; j < modules_features.size(); ++j) {
        module_features_tensor->matrix<float>()(i, j) = modules_features[j];
      }

      data->NextSample();
    }
    data->Finalize();

    OP_REQUIRES_OK(context, hlo_encoder.Finalize());
  }

 private:
  static tf::Status ProduceEmptyOutputs(tf::OpKernelContext* context) {
    tf::Tensor* dummy_tensor;
    tf::Tensor* opcodes_splits;
    tf::Tensor* extra_features_splits;
    tf::Tensor* operand_adj_matrix_shape;
    tf::Tensor* consumer_adj_matrix_shape;
    TF_RETURN_IF_ERROR(context->allocate_output(0, {0}, &dummy_tensor));
    TF_RETURN_IF_ERROR(context->allocate_output(1, {1}, &opcodes_splits));
    TF_RETURN_IF_ERROR(context->allocate_output(2, {0}, &dummy_tensor));
    TF_RETURN_IF_ERROR(
        context->allocate_output(3, {1}, &extra_features_splits));
    // Operand adj matrix
    TF_RETURN_IF_ERROR(context->allocate_output(4, {0}, &dummy_tensor));
    TF_RETURN_IF_ERROR(context->allocate_output(5, {0, 3}, &dummy_tensor));
    TF_RETURN_IF_ERROR(
        context->allocate_output(6, {3}, &operand_adj_matrix_shape));
    // Consumer adj matrix
    TF_RETURN_IF_ERROR(context->allocate_output(7, {0}, &dummy_tensor));
    TF_RETURN_IF_ERROR(context->allocate_output(8, {0, 3}, &dummy_tensor));
    TF_RETURN_IF_ERROR(
        context->allocate_output(9, {3}, &consumer_adj_matrix_shape));

    TF_RETURN_IF_ERROR(context->allocate_output(10, {0}, &dummy_tensor));
    TF_RETURN_IF_ERROR(context->allocate_output(11, {0}, &dummy_tensor));
    TF_RETURN_IF_ERROR(context->allocate_output(12, {0}, &dummy_tensor));
    TF_RETURN_IF_ERROR(context->allocate_output(13, {0}, &dummy_tensor));
    TF_RETURN_IF_ERROR(context->allocate_output(14, {0, 5}, &dummy_tensor));
    TF_RETURN_IF_ERROR(context->allocate_output(15, {0}, &dummy_tensor));
    TF_RETURN_IF_ERROR(context->allocate_output(16, {0}, &dummy_tensor));

    opcodes_splits->flat<int64_t>()(0) = 0;
    extra_features_splits->flat<int64_t>()(0) = 0;
    operand_adj_matrix_shape->flat<int64_t>()(0) = 0;
    operand_adj_matrix_shape->flat<int64_t>()(1) = 0;
    operand_adj_matrix_shape->flat<int64_t>()(2) = 0;
    consumer_adj_matrix_shape->flat<int64_t>()(0) = 0;
    consumer_adj_matrix_shape->flat<int64_t>()(1) = 0;
    consumer_adj_matrix_shape->flat<int64_t>()(2) = 0;

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
  std::string task_;
  std::string split_;
};

REGISTER_OP("EncodeHloTuningData")
    .Input("proto_data: string")
    .Input("source_path: string")
    .Input("tuning_data_type: string")
    .Input("sample_rate: float")
    .Input("samples_limit: float")
    .Output("opcodes_values: uint8")
    .Output("opcodes_splits: int64")
    .Output("node_features_values: float")
    .Output("node_features_splits: int64")
    .Output("operand_adj_matrix_values: uint8")
    .Output("operand_adj_matrix_indices: int64")
    .Output("operand_adj_matrix_shape: int64")
    .Output("consumer_adj_matrix_values: uint8")
    .Output("consumer_adj_matrix_indices: int64")
    .Output("consumer_adj_matrix_shape: int64")
    .Output("normalization_values: int64")
    .Output("compute_times_ns: int64")
    .Output("module_config_counts: int64")
    .Output("module_ids: int64")
    .Output("module_features: float")
    .Attr("task: string")
    .Attr("directed: bool = false")
    .Attr("take_every: int = 1")
    .Attr("include_no_normalization: bool = true")
    .Attr("split: {'train', 'val', 'test', ''} = ''")
    .SetShapeFn([](tf::shape_inference::InferenceContext* c) {
      std::string task;
      CHECK(c->GetAttr("task", &task).ok());
      // `-1` in static shapes below denote dynamic dimensions.
      c->set_output(0, c->Vector(-1));
      c->set_output(1, c->Vector(-1));
      c->set_output(2, c->Matrix(-1, ml_lib::GetNodeFeatureCount(task)));
      c->set_output(3, c->Vector(-1));
      // operand adj matrix
      c->set_output(4, c->Vector(-1));
      c->set_output(5, c->Matrix(-1, 3));
      c->set_output(6, c->Vector(3));
      // consumer adj matrix
      c->set_output(7, c->Vector(-1));
      c->set_output(8, c->Matrix(-1, 3));
      c->set_output(9, c->Vector(3));
      c->set_output(10, c->Vector(-1));
      c->set_output(11, c->Vector(-1));
      c->set_output(12, c->Vector(-1));
      c->set_output(13, c->Vector(-1));
      c->set_output(14, c->Matrix(-1, ml_lib::GetModuleFeatureCount(task)));
      return OkStatus();
    });

REGISTER_KERNEL_BUILDER(Name("EncodeHloTuningData").Device(tf::DEVICE_CPU),
                        EncodeHloTuningData);

}  // namespace
}  // namespace ml_cost_model
}  // namespace xla
