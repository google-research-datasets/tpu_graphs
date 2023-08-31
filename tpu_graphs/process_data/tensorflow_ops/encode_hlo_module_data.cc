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
#include <string>
#include <utility>

#include "tpu_graphs/process_data/xla/hlo_encoder.h"
#include "tpu_graphs/process_data/xla/hlo_opcode.h"
#include "tpu_graphs/process_data/xla/tuning_data_iterator.h"
#include "tpu_graphs/proto/tuning.pb.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/tstring.h"

namespace xla {
namespace ml_cost_model {
namespace {

namespace tf = ::tensorflow;

/**
 * Turns an HLO module proto to an encoding that can be used for training.
 */
class EncodeHloModuleData : public tf::OpKernel {
 public:
  explicit EncodeHloModuleData(tf::OpKernelConstruction* context)
      : tf::OpKernel(context) {
    CHECK(HloOpcodeCount() <= std::numeric_limits<uint8_t>::max());
    OP_REQUIRES_OK(context, context->GetAttr("directed", &directed_));
    OP_REQUIRES_OK(context, context->GetAttr("task", &task_));
  }

  void Compute(tf::OpKernelContext* context) override {
    // Grab the source path, if available
    const tf::tstring& proto_data = context->input(0).flat<tf::tstring>()(0);
    const tf::tstring& source_path = context->input(1).flat<tf::tstring>()(0);
    const tf::tstring& tuning_type_str =
        context->input(2).flat<tf::tstring>()(0);

    LOG(INFO) << "Process " << source_path;

    // Parse the input protobuf.
    TuningDataType tuning_type = GetTuningDataType(tuning_type_str);
    TuningDataIterator::Options options;
    StatusOr<std::unique_ptr<TuningDataIterator>> status_data =
        CreateTuningDataIterator(tuning_type, source_path, proto_data, options);
    OP_REQUIRES_OK(context, status_data.status());
    std::unique_ptr<TuningDataIterator> data = std::move(status_data).value();

    const HloModuleProto& hlo_module_proto = data->GetHloModule();
    const std::vector<int>& config_index_to_node = data->GetConfigIndexToNode();

    ml_lib::SparseHloEncoder hlo_encoder(directed_, task_);
    // Count total nodes and edges in all computations in the given HloModule.
    // We do this first pass to avoid the cost of reallocating
    // incorrectly-sized tensors later.
    ml_lib::HloEncoder::HloModuleStat stat;
    hlo_encoder.CollectStats(hlo_module_proto, &stat);

    StatusOr<int> num_outputs = hlo_encoder.CreateOutputBuilders(stat, context);
    int64_t output_id = num_outputs.value();
    OP_REQUIRES_OK(context, num_outputs.status());

    tf::Tensor* config_index_to_node_tensor;
    tf::Tensor* module_id;

    OP_REQUIRES_OK(
        context,
        context->allocate_output(
            output_id++, {static_cast<long>(config_index_to_node.size())},
            &config_index_to_node_tensor));
    OP_REQUIRES_OK(context,
                   context->allocate_output(output_id++, {}, &module_id));

    // fill in the tensor values
    OP_REQUIRES_OK(context, hlo_encoder.EncodeHloModule(hlo_module_proto));
    for (int i = 0; i < config_index_to_node.size(); ++i) {
      config_index_to_node_tensor->vec<int64_t>()(i) = config_index_to_node[i];
    }
    module_id->scalar<uint64_t>()(0) = data->GetModuleUniqueId();

    OP_REQUIRES_OK(context, hlo_encoder.Finalize());

    tf::Tensor* computation_splits;
    const int64_t num_computations = hlo_encoder.computation_splits().size();
    OP_REQUIRES_OK(context,
                   context->allocate_output(output_id++, {num_computations},
                                            &computation_splits));
    for (int i = 0; i < num_computations; ++i) {
      computation_splits->vec<int64_t>()(i) =
          hlo_encoder.computation_splits()[i];
    }
  }

 private:
  bool directed_;
  std::string task_;
};

REGISTER_OP("EncodeHloModuleData")
    .Input("proto_data: string")
    .Input("source_path: string")
    .Input("tuning_data_type: string")
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
    .Output("config_index_to_node: int64")
    .Output("module_ids: uint64")
    .Output("computation_splits: int64")
    .Attr("task: string")
    .Attr("directed: bool = false")
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
      c->set_output(11, c->Scalar());
      c->set_output(12, c->Vector(-1));
      return OkStatus();
    });

REGISTER_KERNEL_BUILDER(Name("EncodeHloModuleData").Device(tf::DEVICE_CPU),
                        EncodeHloModuleData);

}  // namespace
}  // namespace ml_cost_model
}  // namespace xla