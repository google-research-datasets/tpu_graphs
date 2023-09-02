#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <iostream>
#include <string>

#include "tpu_graphs/process_data/xla/hlo_encoder.h"
#include "tpu_graphs/proto/tuning.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/env.h"

namespace xla {
namespace ml_lib {

namespace py = pybind11;

std::tuple<py::array_t<uint8_t>, py::array_t<float>, py::array_t<int64_t>,
           py::array_t<int64_t>, py::array_t<int64_t>>
ExtractGraphFeatures(const std::string& source_path) {
  // Read HLO graph from protobuf file.
  const std::string task = "module_layout_cost";
  tpu_graphs::ModuleTuningData proto_data;
  CHECK(tf::ReadBinaryProto(tf::Env::Default(), source_path, &proto_data).ok());
  const HloModuleProto& hlo_module_proto = proto_data.module();

  // Count total nodes and edges in all computations in the given HloModule.
  // We do this first pass to avoid the cost of reallocating
  // incorrectly-sized tensors later.
  ml_lib::SparseHloEncoder hlo_encoder(/*directed=*/true, task);
  ml_lib::HloEncoder::HloModuleStat stat;
  hlo_encoder.CollectStats(hlo_module_proto, &stat);
  const int module_count = stat.identifier_vocabs.size();
  const int node_feat_count = GetNodeFeatureCount(task);

  // Set up output tensors for hlo_encoder.
  tf::Tensor opcodes_values(tf::DT_UINT8,
                            tf::TensorShape({stat.total_node_count}));
  tf::Tensor opcodes_splits(tf::DT_INT64, tf::TensorShape({module_count + 1}));
  tf::Tensor extra_features_values(
      tf::DT_FLOAT, tf::TensorShape({stat.total_node_count, node_feat_count}));
  tf::Tensor extra_features_splits(tf::DT_INT64,
                                   tf::TensorShape({module_count + 1}));
  tf::Tensor operand_adj_matrix_values(
      tf::DT_UINT8, tf::TensorShape({stat.total_edge_count}));
  tf::Tensor operand_adj_matrix_indices(
      tf::DT_INT64, tf::TensorShape({stat.total_edge_count, 3}));
  tf::Tensor operand_adj_matrix_shape(tf::DT_INT64, tf::TensorShape({3}));
  // Reverse edges
  tf::Tensor user_adj_matrix_values(tf::DT_UINT8,
                                    tf::TensorShape({stat.total_edge_count}));
  tf::Tensor user_adj_matrix_indices(
      tf::DT_INT64, tf::TensorShape({stat.total_edge_count, 3}));
  tf::Tensor user_adj_matrix_shape(tf::DT_INT64, tf::TensorShape({3}));
  SparseHloEncoder::Outputs outputs = {&opcodes_values,
                                       &opcodes_splits,
                                       &extra_features_values,
                                       &extra_features_splits,
                                       &operand_adj_matrix_values,
                                       &operand_adj_matrix_indices,
                                       &operand_adj_matrix_shape,
                                       &user_adj_matrix_values,
                                       &user_adj_matrix_indices,
                                       &user_adj_matrix_shape};
  hlo_encoder.CreateOutputBuilders(stat.identifier_vocabs, &outputs);

  // Extract graph features.
  CHECK(hlo_encoder.EncodeHloModule(hlo_module_proto).ok());

  // Move outputs to numpy arrays.
  py::array_t<uint8_t> node_opcode(stat.total_node_count,
                                   (const uint8_t*)opcodes_values.data());

  py::array_t<float> node_feat(
      std::vector<ptrdiff_t>{stat.total_node_count, node_feat_count},
      (const float*)extra_features_values.data());

  py::array_t<int64_t> edge_index(
      std::vector<ptrdiff_t>{stat.total_edge_count, 2});
  auto edge_index_mutable = edge_index.mutable_unchecked<2>();
  for (int i = 0; i < stat.total_edge_count; ++i) {
    // module_id is always 0 for layout collection.
    CHECK_EQ(operand_adj_matrix_indices.matrix<int64_t>()(i, 0), 0);
    edge_index_mutable(i, 0) =
        operand_adj_matrix_indices.matrix<int64_t>()(i, 1);
    edge_index_mutable(i, 1) =
        operand_adj_matrix_indices.matrix<int64_t>()(i, 2);
  }

  py::array_t<int64_t> node_config_ids(proto_data.config_index_to_node_size());
  auto node_config_ids_mutable = node_config_ids.mutable_unchecked<1>();
  for (int i = 0; i < proto_data.config_index_to_node_size(); ++i) {
    node_config_ids_mutable(i) = proto_data.config_index_to_node(i);
  }

  const int64_t num_computations = hlo_encoder.computation_splits().size();
  py::array_t<int64_t> node_splits(num_computations);
  auto node_splits_mutable = node_splits.mutable_unchecked<1>();
  for (int i = 0; i < num_computations; ++i) {
    node_splits_mutable(i) = hlo_encoder.computation_splits()[i];
  }

  return std::make_tuple(node_opcode, node_feat, edge_index, node_config_ids,
                         node_splits);
}

}  // namespace ml_lib
}  // namespace xla

PYBIND11_MODULE(graph_features, m) {
  m.def(
      "extract_graph_features", &xla::ml_lib::ExtractGraphFeatures,
      "Extract graph features given a path to protobuf file containing an HLO "
      "graph. Return (node_opcode, node_feat, edge_index, node_config_ids, "
      "node_splits)");
}
