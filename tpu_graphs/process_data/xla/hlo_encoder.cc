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

#include "tpu_graphs/process_data/xla/hlo_encoder.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/match.h"
#include "tpu_graphs/process_data/xla/featurizers.h"
#include "tpu_graphs/process_data/xla/hlo_opcode.h"
#include "tpu_graphs/proto/tuning.pb.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

namespace xla {
namespace ml_lib {

namespace tf = ::tensorflow;
namespace sparse = ::tensorflow::sparse;

namespace {
// Returns a map from computation IDs to their entry instruction IDs.
absl::flat_hash_map<__typeof__(HloComputationProto().id()),
                    __typeof__(HloInstructionProto().id())>
CollectComputationEntries(
    const google::protobuf::RepeatedPtrField<HloComputationProto>& computations) {
  absl::flat_hash_map<__typeof__(HloComputationProto().id()),
                      __typeof__(HloInstructionProto().id())>
      entries;
  entries.reserve(computations.size());
  for (const auto& computation : computations) {
    entries[computation.id()] = computation.root_id();
  }
  return entries;
}

// Put a sparse tensor (expressed as indices and values tensors) into
// canonical (row-major or equivalently lexicographic order in `indices[i]`)
// order in-place. For an example, see:
// https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/sparse/reorder
template <typename T>
static Status ReorderSparseTensor(tf::Tensor* indices, tf::Tensor* values,
                                  tf::TensorShape input_shape) {
  sparse::SparseTensor::ShapeArray std_order(input_shape.dims());
  std::iota(std_order.begin(), std_order.end(), 0);

  sparse::SparseTensor reordered_sp;
  TF_RETURN_IF_ERROR(sparse::SparseTensor::Create(*indices, *values,
                                                  input_shape, &reordered_sp));

  reordered_sp.Reorder<T>(std_order);
  *indices = reordered_sp.indices();
  *values = reordered_sp.values();
  return OkStatus();
}

// Check if the given `indices` matrix has unique rows.
// Requires that `indices` be lexicographically sorted.
bool SparseTensorIndicesAreUnique(tf::Tensor* indices) {
  const auto mat = indices->matrix<int64_t>();
  for (int i = 1; i < indices->dim_size(0); ++i) {
    int8_t cmp = 0;
    for (int j = 0; j < indices->dim_size(1); ++j) {
      if (mat(i - 1, j) < mat(i, j)) {
        // Row (i-1) < Row i.
        cmp = 1;
        break;
      }
      if (mat(i - 1, j) > mat(i, j)) {
        // Row (i-1) > Row i.
        cmp = -1;
        break;
      }
    }
    CHECK_NE(cmp, -1) << "indices isn't sorted";
    if (cmp == 0) {
      // Row (i-1) = Row i. Not unique.
      return false;
    }
  }
  return true;
}
}  // namespace

std::vector<int64_t> DeduplicateField(
    const ::google::protobuf::RepeatedField<int64_t>& source) {
  std::vector<int64_t> vec(source.cbegin(), source.cend());
  std::sort(vec.begin(), vec.end());
  vec.erase(std::unique(vec.begin(), vec.end()), vec.end());
  return vec;
}

template <typename ScalarT>
RaggedTensorBuilder<ScalarT>::RaggedTensorBuilder(tf::Tensor* out_values,
                                                  tf::Tensor* out_splits)
    : values_added_(0),
      splits_added_(0),
      out_values_(out_values),
      out_splits_(out_splits) {
  CHECK_NE(out_values, nullptr);
  CHECK_NE(out_splits, nullptr);
  CHECK_EQ(out_splits->shape().dims(), 1);
  CHECK_GT(out_splits->NumElements(), 0);
}

template <typename ScalarT>
void RaggedTensorBuilder<ScalarT>::AdvanceModule() {
  CHECK_LT(splits_added_ + 1, out_splits_->NumElements());
  out_splits_->vec<int64_t>()(splits_added_) = values_added_;
  ++splits_added_;
}

template <typename ScalarT>
void RaggedTensorBuilder<ScalarT>::AdvanceNode() {}

template <typename ScalarT>
void RaggedTensorBuilder<ScalarT>::push_back(ScalarT new_value) {
  // We should know the final element_cnt at construction and never grow
  // values_
  CHECK_LT(values_added_, out_values_->NumElements());
  out_values_->flat<ScalarT>()(values_added_) = new_value;
  ++values_added_;
}

template <typename ScalarT>
void RaggedTensorBuilder<ScalarT>::Finalize() {
  CHECK_EQ(values_added_, out_values_->NumElements());
  CHECK_EQ(splits_added_ + 1, out_splits_->NumElements());
  CHECK(out_splits_->NumElements() == 0 || out_splits_->vec<int64_t>()(0) == 0)
      << "NumElements() = " << out_splits_->NumElements()
      << "; 0th element = " << out_splits_->vec<int64_t>()(0);

  // Add the final split index
  out_splits_->vec<int64_t>()(splits_added_) = values_added_;

  // RaggedTensorBuilder produces ragged tensors with a ragged outermost
  // dimension but any number (including zero) of non-ragged inner
  // dimensions. During construction, `out_splits_` is filled, at each row
  // change, with the number of values which had been added to that point.
  // However, the final state of `out_splits_` should describe the lengths
  // of each row in the ragged dimension, not the total number of values in
  // that row. In the case where all inner dimensions are of size one or
  // there aren't any inner dimensions, these are equivalent and the below
  // will do not work.
  //
  // For example, if the user is producing a ragged tensor with inner
  // dimension of size 2:
  //
  //   [[[1, 2], [3, 4]],
  //    [[5, 6]]]
  //
  // then, at this point, `out_splits_` will be [0, 4, 6]. Instead, this
  // should be divided by 2 to produce [0, 2, 3].
  if (out_values_->dims() > 1) {
    int64_t inner_dim_product = 1;
    for (int i = 1; i < out_values_->dims(); ++i) {
      inner_dim_product *= out_values_->dim_size(i);
    }
    auto out_splits_eigen = out_splits_->vec<int64_t>();
    for (int i = 0; i < out_splits_->NumElements(); ++i) {
      out_splits_eigen(i) = out_splits_eigen(i) / inner_dim_product;
    }
  }
}

template <typename ScalarT>
Dense2dTensorBuilder<ScalarT>::Dense2dTensorBuilder(tf::Tensor* out_values,
                                                    int64_t column_size)
    : column_size_(out_values->dim_size(1)), out_values_(out_values) {
  CHECK_NE(out_values, nullptr);
  CHECK_EQ(out_values->dims(), 2);
}

template <typename ScalarT>
void Dense2dTensorBuilder<ScalarT>::AdvanceModule() {}

template <typename ScalarT>
void Dense2dTensorBuilder<ScalarT>::AdvanceNode() {
  CHECK_EQ(column_idx_, column_size_);
  column_idx_ = 0;
  ++row_idx_;
}

template <typename ScalarT>
void Dense2dTensorBuilder<ScalarT>::push_back(ScalarT new_value) {
  // We should know the final element_cnt at construction and never grow
  // values_
  CHECK_LT(column_idx_, column_size_);
  out_values_->matrix<ScalarT>()(row_idx_, column_idx_) = new_value;
  ++column_idx_;
}

template <typename ScalarT>
void Dense2dTensorBuilder<ScalarT>::Finalize() {
  CHECK_EQ(row_idx_, out_values_->dim_size(0));
  CHECK_EQ(column_idx_, 0);
}

EdgeListAdjMatrixBuilder::EdgeListAdjMatrixBuilder(
    const std::vector<std::vector<int64_t>>& identifiers,
    tf::Tensor* indices_out, tf::Tensor* values_out, tf::Tensor* shape_out)
    : AdjMatrixBuilder(identifiers),
      indices_out_(indices_out),
      values_out_(values_out),
      shape_out_(shape_out),
      rows_advanced_(0),
      edges_added_(0),
      max_index_(0) {
  // Do some dim. checks
  CHECK(values_out->dim_size(0) == indices_out->dim_size(0));
}

void EdgeListAdjMatrixBuilder::AddEdge(const int64_t from, const int64_t to) {
  CHECK_GE(rows_advanced_, 1)
      << "AdvanceRow must be called at least once before adding an edge";
  CHECK_GT(indices_out_->dim_size(0), 0)
      << "Cannot add edge to adj. matrix with no capacity";
  CHECK_LE(edges_added_ + 1, indices_out_->dim_size(0))
      << "Output tensors are full";
  CHECK_NE(from, to) << "Cannot add self-edges";

  const auto i = edges_added_;
  const auto dim_idx = rows_advanced_ - 1;
  const auto from_idx = identifiers_maps_.at(dim_idx).at(from);
  const auto to_idx = identifiers_maps_.at(dim_idx).at(to);

  auto indices_eigen = indices_out_->matrix<int64_t>();
  indices_eigen(i, 0) = dim_idx;
  indices_eigen(i, 1) = from_idx;
  indices_eigen(i, 2) = to_idx;
  edges_added_ += 1;

  max_index_ = std::max(std::max(max_index_, from_idx), to_idx);
}

Status EdgeListAdjMatrixBuilder::Finalize() {
  // values_out_ is simple... it's just a bunch of ones!
  values_out_->vec<uint8_t>().setConstant(1);

  // shape_out_ for the most part, just needs to be big enough
  tf::TensorShape shape;
  TF_RETURN_IF_ERROR(tf::TensorShapeUtils::MakeShape(
      (int64_t[3]){rows_advanced_, max_index_ + 1, max_index_ + 1}, &shape));
  auto shape_out_eigen = shape_out_->vec<int64_t>();
  for (uint8_t i = 0; i < shape.dims(); i++) {
    shape_out_eigen(i) = shape.dim_size(i);
  }

  // Sparse tensors are generally assumed to be in row-major order, although
  // it is not enforced. Let's reorder here.
  TF_RETURN_IF_ERROR(
      ReorderSparseTensor<uint8_t>(indices_out_, values_out_, shape));

  // Check postconditions
  CHECK_EQ(values_out_->dim_size(0), indices_out_->dim_size(0));
  CHECK_EQ(indices_out_->dim_size(0), edges_added_)
      << "Expected output tensor to be filled (" << indices_out_->dim_size(0)
      << " elements) but was " << edges_added_;
  CHECK_EQ(SparseTensorIndicesAreUnique(indices_out_), true)
      << "Adj. matrix sparse tensor indices were not unique";
  return OkStatus();
}

DenseNeighborIndicesBuilder::DenseNeighborIndicesBuilder(
    const std::vector<std::vector<int64_t>>& identifiers,
    const int64_t module_count, tf::Tensor* module_ids, tf::Tensor* seq_indices,
    tf::Tensor* seq_masks, tf::Tensor* neighbor_indices,
    tf::Tensor* neighbor_masks)
    : AdjMatrixBuilder(identifiers),
      max_neighbor_count_(neighbor_indices->dim_size(1)),
      nodes_per_module_(seq_indices->dim_size(1)),
      module_ids_(module_ids),
      seq_indices_(seq_indices),
      seq_masks_(seq_masks),
      neighbor_indices_(neighbor_indices),
      neighbor_masks_(neighbor_masks),
      neighbor_counts_(module_count,
                       std::vector<int64_t>(nodes_per_module_, 0)) {
  CHECK_EQ(module_ids->dims(), 1);
  CHECK_EQ(seq_indices_->dims(), 2);
  CHECK_EQ(seq_masks_->dims(), 2);
  CHECK_EQ(neighbor_indices->dims(), 2);
  CHECK_EQ(neighbor_masks->dims(), 2);
  CHECK_EQ(neighbor_indices->dim_size(0), neighbor_masks->dim_size(0));
  CHECK_EQ(neighbor_indices->dim_size(1), neighbor_masks->dim_size(1));

  for (int64_t i = 0; i < neighbor_indices->NumElements(); ++i) {
    neighbor_indices->flat<int64_t>()(i) = 0;
    neighbor_masks->flat<float>()(i) = 0;
  }
}

void DenseNeighborIndicesBuilder::AdvanceModule() {
  if (module_id_ >= 0) {
    // Pad the rest of the node indices with zero masks.
    // module_id_ is initialized to -1. When this function is called for the
    // first time, skipping this padding.
    for (int i = node_id_ - node_offset_; i < nodes_per_module_; ++i) {
      seq_indices_->matrix<int32_t>()(module_id_, i) = 0;
      seq_masks_->matrix<bool>()(module_id_, i) = false;
    }
  }
  ++module_id_;
  CHECK_LE(module_id_, neighbor_counts_.size());
  node_offset_ = node_id_;
}

void DenseNeighborIndicesBuilder::AdvanceNode() {
  CHECK_LT(node_id_, module_ids_->dim_size(0));
  module_ids_->flat<int64_t>()(node_id_) = module_id_;

  const int64_t local_i = node_id_ - node_offset_;
  seq_indices_->matrix<int32_t>()(module_id_, local_i) = node_id_;
  seq_masks_->matrix<bool>()(module_id_, local_i) = true;
  ++node_id_;
}

void DenseNeighborIndicesBuilder::AddEdge(const int64_t from,
                                          const int64_t to) {
  const auto from_idx = identifiers_maps_.at(module_id_).at(from);
  const auto to_idx = identifiers_maps_.at(module_id_).at(to);

  int64_t neighbor_idx = neighbor_counts_[module_id_][from_idx]++;
  CHECK_LT(neighbor_idx, max_neighbor_count_)
      << "Instruction id " << from << " has more than " << max_neighbor_count_
      << " neighbors.";
  neighbor_indices_->matrix<int64_t>()(node_offset_ + from_idx, neighbor_idx) =
      node_offset_ + to_idx;
  neighbor_masks_->matrix<float>()(node_offset_ + from_idx, neighbor_idx) = 1;
}

Status DenseNeighborIndicesBuilder::Finalize() {
  AdvanceModule();
  CHECK_EQ(node_id_, module_ids_->dim_size(0));
  CHECK_EQ(module_id_, neighbor_counts_.size());
  return OkStatus();
}

void HloEncoder::AdvanceModule() {
  opcodes_builder_->AdvanceModule();
  extra_features_builder_->AdvanceModule();
  operand_adj_matrix_builder_->AdvanceModule();
  if (user_adj_matrix_builder_) {
    user_adj_matrix_builder_->AdvanceModule();
  }
}

void HloEncoder::AdvanceNode() {
  opcodes_builder_->AdvanceNode();
  extra_features_builder_->AdvanceNode();
  operand_adj_matrix_builder_->AdvanceNode();
  if (user_adj_matrix_builder_) {
    user_adj_matrix_builder_->AdvanceNode();
  }
}

Status HloEncoder::Finalize() {
  opcodes_builder_->Finalize();
  extra_features_builder_->Finalize();
  TF_RETURN_IF_ERROR(operand_adj_matrix_builder_->Finalize());
  if (user_adj_matrix_builder_) {
    TF_RETURN_IF_ERROR(user_adj_matrix_builder_->Finalize());
  }
  return OkStatus();
}

void HloEncoder::PushInstructionFeatures(
    const HloInstructionProto& inst,
    const HloComputationProto& parent_computation) {
  FeaturizeHloInstruction(extra_features_builder_.get(), inst,
                          parent_computation, include_features_);
}

Status HloEncoder::EncodeHloModule(
    const HloModuleProto& module,
    const std::optional<int64_t> computation_unique_id) {
  // Advance each output tensor to a new row for the next module.
  AdvanceModule();

  const auto& run_computations = module.computations();
  const auto computation_root_ids = CollectComputationEntries(run_computations);
  computation_splits_ = {0};
  int64_t instruction_count = 0;
  // Construct opcodes, extra features, and edges into the output tensors
  for (const auto& comp : run_computations) {
    if (computation_unique_id.has_value() &&
        *computation_unique_id != comp.id()) {
      continue;
    }
    for (const auto& inst : comp.instructions()) {
      const uint8_t converted_opcode =
          static_cast<uint8_t>(StringToOpcodeID(inst.opcode()));
      opcodes_builder_->push_back(converted_opcode);

      // Add features for `inst` to `extra_features_builder`
      PushInstructionFeatures(inst, comp);

      // Add edges for operands. Note that this assumes the downstream model
      // is uninterested in |e|.
      for (const auto operand_id : DeduplicateField(inst.operand_ids())) {
        operand_adj_matrix_builder_->AddEdge(inst.id(), operand_id);
        if (directed_) {
          if (user_adj_matrix_builder_) {
            user_adj_matrix_builder_->AddEdge(operand_id, inst.id());
          }
        } else {
          operand_adj_matrix_builder_->AddEdge(operand_id, inst.id());
        }
      }

      // Consider neighbors across computations only when encoding the entire
      // HLO module.
      if (!computation_unique_id.has_value()) {
        for (const auto subcomp_id : inst.called_computation_ids()) {
          operand_adj_matrix_builder_->AddEdge(
              inst.id(), computation_root_ids.at(subcomp_id));
          if (directed_) {
            if (user_adj_matrix_builder_) {
              user_adj_matrix_builder_->AddEdge(
                  computation_root_ids.at(subcomp_id), inst.id());
            }
          } else {
            operand_adj_matrix_builder_->AddEdge(
                computation_root_ids.at(subcomp_id), inst.id());
          }
        }
      }
      AdvanceNode();
    }
    // Add splits for only major computations (discarding parallel computations
    // for fusion, reduce, etc.)
    instruction_count += comp.instructions_size();
    if (absl::StrContains(comp.name(), "cluster") ||
        absl::StrContains(comp.name(), "while") ||
        absl::StrContains(comp.name(), "cond")) {
      computation_splits_.push_back(instruction_count);
    }
    if (computation_unique_id.has_value()) {
      break;
    }
  }
  if (computation_splits_.back() < instruction_count) {
    computation_splits_.push_back(instruction_count);
  }
  return OkStatus();
}

void HloEncoder::CollectStats(const HloModuleProto& module,
                              HloModuleStat* stat) {
  stat->identifier_vocabs.emplace_back();
  uint32_t total_edges_count = 0;
  for (const auto& comp : module.computations()) {
    stat->total_node_count += comp.instructions_size();
    for (const auto& inst : comp.instructions()) {
      // Include edges across computations.
      total_edges_count += DeduplicateField(inst.operand_ids()).size() +
                           inst.called_computation_ids_size();
      stat->identifier_vocabs.back().push_back(inst.id());
    }
  }
  stat->total_edge_count += total_edges_count;

  if (!count_neighbors_) {
    return;
  }

  CHECK_GT(stat->identifier_vocabs.back().size(), 0);
  absl::flat_hash_map<int64_t, int64_t> operand_count;
  absl::flat_hash_map<int64_t, int64_t> user_count;
  const auto computation_root_ids =
      CollectComputationEntries(module.computations());
  for (const auto& comp : module.computations()) {
    for (const auto& inst : comp.instructions()) {
      const std::vector<int64_t> operand_ids =
          DeduplicateField(inst.operand_ids());
      const int64_t parent_size = operand_ids.size();
      if (!operand_count.contains(inst.id())) {
        CHECK_EQ(operand_count[inst.id()], 0);
      }
      operand_count[inst.id()] += parent_size;
      for (const int64_t operand_id : operand_ids) {
        if (directed_) {
          ++user_count[operand_id];
        } else {
          ++operand_count[operand_id];
        }
      }

      // Include edges across computations.
      for (const auto subcomp_id : inst.called_computation_ids()) {
        ++operand_count[inst.id()];
        if (directed_) {
          ++user_count[computation_root_ids.at(subcomp_id)];
        } else {
          ++operand_count[computation_root_ids.at(subcomp_id)];
        }
      }
    }
  }

  for (auto it = operand_count.begin(); it != operand_count.end(); ++it) {
    if (it->second > stat->max_operand_count) {
      stat->max_operand_count = it->second;
    }
  }

  for (auto it = user_count.begin(); it != user_count.end(); ++it) {
    if (it->second > stat->max_user_count) {
      stat->max_user_count = it->second;
    }
  }
}

void HloEncoder::CollectStats(const HloComputationProto& computation,
                              HloModuleStat* stat) {
  stat->identifier_vocabs.emplace_back();
  uint32_t total_edges_count = 0;
  stat->total_node_count += computation.instructions_size();
  for (const auto& inst : computation.instructions()) {
    // Ignore edges across computations.
    total_edges_count += DeduplicateField(inst.operand_ids()).size();
    stat->identifier_vocabs.back().push_back(inst.id());
  }
  stat->total_edge_count = total_edges_count;
  if (!count_neighbors_) {
    return;
  }

  CHECK_GT(stat->identifier_vocabs.back().size(), 0);
  absl::flat_hash_map<int64_t, int64_t> operand_count_map;
  absl::flat_hash_map<int64_t, int64_t> user_count_map;
  for (const auto& inst : computation.instructions()) {
    const std::vector<int64_t> operand_ids =
        DeduplicateField(inst.operand_ids());
    const int64_t parent_size = operand_ids.size();
    if (!operand_count_map.contains(inst.id())) {
      CHECK_EQ(operand_count_map[inst.id()], 0);
    }
    operand_count_map[inst.id()] += parent_size;
    for (const int64_t operand_id : operand_ids) {
      if (directed_) {
        ++user_count_map[operand_id];
      } else {
        ++operand_count_map[operand_id];
      }
    }
  }

  for (const auto& [id, operand_count] : operand_count_map) {
    if (operand_count > stat->max_operand_count) {
      stat->max_operand_count = operand_count;
    }
  }
  for (const auto& [id, user_count] : user_count_map) {
    if (user_count > stat->max_user_count) {
      stat->max_user_count = user_count;
    }
  }
}

tf::StatusOr<int> SparseHloEncoder::CreateOutputBuilders(
    const HloModuleStat& stat, tf::OpKernelContext* context) {
  const int module_count = stat.identifier_vocabs.size();
  Outputs output_tensors;
  TF_RETURN_IF_ERROR(
      context->allocate_output(0, tf::TensorShape({stat.total_node_count}),
                               &output_tensors.opcodes_values));
  TF_RETURN_IF_ERROR(context->allocate_output(
      1, tf::TensorShape({module_count + 1}), &output_tensors.opcodes_splits));
  TF_RETURN_IF_ERROR(context->allocate_output(
      2, tf::TensorShape({stat.total_node_count, GetNodeFeatureCount(task_)}),
      &output_tensors.extra_features_values));
  TF_RETURN_IF_ERROR(
      context->allocate_output(3, tf::TensorShape({module_count + 1}),
                               &output_tensors.extra_features_splits));
  // Operand edges
  int64_t operand_edge_count =
      directed_ ? stat.total_edge_count : 2 * stat.total_edge_count;
  TF_RETURN_IF_ERROR(
      context->allocate_output(4, tf::TensorShape({operand_edge_count}),
                               &output_tensors.operand_adj_matrix_values));
  TF_RETURN_IF_ERROR(
      context->allocate_output(5, tf::TensorShape({operand_edge_count, 3}),
                               &output_tensors.operand_adj_matrix_indices));
  TF_RETURN_IF_ERROR(context->allocate_output(
      6, tf::TensorShape({3}), &output_tensors.operand_adj_matrix_shape));
  // User edges
  int64_t user_edge_count = directed_ ? stat.total_edge_count : 0;
  TF_RETURN_IF_ERROR(
      context->allocate_output(7, tf::TensorShape({user_edge_count}),
                               &output_tensors.user_adj_matrix_values));
  TF_RETURN_IF_ERROR(
      context->allocate_output(8, tf::TensorShape({user_edge_count, 3}),
                               &output_tensors.user_adj_matrix_indices));
  TF_RETURN_IF_ERROR(context->allocate_output(
      9, tf::TensorShape({3}), &output_tensors.user_adj_matrix_shape));

  CreateOutputBuilders(stat.identifier_vocabs, &output_tensors);
  return 10;
}

void SparseHloEncoder::CreateOutputBuilders(
    const std::vector<std::vector<int64_t>>& identifier_vocabs,
    Outputs* output_tensors) {
  opcodes_builder_ = std::make_unique<RaggedTensorBuilder<uint8_t>>(
      output_tensors->opcodes_values, output_tensors->opcodes_splits);
  extra_features_builder_ = std::make_unique<RaggedTensorBuilder<float>>(
      output_tensors->extra_features_values,
      output_tensors->extra_features_splits);
  operand_adj_matrix_builder_ = std::make_unique<EdgeListAdjMatrixBuilder>(
      identifier_vocabs, output_tensors->operand_adj_matrix_indices,
      output_tensors->operand_adj_matrix_values,
      output_tensors->operand_adj_matrix_shape);
  if (output_tensors->user_adj_matrix_indices != nullptr &&
      output_tensors->user_adj_matrix_values != nullptr &&
      output_tensors->user_adj_matrix_shape != nullptr) {
    user_adj_matrix_builder_ = std::make_unique<EdgeListAdjMatrixBuilder>(
        identifier_vocabs, output_tensors->user_adj_matrix_indices,
        output_tensors->user_adj_matrix_values,
        output_tensors->user_adj_matrix_shape);
  } else {
    user_adj_matrix_builder_ = nullptr;
  }
}

void DenseHloEncoder::CreateOutputBuilders(
    const std::vector<std::vector<int64_t>>& identifier_vocabs,
    Outputs* output_tensors) {
  opcodes_builder_ = std::make_unique<Dense1dTensorBuilder<uint8_t>>(
      output_tensors->opcodes_values);
  extra_features_builder_ = std::make_unique<Dense2dTensorBuilder<float>>(
      output_tensors->extra_features_values, GetNodeFeatureCount(task_));
  operand_adj_matrix_builder_ = std::make_unique<DenseNeighborIndicesBuilder>(
      identifier_vocabs, identifier_vocabs.size(), output_tensors->module_ids,
      output_tensors->seq_indices, output_tensors->seq_masks,
      output_tensors->operand_indices, output_tensors->operand_masks);
  if (output_tensors->user_indices != nullptr &&
      output_tensors->user_masks != nullptr) {
    user_adj_matrix_builder_ = std::make_unique<DenseNeighborIndicesBuilder>(
        identifier_vocabs, identifier_vocabs.size(), output_tensors->module_ids,
        output_tensors->seq_indices, output_tensors->seq_masks,
        output_tensors->user_indices, output_tensors->user_masks);
  } else {
    user_adj_matrix_builder_ = nullptr;
  }
}

}  // namespace ml_lib
}  // namespace xla
