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

#ifndef THIRD_PARTY_PY_TPU_GRAPHS_PROCESS_DATA_XLA_HLO_ENCODER_H_
#define THIRD_PARTY_PY_TPU_GRAPHS_PROCESS_DATA_XLA_HLO_ENCODER_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tpu_graphs/process_data/xla/hlo_opcode.h"
#include "tpu_graphs/proto/tuning.pb.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/statusor.h"

namespace xla {
namespace ml_lib {

namespace tf = ::tensorflow;

// Returns a vector with the contents of the google::protobuf::RepeatedField `source`
// sorted and with duplicates removed.
std::vector<int64_t> DeduplicateField(
    const ::google::protobuf::RepeatedField<int64_t>& source);

// An interface class for incrementally constructing tensors for HLO graph.
//
// Example usage:
//
// TensorBuilder builder;
// for (auto module : hlo_modules) {
//   builder.AdvanceModule();
//   for (auto comp : module.computations()) {
//     for (auto inst : comp.instructions()) {
//       builder.push_back(opcode);
//       builder.push_back(feature_vector);
//       builder.AdvanceNode();
//     }
//   }
// }
// builder.Finalize();
template <typename ScalarT>
class TensorBuilder {
 public:
  virtual ~TensorBuilder() = default;

  // Prepares the tensor for the next HLO module.
  virtual void AdvanceModule() = 0;

  // Prepares the tensor for the next node (primitive instruction).
  virtual void AdvanceNode() = 0;

  // Pushes a new scalar into the current row.
  virtual void push_back(ScalarT new_value) = 0;

  // Fills in the out tensors given as arguments to the constructor
  virtual void Finalize() = 0;
};

// An interface class for incrementally constructing tensors suitable for
// composition into a RaggedTensor.
template <typename ScalarT>
class RaggedTensorBuilder : public TensorBuilder<ScalarT> {
 public:
  ~RaggedTensorBuilder() override = default;

  // Takes `out_values` and `out_splits`, the tensors which will be mutated
  // to hold the ragged tensors values and row splits, respectively.
  // `out_values` should have shape (number_of_values * innermost_dim) where
  // innermost_dim is the static (non-ragged) size of the innermost dimension.
  // `out_splits` should have shape (number_of_rows + 1).
  explicit RaggedTensorBuilder(tf::Tensor* out_values, tf::Tensor* out_splits);

  void AdvanceModule() override;
  void AdvanceNode() override;
  void push_back(ScalarT new_value) override;
  void Finalize() override;

 private:
  int64_t values_added_;
  int64_t splits_added_;
  tf::Tensor* const out_values_;
  tf::Tensor* const out_splits_;
};

// A helper class for incrementally constructing a dense 2d tensor.
template <typename ScalarT>
class Dense2dTensorBuilder : public TensorBuilder<ScalarT> {
 public:
  ~Dense2dTensorBuilder() override = default;
  explicit Dense2dTensorBuilder(tf::Tensor* out_values, int64_t column_size);

  void AdvanceModule() override;
  void AdvanceNode() override;
  void push_back(ScalarT new_value) override;
  void Finalize() override;

 private:
  int64_t column_size_;
  int64_t row_idx_ = 0;
  int64_t column_idx_ = 0;
  tf::Tensor* const out_values_;
};

// A helper class for incrementally constructing a dense 1d tensor.
template <typename ScalarT>
class Dense1dTensorBuilder : public TensorBuilder<ScalarT> {
 public:
  ~Dense1dTensorBuilder() override = default;
  explicit Dense1dTensorBuilder(tf::Tensor* out_values)
      : out_values_(out_values) {
    CHECK_EQ(out_values->dims(), 1);
  }

  void AdvanceModule() override {}
  void AdvanceNode() override {}

  void push_back(ScalarT new_value) override {
    CHECK_LT(idx_, out_values_->dim_size(0));
    out_values_->vec<ScalarT>()(idx_++) = new_value;
  }

  void Finalize() override { CHECK_EQ(idx_, out_values_->dim_size(0)); }

 private:
  int64_t idx_ = 0;
  tf::Tensor* const out_values_;
};

// An interface class for constructing a batch of adjacency matrices.
class AdjMatrixBuilder {
 public:
  virtual ~AdjMatrixBuilder() = default;
  explicit AdjMatrixBuilder(
      const std::vector<std::vector<int64_t>>& identifiers) {
    // Build per-decomposed op runs maps from opaque ID to index for faster
    // lookups into `identifiers`
    identifiers_maps_.reserve(identifiers.size());
    for (uint32_t i = 0; i < identifiers.size(); ++i) {
      auto& map = identifiers_maps_.emplace_back();
      map.reserve(identifiers[i].size());
      for (uint32_t j = 0; j < identifiers[i].size(); ++j) {
        map[identifiers[i][j]] = j;
      }
    }
  }

  virtual void AdvanceModule() = 0;
  virtual void AdvanceNode() = 0;

  // Insert an edge for the current batch. The `identifiers` passed to the
  // object constructor will be used to the determine the row and column of
  // the adjacency matrix to update.
  //
  // The AdjMatrixBuilder should not have edges added until the first row has
  // been started (i.e. `AdvanceModule` has been called one or more times).
  virtual void AddEdge(const int64_t from, const int64_t to) = 0;

  // Finalize construction of the tensors given to the constructor. After
  // calling `Finalize`, the `AdjMatrixBuilder` should no longer be used and
  // the given tensors describe a valid sparse tensor.
  virtual Status Finalize() = 0;

 protected:
  // Each `identifiers_maps_[i]` is a mapping from opaque identifiers to
  // unique indices in the adj. matrix at `i`.
  std::vector<absl::flat_hash_map<int64_t, uint32_t>> identifiers_maps_;
};

// A helper class for constructing dense tensors suitable for composition into
// a sparse matrix representing a batch of adjacency matrices.
class EdgeListAdjMatrixBuilder : public AdjMatrixBuilder {
 public:
  ~EdgeListAdjMatrixBuilder() override = default;

  // Takes `indices_out`, `values_out`, and `shape_out`, which are tensors
  // that will be mutated. These correspond to the 3
  // dense tensors used to construct a TensorFlow `SparseTensor.` Also takes a
  // nested vector of vectors `identifiers` which is used as per-batch indices
  // such that row/column indices correspond to that identifier.
  explicit EdgeListAdjMatrixBuilder(
      const std::vector<std::vector<int64_t>>& identifiers,
      tf::Tensor* indices_out, tf::Tensor* values_out, tf::Tensor* shape_out);

  void AdvanceModule() override { rows_advanced_++; }
  void AdvanceNode() override {}
  void AddEdge(const int64_t from, const int64_t to) override;

  Status Finalize() override;

 private:
  tf::Tensor* const indices_out_;
  tf::Tensor* const values_out_;
  tf::Tensor* const shape_out_;
  uint32_t rows_advanced_;
  uint32_t edges_added_;

  // The largest index added to any batch so far. Used by `Finalize` to
  // determine the correct bounding shape for the tensor of adjacency
  // matrices.
  uint32_t max_index_;
};

// A helper class for constructing dense tensors suitable for composition into
// a dense matrix representing a batch of adjacency matrices.
class DenseNeighborIndicesBuilder : public AdjMatrixBuilder {
 public:
  ~DenseNeighborIndicesBuilder() override = default;

  // Takes `module_ids`, `seq_indices`, `seq_masks`, `neighbors_out`, and
  // `neighbor_indices` to be mutated.
  // - `module_ids` is a vector that maps node id to HLO module id.
  // - `seq_indices` is a matrix, where seq_indices[i] is a vector
  // containing indices of node ids belong to module i.
  // - `seq_masks` is a matrix with the same shape as `seq_indices`, where
  // seq_masks[i][j]=false means ignoring seq_indices[i][j] node for module i.
  // - `neighbors_out` is a dense matrix of neighbor indices, where each row
  // contains the list of neighbors for that node.
  // - `neighbor_masks(i,j)` indicates if `neighbors_out(i,j)` should be
  // considered or not.
  explicit DenseNeighborIndicesBuilder(
      const std::vector<std::vector<int64_t>>& identifiers,
      const int64_t module_count, tf::Tensor* module_ids,
      tf::Tensor* seq_indices, tf::Tensor* seq_masks,
      tf::Tensor* neighbor_indices, tf::Tensor* neighbor_masks);

  void AdvanceModule() override;
  void AdvanceNode() override;
  void AddEdge(const int64_t from, const int64_t to) override;
  Status Finalize() override;

 private:
  int64_t module_id_ = -1;
  int64_t node_id_ = 0;
  int64_t node_offset_ = 0;
  int64_t max_neighbor_count_;
  int64_t nodes_per_module_ = 0;
  tf::Tensor* const module_ids_;
  tf::Tensor* const seq_indices_;
  tf::Tensor* const seq_masks_;
  tf::Tensor* const neighbor_indices_;
  tf::Tensor* const neighbor_masks_;

  // `neighbor_counts_[m][n]` tracks the number of neighbors of node n in
  // module m have been added so far.
  std::vector<std::vector<int64_t>> neighbor_counts_;
};

/**
 * Converts HloModuleProto into opcodes, adj. matrices,
 * and extra features.
 **/
class HloEncoder {
 public:
  virtual ~HloEncoder() = default;
  explicit HloEncoder(bool directed, bool count_neighbors,
                      absl::string_view task)
      : directed_(directed),
        task_(task),
        include_features_(GetIncludeFeatureBits(task)),
        count_neighbors_(count_neighbors) {}

  Status Finalize();

  // Encodes the given HLO module with window_config if specified.
  // If computation_unique_id is given, encodes only the specified computation.
  // Otherwise, encodes all computations.
  Status EncodeHloModule(
      const HloModuleProto& module,
      const std::optional<int64_t> computation_unique_id = std::nullopt);

  struct HloModuleStat {
    uint32_t total_edge_count = 0;
    uint32_t total_node_count = 0;
    uint32_t max_operand_count = 1;
    uint32_t max_user_count = 1;
    std::vector<std::vector<int64_t>> identifier_vocabs;
  };

  // Scans `module` and accumulates statistics useful for allocating
  // the right-sized tensors up front.
  // This function can be called multiple times on multiple HLO modules to
  // collect the statistics for an entire batch of HLO modules.
  // `stat` must be initualized properly.
  void CollectStats(const HloModuleProto& module, HloModuleStat* stat);

  // Similar to above but for one computation.
  // Another distinction is that this function ignores neightbor edges across
  // computations (for instructions with called computations).
  void CollectStats(const HloComputationProto& computation,
                    HloModuleStat* stat);

  // Returns a list of instruction positions that indicate the starting/ending
  // of computations. The list ends with the total number of instructions.
  const std::vector<int64_t>& computation_splits() const {
    return computation_splits_;
  }

 protected:
  std::unique_ptr<TensorBuilder<uint8_t>> opcodes_builder_;
  std::unique_ptr<TensorBuilder<float>> extra_features_builder_;
  std::unique_ptr<AdjMatrixBuilder> operand_adj_matrix_builder_;
  std::unique_ptr<AdjMatrixBuilder> user_adj_matrix_builder_;
  const bool directed_;
  const std::string task_;
  const uint8_t include_features_;

 private:
  // Advance each output tensor for the next HLO module.
  void AdvanceModule();

  // Advance each output tensor for the next HLO op (node).
  void AdvanceNode();

  // Add features describing the instruction `inst` to
  // `extra_features_builder_`
  void PushInstructionFeatures(const HloInstructionProto& inst,
                               const HloComputationProto& parent_computation);

  bool count_neighbors_;

  std::vector<int64_t> computation_splits_;
};

/**
 * SparseHloEncoder allocates and produces outputs to the `context`.
 * Let M be the number of HLO modules being encoded.
 * Let N be the number of nodes in all HLO modules being encoded.
 * Let E be the number of edges in all HLO modules being encoded.
 * Let F be GetExtraFeatureCount().
 *
 * The outputs are:
 * (0) opcodes_values: N-length array, opcodes of all nodes.
 * (1) opcodes_splits: (M+1)-length array, node indices indicating the starts
 *     and ends of HLO modules. For example, [0, 10, 20] means node indices
 *     [0, 10) belong to module 0, [10, 20) belong to module 1.
 * (2) extra_features_values: [N x F] matrix.
 * (3) extra_features_splits: same as opcodes_splits.
 * (4) operand_adj_matrix_values: E-length array, all ones.
 * (5) operand_adj_matrix_indices: [E x 3], where adj_matrix_indices[e] =
 *     [<module_id>, <from_node_id>, <to_node_id>].
 * (6) operand_adj_matrix_shape: 3-length array, adj_matrix_shape =
 *     [M, <max_nodes_per_module>, <max_nodes_per_module>].
 * (7-9) same as (4-6) but for users.
 *
 * If directed is false, then operand and user adj matrices are merged into
 * operand adj matrix (4-6), and user adj matrix (7-9) is empty.
 */
class SparseHloEncoder : public HloEncoder {
 public:
  ~SparseHloEncoder() override = default;
  explicit SparseHloEncoder(bool directed, absl::string_view task)
      : HloEncoder(directed, /*count_neighbors=*/false, task) {}

  // Create a number of "builder" objects used by `EncodeHloModule` to construct
  // the operation output tensors..
  //
  // Will allocate the operation's output tensors as a side effect. These are
  // used as the underlying storage for the builders.
  //
  // Return the number of outputs allocated in context.
  tf::StatusOr<int> CreateOutputBuilders(const HloModuleStat& stat,
                                         tf::OpKernelContext* context);

  struct Outputs {
    tf::Tensor* opcodes_values;
    tf::Tensor* opcodes_splits;
    tf::Tensor* extra_features_values;
    tf::Tensor* extra_features_splits;
    tf::Tensor* operand_adj_matrix_values;
    tf::Tensor* operand_adj_matrix_indices;
    tf::Tensor* operand_adj_matrix_shape;
    tf::Tensor* user_adj_matrix_values;
    tf::Tensor* user_adj_matrix_indices;
    tf::Tensor* user_adj_matrix_shape;
  };

  void CreateOutputBuilders(
      const std::vector<std::vector<int64_t>>& identifier_vocabs,
      Outputs* output_tensors);
};

/**
 * DenseHloEncoder allocates and produces outputs to the `context`.
 * Let M be the number of HLO modules being encoded.
 * Let N be the number of nodes in all HLO modules being encoded.
 * Let F be GetExtraFeatureCount().
 * Let D be the maximum number of node's degree (neighbors).
 *   D = 1 when all nodes have no neighbors.
 *
 * The outputs are:
 * (0) module_ids: N-length array, module ids nodes belongs to.
 * (1) opcodes_values: N-length array, opcodes of all nodes.
 * (2) extra_features_values: [N x F] matrix.
 * (3) operand_indices: [N x D] matrix, where each row contains the list of
 *     operands for that node.
 * (4) operand_masks: [N x D] matrix, masks for the previous matrix.
 *     If operand_masks[i][j] = 0, ignore operand_indices[i][j].
 * (5-6) same as (3-4) but for users.
 *
 * If directed is false, then operand and user adj matrices are merged into
 * operand adj matrix (3-4), and user adj matrix (5-6) is empty.
 */
class DenseHloEncoder : public HloEncoder {
 public:
  ~DenseHloEncoder() override = default;
  explicit DenseHloEncoder(bool directed, absl::string_view task)
      : HloEncoder(directed, /*count_neighbors=*/true, task) {}

  struct Outputs {
    tf::Tensor* module_ids;
    tf::Tensor* opcodes_values;
    tf::Tensor* extra_features_values;
    tf::Tensor* operand_indices;
    tf::Tensor* operand_masks;
    tf::Tensor* user_indices;
    tf::Tensor* user_masks;
    tf::Tensor* seq_indices;
    tf::Tensor* seq_masks;
  };

  void CreateOutputBuilders(
      const std::vector<std::vector<int64_t>>& identifier_vocabs,
      Outputs* output_tensors);
};

}  // namespace ml_lib
}  // namespace xla

#endif  // THIRD_PARTY_PY_TPU_GRAPHS_PROCESS_DATA_XLA_HLO_ENCODER_H_
