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

#include "third_party/py/tpu_graphs/process_data/xla/hlo_encoder.h"

#include <cstdint>
#include <vector>

#include "testing/base/public/gunit.h"
#include "third_party/tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "third_party/tensorflow/core/framework/tensor.h"
#include "third_party/tensorflow/core/framework/tensor_shape.h"
#include "third_party/tensorflow/core/framework/tensor_types.h"
#include "third_party/tensorflow/core/framework/types.proto.h"
#include "third_party/tensorflow/tsl/lib/core/status_test_util.h"

namespace xla {
namespace ml_lib {
namespace {

class HloEncoderTest : public HloTestBase {};

TEST_F(HloEncoderTest, DenseNeighborIndicesBuilderOnOneGraphOneNode) {
  std::vector<std::vector<int64_t>> identifiers = {{0}};
  tf::Tensor module_ids(tensorflow::DT_INT64, tf::TensorShape({1}));
  tf::Tensor seq_indices(tensorflow::DT_INT32, tf::TensorShape({1, 1}));
  tf::Tensor seq_masks(tensorflow::DT_BOOL, tf::TensorShape({1, 1}));
  tf::Tensor neighbor_indices(tensorflow::DT_INT64, tf::TensorShape({1, 1}));
  tf::Tensor neighbor_masks(tensorflow::DT_FLOAT, tf::TensorShape({1, 1}));
  DenseNeighborIndicesBuilder builder(identifiers,
                                      /*module_count=*/1, &module_ids,
                                      &seq_indices, &seq_masks,
                                      &neighbor_indices, &neighbor_masks);
  builder.AdvanceModule();
  builder.AdvanceNode();
  TF_ASSERT_OK(builder.Finalize());
  EXPECT_EQ(module_ids.flat<int64_t>()(0), 0);
  EXPECT_EQ(seq_indices.matrix<int32_t>()(0, 0), 0);
  EXPECT_EQ(seq_masks.matrix<bool>()(0, 0), 1);
  EXPECT_EQ(neighbor_indices.matrix<int64_t>()(0, 0), 0);
  EXPECT_EQ(neighbor_masks.matrix<float>()(0, 0), 0);
}

TEST_F(HloEncoderTest, DenseNeighborIndicesBuilderOnMultipleDirectedGraphs) {
  std::vector<std::vector<int64_t>> identifiers = {{0}, {20, 10, 0}};
  tf::Tensor module_ids(tensorflow::DT_INT64, tf::TensorShape({4}));
  tf::Tensor seq_indices(tensorflow::DT_INT32, tf::TensorShape({2, 3}));
  tf::Tensor seq_masks(tensorflow::DT_BOOL, tf::TensorShape({2, 3}));
  tf::Tensor neighbor_indices(tensorflow::DT_INT64, tf::TensorShape({4, 1}));
  tf::Tensor neighbor_masks(tensorflow::DT_FLOAT, tf::TensorShape({4, 1}));
  DenseNeighborIndicesBuilder builder(identifiers,
                                      /*module_count=*/2, &module_ids,
                                      &seq_indices, &seq_masks,
                                      &neighbor_indices, &neighbor_masks);
  // Module 0 has no edge.
  builder.AdvanceModule();
  builder.AdvanceNode();

  // Module 1: 10-->20, 0-->20
  builder.AdvanceModule();
  builder.AdvanceNode();
  builder.AddEdge(10, 20);
  builder.AdvanceNode();
  builder.AddEdge(0, 20);
  builder.AdvanceNode();

  TF_ASSERT_OK(builder.Finalize());
  EXPECT_EQ(module_ids.flat<int64_t>()(0), 0);
  EXPECT_EQ(module_ids.flat<int64_t>()(2), 1);

  EXPECT_EQ(seq_indices.matrix<int32_t>()(0, 0), 0);
  EXPECT_EQ(seq_indices.matrix<int32_t>()(0, 1), 0);
  EXPECT_EQ(seq_indices.matrix<int32_t>()(0, 2), 0);
  EXPECT_EQ(seq_indices.matrix<int32_t>()(1, 0), 1);
  EXPECT_EQ(seq_indices.matrix<int32_t>()(1, 1), 2);
  EXPECT_EQ(seq_indices.matrix<int32_t>()(1, 2), 3);

  EXPECT_EQ(seq_masks.matrix<bool>()(0, 0), 1);
  EXPECT_EQ(seq_masks.matrix<bool>()(0, 1), 0);
  EXPECT_EQ(seq_masks.matrix<bool>()(0, 2), 0);
  EXPECT_EQ(seq_masks.matrix<bool>()(1, 0), 1);
  EXPECT_EQ(seq_masks.matrix<bool>()(1, 1), 1);
  EXPECT_EQ(seq_masks.matrix<bool>()(1, 2), 1);

  // Node 0: points to no other node.
  EXPECT_EQ(neighbor_indices.matrix<int64_t>()(0, 0), 0);
  EXPECT_EQ(neighbor_masks.matrix<float>()(0, 0), 0);

  // Node 1 (id 20): points to no other node.
  EXPECT_EQ(neighbor_indices.matrix<int64_t>()(1, 0), 0);
  EXPECT_EQ(neighbor_masks.matrix<float>()(1, 0), 0);

  // Node 2 (id 10): points to node 1 (id 20)
  EXPECT_EQ(neighbor_indices.matrix<int64_t>()(2, 0), 1);
  EXPECT_EQ(neighbor_masks.matrix<float>()(2, 0), 1);

  // Node 3 (id 0): points to node 1 (id 20)
  EXPECT_EQ(neighbor_indices.matrix<int64_t>()(3, 0), 1);
  EXPECT_EQ(neighbor_masks.matrix<float>()(3, 0), 1);
}

}  // namespace
}  // namespace ml_lib
}  // namespace xla
