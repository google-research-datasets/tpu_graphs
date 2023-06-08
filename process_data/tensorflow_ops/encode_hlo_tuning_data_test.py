# Copyright 2023 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for google3.third_party.py.tpu_graphs.process_data.tensorflow_ops.encode_hlo_tuning_data."""

import tensorflow.compat.v2 as tf

from tpu_graphs.process_data.tensorflow_ops import encode_hlo_tuning_data
from google3.pyglib import resources
from google3.testing.pybase import googletest


class EncodeHloTuningData(googletest.TestCase):

  def test_node_features_size(self):
    path = resources.GetResourceFilename(
        "google3/third_party/py/tpu_graphs/process_data/tensorflow_ops/testdata/op_tuning_data.pb"
    )
    encoded_hlo = encode_hlo_tuning_data.encode_hlo_tuning_data(
        tf.io.read_file(path),
        path,
        tuning_data_type="op_tuning",
        sample_rate=1.0,
        samples_limit=-1,
        task="op_window_cost",
        directed=True,
    )
    num_nodes = tf.shape(encoded_hlo.opcodes_values).numpy()[0]
    self.assertGreater(num_nodes, 0)
    self.assertEqual(
        tf.shape(encoded_hlo.node_features_values).numpy()[0], num_nodes
    )
    self.assertEqual(
        tf.shape(encoded_hlo.node_features_values).numpy()[1],
        140,
    )
    self.assertEqual(
        tf.shape(encoded_hlo.module_ids).numpy()[0],
        tf.shape(encoded_hlo.compute_times_ns).numpy()[0],
    )
    self.assertEqual(
        encoded_hlo.opcodes_splits.numpy().tolist(),
        encoded_hlo.node_features_splits.numpy().tolist(),
    )
    self.assertEqual(encoded_hlo.opcodes_splits.numpy()[-1], num_nodes)
    module_ids = encoded_hlo.module_ids.numpy().tolist()
    module_counters = encoded_hlo.module_features.numpy().tolist()
    self.assertEqual(len(module_ids), len(module_counters))
    self.assertLess(
        tf.reduce_min(encoded_hlo.opcodes_values).numpy(),
        tf.reduce_max(encoded_hlo.opcodes_values).numpy(),
    )
    self.assertLess(
        tf.reduce_min(encoded_hlo.node_features_values).numpy(),
        tf.reduce_max(encoded_hlo.node_features_values).numpy(),
    )
    self.assertLess(
        tf.reduce_min(encoded_hlo.module_features).numpy(),
        tf.reduce_max(encoded_hlo.module_features).numpy(),
    )
    print("opcodes_values:", encoded_hlo.opcodes_values)
    print("node_features_values:", encoded_hlo.node_features_values)
    print("opcodes_splits:", encoded_hlo.opcodes_splits)
    print("node_features_splits:", encoded_hlo.node_features_splits)
    print("operand_adj_matrix_values:", encoded_hlo.operand_adj_matrix_values)
    print("operand_adj_matrix_indices:", encoded_hlo.operand_adj_matrix_indices)
    print("operand_adj_matrix_shape:", encoded_hlo.operand_adj_matrix_shape)
    print("consumer_adj_matrix_values:", encoded_hlo.consumer_adj_matrix_values)
    print(
        "consumer_adj_matrix_indices:", encoded_hlo.consumer_adj_matrix_indices
    )
    print("consumer_adj_matrix_shape:", encoded_hlo.consumer_adj_matrix_shape)
    print("normalization_values:", encoded_hlo.normalization_values)
    print("compute_times_ns:", encoded_hlo.compute_times_ns)
    print("module_ids:", encoded_hlo.module_ids)
    print("module_features:", encoded_hlo.module_features)


if __name__ == "__main__":
  googletest.main()
