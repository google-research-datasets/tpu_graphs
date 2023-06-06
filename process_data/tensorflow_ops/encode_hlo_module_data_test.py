"""Tests for tpu_graphs.process_data.tensorflow_ops.encode_hlo_module_data."""

import tensorflow.compat.v2 as tf

from tpu_graphs.process_data.tensorflow_ops import encode_hlo_module_data
from google3.pyglib import resources
from google3.testing.pybase import googletest


class EncodeHloModuleDataTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    path = resources.GetResourceFilename(
        "google3/third_party/py/tpu_graphs/process_data/tensorflow_ops/testdata/module_tuning_data_layout.pb"
    )
    self.encoded_hlo = encode_hlo_module_data.encode_hlo_module_data(
        tf.io.read_file(path),
        path,
        "module_tuning",
        task="module_layout_cost",
        directed=True,
    )

  def test_non_zero_node(self):
    num_nodes = tf.shape(self.encoded_hlo.opcodes_values).numpy()[0]
    self.assertGreater(num_nodes, 0)
    self.assertEqual(
        tf.shape(self.encoded_hlo.node_features_values).numpy()[0], num_nodes
    )
    encoded_hlo = self.encoded_hlo
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
    print("config_index_to_node:", encoded_hlo.config_index_to_node)
    print("module_ids:", encoded_hlo.module_ids)
    print("computation_splits:", encoded_hlo.computation_splits)

  def test_nodes_equal_last_split(self):
    num_nodes = tf.shape(self.encoded_hlo.opcodes_values).numpy()[0]
    computation_splits = self.encoded_hlo.computation_splits.numpy()
    self.assertEqual(0, computation_splits[0])
    self.assertEqual(num_nodes, computation_splits[-1])

  def test_expected_node_feature_count(self):
    encoded_hlo = self.encoded_hlo
    self.assertEqual(tf.shape(encoded_hlo.node_features_values).numpy()[1], 140)
    self.assertLess(
        tf.reduce_min(encoded_hlo.opcodes_values).numpy(),
        tf.reduce_max(encoded_hlo.opcodes_values).numpy(),
    )
    self.assertLess(
        tf.reduce_min(encoded_hlo.node_features_values).numpy(),
        tf.reduce_max(encoded_hlo.node_features_values).numpy(),
    )

  def test_expected_node_neighbor_size(self):
    self.assertGreater(
        tf.shape(self.encoded_hlo.operand_adj_matrix_values).numpy()[0], 0
    )
    self.assertGreater(
        tf.shape(self.encoded_hlo.consumer_adj_matrix_values).numpy()[0], 0
    )

  def test_expected_tensor_shapes(self):
    self.assertLen(tf.shape(self.encoded_hlo.config_index_to_node).numpy(), 1)
    self.assertGreater(
        tf.shape(self.encoded_hlo.config_index_to_node).numpy()[0], 0
    )
    self.assertEmpty(tf.shape(self.encoded_hlo.module_ids).numpy())  # scalar


if __name__ == "__main__":
  googletest.main()
