# Copyright 2023 The tpu_graphs Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GNN that does forward-pass on entire graph but backprop on a segment."""

import tensorflow as tf
import tensorflow_gnn as tfgnn

from tpu_graphs.baselines.tiles import implicit
from tpu_graphs.baselines.tiles import models


_OpEmbedding = models._OpEmbedding  # pylint: disable=protected-access
_mlp = models._mlp  # pylint: disable=protected-access


class ResModel(tf.keras.Model):
  """GNN with residual connections."""

  def __init__(
      self, num_configs: int, num_ops: int, op_embed_dim: int = 32,
      num_gnns: int = 2, mlp_layers: int = 2,
      hidden_activation: str = 'leaky_relu',
      hidden_dim: int = 32, reduction: str = 'sum'):
    super().__init__()
    self._num_configs = num_configs
    self._num_ops = num_ops
    self._op_embedding = _OpEmbedding(num_ops, op_embed_dim)
    self._prenet = _mlp([hidden_dim] * mlp_layers, hidden_activation)
    self._gc_layers = []
    for _ in range(num_gnns):
      self._gc_layers.append(_mlp([hidden_dim] * mlp_layers, hidden_activation))
    self._postnet = _mlp([hidden_dim, 1], hidden_activation, use_bias=False)

  def call(self, graph: tfgnn.GraphTensor, training: bool = False):
    del training
    return self.forward(graph, self._num_configs)

  def _node_level_forward(
      self, node_features: tf.Tensor,
      config_features: tf.Tensor,
      graph: tfgnn.GraphTensor, num_configs: int,
      edgeset_prefix='') -> tf.Tensor:
    adj_op_op = implicit.AdjacencyMultiplier(
        graph, edgeset_prefix+'feed')  # op->op
    adj_config = implicit.AdjacencyMultiplier(
        graph, edgeset_prefix+'config')  # nconfig->op

    adj_op_op_hat = (adj_op_op + adj_op_op.transpose()).add_eye()
    adj_op_op_hat = adj_op_op_hat.normalize_symmetric()

    x = node_features

    x = tf.stack([x] * num_configs, axis=1)
    config_features = 100 * (adj_config @ config_features)
    x = tf.concat([config_features, x], axis=-1)
    x = self._prenet(x)
    x = tf.nn.leaky_relu(x)

    for layer in self._gc_layers:
      y = x
      y = tf.concat([config_features, y], axis=-1)
      y = tf.nn.leaky_relu(layer(adj_op_op_hat @ y))
      x += y
    return x

  def forward(
      self, graph: tfgnn.GraphTensor, num_configs: int,
      backprop=True) -> tf.Tensor:
    graph = self._op_embedding(graph)

    config_features = graph.node_sets['nconfig']['feats']
    node_features = tf.concat([
        graph.node_sets['op']['feats'],
        graph.node_sets['op']['op_e']
    ], axis=-1)

    x_full = self._node_level_forward(
        node_features=tf.stop_gradient(node_features),
        config_features=tf.stop_gradient(config_features),
        graph=graph, num_configs=num_configs)

    if backprop:
      # TODO(haija, mangpo): Potential place for efficiency improvement. We are
      # now running forward pass on the bug graph. Potential improvement could
      # use stored (precomputed) activations.
      x_backprop = self._node_level_forward(
          node_features=node_features,
          config_features=config_features,
          graph=graph, num_configs=num_configs, edgeset_prefix='sampled_')

      is_selected = graph.node_sets['op']['selected']
      # Need to expand twice as `is_selected` is a vector (num_nodes) but
      # x_{backprop, full} are 3D tensors (num_nodes, num_configs, num_feats).
      is_selected = tf.expand_dims(is_selected, axis=-1)
      is_selected = tf.expand_dims(is_selected, axis=-1)
      x = tf.where(is_selected, x_backprop, x_full)
    else:
      x = x_full

    adj_config = implicit.AdjacencyMultiplier(graph, 'config')

    # Features for configurable nodes.
    config_feats = (adj_config.transpose() @ x)

    # Global pooling
    adj_pool_op_sum = implicit.AdjacencyMultiplier(graph, 'g_op').transpose()
    adj_pool_op_mean = adj_pool_op_sum.normalize_right()
    adj_pool_config_sum = implicit.AdjacencyMultiplier(
        graph, 'g_config').transpose()
    x = self._postnet(tf.concat([
        # (A D^-1) @ Features
        adj_pool_op_mean @ x,
        # l2_normalize( A @ Features )
        tf.nn.l2_normalize(adj_pool_op_sum @ x, axis=-1),
        # l2_normalize( A @ Features )
        tf.nn.l2_normalize(adj_pool_config_sum @ config_feats, axis=-1),
    ], axis=-1))

    x = tf.squeeze(x, -1)

    return x
