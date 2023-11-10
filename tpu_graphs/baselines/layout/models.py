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
      hidden_dim: int = 32,
      adj_normalization='sym',
      dropout='backpropdropout'):
    # dropout can be one of `('fullgraph', 'dropout', 'backpropdropout')`.
    # + fullgraph: all nodes and edges will participate in message-passing.
    # + dropout: most nodes are removed except a subgraph of nodes, that are
    #   topologically close.
    # + backpropdropout: The full graph is used for the forward pass but the
    #   gradients go through only a subgraph.
    super().__init__()
    assert dropout in ('fullgraph', 'dropout', 'backpropdropout')
    self._hidden_activation = hidden_activation
    self.dropout = dropout
    self._num_configs = num_configs
    self._num_ops = num_ops
    self._adj_normalization = adj_normalization
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

    # Implicit representations of adjacency computations, e.g.,
    # A_sym_norm = (D + I)^-0.5   (A + I)   (D + I)^-0.5   [GCN, Kipf & Welling]
    adj_op_symnorm = (
        adj_op_op + adj_op_op.transpose()).add_eye().normalize_symmetric()
    # A_right_norm = A D^-1  , or,  (A+I) (D+I)^-1
    adj_op_rightnorm = adj_op_op.add_eye().normalize_right()
    # A_t_right_norm: transpose A then do above.
    adj_op_t_rightnorm = adj_op_op.add_eye().transpose().normalize_right()
    # NOTE: since they are never multiplied (via op @), they are actually never
    # calculated. Only the ones invoked in `a_time_x` will actually be used on
    # data.

    def a_times_x(x):
      if self._adj_normalization == 'sym':
        return adj_op_symnorm @ x
      elif self._adj_normalization == 'asym':
        return tf.concat(
            [adj_op_rightnorm @ x, adj_op_t_rightnorm @ x], axis=-1)

    activation = tf.keras.layers.Activation(self._hidden_activation)
    x = node_features

    x = tf.stack([x] * num_configs, axis=1)
    config_features = 100 * (adj_config @ config_features)
    x = tf.concat([config_features, x], axis=-1)
    x = self._prenet(x)
    x = activation(x)

    for layer in self._gc_layers:
      y = x
      y = tf.concat([config_features, y], axis=-1)
      y = activation(layer(a_times_x(y)))
      x += y
    return x

  def forward(
      self, graph: tfgnn.GraphTensor, num_configs: int,
      dropout=None) -> tf.Tensor:
    if dropout is None:
      dropout = self.dropout
    graph = self._op_embedding(graph)

    config_features = graph.node_sets['nconfig']['feats']
    node_features = tf.concat([
        graph.node_sets['op']['feats'],
        graph.node_sets['op']['op_e']
    ], axis=-1)

    stop_gradient = tf.stop_gradient
    x: tf.Tensor | None = None
    edgeset_prefix = ''
    if dropout == 'dropout':  # See comment at `__init__(self, `.
      edgeset_prefix = 'sampled_'
      x = self._node_level_forward(
          node_features, config_features, graph,
          num_configs=num_configs, edgeset_prefix=edgeset_prefix)
    elif dropout == 'backpropdropout':
      edgeset_prefix = 'sampled_'
      x_full = self._node_level_forward(
          node_features, config_features, graph, num_configs=num_configs)
      x_full = stop_gradient(x_full)
      x_backprop = self._node_level_forward(
          node_features, config_features, graph,
          num_configs=num_configs, edgeset_prefix=edgeset_prefix)

      is_selected = graph.node_sets['op']['selected']
      # Need to expand twice as `is_selected` is a vector (num_nodes) but
      # x_{backprop, full} are 3D tensors (num_nodes, num_configs, num_feats).
      is_selected = tf.expand_dims(is_selected, axis=-1)
      is_selected = tf.expand_dims(is_selected, axis=-1)
      x = tf.where(is_selected, x_backprop, x_full)
    elif dropout == 'fullgraph':
      x = self._node_level_forward(
          node_features, config_features, graph, num_configs=num_configs)

    assert x is not None

    adj_config = implicit.AdjacencyMultiplier(graph, edgeset_prefix+'config')

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
