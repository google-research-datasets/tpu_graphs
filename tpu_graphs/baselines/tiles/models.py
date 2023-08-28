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

"""Defines GNNs and MLP models for ranking module configurations on tiles data.

The high-level models are:
  + LateJoinResGCN: Applies GNN on op nodes. The GNN output will be concatenated
    with module config features. Finally, MLP outputs scalar that ranks each
    config. Here, GNN is GCN with residual connections.
  + EarlyJoinResGCN: Like above, however, it replicates (==broadcasts) module
    config features on op nodes then applies ResGCN, then applies MLP.
  + EarlyJoinSAGE and LateJoinSAGE: like above, but using GraphSAGE as backbone.

[GCN] Kipf and Welling, ICLR'17.
[GraphSAGE] Hamilton et al, NeurIPS'17.
"""
import abc

import tensorflow as tf
import tensorflow_gnn as tfgnn

from tpu_graphs.baselines.tiles import implicit


class _ConfigFeatureJoiner(abc.ABC):
  """Defines interface for joining config features with op nodes.

  The implementations join features pre- or post-GNN, respectively, named as
  `_EarlyJoin` and `_LateJoin`.
  """

  @abc.abstractmethod
  def get_op_node_features(
      self, graph: tfgnn.GraphTensor, num_configs: int) -> tf.Tensor:
    """Should return feature matrix (or tensor) of op-nodes."""
    raise NotImplementedError()

  def get_penultimate_output(
      self, pooled: tf.Tensor, unused_graph: tfgnn.GraphTensor,
      unused_num_configs: int) -> tf.Tensor:
    """Must return tensor with shape `[batch_size, num_configs, hidden_dim]`."""
    return pooled


def _mlp(dims, hidden_activation, l2reg=1e-4, use_bias=True):
  """Helper function for multi-layer perceptron (MLP)."""
  layers = []
  for i, dim in enumerate(dims):
    if i > 0:
      layers.append(tf.keras.layers.Activation(hidden_activation))
    layers.append(tf.keras.layers.Dense(
        dim, kernel_regularizer=tf.keras.regularizers.l2(l2reg),
        use_bias=use_bias))
  return tf.keras.Sequential(layers)


class _OpEmbedding(tf.keras.Model):
  """Embeds GraphTensor.node_sets['op']['op'] nodes into feature 'op_e'."""

  def __init__(self, num_ops: int, embed_d: int, l2reg: float = 1e-4):
    super().__init__()
    self.embedding_layer = tf.keras.layers.Embedding(
        num_ops, embed_d, activity_regularizer=tf.keras.regularizers.l2(l2reg))

  def call(
      self, graph: tfgnn.GraphTensor,
      training: bool = False) -> tfgnn.GraphTensor:
    op_features = dict(graph.node_sets['op'].features)
    op_features['op_e'] = self.embedding_layer(
        tf.cast(graph.node_sets['op']['op'], tf.int32))
    return graph.replace_features(node_sets={'op': op_features})


class _SAGE(tf.keras.Model, _ConfigFeatureJoiner):
  """Implements GraphSAGE GNN Backbone."""

  def __init__(self, num_configs: int, num_ops: int,
               num_gnns: int = 3, final_mlp_layers: int = 2,
               hidden_activation: str = 'leaky_relu', hidden_dim: int = 64,
               op_embed_dim: int = 64):
    super().__init__()
    self._num_configs = num_configs
    self._op_embedding = _OpEmbedding(num_ops, op_embed_dim)
    self._gnn_layers = []
    for unused_i in range(num_gnns):
      self._gnn_layers.append(_mlp([hidden_dim], hidden_activation))
    self._postnet = _mlp(
        [hidden_dim] * final_mlp_layers + [1], hidden_activation)
    self._activation_fn = getattr(tf.nn, hidden_activation)

  def call(self, graph: tfgnn.GraphTensor, training: bool = False):
    return self.forward(graph, self._num_configs)

  def forward(
      self, graph: tfgnn.GraphTensor, num_configs: int) -> tfgnn.GraphTensor:
    graph = self._op_embedding(graph)
    x = self.get_op_node_features(graph, num_configs)
    bidirectional_adj = implicit.AdjacencyMultiplier(graph, 'feed')
    bidirectional_adj = implicit.Sum(
        bidirectional_adj, bidirectional_adj.transpose())
    for gnn_layer in self._gnn_layers:
      y = bidirectional_adj @ x
      y = tf.concat([y, x], axis=-1)
      y = gnn_layer(y)
      y = self._activation_fn(y)
      y = tf.nn.l2_normalize(y, axis=-1)
      x = y

    pooled = tfgnn.pool_nodes_to_context(graph, 'op', 'sum', feature_value=x)

    pooled = self.get_penultimate_output(pooled, graph, num_configs)
    # Pooled has shape [batch_size, num_configs, hidden_dim]
    # _postnet maps across last channel from hidden_dim to 1.

    return tf.squeeze(self._postnet(pooled), -1)


class _ResGCN(tf.keras.Model, _ConfigFeatureJoiner):
  """Implements GCN backbone with residual connections."""

  def __init__(self, num_configs: int, num_ops: int,
               num_gnns: int = 3, mlp_layers: int = 2,
               hidden_activation: str = 'leaky_relu', hidden_dim: int = 64,
               op_embed_dim: int = 32, directed: bool = False,
               reduction: str = 'sum'):
    super().__init__()
    self._num_configs = num_configs
    self._num_ops = num_ops
    self._op_embedding = _OpEmbedding(num_ops, op_embed_dim)
    self._gc_layers = []
    self._activation_fn = getattr(tf.nn, hidden_activation)
    self._directed = directed
    self._reduction = reduction
    self._prenet = _mlp([hidden_dim, hidden_dim], self._activation_fn)
    self._postnet = _mlp([hidden_dim, 1], self._activation_fn)
    for _ in range(num_gnns):
      if directed:
        configs_mlps = (_mlp([hidden_dim] * mlp_layers, self._activation_fn),
                        _mlp([hidden_dim] * mlp_layers, self._activation_fn),
                        _mlp([hidden_dim] * mlp_layers, self._activation_fn))
      else:
        configs_mlps = (_mlp([hidden_dim] * mlp_layers, self._activation_fn),)
      self._gc_layers.append(tuple(configs_mlps))

  def call(self, graph: tfgnn.GraphTensor, training: bool = False):
    del training
    return self.forward(graph, self._num_configs)

  def forward(
      self, graph: tfgnn.GraphTensor, num_configs: int) -> tfgnn.GraphTensor:
    graph = self._op_embedding(graph)
    x = self.get_op_node_features(graph, num_configs)

    am = implicit.AdjacencyMultiplier(graph, 'feed')
    am = am.add_eye().normalize_right()
    x = self._prenet(x)
    for gc_layer in self._gc_layers:
      y = self._activation_fn(x)
      forward_layer = gc_layer[0]
      if self._directed:
        reverse_layer = gc_layer[1]
        self_layer = gc_layer[2]
        y = (forward_layer(am @ y) + reverse_layer(am.transpose() @ y)
             + self_layer(y))
      else:
        y = forward_layer((am @ y) + (am.transpose() @ y)  + y)

      # Residual connection.
      x += y

    x = self._activation_fn(x)
    pooled = tfgnn.pool_nodes_to_context(
        graph, 'op', self._reduction, feature_value=x)
    # Pooled has shape [batch_size, num_configs, hidden_dim]

    pooled = self.get_penultimate_output(pooled, graph, num_configs)

    return tf.squeeze(self._postnet(pooled), -1)


class _EarlyJoin(_ConfigFeatureJoiner):
  """Joins module configuration features before applying GNN backbone."""

  def get_op_node_features(
      self, graph: tfgnn.GraphTensor, num_configs: int) -> tf.Tensor:
    graph = _EarlyJoin.attach_config_features_on_op_nodes(graph)
    return tf.concat([
        # Shape (num_nodes, num_configs, embedding dim)
        tf.stack([graph.node_sets['op']['op_e']] * num_configs, 1),
        # Shape (num_nodes, num_configs, config feat dim)
        graph.node_sets['op']['config_feats'],
        # Shape (num_nodes, num_configs, op feat dim)
        tf.stack([graph.node_sets['op']['feats']] * num_configs, 1),
    ], axis=-1)

  @staticmethod
  def attach_config_features_on_op_nodes(
      graph: tfgnn.GraphTensor) -> tfgnn.GraphTensor:
    """Replicates config features on every op node."""
    # Shape: [batch_size * num_configs, feature size].
    config_feats = graph.node_sets['config']['feats']
    batch_size = graph.node_sets['config'].sizes.shape[0]
    config_feats = tf.reshape(
        config_feats, [batch_size, -1, config_feats.shape[-1]])
    # shape: (total number of op nodes, config feats dimension)
    op_broadcasted = tfgnn.broadcast_node_to_edges(
        graph, 'g_op', tfgnn.SOURCE, feature_value=config_feats)
    op_features = dict(graph.node_sets['op'].features)
    op_features['config_feats'] = op_broadcasted
    return graph.replace_features(node_sets={'op': op_features})


class _LateJoin(_ConfigFeatureJoiner):
  """Joins module configuration features after applying GNN backbone."""

  def get_op_node_features(
      self, graph: tfgnn.GraphTensor, num_configs: int) -> tfgnn.GraphTensor:
    del num_configs
    return tf.concat([
        # Shape (num_nodes, embedding dim)
        graph.node_sets['op']['op_e'],
        # Shape (num_nodes, op feat dim)
        graph.node_sets['op']['feats'],
    ], axis=-1)

  def get_penultimate_output(
      self, pooled: tf.Tensor, graph: tfgnn.GraphTensor,
      num_configs: int) -> tf.Tensor:
    config_feats = graph.node_sets['config']['feats']
    batch_size = graph.node_sets['config'].sizes.shape[0]
    config_feats = tf.reshape(
        config_feats, [batch_size, -1, config_feats.shape[-1]])
    # Shape like config feats
    pooled = tf.stack([pooled] * num_configs, 1)
    pooled = tf.concat([pooled, config_feats], -1)
    return pooled


class LateJoinResGCN(_LateJoin, _ResGCN):
  pass


class EarlyJoinResGCN(_EarlyJoin, _ResGCN):
  pass


class LateJoinSAGE(_LateJoin, _SAGE):
  pass


class EarlyJoinSAGE(_EarlyJoin, _SAGE):
  pass


class MLP(tf.keras.Model):
  """Embeds op codes, averages features across all-nodes, passing thru MLP."""

  def __init__(
      self, num_configs: int, num_ops: int, op_embed_dim: int = 32,
      mlp_layers: int = 2, hidden_activation: str = 'leaky_relu',
      hidden_dim: int = 64, reduction: str = 'sum'):
    super().__init__()
    self._num_configs = num_configs
    self._num_ops = num_ops
    self._op_embedding = _OpEmbedding(num_ops, op_embed_dim)
    self._reduction = reduction
    layer_dims = [hidden_dim] * mlp_layers
    layer_dims.append(1)
    self._mlp = _mlp(layer_dims, hidden_activation)

  def call(self, graph: tfgnn.GraphTensor, training: bool = False):
    del training
    return self.forward(graph, self._num_configs)

  def forward(
      self, graph: tfgnn.GraphTensor, num_configs: int) -> tfgnn.GraphTensor:
    graph = self._op_embedding(graph)
    op_feats = tf.concat([
        tfgnn.pool_nodes_to_context(
            graph, 'op', self._reduction, feature_name='feats'),
        tfgnn.pool_nodes_to_context(
            graph, 'op', self._reduction, feature_name='op_e'),
    ], axis=-1)

    config_feats = graph.node_sets['config']['feats']
    batch_size = graph.node_sets['config'].sizes.shape[0]
    config_feats = tf.reshape(
        config_feats, [batch_size, -1, config_feats.shape[-1]])
    # Shape like config feats
    op_feats = tf.stack([op_feats] * num_configs, 1)
    op_feats = tf.concat([op_feats, config_feats], -1)
    return tf.squeeze(self._mlp(op_feats), -1)
