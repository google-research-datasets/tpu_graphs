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

"""Computing L(A)-times-X where L is a linear transform of adjacency.

The computation is "implicit" i.e. L(A) is never computed.

NOTE: This file will be moved to tensorflow-gnn codebase.
"""


import tensorflow as tf
import tensorflow_gnn as tfgnn

EPSILON = 1e-6  # To prevent division by 0.


class Multiplier:
  """Holds an (implicit) matrix that can be multiplied with dense matrices."""
  _transpose: 'Multiplier' = None

  def matmul(self, mat: tf.Tensor) -> tf.Tensor:
    raise NotImplementedError()

  def rmatmul(self, mat: tf.Tensor) -> tf.Tensor:
    raise NotImplementedError()

  @property
  def shape(self) -> tuple['int|tf.Tensor', 'int|tf.Tensor']:
    raise NotImplementedError()

  def __matmul__(self, mat: tf.Tensor) -> tf.Tensor:
    tf.assert_equal(self.shape[1], shape(mat)[0])
    return self.matmul(mat)

  def __rmatmul__(self, mat: tf.Tensor) -> tf.Tensor:
    tf.assert_equal(shape(mat)[-1], self.shape[0])
    return self.rmatmul(mat)

  def __add__(self, mat: 'Multiplier') -> 'Multiplier':
    return Sum(self, mat)

  def transpose(self) -> 'Multiplier':
    if self._transpose is None:
      self._transpose = Transpose(self)
    return self._transpose

  def add_eye(self, diag_weight=float(1.0)) -> 'Multiplier':
    tf.assert_equal(self.shape[0], self.shape[1])
    return Sum(self, DiagMatrix(diag_weight * tf.ones([self.shape[0]])))

  def rowsums(self, replace_if_0: 'None|float|tf.Tensor' = None) -> tf.Tensor:
    """Returns vector with shape `num_rows = [self.shape[0]]` that sums rows.

    Args:
      replace_if_0: If None, returns the actual sum, leaving zero-entries as-is.
        Otherwise, zero-entries will be replaced by this value.
    """
    y = self @ tf.ones([self.shape[1]])  # M . 1

    if replace_if_0 is not None:
      y = tf.where(tf.abs(y) < EPSILON, replace_if_0 * tf.ones_like(y), y)
    return y

  def colsums(self, replace_if_0: 'None|float|tf.Tensor' = None) -> tf.Tensor:
    """Returns vector with shape `num_cols = [self.shape[1]]` that sums columns.

    Args:
      replace_if_0: If None, returns the actual sum, leaving zero-entries as-is.
        Otherwise, zero-entries will be replaced by this value.
    """
    y = tf.ones([self.shape[0]]) @ self  # 1^T M  [shape=[cols]]

    if replace_if_0 is not None:
      y = tf.where(tf.abs(y) < EPSILON, replace_if_0 * tf.ones_like(y), y)
    return y

  def normalize_left(self) -> 'Multiplier':
    """Returns a left-stochastic matrix."""
    return Product(self, DiagMatrix(1 / self.colsums(1.0)))

  def normalize_right(self) -> 'Multiplier':
    """Returns a right-stochastic matrix."""
    return Product(DiagMatrix(1 / self.rowsums(1.0)), self)

  def normalize_leftright(self) -> 'Multiplier':
    return Product(
        DiagMatrix(tf.math.rsqrt(self.rowsums(1.0))),
        self,
        DiagMatrix(tf.math.rsqrt(self.colsums(1.0))),
    )

  def normalize_symmetric(self) -> 'Multiplier':
    inv_sqrt_degree = DiagMatrix(tf.math.rsqrt(self.colsums(1.0)))
    return Product(inv_sqrt_degree, self, inv_sqrt_degree)


class Transpose(Multiplier):
  """Defines matrix transpose."""

  def __init__(self, multiplier: Multiplier):
    self._multiplier = multiplier

  def matmul(self, mat: tf.Tensor) -> tf.Tensor:
    # (M'X) == (X'M)'
    return tf.transpose(tf.transpose(mat) @ self._multiplier)

  def rmatmul(self, mat: tf.Tensor) -> tf.Tensor:
    # (XM') == (XM')'' == (M X')'
    return tf.transpose(self._multiplier @ tf.transpose(mat))

  @property
  def shape(self) -> tuple['int|tf.Tensor', 'int|tf.Tensor']:
    transpose_shape = self._multiplier.shape
    return (transpose_shape[1], transpose_shape[0])

  def transpose(self) -> Multiplier:
    return self._multiplier


class DiagMatrix(Multiplier):
  """Defines diagonal matrix."""

  def __init__(self, diag_vector: tf.Tensor):
    assert len(diag_vector.shape) == 1, 'Must be a vector.'
    self._diag_vector = diag_vector
    self._vec_shape = shape(diag_vector)[0]

  def matmul(self, mat: tf.Tensor) -> tf.Tensor:
    return tf.einsum('i,i...->i...', self._diag_vector, mat)

  def rmatmul(self, mat: tf.Tensor) -> tf.Tensor:
    return tf.einsum('i,...i->...i', self._diag_vector, mat)

  @property
  def shape(self) -> tuple['int|tf.Tensor', 'int|tf.Tensor']:
    return (self._vec_shape, self._vec_shape)


class Product(Multiplier):
  """Defines product of multipliers."""

  def __init__(self, *multipliers: Multiplier):
    assert multipliers
    for i in range(1, len(multipliers)):
      tf.assert_equal(multipliers[i - 1].shape[1], multipliers[i].shape[0])

    self._multipliers = multipliers

  def matmul(self, mat: tf.Tensor) -> tf.Tensor:
    for m in self._multipliers[::-1]:
      mat = m @ mat
    return mat

  def rmatmul(self, mat: tf.Tensor) -> tf.Tensor:
    for m in self._multipliers:
      mat = mat @ m
    return mat

  @property
  def shape(self) -> tuple['int|tf.Tensor', 'int|tf.Tensor']:
    return (self._multipliers[0].shape[0], self._multipliers[-1].shape[1])


class Sum(Multiplier):
  """Defines sum of multipliers."""

  def __init__(self, *multipliers: Multiplier):
    assert multipliers
    for i in range(1, len(multipliers)):
      tf.assert_equal(multipliers[i].shape[0], multipliers[0].shape[0])
      tf.assert_equal(multipliers[i].shape[1], multipliers[0].shape[1])
    self._multipliers = multipliers

  def matmul(self, mat: tf.Tensor) -> tf.Tensor:
    return tf.add_n([m @ mat for m in self._multipliers])

  def rmatmul(self, mat: tf.Tensor) -> tf.Tensor:
    return tf.add_n([mat @ m for m in self._multipliers])

  @property
  def shape(self) -> tuple['int|tf.Tensor', 'int|tf.Tensor']:
    return self._multipliers[0].shape


class AdjacencyMultiplier(Multiplier):
  r"""Multiplies (sparse) adjacency with dense matrices.

  Yields adjacency with (rows, cols) == (target, source).

  `adj_multiplier @ x` yields tensor `y` with `y[i]` being `\sum_{j->i} x[j]`.

  Init Args:
      graph:
      sender_tag: If `== tfgnn.SOURCE`, then the (implicit) adjacency will be
        of shape `size_target x size_source`. If `== tfgnn.TARGET`, then `shape`
        should be `size_source x size_target`.
  """

  def __init__(
      self, graph, edge_set_name: tfgnn.EdgeSetName,
      edge_weight_feature_name: 'None|tfgnn.FieldName' = None,
      sender_tag: tfgnn.IncidentNodeTag = tfgnn.SOURCE):
    tfgnn.check_scalar_graph_tensor(graph, 'AdjacencyMultiplier')
    self._sender_tag = sender_tag
    self._receiver_tag: tfgnn.IncidentNodeTag = 1 - sender_tag
    self._edge_set_name = edge_set_name
    self._graph = graph
    self._edge_weight_feature_name = edge_weight_feature_name

  @property
  def shape(self) -> tuple['int|tf.Tensor', 'int|tf.Tensor']:
    """Shape is (size of receiver node set, size of sender node set)."""
    adj = self._graph.edge_sets[self._edge_set_name].adjacency
    sender_node_set_name = adj.node_set_name(self._sender_tag)
    receiver_node_set_name = adj.node_set_name(self._receiver_tag)
    sender_sizes = self._graph.node_sets[sender_node_set_name].sizes
    receiver_sizes = self._graph.node_sets[receiver_node_set_name].sizes
    return (tf.cast(tf.reduce_sum(receiver_sizes), tf.int32),
            tf.cast(tf.reduce_sum(sender_sizes), tf.int32))

  def matmul(self, mat: tf.Tensor):
    edge_level = tfgnn.broadcast_node_to_edges(
        self._graph, self._edge_set_name, self._sender_tag, feature_value=mat)

    if self._edge_weight_feature_name:
      edge_set = self._graph.edge_sets[self._edge_set_name]
      edge_level *= edge_set[self._edge_weight_feature_name]

    return tfgnn.pool_edges_to_node(
        self._graph, self._edge_set_name, self._receiver_tag,
        feature_value=edge_level)

  def rmatmul(self, mat):
    edge_level = tfgnn.broadcast_node_to_edges(
        self._graph, self._edge_set_name, self._receiver_tag,
        feature_value=tf.transpose(mat))

    if self._edge_weight_feature_name:
      edge_set = self._graph.edge_sets[self._edge_set_name]
      edge_level *= edge_set[self._edge_weight_feature_name]

    return tf.transpose(tfgnn.pool_edges_to_node(
        self._graph, self._edge_set_name, self._sender_tag,
        feature_value=edge_level))


def shape(tensor: tf.Tensor) -> 'list[int]|tf.Tensor':
  """Helper function returns shape of eager or symbolic tensors."""
  if any([s is None for s in tensor.shape]):
    return tf.shape(tensor)
  else:
    return tensor.shape
