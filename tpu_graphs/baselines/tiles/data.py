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

"""Functions to read numpy data files to `tf.data.Dataset`.

The high-level function is `get_npz_dataset`, which can be called as:

```
dataset_partitions = get_npz_dataset('path/to/tpu_graphs/tiles')
# Then access: dataset_partitions.{train, vaildation, test}

```
"""

import collections
import functools
import hashlib
import io
import os
from typing import NamedTuple

from absl import flags
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
import tqdm


_TOY_DATA = flags.DEFINE_bool(
    'toy_data', False,
    'If set, then only 100 examples will be used in each of '
    '{train, test, validation} partitions.')


class TileExample(NamedTuple):
  """Single example of tile graph."""
  node_features: tf.Tensor
  node_ops: tf.Tensor
  edges: tf.Tensor
  config_features: tf.Tensor
  config_runtimes: tf.Tensor
  config_runtime_normalizers: tf.Tensor
  tile_id: tf.Tensor
  total_nodes: tf.Tensor
  total_edges: tf.Tensor
  total_configs: tf.Tensor

  def to_graph_tensor(
      self, config_samples: int = -1,
      normalize_runtimes: bool = True) -> tfgnn.GraphTensor:
    """Packages instance tensors (edges, features) into `GraphTensor`.

    Args:
      config_samples: if -1, then all module configurations (and their runtimes)
        are returned. If >=0, then this many module configurations (and their
        corresponding runtimes) are sampled uniformly at random.
      normalize_runtimes: If set (default), runtimes will be normalized by
        dividing over the runtime of "default tile size" (to account for worker
        machine differences).

    Returns:
      GraphTensor with node-sets:
        + op (feats='op': int-categorical, 'feats': float-vector).
          This is the only "real" graph node"
        + configs (feats='feats': float-vector, 'runtimes': float scalar,
                   'normalizers': float scalar).
          These are "fake" nodes. There will be one node per configuration.
        + g (stands for "graph") has one (root) node connecting to all op and
          config nodes.
      and edge-sets:
        + 'feed': directed edges connecting op-node to op-node.
        + 'g_op': edges connecting the singleton "g" node to every "op" node.
        + 'g_config': connecting the singleton "g" node to every "config" node.
    """
    config_features = self.config_features
    config_runtimes = self.config_runtimes
    config_runtime_normalizers = self.config_runtime_normalizers
    num_configs = tf.shape(config_features)[0]

    # If sampling is requested.
    if config_samples >= 0:
      rnd = tf.random.shuffle(tf.range(num_configs, dtype=tf.int32))
      rnd = rnd[:config_samples]
      config_features = tf.gather(config_features, rnd)
      config_runtimes = tf.gather(config_runtimes, rnd)
      config_runtime_normalizers = tf.gather(config_runtime_normalizers, rnd)
      num_configs = tf.shape(config_features)[0]

    if normalize_runtimes:
      config_runtimes /= config_runtime_normalizers

    return tfgnn.GraphTensor.from_pieces(
        node_sets={
            'op': tfgnn.NodeSet.from_fields(
                sizes=tf.expand_dims(self.total_nodes, 0),
                features={
                    'op': self.node_ops,
                    'feats': self.node_features,
                }
            ),
            'config': tfgnn.NodeSet.from_fields(
                features={
                    'feats': config_features,
                    'runtimes': config_runtimes,
                    'normalizers': config_runtime_normalizers,
                },
                sizes=tf.expand_dims(num_configs, 0),
            ),
            'g': tfgnn.NodeSet.from_fields(
                features={'tile_id': tf.expand_dims(self.tile_id, 0)},
                sizes=tf.constant([1]))
        },
        edge_sets={
            'feed': tfgnn.EdgeSet.from_fields(
                sizes=tf.expand_dims(self.total_edges, 0),
                adjacency=tfgnn.Adjacency.from_indices(
                    source=('op', self.edges[:, 0]),
                    target=('op', self.edges[:, 1]))),
            'g_op': tfgnn.EdgeSet.from_fields(
                sizes=tf.expand_dims(self.total_nodes, 0),
                adjacency=tfgnn.Adjacency.from_indices(
                    source=('g', tf.zeros([self.total_nodes], dtype=tf.int32)),
                    target=('op', tf.range(self.total_nodes, dtype=tf.int32)))),
            'g_config': tfgnn.EdgeSet.from_fields(
                sizes=tf.expand_dims(num_configs, 0),
                adjacency=tfgnn.Adjacency.from_indices(
                    source=('g', tf.zeros([num_configs], dtype=tf.int32)),
                    target=('config', tf.range(num_configs, dtype=tf.int32)))),
        })


class NpzDatasetPartition:
  """Holds one data partition (train, test, validation) on device memory."""

  def __init__(self):
    # Populated in `add_npz_file()`.
    self._data_dict: dict[str, list[np.ndarray]] = collections.defaultdict(list)
    self._num_edges: list[int] = [0]    # prepend with 0 to prep for cumsum.
    self._num_configs: list[int] = [0]  # ^^
    self._num_nodes: list[int] = [0]    # ^^

    # Populated in `finalize()`.
    self.node_feat: 'tf.Tensor | None' = None   # indexed by node_ranges.
    self.node_opcode: 'tf.Tensor | None' = None  # ^^
    self.edge_index: 'tf.Tensor | None' = None   # indexed by edge_ranges.
    self.config_feat: 'tf.Tensor | None' = None      # indexed by config_ranges.
    self.config_runtime: 'tf.Tensor | None' = None   # ^^
    self.config_runtime_normalizers: 'tf.Tensor | None' = None  # ^^
    self.tile_id: 'tf.Tensor | None' = None

    # finalize() sets to: cumsum([0, numEdges(graph_1), numEdges(graph_2), ..]).
    self.edge_ranges: 'tf.Tensor | None' = None
    # finalize() sets to: cumsum([0, numNodes(graph_1), numNodes(graph_2), ..]).
    self.node_ranges: 'tf.Tensor | None' = None
    # finalize() sets to: cumsum([0, numModules(graph_1), nModul(graph_2), ..]).
    self.config_ranges: 'tf.Tensor | None' = None

  def save_to_file(self, cache_file: str):
    """Saves dataset as numpy. Can be restored with `load_from_file`."""
    assert self.node_feat is not None, 'finalize() was not invoked'
    assert self.node_opcode is not None
    assert self.edge_index is not None
    assert self.config_feat is not None
    assert self.config_runtime is not None
    assert self.config_runtime_normalizers is not None
    assert self.tile_id is not None
    assert self.edge_ranges is not None
    assert self.node_ranges is not None
    assert self.config_ranges is not None

    np_dict = dict(
        node_feat=self.node_feat.numpy(),
        node_opcode=self.node_opcode.numpy(),
        edge_index=self.edge_index.numpy(),
        config_feat=self.config_feat.numpy(),
        config_runtime=self.config_runtime.numpy(),
        config_runtime_normalizers=self.config_runtime_normalizers.numpy(),
        edge_ranges=self.edge_ranges.numpy(),
        node_ranges=self.node_ranges.numpy(),
        config_ranges=self.config_ranges.numpy()
    )
    bytes_io = io.BytesIO()
    np.savez_compressed(bytes_io, **np_dict)
    with tf.io.gfile.GFile(cache_file, 'wb') as fout:
      fout.write(bytes_io.getvalue())
    print('wrote ' + cache_file)
    tile_ids_file = cache_file + '.tiles.txt'
    with tf.io.gfile.GFile(tile_ids_file, 'w') as fout:
      fout.write(b'\n'.join(self.tile_id.numpy().tolist()).decode())
    print('wrote ' + tile_ids_file)

  def load_from_file(self, cache_file: str):
    """Loads dataset from numpy file."""
    np_dict = np.load(tf.io.gfile.GFile(cache_file, 'rb'))
    self.node_feat = tf.constant(np_dict['node_feat'])
    self.node_opcode = tf.constant(np_dict['node_opcode'])
    self.edge_index = tf.constant(np_dict['edge_index'])
    self.config_feat = tf.constant(np_dict['config_feat'])
    self.config_runtime = tf.constant(np_dict['config_runtime'])
    self.config_runtime_normalizers = tf.constant(
        np_dict['config_runtime_normalizers'])
    self.edge_ranges = tf.constant(np_dict['edge_ranges'])
    self.node_ranges = tf.constant(np_dict['node_ranges'])
    self.config_ranges = tf.constant(np_dict['config_ranges'])
    tile_ids = tf.io.gfile.GFile(cache_file + '.tiles.txt', 'r').readlines()
    self.tile_id = tf.stack([tile_id.rstrip() for tile_id in tile_ids])
    print('loaded from ' + cache_file)

  def add_npz_file(
      self, tile_id: str, npz_file: np.lib.npyio.NpzFile, min_configs: int = 2):
    """Copies data from npz file into this class instance.

    After finishing all calls `add_npz_file()`, user must invoke `finalize()`.

    Args:
      tile_id: the filename (without extension) that npz_file was read from.
      npz_file: Output of np.load on a file from the TpuGraphs Tiles dataset.
      min_configs: The file be incorporated only if the number of module
        configurations is equal or greater than this.
    """
    npz_data = dict(npz_file.items())
    num_configs = npz_data['config_feat'].shape[0]
    if num_configs < min_configs:
      print('skipping tile with only %i configurations' % num_configs)
      return
    for key, ndarray in npz_data.items():
      self._data_dict[key].append(ndarray)
    self._data_dict['tile_id'].append(np.array(tile_id))
    num_nodes = npz_data['node_feat'].shape[0]
    num_edges = npz_data['edge_index'].shape[0]
    assert num_nodes == npz_data['node_opcode'].shape[0]
    assert num_configs == npz_data['config_runtime'].shape[0]
    assert num_configs == npz_data['config_runtime_normalizers'].shape[0]
    self._num_nodes.append(num_nodes)
    self._num_edges.append(num_edges)
    self._num_configs.append(num_configs)

  def finalize(self):
    self.tile_id = tf.stack(self._data_dict['tile_id'], axis=0)
    self.node_feat = tf.concat(self._data_dict['node_feat'], axis=0)
    self.node_opcode = tf.concat(self._data_dict['node_opcode'], axis=0)
    self.edge_index = tf.concat(self._data_dict['edge_index'], axis=0)
    self.config_feat = tf.concat(self._data_dict['config_feat'], axis=0)
    self.config_runtime = tf.concat(self._data_dict['config_runtime'], axis=0)
    self.config_runtime_normalizers = tf.concat(
        self._data_dict['config_runtime_normalizers'], axis=0)
    self.edge_ranges = tf.cumsum(self._num_edges)
    self.node_ranges = tf.cumsum(self._num_nodes)
    self.config_ranges = tf.cumsum(self._num_configs)

  def get_item(self, index: int) -> TileExample:
    node_start = self.node_ranges[index]
    node_end = self.node_ranges[index + 1]
    edge_start = self.edge_ranges[index]
    edge_end = self.edge_ranges[index + 1]
    config_start = self.config_ranges[index]
    config_end = self.config_ranges[index + 1]

    return TileExample(
        node_features=self.node_feat[node_start:node_end],
        node_ops=self.node_opcode[node_start:node_end],
        edges=self.edge_index[edge_start:edge_end],
        config_features=self.config_feat[config_start:config_end],
        config_runtimes=self.config_runtime[config_start:config_end],
        config_runtime_normalizers=(
            self.config_runtime_normalizers[config_start:config_end]),
        tile_id=self.tile_id[index],
        total_nodes=node_end - node_start,
        total_edges=edge_end - edge_start,
        total_configs=config_end - config_start)

  def get_graph_tensors_dataset(
      self, config_samples: int = -1) -> tf.data.Dataset:
    if self.edge_ranges is None:
      raise ValueError('finalize() was not invoked.')
    dataset = tf.data.Dataset.range(self.edge_ranges.shape[0] - 1)
    dataset = dataset.map(self.get_item, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(
        functools.partial(TileExample.to_graph_tensor,
                          config_samples=config_samples))
    return dataset


def get_npz_split(
    split_path: str, min_configs=2, cache_dir=None) -> NpzDatasetPartition:
  """Returns data for a single partition."""
  glob_pattern = os.path.join(split_path, '*.npz')
  files = tf.io.gfile.glob(glob_pattern)
  if not files:
    raise ValueError('No files matched: ' + glob_pattern)
  if _TOY_DATA.value:
    files = files[:100]

  cache_filename = None
  if cache_dir:
    if not tf.io.gfile.exists(cache_dir):
      tf.io.gfile.makedirs(cache_dir)
    filename_hash = hashlib.md5(
        f'{split_path}:{min_configs}:{_TOY_DATA.value}'.encode()).hexdigest()
    cache_filename = os.path.join(cache_dir, f'{filename_hash}-cache.npz')
    print('dataset cache file: ', cache_filename)

  npz_dataset = NpzDatasetPartition()
  if cache_filename and tf.io.gfile.exists(cache_filename):
    npz_dataset.load_from_file(cache_filename)
  else:
    for filename in tqdm.tqdm(files):
      np_data = np.load(tf.io.gfile.GFile(filename, 'rb'))
      tile_id = os.path.splitext(os.path.basename(filename))[0]
      npz_dataset.add_npz_file(tile_id, np_data, min_configs=min_configs)
    npz_dataset.finalize()
    if cache_filename:
      npz_dataset.save_to_file(cache_filename)

  return npz_dataset


class NpzDataset(NamedTuple):
  """Contains all partitions of the dataset."""
  train: NpzDatasetPartition
  validation: NpzDatasetPartition
  test: NpzDatasetPartition

  @property
  def num_ops(self):
    return int(
        tf.reduce_max([
            tf.reduce_max(self.train.node_opcode),
            tf.reduce_max(self.validation.node_opcode),
            tf.reduce_max(self.test.node_opcode),
        ]).numpy()) + 1

  def _get_normalizer(self, feature_matrix) -> tuple[
      tf.Tensor, tf.Tensor, tf.Tensor]:
    max_feat = tf.reduce_max(feature_matrix, axis=0, keepdims=True)
    min_feat = tf.reduce_min(feature_matrix, axis=0, keepdims=True)
    return min_feat[0] != max_feat[0], min_feat, max_feat

  def _apply_normalizer(self, feature_matrix, used_columns, min_feat, max_feat):
    feature_matrix = tf.boolean_mask(feature_matrix, used_columns, axis=1)
    min_feat = tf.boolean_mask(min_feat, used_columns, axis=1)
    max_feat = tf.boolean_mask(max_feat, used_columns, axis=1)
    return (feature_matrix - min_feat) / (max_feat - min_feat)

  def normalize(self):
    """Removes constant features and normalizes remaining onto [0, 1].

    The statistics are computed only from train partition then applied to all
    partitions {train, test, validation}.
    """
    normalizer_args = self._get_normalizer(self.train.node_feat)
    self.train.node_feat = self._apply_normalizer(
        self.train.node_feat, *normalizer_args)
    self.validation.node_feat = self._apply_normalizer(
        self.validation.node_feat, *normalizer_args)
    self.test.node_feat = self._apply_normalizer(
        self.test.node_feat, *normalizer_args)

    normalizer_args = self._get_normalizer(self.train.config_feat)
    self.train.config_feat = self._apply_normalizer(
        self.train.config_feat, *normalizer_args)
    self.validation.config_feat = self._apply_normalizer(
        self.validation.config_feat, *normalizer_args)
    self.test.config_feat = self._apply_normalizer(
        self.test.config_feat, *normalizer_args)


def get_npz_dataset(
    root_path: str, min_train_configs=-1,
    cache_dir: 'None | str' = None) -> NpzDataset:
  """Returns {train, test, validation} partitions of tiles dataset collection.

  All partitions will be normalized: statistics are computed from training set
  partition and applied to all partitions.

  Args:
    root_path: Path where dataset lives. It must have subdirectories 'train',
      'test' and 'valid'.
    min_train_configs: If > 0, then tile examples will be filtered to have at
      least this many configurations (features and runtimes).
    cache_dir: If given, the many files for each of {train, test, validation}
      will be stored as one file (makes loading faster, for future runs).
  """
  npz_dataset = NpzDataset(
      train=get_npz_split(
          os.path.join(root_path, 'train'), min_configs=min_train_configs,
          cache_dir=cache_dir),
      validation=get_npz_split(
          os.path.join(root_path, 'valid'), cache_dir=cache_dir),
      test=get_npz_split(
          os.path.join(root_path, 'test'), cache_dir=cache_dir))
  npz_dataset.normalize()
  return npz_dataset
