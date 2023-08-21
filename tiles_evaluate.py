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

"""Evaluates model on all validation graphs, grouping metrics by benchmark."""

import collections
import gzip
import json
import os

from absl import app
from absl import flags
import tensorflow as tf
# So that keras.models.load_model() can re-instantiate layers of saved model.
import tensorflow_gnn as tfgnn  # pylint: disable=unused-import.
import tensorflow_ranking as tfr  # pylint: disable=unused-import.
from tpu_graphs.baselines.tiles import data
from tpu_graphs.baselines.tiles import metrics
from tpu_graphs.baselines.tiles import models
import tqdm
unused_modules = [tfr, tfgnn]

_MODEL_DIRS = flags.DEFINE_string(
    'dirs', None,
    'Comma-separated list of model directories to evaluate. '
    'The per-benchmark average will be printed', required=True)
_DATA_ROOT = flags.DEFINE_string(
    'data_root', '~/data/tpugraphs/npz/tile/xla',
    'Root directory containing dataset. It must contain subdirectories '
    '{train, test, validation}, each having many .npz files')
_CACHE_DIR = flags.DEFINE_string(
    'cache_dir', '~/data/tpugraphs/cache/tile/xla',
    'If given, dataset tensors will be cached here for faster loading.')


def main(unused_argv: list[str]) -> None:
  errors_by_benchmark = {
      1: collections.defaultdict(list), 5: collections.defaultdict(list)}
  dataset = data.get_npz_dataset(
      os.path.expanduser(_DATA_ROOT.value),
      cache_dir=os.path.expanduser(_CACHE_DIR.value))
  ds = dataset.validation.get_graph_tensors_dataset()

  for dirpath in tqdm.tqdm(_MODEL_DIRS.value.split(',')):
    # Load keras model.
    with tf.keras.saving.custom_object_scope(
        # Model was compiled with a loss before it was saved.
        # Override `load_model` in this scope to reconstruct loss object.
        {'CombinedLoss': metrics.CombinedLoss}):
      keras_model = tf.keras.models.load_model(dirpath)

    jsonz_file = dirpath.replace('/model_', '/run_') + '.jsonz'
    with gzip.open(open(jsonz_file, 'rb'), 'rb') as fin:
      json_data = json.loads(fin.read().decode())
      model_name = json_data['args']['model']
      model_kwargs = json.loads(json_data['args']['model_kwargs_json'])
    model_class = getattr(models, model_name)

    # Load pythonic model.
    model = model_class(
        num_configs=json_data['args']['configs'], num_ops=dataset.num_ops,
        **model_kwargs)

    # Instantiate `model`` parameters (to copy from `keras_model`).
    sample_graph, = ds.take(1)  # Example graph to invoke `model.forward`.
    num_configs = int(sample_graph.node_sets['config'].sizes[0])
    model.forward(sample_graph, num_configs)
    del sample_graph  # No longer need a toy example.

    target_vars = model.trainable_variables
    source_vars = keras_model.trainable_variables
    assert len(target_vars) == len(source_vars)
    for tv, sv in zip(target_vars, source_vars):
      assert sv.shape == tv.shape
      tv.assign(sv)

    for graph in tqdm.tqdm(ds):
      num_configs = int(graph.node_sets['config'].sizes[0])
      preds = model.forward(graph, num_configs)
      # Batch size 1: one inference graph at a time (with all configs).
      preds = tf.squeeze(preds, 0)
      runtimes = graph.node_sets['config']['runtimes']
      time_best = tf.reduce_min(runtimes)
      sorted_indices = tf.argsort(preds)
      benchmark = (graph.node_sets['g']['tile_id']
                   .numpy()[0].decode().rsplit('_', 1)[0])
      for k in [1, 5]:
        time_model_candidates = tf.gather(runtimes, sorted_indices[:k])
        best_of_candidates = tf.reduce_min(time_model_candidates)
        error = float((best_of_candidates - time_best) / time_best)
        errors_by_benchmark[k][benchmark].append(error)

  print(json.dumps(
      # Map inner keys to floats (equivalent to tf.nest.map_structure).
      # Maps {'0', '1'} -> {bert_64*, other benchmarks, .. } -> {errors of runs}
      # Replace each inner value list with mean and convert to float():
      {k1: {k2: float(tf.reduce_mean(v2).numpy()) for k2, v2 in v.items()}
       for k1, v in errors_by_benchmark.items()},
      indent=2, sort_keys=True))


if __name__ == '__main__':
  app.run(main)
