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

"""Library for running train-and-eval loop on tiles dataset."""

import gzip
import io
import json
import os
import pdb
from typing import Any

from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow_gnn as tfgnn
import tensorflow_ranking as tfr
from tpu_graphs.baselines.layout import data
from tpu_graphs.baselines.layout import models
from tpu_graphs.baselines.layout import train_args
import tqdm


_DATA_ROOT = flags.DEFINE_string(
    'data_root', '~/data/tpugraphs/npz/layout',
    'Root directory containing dataset. It must contain subdirectories '
    '{train, test, valid}, each having many .npz files')
_CACHE_DIR = flags.DEFINE_string(
    'cache_dir', '~/data/tpugraphs/cache/layout',
    'If given, dataset tensors will be cached here for faster loading. Files '
    'with name "<hash>.npz" will be written, where <hash> is a hash of the '
    'filepattern of training data, i.e., it depends on the collection e.g., '
    '{xla:default} and partition {train, test, valid}.')
_PDB = flags.DEFINE_integer(
    'debug', -1, 'If >0, pdb debugger will be entered after this many epochs.')


def _graph_and_label(graph: tfgnn.GraphTensor):
  # Return runtimes divded over large number: only ranking is required. The
  # runtimes are in the 100K range
  label = tf.cast(graph.node_sets['g']['runtimes'], tf.float32) / 1e7
  return graph, label


def save_model(
    model: tf.keras.Model, run_info: dict[str, Any], out_dir: str,
    args: train_args.TrainArgs):
  """Writes `model` and `run_info` onto `out_dir`/*`args.compute_hash()`*."""
  args_hash = args.compute_hash()

  # Save run file.
  out_run_file = os.path.join(out_dir, f'run_{args_hash}.jsonz')
  bytes_io = io.BytesIO()
  with gzip.open(bytes_io, 'wb') as fout:
    fout.write(json.dumps(run_info).encode())
  with tf.io.gfile.GFile(out_run_file, 'wb') as fout:
    fout.write(bytes_io.getvalue())
  logging.info('wrote %s', out_run_file)

  # Keras model.
  out_model_file = os.path.join(out_dir, f'model_{args_hash}')
  model.save(out_model_file)
  logging.info('wrote %s', out_model_file)


_INFERENCE_CONFIGS_BATCH_SIZE = 500  # For producing inference csv, post-train.


def train(args: train_args.TrainArgs):
  """Training loop. `train_args.py` contains description of arguments."""
  out_dir = os.path.expanduser(args.out_dir)
  if not tf.io.gfile.exists(out_dir):
    tf.io.gfile.makedirs(out_dir)

  # Will be written in out_dir.
  run_info = dict(
      train_curve=dict(
          epoch=[], train_loss=[], train_opa=[], val_loss=[], val_opa=[]),
      final_opa=dict(),
      args=args._asdict(),
  )

  # Input training data.
  data_root_dir = os.path.join(
      os.path.expanduser(_DATA_ROOT.value), args.source, args.search)
  num_configs = args.configs
  dataset_partitions = data.get_npz_dataset(
      data_root_dir, min_train_configs=num_configs,
      max_train_configs=args.max_configs,
      cache_dir=os.path.expanduser(_CACHE_DIR.value))
  batch_size = args.batch_size

  train_ds = (
      dataset_partitions.train.get_graph_tensors_dataset(
          num_configs, max_nodes=args.keep_nodes)
      .shuffle(100, reshuffle_each_iteration=True)
      .batch(batch_size, drop_remainder=False)
      .map(tfgnn.GraphTensor.merge_batch_to_components)
      .map(_graph_and_label))

  model = models.ResModel(num_configs, dataset_partitions.num_ops)

  loss = tfr.keras.losses.ListMLELoss()  # (temperature=10)
  opt = tf.keras.optimizers.Adam(
      learning_rate=args.learning_rate, clipnorm=args.clip_norm)

  model.compile(loss=loss, optimizer=opt, metrics=[
      tfr.keras.metrics.OPAMetric(name='opa_metric'),
  ])

  valid_ds = (
      dataset_partitions.validation.get_graph_tensors_dataset(
          num_configs)
      .batch(batch_size, drop_remainder=False)
      .map(tfgnn.GraphTensor.merge_batch_to_components)
      .map(_graph_and_label))

  best_params = None
  best_val_opa = -1
  best_val_at_epoch = -1
  train_curve = run_info['train_curve']  # For short.
  for i in range(args.epochs):
    old_alsologtostderr = flags.FLAGS.alsologtostderr
    flags.FLAGS.alsologtostderr = True
    history = model.fit(
        train_ds, epochs=1, verbose=1, validation_data=valid_ds,
        validation_freq=1)
    if _PDB.value == i:
      pdb.set_trace()  # pylint: disable=forgotten-debug-statement

    flags.FLAGS.alsologtostderr = old_alsologtostderr
    train_curve['epoch'].append(i)
    train_curve['train_loss'].append(history.history['loss'][-1])
    train_curve['train_opa'].append(history.history['opa_metric'][-1])
    train_curve['val_loss'].append(history.history['val_loss'][-1])
    train_curve['val_opa'].append(history.history['val_opa_metric'][-1])
    val_opa = history.history['val_opa_metric'][-1]
    if val_opa > best_val_opa:
      best_val_opa = val_opa
      best_val_at_epoch = i
      best_params = {v.ref: v + 0 for v in model.trainable_variables}
      logging.info(' * [@%i] Validation (NEW BEST): %s', i, str(val_opa))
      # Write model and train metrics (in `run_info`).
      save_model(model, run_info, out_dir, args)
    elif args.early_stop > 0 and i - best_val_at_epoch >= args.early_stop:
      logging.info('[@%i] Best accuracy was attained at epoch %i. Stopping.',
                   i, best_val_at_epoch)
      break

  # Restore best parameters.
  assert best_params is not None
  for v in model.trainable_variables:
    v.assign(best_params[v.ref])

  print('\n\n   Running inference on test set ...\n\n')
  test_rankings = []

  assert dataset_partitions.test.graph_id is not None
  for graph in tqdm.tqdm(dataset_partitions.test.iter_graph_tensors(),
                         total=dataset_partitions.test.graph_id.shape[-1],
                         desc='Inference'):
    num_configs = graph.node_sets['g']['runtimes'].shape[-1]
    all_scores = []
    for i in range(0, num_configs, _INFERENCE_CONFIGS_BATCH_SIZE):
      end_i = min(i + _INFERENCE_CONFIGS_BATCH_SIZE, num_configs)
      # Take a cut of the configs.
      node_set_g = graph.node_sets['g']
      subconfigs_graph = tfgnn.GraphTensor.from_pieces(
          edge_sets=graph.edge_sets,
          node_sets={
              'op': graph.node_sets['op'],
              'nconfig': tfgnn.NodeSet.from_fields(
                  sizes=graph.node_sets['nconfig'].sizes,
                  features={
                      'feats': graph.node_sets['nconfig']['feats'][:, i:end_i],
                  }),
              'g': tfgnn.NodeSet.from_fields(
                  sizes=tf.constant([1]),
                  features={
                      'graph_id': node_set_g['graph_id'],
                      'runtimes': node_set_g['runtimes'][:, i:end_i],
                      'kept_node_ratio': node_set_g['kept_node_ratio'],
                  })
          })
      h = model.forward(subconfigs_graph, num_configs=(end_i - i),
                        backprop=False)
      all_scores.append(h[0])
    all_scores = tf.concat(all_scores, axis=0)
    graph_id = graph.node_sets['g']['graph_id'][0].numpy().decode()
    sorted_indices = tf.strings.join(
        tf.strings.as_string(tf.argsort(all_scores)), ';').numpy().decode()
    test_rankings.append((graph_id, sorted_indices))

  with tf.io.gfile.GFile(args.results_csv, 'w') as fout:
    fout.write('ID,TopConfigs\n')
    for graph_id, ranks in test_rankings:
      fout.write(f'layout:{args.source}:{args.search}:{graph_id},{ranks}\n')
  print('\n\n   ***  Wrote', args.results_csv, '\n\n')

