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

import functools
import gzip
import io
import json
import os
from typing import Callable, Any

from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow_gnn as tfgnn
import tensorflow_ranking as tfr
from tpu_graphs.baselines.tiles import data
from tpu_graphs.baselines.tiles import metrics
from tpu_graphs.baselines.tiles import models
from tpu_graphs.baselines.tiles import train_args
import tqdm


_DATA_ROOT = flags.DEFINE_string(
    'data_root', '~/data/tpugraphs/npz/tile/xla',
    'Root directory containing dataset. It must contain subdirectories '
    '{train, test, validation}, each having many .npz files')
_CACHE_DIR = flags.DEFINE_string(
    'cache_dir', '~/data/tpugraphs/cache/tile',
    'If given, dataset tensors will be cached here for faster loading.')


def _graph_and_label(graph: tfgnn.GraphTensor, batch_size=10, num_configs=2):
  label = tf.reshape(
      graph.node_sets['config']['runtimes'], [batch_size, num_configs])
  return graph, label


# Used for validation. For training, data.py accepts `min_train_configs`.
def _graph_has_enough_configs(graph: tfgnn.GraphTensor, num_configs=2):
  """To used to filter validation dataset."""
  return graph.node_sets['config'].sizes[0] >= num_configs


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


def train(args: train_args.TrainArgs):
  """Training loop. `train_args.py` contains description of arguments."""
  out_dir = os.path.expanduser(args.out_dir)
  if not tf.io.gfile.exists(out_dir):
    tf.io.gfile.makedirs(out_dir)

  # Will be written in out_dir.
  run_info = dict(
      train_curve=dict(
          epoch=[], train_loss=[], train_opa=[], val_loss=[], val_opa=[]),
      final_error=dict(),
      args=args._asdict(),
  )

  # Input training data.
  data_root_dir = os.path.expanduser(_DATA_ROOT.value)
  num_configs = args.configs
  dataset_partitions = data.get_npz_dataset(
      data_root_dir, min_train_configs=num_configs,
      cache_dir=os.path.expanduser(_CACHE_DIR.value))
  batch_size = args.batch_size
  train_ds = (
      dataset_partitions.train.get_graph_tensors_dataset(num_configs)
      .shuffle(5000, reshuffle_each_iteration=True)
      .batch(batch_size, drop_remainder=True)
      .map(tfgnn.GraphTensor.merge_batch_to_components))

  # Model.
  model_class = getattr(models, args.model)
  model_kwargs = json.loads(args.model_kwargs_json)
  num_ops = dataset_partitions.num_ops
  model = model_class(num_configs, num_ops, **model_kwargs)

  loss = metrics.CombinedLoss(metrics.parse_loss_str(args.losses))
  opt = tf.keras.optimizers.Adam(
      learning_rate=args.learning_rate, clipnorm=args.clip_norm)

  model.compile(loss=loss, optimizer=opt, metrics=[
      tfr.keras.metrics.OPAMetric(name='opa_metric'),
  ])
  attach_labels_fn = functools.partial(
      _graph_and_label, batch_size=batch_size, num_configs=num_configs)
  train_ds = train_ds.map(attach_labels_fn)

  valid_ds = (
      dataset_partitions.validation.get_graph_tensors_dataset(num_configs)
      # Get an extra 5% as we follow with `filter()`.
      .take(int(args.validate_batches * batch_size * 1.05))
      .filter(
          functools.partial(_graph_has_enough_configs, num_configs=num_configs))
      .batch(batch_size, drop_remainder=True)
      .map(tfgnn.GraphTensor.merge_batch_to_components)
      .map(attach_labels_fn))

  best_params = None
  best_val_opa = -1
  best_val_at_epoch = -1
  train_curve = run_info['train_curve']  # For short.
  for i in range(args.epochs):
    old_alsologtostderr = flags.FLAGS.alsologtostderr
    flags.FLAGS.alsologtostderr = True
    history = model.fit(
        train_ds, epochs=1, verbose=1, validation_data=valid_ds)
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

  # Run on full validation.
  run_info['final_error']['val'] = metrics.top_error_performance(
      dataset_partitions.validation.get_graph_tensors_dataset(), model.forward)

  # Run on test set.
  test_ds = dataset_partitions.test.get_graph_tensors_dataset()
  if args.test_mode == 'metrics':
    run_info['final_error']['test'] = metrics.top_error_performance(
        test_ds, model.forward)
  elif args.test_mode == 'predictions':
    module_ids, ranks = rank_config_indices(test_ds, model.forward)

    write_least_runtimes_csv(args.results_csv, module_ids, ranks)

    ### Add test predictions into run_info file.
    run_info['test_predictions'] = {}
    module_ids = module_ids.numpy().tolist()
    predictions = ranks.numpy().tolist()
    for module_id, module_predictions in zip(module_ids, predictions):
      module_id = bytes(module_id).decode()
      run_info['test_predictions'][module_id] = module_predictions

  save_model(model, run_info, out_dir, args)


def rank_config_indices(
    test_ds: tf.data.Dataset,
    model_fn: Callable[[tfgnn.GraphTensor, int], tf.Tensor],
    top_ranks=10
    ) -> tuple[tf.Tensor, tf.Tensor]:
  """Module IDs and config indices that `model_fn` assigns lowest scores.

  Args:
    test_ds: Test dataset containing `GraphTensor` instances. Each instance must
      have node sets `'config'` and `'g'` (with feature 'tile_id')
    model_fn: Callable (e.g., tf.Keras model) that will be invoked on every item
      in `test_ds` and the number of configurations (=N). It is expeted to
      return tensor of shape (1, N). The least indices will be output.
    top_ranks: Only this many least indices will be kept.

  Returns:
    Two `tf.Tensor` instances. The first is a vector with entry `[i]` being the
    `graph.node_sets['g']['tile_id']` of the `i`th element of `test_ds`. The
    second is a matrix with width `top_ranks`, where row `[i]` being the least
    `top_ranks` indices when invoking `model_fn` on `graph`.
  """
  all_sorted_indices = []
  all_module_ids = []
  for graph in tqdm.tqdm(test_ds, desc='Generating Predictions'):
    num_configs = int(graph.node_sets['config'].sizes[0].numpy())
    preds = model_fn(graph, num_configs)
    preds = tf.squeeze(preds, 0)  # Remove batch size (of 1)
    sorted_indices = tf.argsort(preds)
    sorted_indices = tf.concat([  # zero-pad.
        sorted_indices, tf.zeros([top_ranks], dtype=sorted_indices.dtype)
    ], axis=0)
    sorted_indices = sorted_indices[:top_ranks]
    all_sorted_indices.append(sorted_indices)
    all_module_ids.append(graph.node_sets['g']['tile_id'][0])

  return tf.stack(all_module_ids, axis=0), tf.stack(all_sorted_indices, axis=0)


def write_least_runtimes_csv(
    out_csv_filepath: str, module_ids: tf.Tensor, ranks: tf.Tensor):
  """Writes CSV file with line `i` containing module_ids[i] and ranks[i]."""
  csv_ranks = tf.strings.join(
      tf.strings.as_string(tf.transpose(ranks)), ';')

  stack_join = lambda x, delim: tf.strings.join(tf.stack(x), delim)
  with tf.io.gfile.GFile(out_csv_filepath, 'w') as fout:
    fout.write('ID,TopConfigs\n')
    id_vector = stack_join(
        [tf.fill(module_ids.shape, 'tile:xla'), module_ids], ':')
    csv_lines = stack_join([id_vector, csv_ranks], ',')
    fout.write(stack_join(csv_lines, '\n').numpy().decode('utf-8'))
  print('\n\n   ***  Wrote', out_csv_filepath, '\n\n')
