"""Library for running train-and-eval loop on tiles dataset."""

import functools
import gzip
import io
import json
import os
from typing import Callable

from absl import flags
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tpu_graphs.baselines.tiles import data
from tpu_graphs.baselines.tiles import metrics
from tpu_graphs.baselines.tiles import models
from tpu_graphs.baselines.tiles import train_args
import tqdm


_DATA_ROOT = flags.DEFINE_string(
    'data_root', '~/data/tpugraphs_tiles',
    'Root directory containing dataset. It must contain subdirectories '
    '{train, test, validation}, each having many .npz files')
_CACHE_DIR = flags.DEFINE_string(
    'cache_dir', '~/data/cache/tpugraphs_tiles',
    'If given, dataset tensors will be cached here for faster loading.')


def graph_and_label(graph: tfgnn.GraphTensor, batch_size=10, num_configs=2):
  label = tf.reshape(
      graph.node_sets['config']['runtimes'], [batch_size, num_configs])
  return graph, label


def train(args: train_args.TrainArgs):
  """Training loop. `train_args.py` contains description of arguments."""
  out_dir = os.path.expanduser(args.out_dir)
  if not tf.io.gfile.exists(out_dir):
    tf.io.gfile.makedirs(out_dir)

  # Will be written in out_dir.
  run_info = dict(
      train_curve=dict(epoch=[], train_loss=[], valid_error=[]),
      args=args._asdict(),
  )

  # Input training data.
  data_root_dir = os.path.expanduser(_DATA_ROOT.value)
  num_configs = args.configs
  dataset_partitions = data.get_npz_dataset(
      data_root_dir, min_train_configs=num_configs,
      cache_dir=os.path.expanduser(_CACHE_DIR.value))
  dataset_partitions.normalize()
  batch_size = args.batch_size
  train_ds = dataset_partitions.train.get_graph_tensors_dataset(num_configs)
  train_ds = train_ds.shuffle(5000, reshuffle_each_iteration=True)
  train_ds = train_ds.batch(batch_size, drop_remainder=True).map(
      tfgnn.GraphTensor.merge_batch_to_components)

  # Model.
  model_class = getattr(models, args.model)
  model_kwargs = json.loads(args.model_kwargs_json)
  num_ops = dataset_partitions.num_ops
  model = model_class(num_configs, num_ops, **model_kwargs)

  loss = metrics.CombinedLoss(metrics.parse_loss_str(args.losses))
  opt = tf.keras.optimizers.Adam(
      learning_rate=args.learning_rate, clipnorm=args.clip_norm)
  model.compile(loss=loss, optimizer=opt)
  train_ds = train_ds.map(
      functools.partial(
          graph_and_label, batch_size=batch_size, num_configs=num_configs))

  valid_ds = dataset_partitions.validation.get_graph_tensors_dataset()
  if args.validate_batches > 0:
    valid_ds = valid_ds.take(args.validate_batches)
  best_valid_error = 99999

  best_params = None
  for i in range(args.epochs):
    flags.FLAGS.alsologtostderr = True
    history = model.fit(train_ds, epochs=1, verbose=1)
    flags.FLAGS.alsologtostderr = False
    run_info['train_curve']['epoch'].append(i)
    run_info['train_curve']['train_loss'].append(history.history['loss'][-1])
    valid_error = metrics.top_error_performance(valid_ds, model.forward)

    if valid_error[1] < best_valid_error:
      best_valid_error = valid_error[1]
      best_params = {v.ref: v + 0 for v in model.trainable_variables}
      print(' ** Validation (NEW BEST): ', valid_error)
    else:
      print(' ** Validation: ', valid_error)

  # Restore best parameters.
  assert best_params is not None
  for v in model.trainable_variables:
    v.assign(best_params[v.ref])

  # Run on test set.
  test_ds = dataset_partitions.test.get_graph_tensors_dataset()
  if args.test_mode == 'metrics':
    test_error = metrics.top_error_performance(test_ds, model.forward)
    run_info['train_curve']['final_test_error'] = test_error
  elif args.test_mode == 'predictions':
    module_ids, predictions = get_predictions(test_ds, model.forward)
    run_info['test_predictions'] = {}
    module_ids = module_ids.numpy().tolist()
    predictions = predictions.numpy().tolist()
    for module_id, module_predictions in zip(module_ids, predictions):
      module_id = bytes(module_id).decode()
      run_info['test_predictions'][module_id] = module_predictions
  out_file = os.path.join(out_dir, f'run_{args.compute_hash()}.jsonz')

  bytes_io = io.BytesIO()
  with gzip.open(bytes_io, 'wb') as fout:
    fout.write(json.dumps(run_info).encode())
  with tf.io.gfile.GFile(out_file, 'wb') as fout:
    fout.write(bytes_io.getvalue())
  print('wrote ' + out_file)


def get_predictions(
    test_ds: tf.data.Dataset,
    model_fn: Callable[[tfgnn.GraphTensor, int], tf.Tensor]
    ) -> tuple[tf.Tensor, tf.Tensor]:
  """Module IDs and config indices that `model_fn` assigns lowest scores."""
  all_sorted_indices = []
  all_module_ids = []
  for graph in tqdm.tqdm(test_ds, desc='Generating Predictions'):
    num_configs = int(graph.node_sets['config'].sizes[0].numpy())
    preds = model_fn(graph, num_configs)
    preds = tf.squeeze(preds, 0)  # Remove batch size (of 1)
    sorted_indices = tf.argsort(preds)
    sorted_indices = tf.concat([  # zero-pad.
        sorted_indices, tf.zeros([10], dtype=sorted_indices.dtype)
    ], axis=0)
    sorted_indices = sorted_indices[:10]
    all_sorted_indices.append(sorted_indices)
    all_module_ids.append(graph.node_sets['g']['tile_id'][0])

  return tf.stack(all_module_ids, axis=0), tf.stack(all_sorted_indices, axis=0)
