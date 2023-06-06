"""Computes evals if the selection was uniformly at random."""

import os
from typing import Any

from absl import app
from absl import flags
import tensorflow as tf
from tpu_graphs.baselines.tiles import data
from tpu_graphs.baselines.tiles import metrics

_DATA_ROOT = flags.DEFINE_string(
    'data_root', '~/data/tpugraphs_tiles',
    'Root directory containing dataset. It must contain subdirectories '
    '{train, test, validation}, each having many .npz files')
_CACHE_DIR = flags.DEFINE_string(
    'cache_dir', '~/data/cache/tpugraphs_tiles',
    'If given, dataset tensors will be cached here for faster loading.')


def random_prediction(example: Any, num_modules: int) -> tf.Tensor:
  del example
  return tf.random.uniform(shape=[1, num_modules], minval=0.0, maxval=100.0)


def main(unused_argv) -> None:
  data_root_dir = os.path.join(os.path.expanduser(_DATA_ROOT.value), 'test')
  test_split = data.get_npz_split(
      data_root_dir, cache_dir=os.path.expanduser(_CACHE_DIR.value))
  test_ds = test_split.get_graph_tensors_dataset()
  errors = metrics.top_error_performance(test_ds, random_prediction)
  print('; '.join([f'err{k}={e}' for k, e in errors.items()]))


if __name__ == '__main__':
  app.run(main)
