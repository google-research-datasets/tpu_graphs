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

"""Computes evals if the selection was uniformly at random."""

import collections
import os
from typing import Any

from absl import app
from absl import flags
import tensorflow as tf
from tpu_graphs.baselines.tiles import data
from tpu_graphs.baselines.tiles import metrics
import tqdm

_DATA_ROOT = flags.DEFINE_string(
    'data_root', '~/data/tpugraphs_tiles',
    'Root directory containing dataset. It must contain subdirectories '
    '{train, test, validation}, each having many .npz files')
_CACHE_DIR = flags.DEFINE_string(
    'cache_dir', '~/data/cache/tpugraphs_tiles',
    'If given, dataset tensors will be cached here for faster loading.')
_REPEATS = flags.DEFINE_integer(
    'repeats', 10, 'Number of times to repeat experiment.')


def random_prediction(example: Any, num_modules: int) -> tf.Tensor:
  del example
  return tf.random.uniform(shape=[1, num_modules], minval=0.0, maxval=100.0)


def main(unused_argv) -> None:
  splits = ('test', 'valid')
  for split_name in splits:
    data_root_dir = os.path.join(
        os.path.expanduser(_DATA_ROOT.value), split_name)
    split = data.get_npz_split(
        data_root_dir, cache_dir=os.path.expanduser(_CACHE_DIR.value))
    test_ds = split.get_graph_tensors_dataset()
    stats = collections.defaultdict(list)
    for unused_i in tqdm.tqdm(range(_REPEATS.value)):
      errors = metrics.top_error_performance(test_ds, random_prediction)
      for key, error in errors.items():
        stats[key].append(error)
    for k in stats:
      print('Split=%s; TopK=%s; Err=%g' % (
          split_name, k, sum(stats[k])/len(stats)))


if __name__ == '__main__':
  app.run(main)
