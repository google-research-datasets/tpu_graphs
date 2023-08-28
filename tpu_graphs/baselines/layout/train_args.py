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

"""Defines flags and corresponding `TrainArgs` used by training loop."""

import hashlib
import json
import os
import time
from typing import NamedTuple

from absl import flags

_EPOCHS = flags.DEFINE_integer('epochs', 500, 'number of train epochs.')
_EARLY_STOP = flags.DEFINE_integer(
    'early_stop', 40,
    'If held-out validation does not improve after this many epochs, then '
    'training will stop.')
_LEARNING_RATE = flags.DEFINE_float(
    'lr', 1e-3, 'Learning rate for Adam optimizer.')
_CLIP_NORM = flags.DEFINE_float(
    'clip_norm', 1e-2, 'Max L2 norm of gradient per tensor.')
_NUM_CONFIGS = flags.DEFINE_integer(
    'configs', 16, 'Number of configurations to consider in ranked-list.')
_MAX_CONFIGS = flags.DEFINE_integer(
    'max_configs', -1,
    'Maximum number of configurations in train and validation partitions to '
    'keep during pre-processing step. This reduces the dataset size. Only '
    'active if > 0. The configurations will be selected as follows. Best and '
    'worst configurations will be selected, as will as some in the middle. '
    'This option is useful to make the dataset fit in memory.')
_KEEP_NODES = flags.DEFINE_integer(
    'keep_nodes', 5000,
    'Sets the number of nodes to keep for Graph-Segmented-Training')
_BATCH = flags.DEFINE_integer(
    'batch', 8,
    'Batch size: number of subgraphs, each with `--configs` configurations.')
_OUTPUT_DIR = flags.DEFINE_string(
    'out_dir', '~/out/tpugraphs_layout',
    'Output metrics and trained models will be written here.')
_RESULTS_CSV = flags.DEFINE_string(
    'results_csv', '',
    'Path to output CSV file to contain inference on test examples. '
    'If not set, defaults to '
    '<--out_dir>/results_<timestamp>_<--source>_<--search>.csv.')
_VALIDATE_BATCHES = flags.DEFINE_integer(
    'validate_batches', 10,
    'If set to >0, then only this many batches will be used to compute '
    'validation error while training. Nonetheless, full validation will be '
    'computed *after* training, but using the best model parameters computed '
    'on this many batches')
_RUN_ID = flags.DEFINE_string(
    'run_id', '', 'Can be used for tagging the experiment.')
_SOURCE = flags.DEFINE_string(
    'source', 'xla', 'The graphs collection. You may use "xla" or "nlp".')
_SEARCH = flags.DEFINE_string(
    'search', 'random', 'The optimization search space. "random" or "default".')


class TrainArgs(NamedTuple):
  """Bundles flags for model specification and training loop."""
  # Data
  source: str  # One of "nlp" or "xla"
  search: str  # One of "random" or "default"

  # Training loop.
  epochs: int
  batch_size: int
  configs: int
  max_configs: int
  early_stop: int
  keep_nodes: int

  # Optimization.
  learning_rate: float
  clip_norm: float

  # Inference.
  out_dir: str
  results_csv: str
  validate_batches: int

  # To run multiple experiments.
  run_id: str

  def compute_hash(self) -> str:
    """Returns psuedo-random string that uniquely identifies flag arguments."""
    json_args = json.dumps(self._asdict(), sort_keys=True).encode()
    return hashlib.md5(json_args).hexdigest()


def _get_results_csv_or_default() -> str:
  """Returns path for CSV file where inference results should be saved.

  Returns:
    If flag --results_csv is set, it returns it. Otherwise, returns
    f"~/{--out_dir}/results_{timestamp}_{--source}_{--search}.csv".
  """
  results_csv = _RESULTS_CSV.value
  if not results_csv:
    source, search = _SOURCE.value, _SEARCH.value
    results_csv = os.path.join(
        os.path.expanduser(_OUTPUT_DIR.value),
        f'results_{int(time.time() * 1000)}_{source}_{search}.csv')

  dirname = os.path.dirname(results_csv)
  if not os.path.exists(dirname):
    os.makedirs(dirname)
  return results_csv


def get_args() -> TrainArgs:
  return TrainArgs(
      source=_SOURCE.value, search=_SEARCH.value,
      epochs=_EPOCHS.value, batch_size=_BATCH.value,
      early_stop=_EARLY_STOP.value, keep_nodes=_KEEP_NODES.value,
      learning_rate=_LEARNING_RATE.value, clip_norm=_CLIP_NORM.value,
      configs=_NUM_CONFIGS.value, max_configs=_MAX_CONFIGS.value,
      out_dir=_OUTPUT_DIR.value, validate_batches=_VALIDATE_BATCHES.value,
      results_csv=_get_results_csv_or_default(), run_id=_RUN_ID.value)
