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

_EPOCHS = flags.DEFINE_integer('epochs', 100, 'number of train epochs.')
_EARLY_STOP = flags.DEFINE_integer(
    'early_stop', 20,
    'If held-out validation does not improve after this many epochs, then '
    'training will stop. Must be divisible by `--eval_every`.')
_EVAL_EVERY = flags.DEFINE_integer(
    'eval_every', 5, 'Eval every this many.')
_LOSSES = flags.DEFINE_string(
    'losses', 'ListMLELoss:1,MSE:0.02',
    'Comma-separated list of "lossName:lossWeight", per `metrics.py`.')
_LEARNING_RATE = flags.DEFINE_float(
    'lr', 1e-3, 'Learning rate for Adam optimizer.')
_CLIP_NORM = flags.DEFINE_float(
    'clip_norm', 1e-3, 'Max L2 norm of gradient per tensor.')
_NUM_CONFIGS = flags.DEFINE_integer(
    'configs', 10, 'Number of configurations to consider in ranked-list.')
_BATCH = flags.DEFINE_integer(
    'batch', 10,
    'Batch size: number of subgraphs, each with `--configs` configurations.')
_MODEL = flags.DEFINE_string(
    'model', 'EarlyJoinSAGE',
    'Name of model. Must be a class in models.py. E.g., LateJoinResGCN, '
    'EarlyJoinResGCN, LateJoinSAGE, EarlyJoinSAGE, MLP.')
_MODEL_KWARGS_JSON = flags.DEFINE_string(
    'model_kwargs_json', '{}',
    'JSON-serialized dictionary with model class constructor arguments.')
_TEST_MODE = flags.DEFINE_enum(
    'test_mode', 'predictions', ['metrics', 'predictions'],
    'Whether to compute test metrics or output predictions csv file. '
    'If set to "metrics", then test graphs must contain `config_runtimes` and '
    '`config_runtime_normalizers` and test metrics will be added to output '
    'file (written in `--out_dir`). If set to "predictions", then csv file '
    'will be written containing predictions.')
_OUTPUT_DIR = flags.DEFINE_string(
    'out_dir', '~/out/tpugraphs_tiles',
    'Output metrics and trained models will be written here.')
_RESULTS_CSV = flags.DEFINE_string(
    'results_csv', '',
    'Path to output CSV file to contain inference on test examples. '
    'If not set, defaults to <--out_dir>/results_<timestamp>.csv.')
_VALIDATE_BATCHES = flags.DEFINE_integer(
    'validate_batches', -1,
    'If set to >0, then only this many batches will be used to compute '
    'validation error while training. Nonetheless, full validation will be '
    'computed *after* training, but using the best model parameters computed '
    'on this many batches')
_RUN_ID = flags.DEFINE_string(
    'run_id', '', 'Can be used for tagging the experiment.')


class TrainArgs(NamedTuple):
  """Bundles flags for model specification and training loop."""
  # Training loop.
  epochs: int
  eval_every: int  # DEPRECATED. To be removed.
  batch_size: int
  configs: int
  early_stop: int

  # Optimization.
  losses: str
  learning_rate: float
  clip_norm: float

  # Model.
  model: str
  model_kwargs_json: str

  # Inference.
  test_mode: str
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
    f"~/{--out_dir}/results_{timestamp}.csv".
  """
  results_csv = _RESULTS_CSV.value
  if not results_csv:
    results_csv = os.path.join(
        os.path.expanduser(_OUTPUT_DIR.value),
        f'results_{int(time.time() * 1000)}.csv')

  dirname = os.path.dirname(results_csv)
  if not os.path.exists(dirname):
    os.makedirs(dirname)
  return results_csv


def get_args() -> TrainArgs:
  return TrainArgs(
      epochs=_EPOCHS.value, eval_every=_EVAL_EVERY.value, losses=_LOSSES.value,
      batch_size=_BATCH.value, configs=_NUM_CONFIGS.value,
      early_stop=_EARLY_STOP.value, learning_rate=_LEARNING_RATE.value,
      clip_norm=_CLIP_NORM.value, model=_MODEL.value,
      model_kwargs_json=_MODEL_KWARGS_JSON.value, test_mode=_TEST_MODE.value,
      out_dir=_OUTPUT_DIR.value, validate_batches=_VALIDATE_BATCHES.value,
      results_csv=_get_results_csv_or_default(), run_id=_RUN_ID.value)
