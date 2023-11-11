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

"""Evaluates layout model on validation or test benchmarks.

Script writes a CSV file with one line per benchmark. The line contains the
kendal correlation, as well as the slow-down when taking the best of (predicted)
top `K` configurations, as compared the fastest configuration, for `K` in
`(1, 10, 100)`. Optionally, if `--time` is set, then the average inference time
(of batch size determined by --batch) will be printed per benchmark graph.

Finally, you can pass `--test` flag to run on test partition, **but only if**
you have the secret test files locally (i.e., you are provided the files by the
dataset organizers).


# Usage Example

```sh
E='python layout_evaluate.py'
$E --dirs ~/out/tpugraphs_layout/model_81f19f85346ed8d36fac7b59e9a8bec9
```
where `model_81f19f85346ed8d36fac7b59e9a8bec9` is written by `layout_train.py`
(path is printed on STDOUT). By default (when supplying on --dirs), the model
will be evaluated on same subcollection it was trained on ("xla" or "nlp", and,
"random" or "default"). It is possible to train on a collection and evaluating
on another by supplying to this binary the flags `--source` and `--search`.

The above invocation writes CSV file
`~/out/tpugraphs_layout/validation_results_81f19f85346ed8d36fac7b59e9a8bec9.csv`
with one line per benchmark. The script also computes average metric across all
benchmarks.
"""

import collections
import gzip
import json
import os
import time

from absl import app
from absl import flags
import numpy as np
import scipy.stats
import tensorflow as tf
# So that keras.models.load_model() can re-instantiate layers of saved model.
import tensorflow_gnn as tfgnn
import tensorflow_ranking as tfr
from tpu_graphs.baselines.layout import data
from tpu_graphs.baselines.layout import models
from tpu_graphs.baselines.layout.eval_indices import validation
import tqdm
unused_modules = [tfr, tfgnn]

_MODEL_DIRS = flags.DEFINE_string(
    'dirs', None,
    'Comma-separated list of model directories to evaluate. '
    'The per-benchmark average will be printed', required=True)
_DATA_ROOT = flags.DEFINE_string(
    'data_root', '~/data/tpugraphs/npz/layout',
    'Root directory containing dataset. It must contain subdirectories '
    '{nlp, xla}, each with subdirectories {train, test, valid}, each having '
    'many .npz files')
_SECRET_TEST_DATA_ROOT = flags.DEFINE_string(
    'secret_path', os.path.expanduser('~/data/tpu_graphs/final/npz/layout'),
    'Used if --test is activated. It must be directory path containing files '
    '`{nlp|xla}/{random|uniform}/test_export/secret/*.npz`.')
_RUN_ON_SECRET_TEST = flags.DEFINE_bool(
    'test', False, 'If set, will be run on secret test.')
_PRINT_INFERENCE_TIME = flags.DEFINE_bool(
    'time', False,
    'If set, the mean wallclock time to run the model will be printed, one '
    'line per benchmark graph.')
_CACHE_DIR = flags.DEFINE_string(
    'cache_dir', '~/data/tpugraphs/cache/layout',
    'If given, dataset tensors will be cached here for faster loading. Files '
    'with name "<hash>.npz" will be written, where <hash> is a hash of the '
    'file pattern of training data, i.e., it depends on the collection e.g., '
    '{xla:default} and partition {train, test, valid}.')
_MODEL_KWARGS_JSON = flags.DEFINE_string(
    'model_kwargs_json', '',
    'If set, must be a JSON-encoded dict that would be parsed and sent to '
    'model constructor as **kwargs. If not given, the arguments will be loaded '
    'from the .jsonz file associated with model directory.')
_SOURCE = flags.DEFINE_string(
    'source', '', 'If set, must be "nlp" or "xla". If skipped, eval will be '
    'computed over the validation data belonging source that the model is '
    'trained on.')
_SEARCH = flags.DEFINE_string(
    'search', '', 'If set, must be "random" or "default". If skipped, '
    'eval will be computed over the validation data belonging search-space '
    'that the model is trained on.')
_SAMPLE_CONFIGS = flags.DEFINE_bool(
    'sample', True, 'If set (default), only some configurations will be '
    'evaluated, as given by eval_indices/{validation.py, *.json}.')
_BATCH_SIZE = flags.DEFINE_integer(
    'batch', 100, 'Batch size for inference. This many configurations will be '
    'scored at-once **for the same benchmark**.')


def main(unused_argv: list[str]) -> None:
  results_on = 'test' if _RUN_ON_SECRET_TEST.value else 'validation'
  for dirpath in tqdm.tqdm(_MODEL_DIRS.value.split(',')):
    dirpath = os.path.expanduser(dirpath)
    jsonz_file = dirpath.replace('/model_', '/run_') + '.jsonz'
    out_results_csv = (
        dirpath.replace('/model_', f'/{results_on}_results_'))
    if _SAMPLE_CONFIGS.value:
      out_results_csv += '_sample'
    out_results_csv += '.csv'
    jsonz_data = json.loads(
        gzip.open(tf.io.gfile.GFile(jsonz_file, 'rb'), 'rb').read().decode())
    if _SEARCH.value and _SOURCE.value:
      source = _SOURCE.value
      search = _SEARCH.value
    else:
      source = jsonz_data['args']['source']
      search = jsonz_data['args']['search']

    if _SAMPLE_CONFIGS.value and not _RUN_ON_SECRET_TEST.value:
      config_indices = validation.get_eval_indices(source, search)
    else:
      config_indices = None

    data_root_dir = os.path.join(
        os.path.expanduser(_DATA_ROOT.value), source, search)

    dataset = data.get_npz_dataset(
        data_root_dir,
        cache_dir=os.path.expanduser(_CACHE_DIR.value),
        max_train_configs=1000,
        max_valid_configs=-1)

    keras_model = tf.keras.models.load_model(dirpath)

    # Load pythonic model.
    if _MODEL_KWARGS_JSON.value:
      model_kwargs = json.loads(_MODEL_KWARGS_JSON.value)
    else:
      model_kwargs = json.loads(
          jsonz_data['args'].get('model_kwargs_json', '{}'))

    if 'segment' in model_kwargs:  # Argument `segment` is renamed to `dropout`.
      model_kwargs['dropout'] = model_kwargs.pop('segment')
    model = models.ResModel(
        num_configs=jsonz_data['args']['configs'],
        num_ops=dataset.num_ops,
        **model_kwargs)

    # Instantiate `model` parameters (to copy from `keras_model`), so that we
    # can instantiate `model.forward` and therefore be able to run any number of
    # configurations, even if different than the one that keras_model was
    # compiled with.
    sample_num_configs = 2
    sample_graph = dataset.train.get_item(0).to_graph_tensor(sample_num_configs)
    model.forward(sample_graph, sample_num_configs)
    del sample_graph, sample_num_configs  # No longer need a toy example.

    target_vars = model.trainable_variables
    source_vars = keras_model.trainable_variables
    assert len(target_vars) == len(source_vars)
    for tv, sv in zip(target_vars, source_vars):
      # The function `get_npz_dataset()` invokes `.normalize()` which centers
      # the data (subtract mean and divide over std). `normalize()` also removes
      # features that are **constant** across all examples. When using
      # --toy_data, only 3 graphs are loaded and more features will appear
      # "constant".
      assert sv.shape == tv.shape, (
          'Are you evaluating model trained with --toy_data?')
      tv.assign(sv)

    csv_lines = [
        'graph,kendaltau,slowdown1,slowdown10,slowdown100'
    ]

    measurements = collections.defaultdict(list)
    partition = dataset.validation
    if _RUN_ON_SECRET_TEST.value:
      partition = dataset.test
    assert partition.graph_id is not None
    all_inference_wallclock_time = []
    for graph_idx in tqdm.tqdm(
        range(partition.graph_id.shape[-1]),
        desc='Inference on ' + results_on):
      layout_example = partition.get_item(graph_idx)
      graph_id = layout_example.graph_id.numpy().decode()
      runtimes = layout_example.config_runtimes

      if _RUN_ON_SECRET_TEST.value:
        secret_npz_path = os.path.join(
            _SECRET_TEST_DATA_ROOT.value, source, search,
            'test_export', 'secret', graph_id + '.npz')
        secret_data = np.load(secret_npz_path)
        graph_id = os.path.basename(str(secret_data['input_file'])).split(
            '.npz')[0]
        runtimes = secret_data['config_runtime']
      else:
        runtimes = tf.gather(runtimes, config_indices[graph_id])

      keep_nodes = (
          jsonz_data['args']['keep_nodes']
          if model_kwargs.get('dropout', '') == 'dropout'
          else -1)
      if _RUN_ON_SECRET_TEST.value:
        preds, model_wallclock_time = infer_model_on_example(
            model, layout_example, keep_nodes)
      else:
        preds, model_wallclock_time = infer_model_on_example(
            model, layout_example, keep_nodes,
            config_indices=config_indices[graph_id])

      all_inference_wallclock_time.append((graph_id, model_wallclock_time))

      time_best = tf.reduce_min(runtimes)

      kendalltau = scipy.stats.kendalltau(preds, runtimes).correlation
      csv_line: list[str] = [
          graph_id,
          '%f' % kendalltau,
      ]

      measurements['kendalltau'].append(kendalltau)

      sorted_indices = tf.argsort(preds)
      # Slow downs
      for k in [1, 10, 100]:
        time_model_candidates = tf.gather(runtimes, sorted_indices[:k])
        best_of_candidates = tf.reduce_min(time_model_candidates)
        error = float((best_of_candidates - time_best) / time_best)
        csv_line.append('%f' % error)
        measurements[f'slowdown{k}'].append(error)

      csv_lines.append(','.join(csv_line))

    with tf.io.gfile.GFile(out_results_csv, 'w') as f:
      f.write('\n'.join(csv_lines))
    print('\n\n *** Wrote ' + out_results_csv)
    print('Average measurements:')
    for k, m in measurements.items():
      print('%s: %f' % (k, sum(m)/len(m)))

    if _PRINT_INFERENCE_TIME.value:
      print('\n\n *** Wallclock time to run model (with batch size %i)' %
            _BATCH_SIZE.value)
      for graph_id, wc_time in all_inference_wallclock_time:
        print(graph_id, wc_time, 'seconds')


def infer_model_on_example(
    model: tf.keras.Model, example: data.LayoutExample, keep_nodes: int = -1,
    config_indices=None, repeats_if_dropout: int = 3
    ) -> tuple[tf.Tensor, float]:
  """Runs model on all configrations of `example` output and wallclock times.

  Args:
    model: should implement the `forward` function that accepts `GraphTensor`
      instance.
    example: contains layout example with possibly many configurations.
    keep_nodes: If set to -1 or if `repeats_if_dropout == False`, then entire
      graph will be given to model. If >0 and repeats_if_dropout, then this many
      nodes will be sampled per subgraph (where number of sampled subgraphs ==
      `repeats_if_dropout`).
    config_indices: If given, only configuration at these indices will be given
      to the `model.forward` invocation. If not given, all configurations will
      be passed to `model.forward` -- each invocation passes `_BATCH_SIZE`
      configurations.
    repeats_if_dropout: If model was trained with segment dropout, then this
      many subgraphs (sampled uniformly at random) will be run through the
      model. The logits (model output) will be averaged across repeats. If set,
      then `keep_nodes` should be >0, which determines the number of nodes to
      keep (i.e., size of sampled subgraph) to pass to model.

  Returns:
    tuple: (inference model scores vector, time to run the `model.forward`
            function per batch).
    The length of the vector will be `config_indices` (if it is provided) or
    the number of configurations in `example`.
  """
  repeats = 1 if keep_nodes == -1 else repeats_if_dropout
  graph_repeat_preds = []
  wallclock_times = []
  for unused_r in range(repeats):
    graph = example.to_graph_tensor(max_nodes=keep_nodes)
    if config_indices is not None:
      graph = data.sub_configs(graph, config_indices)
    num_configs = graph.node_sets['g']['runtimes'].shape[-1]
    batch_scores = []
    for i in range(0, num_configs, _BATCH_SIZE.value):
      end_i = min(i + _BATCH_SIZE.value, num_configs)
      # Take a cut of the configs.
      subconfigs_graph = data.sub_configs(graph, slice(i, end_i))
      starttime = time.time()
      h = model.forward(subconfigs_graph, num_configs=(end_i - i))
      wallclock_times.append(time.time() - starttime)
      batch_scores.append(h[0])
    graph_repeat_preds.append(tf.concat(batch_scores, axis=0))
  mean_wc_time = float(np.mean(wallclock_times))
  return tf.reduce_mean(graph_repeat_preds, axis=0), mean_wc_time


if __name__ == '__main__':
  app.run(main)
