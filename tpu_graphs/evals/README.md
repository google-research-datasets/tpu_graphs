# Evaluation

This page contains guidelines on the evaluation procedure for TpuGraphs.

This page assumes that you have already [downloaded the dataset](https://github.com/google-research-datasets/tpu_graphs#dataset).

## Tile collection

Since the tile graphs are small (e.g., tens of nodes).
Further, the configuration features are at the graph level, and the number of
available configurations is relatively small (e.g., a few, to hundreds), we
report *slowdown metrics* on all configurations. Please refer to
[our paper](https://openreview.net/forum?id=plAix1NxhU) or
[tiles_evaluate.py](https://github.com/google-research-datasets/tpu_graphs/blob/main/tiles_evaluate.py)

## Layout collections

On the other hand, the layout graphs are larger (up to hundreds of thousands of
nodes). Further, the configuration features are at the node level, and the
number of available configurations is relatively large (e.g., up to hundreds of
thousands of configurations). Therefore, we choose **only** a 1000
configurations to score and report metrics on, in our main paper. This should
decrease the burden for the academic community for training and evaluating
models, especially for reporting experimental metrics.

For every **validation** graph in every subcollection {xla|nlp}:{default|random}
we pre-compute indices of configuration features and their corresponding
runtimes.
Specifically, the indices for each graph is available in json format, at:

https://github.com/google-research-datasets/tpu_graphs/tree/main/tpu_graphs/baselines/layout/eval_indices


The following code snippet reads-in a validation graph and restricts to the
validation indices.

```py
import os
import json
import numpy as np

# Assume that you did `git clone` inside of `~/code`:
_JSON_ROOT_DIR = os.path.expanduser(
    '~/code/tpu_graphs/tpu_graphs/baselines/layout/eval_indices')
# Assume data was downloaded per
# https://github.com/google-research-datasets/tpu_graphs#dataset:
_LAYOUT_DATA_ROOT = os.path.expanduser('~/data/tpugraphs/npz/layout')

_JSON_DATA = {
  ('nlp', 'default'): json.load(open(f'{_JSON_ROOT_DIR}/nlp_default.json')),
  ('nlp', 'random'): json.load(open(f'{_JSON_ROOT_DIR}/nlp_random.json')),
  ('xla', 'default'): json.load(open(f'{_JSON_ROOT_DIR}/xla_default.json')),
  ('xla', 'random'): json.load(open(f'{_JSON_ROOT_DIR}/xla_random.json')),
}


def read_validation_graph(source, search, graph_name):
  npz_path = os.path.join(
      _LAYOUT_DATA_ROOT, source, search, 'valid', graph_name+'.npz')
  npz_data = dict(np.load(npz_path))
  ids = _JSON_DATA[(source, search)][graph_name]
  np.random.shuffle(ids)
  npz_data['config_runtime'] = npz_data['config_runtime'][ids]
  npz_data['node_config_feat'] = npz_data['node_config_feat'][ids]
  return npz_data


print(read_validation_graph('xla', 'random', 'resnet50.4x4.fp16'))

```