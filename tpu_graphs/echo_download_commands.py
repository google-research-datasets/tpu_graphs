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

"""Script to print shell command lines to download tpu-graphs dataset."""

filenames = (
    [f'npz_tile_xla_{split}.tar' for split in ('train', 'test', 'valid')])

for source in ('xla', 'nlp'):
  for search in ('default', 'random'):
    for split in ('train', 'valid', 'test'):
      filenames.append(f'npz_layout_{source}_{search}_{split}.tar')

print('mkdir -p ~/data/tpugraphs')
print('cd ~/data/tpugraphs')
print('\n\n')
for f in filenames:
  print(f'ls {f} || '
        f'curl http://download.tensorflow.org/data/tpu_graphs/v0/{f} > {f}')
  print(f'tar xvf {f}')
