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

"""Returns indices of configurations for validation split."""

import json
import os
from typing import Dict, List


def get_eval_indices(source: str, search: str) -> Dict[str, List[int]]:
  if source not in ('nlp', 'xla'):
    raise ValueError('`source` must be "xla" or "nlp". Got: ' + source)
  if search not in ('default', 'random'):
    raise ValueError('`search` must be "default" or "random". Got: ' + source)
  json_filename = os.path.join(os.path.dirname(__file__),
                               f'{source}_{search}.json')
  return json.loads(open(json_filename).read())
