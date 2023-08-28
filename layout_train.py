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

# Copyright 2023 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Binary for invoking the training loop.

# Usage Example
```sh
BIN='python baselines/layout/layout_train.py'

$BIN --source xla --search random --epochs 10 --max_configs 1000
$BIN --source xla --search default --epochs 10 --max_configs 1000
$BIN --source nlp --search random --epochs 10 --max_configs 1000
$BIN --source nlp --search default --epochs 10 --max_configs 1000
"""

from collections.abc import Sequence

from absl import app

from tpu_graphs.baselines.layout import train_args
from tpu_graphs.baselines.layout import train_lib


def main(unused_argv: Sequence[str]) -> None:
  train_lib.train(train_args.get_args())


if __name__ == '__main__':
  app.run(main)
