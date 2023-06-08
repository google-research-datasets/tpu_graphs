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

"""Tests for tpu_graphs.process_data.tensorflow_ops.encode_hlo_config_data."""

import tensorflow.compat.v2 as tf

from tpu_graphs.process_data.tensorflow_ops import encode_hlo_config_data
from google3.pyglib import resources
from google3.pyglib import retry
from google3.testing.pybase import googletest


class EncodeHloConfigDataTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    self.path = resources.GetResourceFilename(
        "google3/third_party/py/tpu_graphs/process_data/tensorflow_ops/testdata/module_tuning_data_layout.pb"
    )
    self.encoded_hlo = encode_hlo_config_data.encode_hlo_config_data(
        tf.io.read_file(self.path),
        self.path,
        "module_tuning",
        sample_rate=1.0,
        samples_limit=-1,
        batch_size=0,
        task="module_layout_cost",
    )

  def test_first_dimension_equal_num_configs(self):
    encoded_hlo = self.encoded_hlo
    num_configs = tf.shape(encoded_hlo.config_features).numpy()[0]
    self.assertEqual(
        num_configs, tf.shape(encoded_hlo.compute_times_ns).numpy()[0]
    )
    self.assertEqual(
        num_configs, tf.shape(encoded_hlo.normalization_values).numpy()[0]
    )
    self.assertEqual(
        num_configs, tf.shape(encoded_hlo.module_config_counts).numpy()[0]
    )
    self.assertEqual(num_configs, tf.shape(encoded_hlo.module_ids).numpy()[0])

  def test_non_zero_node(self):
    num_nodes = tf.shape(self.encoded_hlo.config_features).numpy()[1]
    self.assertGreater(num_nodes, 0)
    encoded_hlo = self.encoded_hlo
    print("normalization_values:", encoded_hlo.normalization_values)
    print("compute_times_ns:", encoded_hlo.compute_times_ns)
    print("module_ids:", encoded_hlo.module_ids)
    print("config_features:", encoded_hlo.config_features)

  def test_expected_config_features_size(self):
    config_features_size = tf.shape(self.encoded_hlo.config_features).numpy()[2]
    self.assertEqual(config_features_size, 18)

  def test_different_config_features(self):
    encoded_hlo = self.encoded_hlo
    self.assertNotEqual(
        encoded_hlo.config_features[0].numpy().tolist(),
        encoded_hlo.config_features[5].numpy().tolist(),
    )
    self.assertNotEqual(
        encoded_hlo.config_features[5].numpy().tolist(),
        encoded_hlo.config_features[10].numpy().tolist(),
    )
    self.assertNotEqual(
        encoded_hlo.config_features[10].numpy().tolist(),
        encoded_hlo.config_features[15].numpy().tolist(),
    )
    self.assertLess(
        tf.reduce_min(encoded_hlo.config_features).numpy(),
        tf.reduce_max(encoded_hlo.config_features).numpy(),
    )

  def test_no_sampling_has_one_max(self):
    max_runtime = tf.reduce_max(self.encoded_hlo.compute_times_ns)
    is_max = tf.equal(self.encoded_hlo.compute_times_ns, max_runtime)
    max_count = tf.reduce_sum(tf.cast(is_max, dtype=tf.int32))
    tf.assert_equal(max_count, 1)

  def _get_compute_times(self, shuffle: bool) -> list[int]:
    encoded_hlo = encode_hlo_config_data.encode_hlo_config_data(
        tf.io.read_file(self.path),
        self.path,
        "module_tuning",
        sample_rate=1.0,
        samples_limit=-1,
        batch_size=0,
        task="module_layout_cost",
        shuffle=shuffle,
    )
    return encoded_hlo.compute_times_ns.numpy().tolist()

  def test_no_shuffle_same_order(self):
    tf.assert_equal(
        self._get_compute_times(shuffle=False),
        self._get_compute_times(shuffle=False),
    )

  def test_default_shuffle_different_orders(self):
    @retry.retry_on_exception(
        retry_value=self.failureException, retry_intervals=[0] * 3
    )
    def assert_sometimes_different(ref, get_result):
      val = get_result()
      self.assertNotEqual(ref, val)
      return val

    ref_compute_times = self._get_compute_times(shuffle=True)
    get_compute_times = lambda: self._get_compute_times(shuffle=True)
    compute_times = assert_sometimes_different(
        ref_compute_times, get_compute_times
    )
    self.assertCountEqual(compute_times, ref_compute_times)


if __name__ == "__main__":
  googletest.main()
