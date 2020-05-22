  
# Lint as: python3
# Copyright 2019, The TensorFlow Federated Authors.
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
"""Utils for testing executors."""

from absl.testing import absltest
from absl.testing import parameterized

import asyncio
import tensorflow_federated as tff

from tensorflow_federated.python.core.impl.executors import federating_executor

from tff_aggregations import channels


class AsyncTestCase(absltest.TestCase):
  """A test case that manages a new event loop for each test.

  Each test will have a new event loop instead of using the current event loop.
  This ensures that tests are isolated from each other and avoid unexpected side
  effects.

  Attributes:
    loop: An `asyncio` event loop.
  """

  def setUp(self):
    super().setUp()
    self.loop = asyncio.new_event_loop()

    # If `setUp()` fails, then `tearDown()` is not called; however cleanup
    # functions will be called. Register the newly created loop `close()`
    # function here to ensure it is closed after each test.
    self.addCleanup(self.loop.close)

  def run_sync(self, coro):
    return self.loop.run_until_complete(coro)


class MockStrategy(federating_executor.CentralizedIntrinsicStrategy):
  def __init__(self, parent_executor, channel_grid=None):
    super().__init__(parent_executor)
    self.channel_grid = channel_grid

    self.channel_grid.setup_channels(strategy=self)

