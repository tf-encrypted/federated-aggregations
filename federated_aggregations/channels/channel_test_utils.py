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
"""Utils for testing channels."""

from absl.testing import absltest
from absl.testing import parameterized
import asyncio

import tensorflow_federated as tff
from tensorflow_federated.python.core.impl.executors import federating_executor

from federated_aggregations.channels import channel_grid as grid
from federated_aggregations.channels import channel as ch


def create_bottom_stack():
  executor = tff.framework.EagerTFExecutor()
  return tff.framework.ReferenceResolvingExecutor(executor)


def create_test_executor(
    number_of_clients: int = 3,
    channel_grid: grid.ChannelGrid = None,
    channel: ch.Channel = ch.EasyBoxChannel):
  if channel_grid is None:
    channel_grid = grid.ChannelGrid({(tff.CLIENTS, tff.SERVER): channel})

  def intrinsic_strategy_fn(executor):
    return MockStrategy(executor, channel_grid)

  return tff.framework.FederatingExecutor({
      tff.SERVER: create_bottom_stack(),
      tff.CLIENTS: [create_bottom_stack() for _ in range(number_of_clients)],
      None: create_bottom_stack()},
      intrinsic_strategy_fn=intrinsic_strategy_fn)


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
