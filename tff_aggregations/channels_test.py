from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

import asyncio
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.core.impl.executors import federating_executor
from tensorflow_federated.python.core.impl.types import placement_literals

from tff_aggregations import channels
from tff_aggregations import channels_test_utils

def create_bottom_stack():
  executor = tff.framework.EagerTFExecutor()
  return tff.framework.ReferenceResolvingExecutor(executor)


def create_test_executor(
    number_of_clients: int = 3,
    channel_grid: channels.ChannelGrid = None,
    channel: channels.Channel = channels.EasyBoxChannel):
  if channel_grid is None:
    channel_grid = channels.ChannelGrid({(tff.CLIENTS, tff.SERVER): channel})

  def intrinsic_strategy_fn(executor):
    return channels_test_utils.MockStrategy(executor, channel_grid)

  return tff.framework.FederatingExecutor({
      tff.SERVER: create_bottom_stack(),
      tff.CLIENTS: [create_bottom_stack() for _ in range(number_of_clients)],
      None: create_bottom_stack()},
      intrinsic_strategy_fn=intrinsic_strategy_fn)


class ChannelGridTest(channels_test_utils.AsyncTestCase):
  def test_channel_grid_setup(self):
    channel_grid = channels.ChannelGrid(
        {(tff.CLIENTS, tff.SERVER): channels.PlaintextChannel})
    ex = create_test_executor(channel_grid=channel_grid)
    self.run_sync(channel_grid.setup_channels(ex.intrinsic_strategy))

    channel = channel_grid[(tff.CLIENTS, tff.SERVER)]

    assert isinstance(channel, channels.PlaintextChannel)


class EasyBoxChannelTest(channels_test_utils.AsyncTestCase):
  def test_generate_aggregator_keys(self):
    fed_ex = create_test_executor()
    strategy = fed_ex.intrinsic_strategy
    channel_grid = strategy.channel_grid
    self.run_sync(channel_grid.setup_channels(fed_ex.intrinsic_strategy))
    
    channel = channel_grid[(tff.CLIENTS, tff.SERVER)]
    pk_clients, sk_clients = channel.key_references.get_key_pair(tff.CLIENTS)
    pk_server, sk_server = channel.key_references.get_key_pair(tff.SERVER)

    self.assertEqual(str(pk_clients.type_signature), '<uint8[32]@SERVER,uint8[32]@SERVER,uint8[32]@SERVER>')
    self.assertEqual(str(sk_clients.type_signature), '{uint8[32]}@CLIENTS')
    self.assertEqual(str(pk_server.type_signature), 'uint8[32]@CLIENTS')
    self.assertEqual(str(sk_server.type_signature), 'uint8[32]@SERVER')

  def test_encryption_decryption(self):
    fed_ex = create_test_executor()
    strategy = fed_ex.intrinsic_strategy
    channel_grid = strategy.channel_grid
    self.run_sync(channel_grid.setup_channels(fed_ex.intrinsic_strategy))

    channel = channel_grid[(placement_literals.CLIENTS,
                            placement_literals.SERVER)]
    val = self.run_sync(fed_ex.create_value([2.0] * 3,
        tff.FederatedType(tf.float32, tff.CLIENTS)))
    val_enc = self.run_sync(channel.send(val))
    val_dec = self.run_sync(channel.receive(val_enc))
    dec_tf_tensor = self.run_sync(val_dec.compute())

    self.assertEqual(dec_tf_tensor, tf.constant(2.0, dtype=tf.float32))