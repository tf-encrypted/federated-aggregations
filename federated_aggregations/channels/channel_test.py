from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

import asyncio
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.core.impl.executors import federating_executor
from tensorflow_federated.python.core.impl.types import placement_literals
from tensorflow_federated.python.core.impl.types import type_conversions

from federated_aggregations.channels import channel as ch
from federated_aggregations.channels import channel_grid as grid
from federated_aggregations.channels import channel_test_utils as utils


class PlaintextChannelTest(utils.AsyncTestCase):
  @parameterized.named_parameters(
      ("clients_to_server", [2.0] * 3, tff.CLIENTS, tff.SERVER,
          [tf.constant(2.0, dtype=tf.float32)] * 3),
      ("server_to_clients", 2.0, tff.SERVER, tff.CLIENTS,
          tf.constant(2.0, dtype=tf.float32)))
  def test_move_from(self, value, source_placement, target_placement, expected):
    fed_ex = utils.create_test_executor(channel=ch.PlaintextChannel)
    strategy = fed_ex.intrinsic_strategy
    channel_grid = strategy.channel_grid
    self.run_sync(channel_grid.setup_channels(strategy))

    channel = channel_grid[(tff.CLIENTS, tff.SERVER)]
    val = self.run_sync(fed_ex.create_value(value,
        tff.FederatedType(tf.float32, source_placement)))
    sent = self.run_sync(channel.send(val))
    received = self.run_sync(channel.receive(sent))
    result = self.run_sync(received.compute())

    expected_type = type_conversions.infer_type(expected)
    assert isinstance(sent, federating_executor.FederatingExecutorValue)
    self.assertEqual(sent.type_signature,
        tff.FederatedType(tf.float32, source_placement))
    assert isinstance(received, federating_executor.FederatingExecutorValue)
    self.assertEqual(received.type_signature,
        tff.FederatedType(expected_type, target_placement, all_equal=True))
    if isinstance(expected, list):
      result = anonymous_tuple.flatten(result)
    self.assertEqual(result, expected)


class EasyBoxChannelTest(utils.AsyncTestCase):
  def test_generate_aggregator_keys(self):
    fed_ex = utils.create_test_executor()
    strategy = fed_ex.intrinsic_strategy
    channel_grid = strategy.channel_grid
    self.run_sync(channel_grid.setup_channels(strategy))
    
    channel = channel_grid[(tff.CLIENTS, tff.SERVER)]
    pk_clients, sk_clients = channel.key_references.get_key_pair(tff.CLIENTS)
    pk_server, sk_server = channel.key_references.get_key_pair(tff.SERVER)

    self.assertEqual(str(pk_clients.type_signature), '<uint8[32]@SERVER,uint8[32]@SERVER,uint8[32]@SERVER>')
    self.assertEqual(str(sk_clients.type_signature), '{uint8[32]}@CLIENTS')
    self.assertEqual(str(pk_server.type_signature), 'uint8[32]@CLIENTS')
    self.assertEqual(str(sk_server.type_signature), 'uint8[32]@SERVER')

  @parameterized.named_parameters(
      ("clients_to_server", [2.0] * 3, tff.CLIENTS, tff.SERVER,
          [tf.constant(2.0, dtype=tf.float32)] * 3),
      ("server_to_clients", 2.0, tff.SERVER, tff.CLIENTS,
          tf.constant(2.0, dtype=tf.float32)))
  def test_move_from(self, value, source_placement, target_placement, expected):
    fed_ex = utils.create_test_executor()
    strategy = fed_ex.intrinsic_strategy
    channel_grid = strategy.channel_grid
    self.run_sync(channel_grid.setup_channels(fed_ex.intrinsic_strategy))

    channel = channel_grid[(placement_literals.CLIENTS,
                            placement_literals.SERVER)]
    val = self.run_sync(fed_ex.create_value(value,
        tff.FederatedType(tf.float32, source_placement)))
    val_enc = self.run_sync(channel.send(val))
    val_dec = self.run_sync(channel.receive(val_enc))
    dec_tf_tensor = self.run_sync(val_dec.compute())

    self.assertEqual(dec_tf_tensor, expected)
