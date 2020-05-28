from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

import asyncio
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.core.impl.executors import federating_executor
from tensorflow_federated.python.core.impl.types import placement_literals

from tff_aggregations.channels import channel as ch
from tff_aggregations.channels import channel_grid as grid
from tff_aggregations.channels import channel_test_utils as utils


class PlaintextChannelTest(utils.AsyncTestCase):
  def test_send_receive(self):
    fed_ex = utils.create_test_executor(channel=ch.PlaintextChannel)
    strategy = fed_ex.intrinsic_strategy
    channel_grid = strategy.channel_grid
    self.run_sync(channel_grid.setup_channels(strategy))

    channel = channel_grid[(tff.CLIENTS, tff.SERVER)]
    val = self.run_sync(fed_ex.create_value([2.0] * 3,
        tff.FederatedType(tf.float32, tff.CLIENTS)))
    sent = self.run_sync(channel.send(val))
    received = self.run_sync(channel.receive(sent))
    result = self.run_sync(received.compute())

    assert isinstance(sent, federating_executor.FederatingExecutorValue)
    self.assertEqual(received.type_signature, tff.FederatedType(tf.float32, tff.CLIENTS))
    assert isinstance(received, federating_executor.FederatingExecutorValue)
    self.assertEqual(received.type_signature, tff.FederatedType(tf.float32, tff.SERVER))
    self.assertEqual(result, tf.constant([2.0] * 3, dtype=tf.float32))


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

  def test_encryption_decryption(self):
    fed_ex = utils.create_test_executor()
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
