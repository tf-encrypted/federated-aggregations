from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

import asyncio
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.core.impl.compiler import placement_literals
from tensorflow_federated.python.core.impl.executors import federating_executor

from tff_aggregations import channels
from tff_aggregations import channels_test_utils


def create_test_executor(number_of_clients: int = 3):

  def create_bottom_stack():
    executor = tff.framework.EagerTFExecutor()
    # NOTE why do we need ReferenceResolvingExecutor?
    # return tff.framework.ReferenceResolvingExecutor(executor)
    return executor

  def intrinsic_strategy_fn(executor):
    return federating_executor.CentralizedIntrinsicStrategy(executor)

  return tff.framework.FederatingExecutor({
      tff.SERVER: create_bottom_stack(),
      tff.CLIENTS: [create_bottom_stack() for _ in range(number_of_clients)],
      None: create_bottom_stack()},
      intrinsic_strategy_fn=intrinsic_strategy_fn)

class ChannelGridTest(parameterized.TestCase):

  def test_channel_grid(self):
    channel_grid = channels.ChannelGrid({
          (placement_literals.CLIENTS, placement_literals.SERVER):
              channels.PlaintextChannel()
      })

    channel = channel_grid[(placement_literals.CLIENTS,
                            placement_literals.SERVER)]

    assert isinstance(channel, channels.PlaintextChannel)

class EasyBoxChannelTest(channels_test_utils.AsyncTestCase):

  def test_generate_aggregator_keys(self):

    fed_ex = create_test_executor()
    strat = fed_ex.intrinsic_strategy

    channel_grid = channels.ChannelGrid({
        (placement_literals.CLIENTS, placement_literals.SERVER):
            channels.EasyBoxChannel(
                parent_executor=strat,
                sender_placement=placement_literals.CLIENTS,
                receiver_placement=placement_literals.SERVER)
    })

    channel = channel_grid[(placement_literals.CLIENTS,
                            placement_literals.SERVER)]

    self.run_sync(channel.setup())
    key_references = channel.key_references

    pk_c = key_references.get_public_key(placement_literals.CLIENTS.name)
    sk_c = key_references.get_secret_key(placement_literals.CLIENTS.name)
    pk_a = key_references.get_public_key(placement_literals.SERVER.name)
    sk_a = key_references.get_secret_key(placement_literals.SERVER.name)

    self.assertEqual(str(pk_c.type_signature), '{uint8[32]}@SERVER')
    self.assertEqual(str(sk_c.type_signature), '{uint8[32]}@CLIENTS')
    # NOTE check why type_signature is not consistent
    self.assertEqual(str(pk_a.type_signature), 'uint8[32]@CLIENTS')
    self.assertEqual(str(sk_a.type_signature), '{uint8[32]}@SERVER')

  def test_encryption_decryption(self):

    fed_ex = create_test_executor(1)
    strat = fed_ex.intrinsic_strategy

    # Will have to figure out how EasyBoxChannel gets access to 
    # target_executors, and methods from CentralizedIntrinsicStrategy
    # if it's instanciated from outside the PaillierStrategy
    # (currently provided from fed_ex.intrinsic_strategy)
    channel_grid = channels.ChannelGrid({
        (placement_literals.CLIENTS, placement_literals.SERVER):
            channels.EasyBoxChannel(
                parent_executor=strat,
                sender_placement=placement_literals.CLIENTS,
                receiver_placement=placement_literals.SERVER)
    })

    channel = channel_grid[(placement_literals.CLIENTS,
                            placement_literals.SERVER)]

    self.run_sync(channel.setup())

    val = self.run_sync(
        fed_ex.create_value([2.0], tff.FederatedType(tf.float32, tff.CLIENTS)))

    val_enc = self.run_sync(
        channel.send(val.internal_representation[0]))

    val_dec = self.run_sync(
        channel.receive(val_enc))

    dec_tf_tensor = val_dec[0].internal_representation

    self.assertEqual(dec_tf_tensor, tf.constant(2.0, dtype=tf.float32))