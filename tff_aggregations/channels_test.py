from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

import asyncio

from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl.compiler import placement_literals
from tensorflow_federated.python.core.impl.executors import federating_executor

from tff_aggregations import channels
from tff_aggregations.paillier import factory
from tff_aggregations.paillier import placement as paillier_placement


# NOTE AsycTestCase should be imported from tff
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


# class EasyBoxChannelTest(executor_service_utils.AsyncTestCase):
class EasyBoxChannelTest(executor_test_utils.AsyncTestCase):

  def test_generate_aggregator_keys(self):
    # strategy = federating_executor.TrustedAggregatorIntrinsicStrategy
    # loop, ex = _make_test_runtime(intrinsic_strategy_fn=strategy)
    # strat_ex = ex.intrinsic_strategy

    strat_ex = federating_executor.CentralizedIntrinsicStrategy

    channel_grid = channels.ChannelGrid({
        (placement_literals.CLIENTS, paillier_placement.PAILLIER):
            channels.EasyBoxChannel(
                parent_executor=strat_ex,
                sender_placement=placement_literals.CLIENTS,
                receiver_placement=paillier_placement.PAILLIER)
    })

    channel = channel_grid[(placement_literals.CLIENTS,
                            paillier_placement.PAILLIER)]

    self.run_sync(channel.setup())
    
    # loop.run_until_complete(channel.setup())
    # key_references = channel.key_references

    # pk_c = key_references.get_public_key(placement_literals.CLIENTS.name)
    # sk_c = key_references.get_secret_key(placement_literals.CLIENTS.name)
    # pk_a = key_references.get_public_key(placement_literals.AGGREGATORS.name)
    # sk_a = key_references.get_secret_key(placement_literals.AGGREGATORS.name)

    # self.assertEqual(str(pk_c.type_signature), '{uint8[32]}@AGGREGATORS')
    # self.assertEqual(str(sk_c.type_signature), '{uint8[32]}@CLIENTS')
    # self.assertEqual(str(pk_a.type_signature), '{uint8[32]}@CLIENTS')
    # self.assertEqual(str(sk_a.type_signature), '{uint8[32]}@AGGREGATORS')

#   def test_encryption_decryption(self):

#     strategy = federating_executor.TrustedAggregatorIntrinsicStrategy
#     loop, ex = _make_test_runtime(intrinsic_strategy_fn=strategy)
#     strat_ex = ex.intrinsic_strategy

#     channel_grid = channel_base.ChannelGrid({
#         (placement_literals.AGGREGATORS, placement_literals.CLIENTS):
#             federating_executor.EasyBoxChannel(
#                 parent_executor=strat_ex,
#                 sender_placement=placement_literals.CLIENTS,
#                 receiver_placement=placement_literals.AGGREGATORS)
#     })

#     channel = channel_grid[(placement_literals.CLIENTS,
#                             placement_literals.AGGREGATORS)]
#     loop.run_until_complete(channel.setup())

#     val = loop.run_until_complete(
#         ex.create_value([2.0], type_factory.at_clients(tf.float32)))

#     val_enc = loop.run_until_complete(
#         channel.send(val.internal_representation[0]))

#     val_dec = loop.run_until_complete(
#         channel.receive(val_enc))

#     dec_tf_tensor = val_dec[0].internal_representation

#     self.assertEqual(dec_tf_tensor, tf.constant(2.0, dtype=tf.float32))