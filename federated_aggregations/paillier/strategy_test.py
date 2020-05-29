from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow_federated.python.core.impl.executors import execution_context

from federated_aggregations import channels
from federated_aggregations.paillier import factory
from federated_aggregations.paillier import placement
from federated_aggregations.paillier import strategy


def _install_executor(executor_factory_instance):
  context = execution_context.ExecutionContext(executor_factory_instance)
  return tff.framework.get_context_stack().install(context)


def create_test_executor(number_of_clients: int = 3):

  def create_bottom_stack():
    executor = tff.framework.EagerTFExecutor()
    return tff.framework.ReferenceResolvingExecutor(executor)

  channel_grid = channels.ChannelGrid({
      (tff.CLIENTS, placement.PAILLIER): channels.StubChannel(),
      (tff.CLIENTS, tff.SERVER): channels.StubChannel(),
      (placement.PAILLIER, tff.SERVER): channels.StubChannel()})

  def intrinsic_strategy_fn(executor):
    return strategy.PaillierStrategy(executor, channel_grid)

  return tff.framework.FederatingExecutor({
      tff.SERVER: create_bottom_stack(),
      tff.CLIENTS: [create_bottom_stack() for _ in range(number_of_clients)],
      None: create_bottom_stack()},
      intrinsic_strategy_fn=intrinsic_strategy_fn)


class PaillierStrategyTest(parameterized.TestCase):
  @parameterized.named_parameters(
      ('paillier_executor_factory_none_clients',
      factory.local_paillier_executor_factory()),
      ('paillier_executor_factory_five_clients',
      factory.local_paillier_executor_factory(num_clients=5)),
  )
  def test_federated_secure_sum(self, factory):
    @tff.federated_computation(tff.FederatedType(tf.int32, tff.CLIENTS))
    def secure_paillier_addition(x):
      bitwidth = 8  # assume upper-bound of 2^8 on summation result
      return tff.federated_secure_sum(x, bitwidth)

    with _install_executor(factory):
      result = secure_paillier_addition([1, 2, 3, 4, 5])

    self.assertAlmostEqual(result, 15.0)
