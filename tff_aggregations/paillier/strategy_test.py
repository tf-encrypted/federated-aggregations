from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow_federated as tff

from tff_aggregations import channels
from tff_aggregations.paillier import strategy
from tff_aggregations.paillier import placement


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
      ('paillier_executor_factory_three_clients',
      factory.local_paillier_executor_factory(num_clients=3)),
  )
  def test_federated_secure_sum(self):
    @tff.federated_computation(tf.FederatedType(tf.float32, tff.CLIENTS))
    def secure_sum(x):
      return tff.federated_secure_sum(x, 0)

    

