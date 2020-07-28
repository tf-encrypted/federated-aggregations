from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
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


def make_integer_secure_sum(input_shape):
  if input_shape is None:
    member_type = tf.int32
  else:
    member_type = tff.TensorType(tf.int32, input_shape)
  @tff.federated_computation(tff.FederatedType(member_type, tff.CLIENTS))
  def secure_paillier_addition(x):
    return tff.federated_secure_sum(x, 64)
  
  return secure_paillier_addition


class PaillierAggregatingStrategyTest(parameterized.TestCase):
  @parameterized.named_parameters(
      ('paillier_executor_factory_none_clients',
      factory.local_paillier_executor_factory()),
      ('paillier_executor_factory_five_clients',
      factory.local_paillier_executor_factory(num_clients=5)))
  def test_federated_secure_sum_with(self, factory):
    secure_paillier_addition = make_integer_secure_sum(None)
    with _install_executor(factory):
      result = secure_paillier_addition([1, 2, 3, 4, 5])
    self.assertAlmostEqual(result, 15.0)

  @parameterized.named_parameters(
      (('rank{}'.format(i), [2] * i) for i in range(1, 6)))
  def test_secure_sum_inputs(self, input_shape):
    input_tensor = np.ones(input_shape, dtype=np.int32)
    NUM_CLIENTS = 5
    expected = input_tensor * NUM_CLIENTS
    secure_paillier_addition = make_integer_secure_sum(input_shape)
    with _install_executor(factory.local_paillier_executor_factory()):
      result = secure_paillier_addition([input_tensor] * NUM_CLIENTS)
    np.testing.assert_almost_equal(result, expected)

  @parameterized.named_parameters(
      (('{}'.format(n), n) for n in [5, 20, 50]))
  def test_secure_sum_many_clients(self, num_clients):
    secure_paillier_addition = make_integer_secure_sum([1, 1])
    with _install_executor(factory.local_paillier_executor_factory()):
      result = secure_paillier_addition([[[1]]] * num_clients)
    self.assertAlmostEqual(result, num_clients)

  @parameterized.named_parameters(
      ('{}x{}'.format(r, c), r, c) for r, c in [(1, 1), (2, 2), (5, 5), (10, 10)])
  def test_secure_sum_larger_matrices(self, first_dim, second_dim):
    NUM_CLIENTS = 5
    shape = (first_dim, second_dim)
    input_tensor = np.ones(shape, dtype=np.int32)
    member_type = tff.TensorType(tf.int32, shape)
    @tff.federated_computation(tff.FederatedType(member_type, tff.CLIENTS))
    def secure_paillier_addition(x):
      return tff.federated_secure_sum(x, 64)

    with _install_executor(factory.local_paillier_executor_factory()):
      result = secure_paillier_addition([input_tensor] * NUM_CLIENTS)
    expected = input_tensor * NUM_CLIENTS
    np.testing.assert_almost_equal(result, expected)
