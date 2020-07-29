from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow_federated.python.core.impl.executors import execution_context
from tensorflow_federated.python.core.impl.executors import executor_factory

from federated_aggregations.paillier import factory

def _temperature_sensor_example_next_fn():

  @tff.tf_computation(
      tff.SequenceType(tf.float32), tf.float32)
  def count_over(ds, t):
    return ds.reduce(
        np.float32(0), lambda n, x: n + tf.cast(tf.greater(x, t), tf.float32))

  @tff.tf_computation(tff.SequenceType(tf.float32))
  def count_total(ds):
    return ds.reduce(np.float32(0.0), lambda n, _: n + 1.0)

  @tff.federated_computation(
      tff.FederatedType(tff.SequenceType(tf.float32), tff.CLIENTS),
      tff.FederatedType(tf.float32, tff.SERVER))
  def comp(temperatures, threshold):
    return tff.federated_mean(
        tff.federated_map(
            count_over,
            tff.federated_zip(
                [temperatures,
                 tff.federated_broadcast(threshold)])),
        tff.federated_map(count_total, temperatures))

  return comp


def _install_executor(executor_factory_instance):
  context = execution_context.ExecutionContext(executor_factory_instance)
  return tff.framework.get_context_stack().install(context)


class ExecutorMock(mock.MagicMock, tff.framework.Executor):

  def create_value(self, *args):
    pass

  def create_call(self, *args):
    pass

  def create_selection(self, *args):
    pass

  def create_struct(self, *args):
    pass

  def close(self, *args):
    pass


class ExecutorStacksTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('paillier_executor_factory', factory.local_paillier_executor_factory),
  )
  def test_construction_with_no_args(self, executor_factory_fn):
    executor_factory_impl = executor_factory_fn()
    self.assertIsInstance(executor_factory_impl,
                          executor_factory.ExecutorFactoryImpl)

  @parameterized.named_parameters(
      ('paillier_executor_factory_none_clients',
       factory.local_paillier_executor_factory()),
      ('paillier_executor_factory_three_clients',
       factory.local_paillier_executor_factory(num_clients=3)),
  )
  def test_execution_of_temperature_sensor_example(self, executor):
    comp = _temperature_sensor_example_next_fn()
    to_float = lambda x: tf.cast(x, tf.float32)
    temperatures = [
        tf.data.Dataset.range(10).map(to_float),
        tf.data.Dataset.range(20).map(to_float),
        tf.data.Dataset.range(30).map(to_float),
    ]
    threshold = 15.0

    with _install_executor(executor):
      result = comp(temperatures, threshold)

    self.assertAlmostEqual(result, 8.333, places=3)

  @parameterized.named_parameters(
      ('paillier_executor_factory_none_clients',
       factory.local_paillier_executor_factory()),
      ('paillier_executor_factory_one_client',
       factory.local_paillier_executor_factory(num_clients=1)),
  )
  def test_execution_of_tensorflow(self, executor):

    @tff.tf_computation
    def comp():
      return tf.math.add(5, 5)

    with _install_executor(executor):
      result = comp()

    self.assertEqual(result, 10)


  @parameterized.named_parameters(
      ('paillier_executor_factory', factory.local_paillier_executor_factory),
  )
  def test_create_executor_raises_with_wrong_cardinalities(
      self, executor_factory_fn):
    executor_factory_impl = executor_factory_fn(num_clients=5)
    cardinalities = {
        tff.SERVER: 1,
        None: 1,
        tff.CLIENTS: 1,
    }
    with self.assertRaises(ValueError,):
      executor_factory_impl.create_executor(cardinalities)

if __name__ == '__main__':
  absltest.main()
