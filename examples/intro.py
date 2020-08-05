import tensorflow as tf
import tensorflow_federated as tff
from federated_aggregations import paillier

NUM_CLIENTS = 5
paillier_factory = paillier.local_paillier_executor_factory()
tff.framework.set_default_executor(paillier_factory)

@tff.federated_computation(
    tff.FederatedType(tff.TensorType(tf.int32, [2]), tff.CLIENTS))
def secure_paillier_addition(x):
  return tff.federated_secure_sum(x, 32)

x = [[i, i + 1] for i in range(NUM_CLIENTS)]
result = secure_paillier_addition(x)
print(result)
