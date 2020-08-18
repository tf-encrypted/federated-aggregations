import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from federated_aggregations import paillier

NUM_CLIENTS = 5

paillier_factory = paillier.local_paillier_executor_factory(NUM_CLIENTS)
paillier_context = tff.framework.ExecutionContext(paillier_factory)
tff.framework.set_default_context(paillier_context)

@tff.federated_computation(tff.FederatedType(tff.TensorType(tf.int32, [2]), tff.CLIENTS), tff.TensorType(tf.int32))
def secure_paillier_addition(x, bitwidth):
  return tff.federated_secure_sum(x, bitwidth)

base = np.array([1, 2], np.int32)
x = [base + i for i in range(NUM_CLIENTS)]
result = secure_paillier_addition(x, 32)
print(result)
