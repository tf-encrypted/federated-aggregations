import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from tff_aggregations import paillier

NUM_CLIENTS = 5

tff.framework.set_default_executor(paillier.local_paillier_executor_factory())

@tff.federated_computation(tff.FederatedType(tff.TensorType(tf.int32, [2]), tff.CLIENTS))
def secure_paillier_addition(x):
  bitwidth = 6  # assume upper-bound of 2^8 on summation result
  return tff.federated_secure_sum(x, bitwidth)

base = np.array([1, 1], np.int32)
x = [base + i for i in range(NUM_CLIENTS)]  # adds up to 15
result = secure_paillier_addition(x)
print(result)
