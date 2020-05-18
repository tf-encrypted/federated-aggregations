import tensorflow as tf
import tensorflow_federated as tff

from tff_aggregations import paillier

NUM_CLIENTS = 5

tff.framework.set_default_executor(paillier.local_paillier_executor_factory())

@tff.federated_computation(tff.FederatedType(tf.int32, tff.CLIENTS))
def secure_paillier_addition(x):
  bitwidth = 6  # assume upper-bound of 2^8 on summation result
  return tff.federated_secure_sum(x, bitwidth)

x = [1, 2, 3, 4, 5]  # adds up to 15
result = secure_paillier_addition(x)
print(result)
