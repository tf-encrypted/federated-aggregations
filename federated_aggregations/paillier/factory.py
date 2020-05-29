import functools

import tensorflow as tf
import tensorflow_federated as tff
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.executors import executor_stacks
from tensorflow_federated.python.core.impl.types import placement_literals

from federated_aggregations import channels
from federated_aggregations.paillier import placement as paillier_placement
from federated_aggregations.paillier import strategy as paillier_strategy


# separation of setup & sending -- setup phase only needs to happen once
#
# pass dict with metadata describing keys (optional metadata)
# Channel.setup uses this metadata to either load/generate keys via TF ops
# start with None: generate, eventually allow key metadata: load
#
# setup includes both generating/loading key pairs & exchanging the key pairs

# TODO: add more factory functions, including:
#   - composite executory factory
#   - worker pool factory (for use with RemoteExecutor)

def local_paillier_executor_factory(
    num_clients=None,
    num_client_executors=32,
    server_tf_device=None,
    aggregator_tf_device=None,
    client_tf_devices=tuple(),
):

  # TODO consider parameterizing this function with channel_grid
  channel_grid = channels.ChannelGrid({
      (tff.CLIENTS,
       paillier_placement.PAILLIER): channels.EasyBoxChannel,
      (tff.CLIENTS, 
       tff.SERVER): channels.PlaintextChannel,
      (paillier_placement.PAILLIER, 
       tff.SERVER): channels.PlaintextChannel})

  def intrinsic_strategy_fn(executor):
    return paillier_strategy.PaillierStrategy(executor, channel_grid,
        paillier_strategy.paillier_keygen(bitlength=2048))

  device_scheduler = _AggregatorDeviceScheduler(
      server_tf_device, aggregator_tf_device, client_tf_devices)

  stack_func = functools.partial(
      _create_paillier_federated_stack,
      num_client_executors=num_client_executors,
      aggregator_device_scheduler=device_scheduler,
      intrinsic_strategy_fn=intrinsic_strategy_fn)

  if num_clients is None:
    return _create_inferred_cardinality_factory(
        stack_func)

  return _create_explicit_cardinality_factory(
      num_clients, stack_func)

def _create_explicit_cardinality_factory(
    num_clients, stack_func):

  def _make_factory(cardinalities):
    num_requested_clients = cardinalities.get(tff.CLIENTS)
    if num_requested_clients is not None and num_requested_clients != num_clients:
      raise ValueError('Expected to construct an executor with {} clients, '
                       'but executor is hardcoded for {}'.format(
                           num_requested_clients, num_clients))
    return stack_func(num_clients)

  return tff.framework.create_executor_factory(_make_factory)


def _create_inferred_cardinality_factory(stack_func):

  def _make_factory(cardinalities):
    py_typecheck.check_type(cardinalities, dict)
    for k, v in cardinalities.items():
      py_typecheck.check_type(k, placement_literals.PlacementLiteral)
      if k not in [tff.CLIENTS, tff.SERVER, paillier_placement.PAILLIER]:
        raise ValueError('Unsupported placement: {}.'.format(k))
      if v <= 0:
        raise ValueError(
            'Cardinality must be at '
            'least one; you have passed {} for placement {}.'.format(v, k))
    return stack_func(cardinalities.get(tff.CLIENTS, 0))

  return tff.framework.create_executor_factory(_make_factory)


def _create_paillier_federated_stack(num_clients, num_client_executors,
    aggregator_device_scheduler, intrinsic_strategy_fn):

  client_bottom_stacks = [
      executor_stacks._create_bottom_stack(
          device=aggregator_device_scheduler.next_client_device())
      for _ in range(num_client_executors)
  ]
  executor_dict = {
      tff.CLIENTS: [
          client_bottom_stacks[k % len(client_bottom_stacks)]
          for k in range(num_clients)
      ],
      tff.SERVER: executor_stacks._create_bottom_stack(
          device=aggregator_device_scheduler.server_device()),
      paillier_placement.PAILLIER: executor_stacks._create_bottom_stack(
          device=aggregator_device_scheduler.aggregator_device()),
      None: executor_stacks._create_bottom_stack(
          device=aggregator_device_scheduler.server_device()),
  }
  return executor_stacks._complete_stack(
      tff.framework.FederatingExecutor(
          executor_dict, intrinsic_strategy_fn=intrinsic_strategy_fn))


class _AggregatorDeviceScheduler(executor_stacks._DeviceScheduler):
  """Assign server and clients to devices. Useful in multi-GPU environment."""

  def __init__(self, server_tf_device, aggregator_tf_device, client_tf_devices):
    """Initialize with server and client TF device placement.

    Args:
      server_tf_device: A `tf.config.LogicalDevice` to place server and other
        computation without explicit TFF placement.
      aggregator_tf_device: A `tf.config.LogicalDevice` to place Paillier
        aggregator computations. This is currently required to be CPU device
        by the Paillier primitives.
      client_tf_devices: List/tuple of `tf.config.LogicalDevice` to place
        clients for simulation. Possibly accelerators returned by
        `tf.config.list_logical_devices()`.
    """
    super().__init__(server_tf_device, client_tf_devices)
    if aggregator_tf_device is None:
      self._aggregator_device = None
    else:
      py_typecheck.check_type(aggregator_tf_device, tf.config.LogicalDevice)
      self._aggregator_device = aggregator_tf_device.name
  
  def aggregator_device(self):
    return self._aggregator_device
