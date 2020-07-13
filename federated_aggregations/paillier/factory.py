import functools
from typing import List, Callable, Optional, Sequence

import tensorflow as tf
import tensorflow_federated as tff
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.executors import eager_tf_executor
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.executors import executor_factory
from tensorflow_federated.python.core.impl.executors import executor_stacks
from tensorflow_federated.python.core.impl.executors import federating_executor
from tensorflow_federated.python.core.impl.executors import sizing_executor
from tensorflow_federated.python.core.impl.types import placement_literals

from federated_aggregations import channels
from federated_aggregations.paillier import placement as paillier_placement
from federated_aggregations.paillier import strategy as paillier_strategy
from federated_aggregations.paillier import computations as paillier_comp


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

class AggregatingUnplacedExecutorFactory(executor_stacks.UnplacedExecutorFactory):

  def __init__(
      self,
      *,
      use_caching: bool,
      server_device: Optional[tf.config.LogicalDevice] = None,
      aggregator_device: Optional[tf.config.LogicalDevice] = None,
      client_devices: Optional[Sequence[tf.config.LogicalDevice]] = ()):
    super().__init__(
        use_caching=use_caching,
        server_device=server_device,
        client_devices=client_devices)
    self._aggregator_device = aggregator_device

  def create_executor(
      self,
      *,
      cardinalities: Optional[executor_factory.CardinalitiesType] = None,
      placement: Optional[placement_literals.PlacementLiteral] = None
  ) -> executor_base.Executor:
    if cardinalities:
      raise ValueError(
          'Unplaced executors cannot accept nonempty cardinalities as '
          'arguments. Received cardinalities: {}.'.format(cardinalities))
    if placement == paillier_placement.PAILLIER:
      ex = eager_tf_executor.EagerTFExecutor(device=self._aggregator_device)
      return executor_stacks._wrap_executor_in_threading_stack(ex)
    return super().create_executor(
        cardinalities=cardinalities, placement=placement)


class PaillierAggregatingExecutorFactory(executor_stacks.FederatingExecutorFactory):

  def create_executor(
      self, cardinalities: executor_factory.CardinalitiesType
  ) -> executor_base.Executor:
    """Constructs a federated executor with requested cardinalities."""
    num_clients = self._validate_requested_clients(cardinalities)
    client_stacks = [
        self._unplaced_executor_factory.create_executor(
            cardinalities={}, placement=placement_literals.CLIENTS)
        for _ in range(self._num_client_executors)
    ]
    if self._use_sizing:
      client_stacks = [
          sizing_executor.SizingExecutor(ex) for ex in client_stacks
      ]
      self._sizing_executors.extend(client_stacks)
    paillier_stack = self._unplaced_executor_factory.create_executor(
        cardinalities={}, placement=paillier_placement.PAILLIER)
    if self._use_sizing:
      paillier_stack = sizing_executor.SizingExecutor(paillier_stack)
    # Set up secure channel between clients & Paillier executor
    secure_channel_grid = channels.ChannelGrid({
      (tff.CLIENTS,
      # TODO: replace this line with the one after it
       paillier_placement.PAILLIER): channels.PlaintextChannel,
      #  paillier_placement.PAILLIER): channels.EasyBoxChannel,
      (tff.CLIENTS, 
       tff.SERVER): channels.PlaintextChannel,
      (paillier_placement.PAILLIER, 
       tff.SERVER): channels.PlaintextChannel})
    # Build a FederatingStrategy factory for Paillier aggregation with the secure channel setup
    strategy_factory = paillier_strategy.PaillierAggregatingStrategy.factory(
        {
            placement_literals.CLIENTS: [
                client_stacks[k % len(client_stacks)]
                for k in range(num_clients)
            ],
            placement_literals.SERVER:
                self._unplaced_executor_factory.create_executor(
                    cardinalities={}, placement=placement_literals.SERVER),
            paillier_placement.PAILLIER: paillier_stack,
        },
        channel_grid=secure_channel_grid,
        # NOTE: we let the server generate it's own key here, but for proper
        # deployment we would want to supply a key verified by proper PKI
        key_inputter=paillier_comp.make_keygen(bitlength=2048))
    unplaced_executor = self._unplaced_executor_factory.create_executor(
        cardinalities={})
    executor = federating_executor.FederatingExecutor(
        strategy_factory, unplaced_executor)
    return executor_stacks._wrap_executor_in_threading_stack(executor)


def local_paillier_executor_factory(
    num_clients=None,
    num_client_executors=32,
    server_tf_device=None,
    aggregator_tf_device=None,
    client_tf_devices=tuple()):
  """Like tff.framework.local_executor_factory, but with Paillier aggregation.
  
  The resulting factory function does not implement composing executor stacks,
  so there is no max_fanout argument.

  Args:
    num_clients: The number of clients. If specified, the executor factory
      function returned by `local_paillier_executor_factory` will be configured
      to have exactly `num_clients` clients. If unspecified (`None`), then the
      function returned will attempt to infer cardinalities of all placements
      for which it is passed values.
    num_client_executors: The number of distinct client executors to run
      concurrently; executing more clients than this number results in
      multiple clients having their work pinned on a single executor in a
      synchronous fashion.
    server_tf_device: A `tf.config.LogicalDevice` to place server and
      other computation without explicit TFF placement.
    aggregator_tf_device: A `tf.config.LogicalDevice` to place computation
      of the Paillier aggregation. See README for a clearer description.
    client_tf_devices: List/tuple of `tf.config.LogicalDevice` to place clients
      for simulation. Possibly accelerators returned by
      `tf.config.list_logical_devices()`.
  """
  # TODO consider parameterizing this function with channel_grid
  if server_tf_device is not None:
    py_typecheck.check_type(server_tf_device, tf.config.LogicalDevice)
  py_typecheck.check_type(client_tf_devices, (tuple, list))
  py_typecheck.check_type(num_client_executors, int)
  if num_clients is not None:
    py_typecheck.check_type(num_clients, int)
  unplaced_ex_factory = AggregatingUnplacedExecutorFactory(
      use_caching=True,
      server_device=server_tf_device,
      client_devices=client_tf_devices)
  paillier_aggregating_executor_factory = PaillierAggregatingExecutorFactory(
      num_client_executors=num_client_executors,
      unplaced_ex_factory=unplaced_ex_factory,
      num_clients=num_clients,
      use_sizing=False)
  factory_fn = paillier_aggregating_executor_factory.create_executor
  return tff.framework.create_executor_factory(
      paillier_aggregating_executor_factory.create_executor)
