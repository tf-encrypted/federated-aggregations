import asyncio

import tensorflow_federated as tff
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl.compiler import placement_literals
from tensorflow_federated.python.core.impl.executors import executor_utils
from tensorflow_federated.python.core.impl.executors import federating_executor
from tf_encrypted.primitives import paillier

from tff_aggregations import channels
from tff_aggregations.paillier import placement as paillier_placement

class PaillierStrategy(federating_executor.CentralizedIntrinsicStrategy):  # TODO: change to tff.framework.CentralizedIntrinsicStrategy
  def __init__(self, parent_executor, channel_grid):
    super().__init__(parent_executor)
    self.channel_grid = channel_grid
  
  @classmethod
  def validate_executor_placements(cls, executor_placements):
    py_typecheck.check_type(executor_placements, dict)
    for k, v in executor_placements.items():
      if k is not None:
        py_typecheck.check_type(k, placement_literals.PlacementLiteral)
      py_typecheck.check_type(v, (list, tff.framework.Executor))
      if isinstance(v, list):
        for e in v:
          py_typecheck.check_type(e, tff.framework.Executor)
      for pl in [None, tff.SERVER, paillier_placement.PAILLIER]:
        if pl in executor_placements:
          ex = executor_placements[pl]
          if isinstance(ex, list):
            pl_cardinality = len(ex)
            if pl_cardinality != 1:
              raise ValueError(
                  'Unsupported cardinality for placement "{}": {}.'.format(
                      pl, pl_cardinality))
  
  async def _move(self, value, value_type, source_placement, target_placement):
    target_executors = self._get_child_executors(target_placement)
    channel = self.channel_grid[(source_placement, target_placement)]
    msg = await channel.send(value)
    res = await channel.receive(msg)
    return [await ex.create_value(res, value_type) for ex in target_executors]

  async def federated_secure_sum(self, arg):
    # TODO
    pass