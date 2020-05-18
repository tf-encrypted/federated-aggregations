import asyncio

import tensorflow as tf
import tensorflow_federated as tff
from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl.compiler import placement_literals
from tensorflow_federated.python.core.impl.executors import executor_utils
from tensorflow_federated.python.core.impl.executors import federating_executor
from tf_encrypted.primitives import paillier

from tff_aggregations import channels
from tff_aggregations.paillier import placement as paillier_placement


def paillier_keygen(bitlength=None):
  # TODO this bitlength is currently fixed during executor factory construction
  # but it technically only needs to be fixed at each _paillier_setup phase.
  # would we ever need to rerun the setup phase with a new bitlength?

  @tff.tf_computation
  def _keygen():
    ek, dk = paillier.gen_keypair(bitlength)
    return ek.export(), dk.export()

  return _keygen


def _check_key_inputter(fn_value):
  fn_type = fn_value.type_signature
  py_typecheck.check_type(fn_type, computation_types.FunctionType)
  try:
    py_typecheck.check_len(fn_type.result, 2)
  except ValueError:
    raise ValueError(
        'Expected 2 elements in the output of key_inputter, '
        'found {}.'.format(len(fn_type.result)))
  ek_type, dk_type = fn_type.result
  py_typecheck.check_type(ek_type, tff.TensorType)
  py_typecheck.check_type(dk_type, computation_types.NamedTupleType)
  try:
    py_typecheck.check_len(dk_type, 2)
  except ValueError:
    raise ValueError(
        'Expected a two element tuple for the decryption key from '
        'key_inputter, found {} elements.'.format(len(fn_type.result)))
  py_typecheck.check_type(dk_type[0], tff.TensorType)
  py_typecheck.check_type(dk_type[1], tff.TensorType)


class PaillierStrategy(federating_executor.CentralizedIntrinsicStrategy):  # TODO: change to tff.framework.CentralizedIntrinsicStrategy
  def __init__(self, parent_executor, channel_grid, key_inputter):
    super().__init__(parent_executor)
    self.channel_grid = channel_grid
    self._requires_setup = True  # lazy key setup
    self._key_inputter = key_inputter

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
  
  async def _move(self, value, source_placement, target_placement):
    del source_placement
    return await self._place(value, target_placement)

  async def _paillier_setup(self):
    key_inputter = await self.executor.create_value(self._key_inputter)
    _check_key_inputter(key_inputter)
    fed_output = await self._eval(key_inputter, tff.SERVER, all_equal=True)
    output = fed_output.internal_representation[0]

    server_executor = self._get_child_executors(tff.SERVER, index=0)
    ek_ref = await server_executor.create_selection(output, index=0)
    dk_ref = await server_executor.create_selection(output, index=1)

    ek = federating_executor.FederatingExecutorValue(
        ek_ref, ek_ref.type_signature)
    self.encryption_key_clients = await self._move(
        ek, tff.SERVER, tff.CLIENTS)
    self.encryption_key_paillier = await self._move(
        ek, tff.SERVER, paillier_placement.PAILLIER)

    self.decryption_key = federating_executor.FederatingExecutorValue(dk_ref,
        tff.FederatedType(dk_ref.type_signature, tff.SERVER, all_equal=True))

  async def federated_secure_sum(self, arg):
    self._check_arg_is_anonymous_tuple(arg)
    py_typecheck.check_len(arg.internal_representation, 2)
    value_type = arg.type_signature[0]
    py_typecheck.check_type(value_type, tff.FederatedType)
    bitwidth_type = arg.type_signature[1]
    py_typecheck.check_type(bitwidth_type, tff.TensorType)

    if self._requires_setup:
      await self._paillier_setup()
      self._requires_setup = False

    value, bitwidth = arg

    # TODO
    #   1. If not done before, call self._paillier_setup()
    #       Results ek@CLIENTS, ek@PAILLIER, dk@SERVER
    #   2. encrypt(ek@CLIENTS, {value}@CLIENTS) -> {v_enc}@CLIENTS
    #   3. _move({v_enc}@CLIENTS, PAILLIER) -> <v_enc.>@PAILLIER
    #   4. Create call partial(paillier.add, ek@PAILLIER) -> paillier_binary_op
    #   5. Define res_enc = paillier.encrypt(0, ek@PAILLIER)
    #   6. For v_enc in <v_enc.>@PAILLIER:
    #       Create tuple (v_enc@PAILLIER, res_enc@PAILLIER) -> args@PAILLIER
    #       Create call paillier_binary_op(args@PAILLIER) -> res_enc@PAILLIER
    #   7. _move(res_enc@PAILLIER, SERVER) -> res_enc@SERVER
    #   8. Create tuple (dk@SERVER, res_enc@SERVER)
    #   9. Create call decrypt(dk@SERVER, res_enc@SERVER) -> res@SERVER
