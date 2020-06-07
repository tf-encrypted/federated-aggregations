import asyncio
from collections import OrderedDict

import tensorflow_federated as tff
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl.executors import federating_executor
from tensorflow_federated.python.core.impl.types import placement_literals
from tensorflow_federated.python.core.impl.types import type_analysis

from federated_aggregations import utils
from federated_aggregations.channels import channel
from federated_aggregations.paillier import placement as paillier_placement
from federated_aggregations.paillier import computations as paillier_comp


def _check_key_inputter(fn_value):
  fn_type = fn_value.type_signature
  py_typecheck.check_type(fn_type, tff.FunctionType)
  try:
    py_typecheck.check_len(fn_type.result, 2)
  except ValueError:
    raise ValueError(
        'Expected 2 elements in the output of key_inputter, '
        'found {}.'.format(len(fn_type.result)))
  ek_type, dk_type = fn_type.result
  py_typecheck.check_type(ek_type, tff.TensorType)
  py_typecheck.check_type(dk_type, tff.NamedTupleType)
  try:
    py_typecheck.check_len(dk_type, 2)
  except ValueError:
    raise ValueError(
        'Expected a two element tuple for the decryption key from '
        'key_inputter, found {} elements.'.format(len(fn_type.result)))
  py_typecheck.check_type(dk_type[0], tff.TensorType)
  py_typecheck.check_type(dk_type[1], tff.TensorType)


# TODO: change superclass to tff.framework.DefaultFederatingStrategy 
#       when the FederatingStrategy change lands on master
class PaillierStrategy(federating_executor.CentralizedIntrinsicStrategy):
  def __init__(self, parent_executor, channel_grid, key_inputter):
    super().__init__(parent_executor)
    self.channel_grid = channel_grid
    self._requires_setup = True  # lazy key setup
    self._key_inputter = key_inputter
    self._paillier_encryptor = paillier_comp.make_encryptor()
    self._paillier_decryptor_cache = {}
    self._paillier_sequence_sum = paillier_comp.make_sequence_sum()

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
    await self.channel_grid.setup_channels(self)
    channel = self.channel_grid[(source_placement, target_placement)]
    return await channel.transfer(value)

  async def _paillier_setup(self):
    # Load paillier keys on server
    key_inputter = await self.executor.create_value(self._key_inputter)
    _check_key_inputter(key_inputter)
    fed_output = await self._eval(key_inputter, tff.SERVER, all_equal=True)
    output = fed_output.internal_representation[0]
    # Broadcast encryption key to all placements
    server_executor = self._get_child_executors(tff.SERVER, index=0)
    ek_ref = await server_executor.create_selection(output, index=0)
    ek = federating_executor.FederatingExecutorValue(ek_ref,
        tff.FederatedType(ek_ref.type_signature, tff.SERVER, True))
    placed = await asyncio.gather(
      self._move(ek, tff.SERVER, tff.CLIENTS),
      self._move(ek, tff.SERVER, paillier_placement.PAILLIER))
    self.encryption_key_server = ek
    self.encryption_key_clients = placed[0]
    self.encryption_key_paillier = placed[1]
    # Keep decryption key on server with formal placement
    dk_ref = await server_executor.create_selection(output, index=1)
    self.decryption_key = federating_executor.FederatingExecutorValue(dk_ref,
        tff.FederatedType(dk_ref.type_signature, tff.SERVER, all_equal=True))

  async def federated_secure_sum(self, arg):
    self._check_arg_is_anonymous_tuple(arg)
    py_typecheck.check_len(arg.internal_representation, 2)
    value_type = arg.type_signature[0]
    type_analysis.check_federated_type(value_type, placement=tff.CLIENTS)
    py_typecheck.check_type(value_type.member, tff.TensorType)
    bitwidth_type = arg.type_signature[1]
    py_typecheck.check_type(bitwidth_type, tff.TensorType)
    # Stash input dtype for later
    input_tensor_dtype = value_type.member.dtype
    # Paillier setup phase
    if self._requires_setup:
      await self._paillier_setup()
      self._requires_setup = False
    # Encrypt summands on tff.CLIENTS
    clients_value = await self.executor.create_selection(arg, index=0)
    encrypted_values = await self._compute_paillier_encryption(
        self.encryption_key_clients, clients_value)
    # Perform Paillier sum on ciphertexts
    encrypted_values = await self._move(encrypted_values,
        tff.CLIENTS, paillier_placement.PAILLIER)
    encrypted_sum = await self._compute_paillier_sum(
        self.encryption_key_paillier, encrypted_values)
    # Move to server and decrypt the result
    encrypted_sum = await self._move(encrypted_sum,
        paillier_placement.PAILLIER, tff.SERVER)
    return await self._compute_paillier_decryption(
        self.decryption_key,
        self.encryption_key_server,
        encrypted_sum,
        export_dtype=input_tensor_dtype)

  async def _compute_paillier_encryption(self,
      client_encryption_keys: federating_executor.FederatingExecutorValue,
      clients_value: federating_executor.FederatingExecutorValue):
    client_children = self._get_child_executors(tff.CLIENTS)
    num_clients = len(client_children)
    py_typecheck.check_len(client_encryption_keys.internal_representation,
        num_clients)
    py_typecheck.check_len(clients_value.internal_representation, num_clients)
    encryptor_proto, encryptor_type = utils.lift_to_computation_spec(
        self._paillier_encryptor,
        input_arg_type=tff.NamedTupleType((
            client_encryption_keys.type_signature.member,
            clients_value.type_signature.member)))
    encryptor_fns = asyncio.gather(*[
        c.create_value(encryptor_proto, encryptor_type)
        for c in client_children])
    encryptor_args = asyncio.gather(*[c.create_tuple((ek, v)) for c, ek, v in zip(
        client_children,
        client_encryption_keys.internal_representation,
        clients_value.internal_representation)])
    encryptor_fns, encryptor_args = await asyncio.gather(
        encryptor_fns, encryptor_args)
    encrypted_values = await asyncio.gather(*[
      c.create_call(fn, arg) for c, fn, arg in zip(
          client_children, encryptor_fns, encryptor_args)])
    return federating_executor.FederatingExecutorValue(encrypted_values,
        tff.FederatedType(encryptor_type.result, tff.CLIENTS,
            clients_value.type_signature.all_equal))

  async def _compute_paillier_sum(self,
      encryption_key: federating_executor.FederatingExecutorValue,
      values: federating_executor.FederatingExecutorValue):
    paillier_child = self._get_child_executors(
        paillier_placement.PAILLIER, index=0)
    sum_proto, sum_type = utils.lift_to_computation_spec(
        self._paillier_sequence_sum,
        input_arg_type=tff.NamedTupleType((
            encryption_key.type_signature.member,
            tff.NamedTupleType([vt.member for vt in values.type_signature]))))
    sum_fn = paillier_child.create_value(sum_proto, sum_type)
    sum_arg = paillier_child.create_tuple((
        encryption_key.internal_representation,
        await paillier_child.create_tuple(values.internal_representation)))
    sum_fn, sum_arg = await asyncio.gather(sum_fn, sum_arg)
    encrypted_sum = await paillier_child.create_call(sum_fn, sum_arg)
    return federating_executor.FederatingExecutorValue(encrypted_sum,
        tff.FederatedType(sum_type.result, paillier_placement.PAILLIER, True))

  async def _compute_paillier_decryption(self,
      decryption_key: federating_executor.FederatingExecutorValue,
      encryption_key: federating_executor.FederatingExecutorValue,
      value: federating_executor.FederatingExecutorValue,
      export_dtype):
    server_child = self._get_child_executors(tff.SERVER, index=0)
    import pdb; pdb.set_trace()
    decryptor_arg_spec = (decryption_key.type_signature.member,
        encryption_key.type_signature.member,
        value.type_signature.member)
    decryptor_proto, decryptor_type = utils.materialize_computation_from_cache(
        paillier_comp.make_decryptor,
        self._paillier_decryptor_cache,
        arg_spec=decryptor_arg_spec,
        dtype=export_dtype)
    decryptor_fn = server_child.create_value(decryptor_proto, decryptor_type)
    decryptor_arg = server_child.create_tuple((
        decryption_key.internal_representation,
        encryption_key.internal_representation,
        value.internal_representation))
    decryptor_fn, decryptor_arg = await asyncio.gather(decryptor_fn, decryptor_arg)
    decrypted_value = await server_child.create_call(decryptor_fn, decryptor_arg)
    return federating_executor.FederatingExecutorValue(decrypted_value,
        tff.FederatedType(decryptor_type.result, tff.SERVER, True))
