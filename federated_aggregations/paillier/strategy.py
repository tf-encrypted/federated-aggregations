import asyncio
from collections import OrderedDict

import tensorflow as tf
import tensorflow_federated as tff
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl.executors import federating_executor
from tensorflow_federated.python.core.impl.types import placement_literals
from tf_encrypted.primitives import paillier

from federated_aggregations import utils
from federated_aggregations.channels import channel
from federated_aggregations.paillier import placement as paillier_placement


def paillier_keygen(bitlength=2048):

  @tff.tf_computation
  def _keygen():
    encryption_key, decryption_key = paillier.gen_keypair(bitlength)
    ek_raw = encryption_key.export(dtype=tf.string)
    dk_raw = decryption_key.export(dtype=tf.string)
    return ek_raw, dk_raw

  return _keygen


def _paillier_encrypt():

  @tff.tf_computation
  def _encrypt(encryption_key_raw, plaintext):
    ek = paillier.EncryptionKey(encryption_key_raw)
    ciphertext = paillier.encrypt(ek, plaintext)
    return ciphertext.export(dtype=tf.string)

  return _encrypt


def _paillier_decrypt(dtype):
  # FIXME figure out how different dtypes are chosen;
  #    maybe a dtype-specific fn cache on the strategy at runtime?
  # could also pass it directly to the tf_computation as a tf.string tensor,
  # but that feels like a hack

  @tff.tf_computation((tf.string, tf.string), tf.string, tf.string)
  def _decrypt(decryption_key_raw, encryption_key_raw, ciphertext_raw):
    dk = paillier.DecryptionKey(*decryption_key_raw)
    ek = paillier.EncryptionKey(encryption_key_raw)
    ciphertext = paillier.Ciphertext(ek, ciphertext_raw)
    return paillier.decrypt(dk, ciphertext, dtype)

  return _decrypt


def _paillier_sequence_sum():

  @tff.tf_computation
  def _sequence_sum(encryption_key_raw, *summands_raw):
    ek = paillier.EncryptionKey(encryption_key_raw)
    result = paillier.Ciphertext(ek, summands_raw[0])
    for summand in summands_raw[1:]:
      summand = paillier.Ciphertext(ek, summand)
      result = paillier.add(ek, result, summand, do_refresh=False)
    refreshed_sum = paillier.refresh(ek, result)
    return refreshed_sum.export(dtype=tf.string)

  return _sequence_sum


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


# TODO: change superclass to tff.framework.CentralizedIntrinsicStrategy
class PaillierStrategy(federating_executor.CentralizedIntrinsicStrategy):
  def __init__(self, parent_executor, channel_grid, key_inputter):
    super().__init__(parent_executor)
    self.channel_grid = channel_grid
    self._requires_setup = True  # lazy key setup
    self._key_inputter = key_inputter
    self._paillier_encryptor = _paillier_encrypt()
    self._paillier_decryptor_cache = OrderedDict()
    self._paillier_sequence_sum = _paillier_sequence_sum()

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
    if self.channel_grid.requires_setup:
      await self.channel_grid.setup_channels(self)
    channel = self.channel_grid[(source_placement, target_placement)]
    sent = await channel.send(value)
    return await channel.receive(sent)

  async def _paillier_setup(self):
    # Load paillier keys on server
    key_inputter = await self.executor.create_value(self._key_inputter)
    _check_key_inputter(key_inputter)
    fed_output = await self._eval(key_inputter, tff.SERVER, all_equal=True)
    output = fed_output.internal_representation[0]

    # Move encryption keys to PAILLIER service & CLIENTS
    server_executor = self._get_child_executors(tff.SERVER, index=0)
    ek_ref = await server_executor.create_selection(output, index=0)
    ek = federating_executor.FederatingExecutorValue(
        ek_ref, ek_ref.type_signature)
    self.encryption_key_clients = await self._move(
        ek, tff.SERVER, tff.CLIENTS)
    self.encryption_key_paillier = await self._move(
        ek, tff.SERVER, paillier_placement.PAILLIER)

    # Keep decryption key on server via formal placement
    dk_ref = await server_executor.create_selection(output, index=1)
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
      # Paillier setup phase
      await self._paillier_setup()
      self._requires_setup = False

    # Prepare paillier encryption key & client values for encryption eval
    client_keys = self.encryption_key_clients.internal_representation
    client_keys_type = self.encryption_key_clients.type_signature
    client_values = await self.executor.create_selection(arg, index=0)
    client_values_ir = client_values.internal_representation
    client_values_type = client_values.type_signature
    import pdb; pdb.set_trace()
    client_values_dtype = client_values_type.member.dtype
    zip_arg = federating_executor.FederatingExecutorValue(
        anonymous_tuple.AnonymousTuple(
              ((None, client_keys), (None, client_values_ir))),
        tff.NamedTupleType((client_keys_type, client_values_type)))
    encryptor_arg = await self.executor._compute_intrinsic_federated_zip_at_clients(
        zip_arg)

    # Map encryptor onto encryption key & client values
    encryptor_proto, encryptor_type = utils._lift_to_computation_spec(
        self._paillier_encryptor,
        input_arg_type=tff.NamedTupleType((
            client_keys_type.member, client_values_type.member)))
    map_arg = federating_executor.FederatingExecutorValue(
        anonymous_tuple.AnonymousTuple(
            ((None, encryptor_proto),
            (None, encryptor_arg.internal_representation))),
        tff.NamedTupleType((encryptor_type, encryptor_arg.type_signature)))
    encrypted_values = await self.executor._compute_intrinsic_federated_map(
        map_arg)

    # TODO compute _paillier_sequence_sum on encrypted_values
    encrypted_values = await self._move(encrypted_values,
        tff.CLIENTS, paillier_placement.PAILLIER)
    sum_proto, sum_type = _lift_to_computation_spec(
        self._paillier_sequence_sum,
        input_arg_type=tff.NamedTupleType((
            self.encryption_key_paillier.type_signature.member,
            encrypted_values.type_signature.member)))

    # TODO perform _paillier_decrypt on sum result
