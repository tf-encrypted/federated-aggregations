import abc
import asyncio
import itertools

import tensorflow_federated as tff
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.executors import federating_executor
from tensorflow_federated.python.core.impl.types import placement_literals
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_conversions
from tf_encrypted.primitives.sodium import easy_box

from federated_aggregations import utils
from federated_aggregations.channels import key_store


class Channel(metaclass=abc.ABCMeta):
  @abc.abstractmethod
  async def transfer(self, value):
    pass

  @abc.abstractmethod
  async def send(self, value, source, recipient):
    pass

  @abc.abstractmethod
  async def receive(self, value, source, recipient):
    pass

  @abc.abstractmethod
  async def setup(self):
    pass


class BaseChannel(Channel):
  def __init__(
      self,
      strategy,
      *placements: utils.PlacementPair):
    self.strategy = strategy
    self.placements = placements

  async def transfer(self, value):
    _check_value_placement(value, self.placements)
    sender_placement = value.type_signature.placement
    receiver_placement = _get_other_placement(
        sender_placement, self.placements)
    sent = await self.send(value, sender_placement, receiver_placement)
    rcv_children = self.strategy._get_child_executors(receiver_placement)
    message = await sent.compute()
    message_type = type_conversions.infer_type(message)
    if receiver_placement is tff.CLIENTS:
      if isinstance(message_type, tff.NamedTupleType):
        iterator = zip(rcv_children, message, message_type)
        member_type = message_type[0]
        all_equal = False
      else:
        iterator = zip(rcv_children,
            itertools.repeat(message),
            itertools.repeat(message_type))
        member_type = message_type
        all_equal = True
      message_value = federating_executor.FederatingExecutorValue(
        await asyncio.gather(*[c.create_value(m, t) for c, m, t in iterator]),
        tff.FederatedType(member_type, receiver_placement, all_equal))
    else:
      rcv_child = rcv_children[0]
      message_value = federating_executor.FederatingExecutorValue(
          anonymous_tuple.from_container(
              await asyncio.gather(*[
                  rcv_child.create_value(m, t)
                  for m, t in zip(message, message_type)])),
          tff.NamedTupleType([
              tff.FederatedType(mt, receiver_placement, True)
              for mt in message_type]))
    return await self.receive(message_value, sender_placement, receiver_placement)


class PlaintextChannel(BaseChannel):
  async def send(self, value, source, recipient):
    del source, recipient
    return value

  async def receive(self, value, source, recipient):
    del source, recipient
    return value

  async def setup(self):
    pass


class EasyBoxChannel(BaseChannel):
  def __init__(
      self,
      strategy,
      *placements: utils.PlacementPair):
    super().__init__(strategy, *placements)
    self.key_references = key_store.KeyStore()
    self._requires_setup = True
    self._key_generator = None  # lazy key generation
    self._encryptor_cache = {}
    self._decryptor_cache = {}

  async def send(self, value, sender_placement, receiver_placement):
    return await self._encrypt_values_on_sender(
        value, sender_placement, receiver_placement)

  async def receive(self, value, sender_placement, receiver_placement):
    return await self._decrypt_values_on_receiver(
        value, sender_placement, receiver_placement)

  async def setup(self):
    if self._requires_setup:
      p0, p1 = self.placements
      await asyncio.gather(*[
          self._generate_keys(p0),
          self._generate_keys(p1)])
      await asyncio.gather(*[
          self._share_public_key(p0, p1),
          self._share_public_key(p1, p0)])
      self._requires_setup = False

  async def _encrypt_values_on_sender(self, val, sender, receiver):
    # Check proper key placement
    sk_sender = self.key_references.get_secret_key(sender)
    pk_receiver = self.key_references.get_public_key(receiver)
    type_analysis.check_federated_type(val.type_signature, placement=sender)
    type_analysis.check_federated_type(sk_sender.type_signature, placement=sender)
    assert sk_sender.type_signature.placement is sender
    assert pk_receiver.type_signature.placement is sender
    # Materialize encryptor function definition & type spec
    input_type = val.type_signature.member
    self._input_type_cache = input_type
    pk_rcv_type = pk_receiver.type_signature.member
    sk_snd_type = sk_sender.type_signature.member
    if isinstance(pk_rcv_type, tff.NamedTupleType):
      pk_element_type = pk_rcv_type[0]
    else:
      pk_element_type = pk_rcv_type
    encryptor_arg_spec = (input_type, pk_element_type, sk_snd_type)
    encryptor_proto, encryptor_type = _materialize_function_with_cache(
        _encrypt_tensor, self._encryptor_cache, encryptor_arg_spec)
    # apply encryption on sender placement
    if receiver is tff.CLIENTS:
      ###
      # since CLIENTS is the only placement with cardinality>1,
      # we can safely assume receiver is CLIENTS and sender has cardinality=1
      ###
      # Case 1: receiver=CLIENTS
      #     plaintext: Fed(Tensor, sender, all_equal=True)
      #     pk_receiver: Fed(Tuple(Tensor), sender, all_equal=True)
      #     sk_sender: Fed(Tensor, sender, all_equal=True)
      #   Returns:
      #     encrypted_values: Tuple(Fed(Tensor, sender, all_equal=True))
      rcv_children = self.strategy._get_child_executors(receiver)
      snd_children = self.strategy._get_child_executors(sender)
      py_typecheck.check_len(snd_children, 1)
      snd_child = snd_children[0]
      # Check and prepare encryption arguments
      py_typecheck.check_len(val.internal_representation, 1)
      py_typecheck.check_len(pk_receiver.internal_representation,
          len(rcv_children))
      py_typecheck.check_len(sk_sender.internal_representation, 1)
      v = val.internal_representation[0]
      sk = sk_sender.internal_representation[0]
      # Encrypt values and return them
      encryptor_fn = await snd_child.create_value(encryptor_proto, encryptor_type)
      encryptor_args = await asyncio.gather(*[
          snd_child.create_tuple([v, this_pk, sk])
          for this_pk in pk_receiver.internal_representation])
      encrypted_values = await asyncio.gather(*[
          snd_child.create_call(encryptor_fn, arg) for arg in encryptor_args])
      encrypted_value_types = [encryptor_type.result] * len(encrypted_values)
      return federating_executor.FederatingExecutorValue(
          anonymous_tuple.from_container(encrypted_values),
          tff.NamedTupleType([tff.FederatedType(evt, sender, all_equal=False)
              for evt in encrypted_value_types]))
    # Case 2: sender=CLIENTS
    #     plaintext: Fed(Tensor, CLIENTS, all_equal=False)
    #     pk_receiver: Fed(Tensor, CLIENTS, all_equal=True)
    #     sk_sender: Fed(Tensor, CLIENTS, all_equal=False)
    #   Returns:
    #     encrypted_values: Fed(Tensor, CLIENTS, all_equal=False)
    # Check and prepare encryption arguments
    snd_children = self.strategy._get_child_executors(sender)
    rcv_children = self.strategy._get_child_executors(receiver)
    py_typecheck.check_len(rcv_children, 1)
    federated_value_internals = [
        val.internal_representation,
        pk_receiver.internal_representation,
        sk_sender.internal_representation]
    for v in federated_value_internals:
      py_typecheck.check_len(v, len(snd_children))
    # Encrypt values and return them
    encryptor_fns = asyncio.gather(*[
        snd_child.create_value(encryptor_proto, encryptor_type)
        for snd_child in snd_children])
    encryptor_args = asyncio.gather(*[
        snd_child.create_tuple([v, pk, sk])
        for v, pk, sk, snd_child in zip(
            *federated_value_internals, snd_children)])
    encryptor_fns, encryptor_args = await asyncio.gather(
        encryptor_fns, encryptor_args)
    encrypted_values = [
        snd_child.create_call(encryptor, arg)
        for encryptor, arg, snd_child in zip(
            encryptor_fns, encryptor_args, snd_children)]
    return federating_executor.FederatingExecutorValue(
        await asyncio.gather(*encrypted_values),
        tff.FederatedType(encryptor_type.result, sender,
            all_equal=val.type_signature.all_equal))

  async def _decrypt_values_on_receiver(self, val, sender, receiver):
    pk_sender = self.key_references.get_public_key(sender)
    sk_receiver = self.key_references.get_secret_key(receiver)
    type_analysis.check_federated_type(pk_sender.type_signature,
        placement=receiver)
    type_analysis.check_federated_type(sk_receiver.type_signature,
        placement=receiver)
    pk_snd_type = pk_sender.type_signature.member
    sk_rcv_type = sk_receiver.type_signature.member
    if receiver is tff.CLIENTS:
      input_type = val.type_signature.member
      # input_type[0] is a tff.TensorType, thus input_type represents a tuple 
      # needed for a single value to be decrypted
      py_typecheck.check_type(input_type[0], tff.TensorType)
      py_typecheck.check_type(pk_snd_type, tff.TensorType)
      input_element_type = input_type
      pk_element_type = pk_snd_type
    else:
      py_typecheck.check_type(val.type_signature, tff.NamedTupleType)
      type_analysis.check_federated_type(val.type_signature[0],
          placement=receiver, all_equal=True)
      input_type = val.type_signature[0].member
      # each input_type is a tuple needed for one value to be decrypted
      py_typecheck.check_type(input_type, tff.NamedTupleType)
      py_typecheck.check_type(pk_snd_type, tff.NamedTupleType)
      py_typecheck.check_len(input_type, len(pk_snd_type))
      input_element_type = input_type
      pk_element_type = pk_snd_type[0]
    decryptor_arg_spec = (input_element_type, pk_element_type, sk_rcv_type)
    decryptor_proto, decryptor_type = _materialize_function_with_cache(
        _decrypt_tensor,
        self._decryptor_cache,
        decryptor_arg_spec,
        orig_tensor_dtype=self._input_type_cache.dtype)
    if receiver is tff.CLIENTS:
      # we actually only care that cardinality of placement is > 1, so the
      # alternative conditions are assumed to be equivalent. that is,
      #      receiver != CLIENTS iff len(rcv_children) == 1
      rcv_children = self.strategy._get_child_executors(receiver)
      federated_value_internals = [
        val.internal_representation,
        pk_sender.internal_representation,
        sk_receiver.internal_representation]
      for fv in federated_value_internals:
        py_typecheck.check_len(fv, len(rcv_children))
      decryptor_fns = asyncio.gather(*[
        rcv_child.create_value(decryptor_proto, decryptor_type)
        for rcv_child in rcv_children])
      decryptor_args = asyncio.gather(*[
          rcv_child.create_tuple([v, pk, sk])
          for v, pk, sk, rcv_child in zip(
              *federated_value_internals, rcv_children)])
      decryptor_fns, decryptor_args = await asyncio.gather(
          decryptor_fns, decryptor_args)
      decrypted_values = [
          rcv_child.create_call(decryptor, arg)
          for decryptor, arg, rcv_child in zip(
              decryptor_fns, decryptor_args, rcv_children)]
      return federating_executor.FederatingExecutorValue(
          await asyncio.gather(*decrypted_values),
          tff.FederatedType(decryptor_type.result, receiver,
              all_equal=val.type_signature.all_equal))
    # sender==CLIENTS, receiver has cardinality 1
    #   val: Tuple(Fed(Tuple(Tensor), receiver, all_equal=True))
    #   pk_sender: Fed(Tuple(Tensor), receiver, all_equal=True)
    #   sk_receiver: Fed(Tensor, receiver, all_equal=True)
    # return decrypted_values: Fed(Tuple(Tensor), receiver, all_equal=True)
    snd_children = self.strategy._get_child_executors(sender)
    rcv_children = self.strategy._get_child_executors(receiver)
    py_typecheck.check_len(rcv_children, 1)
    rcv_child = rcv_children[0]
    py_typecheck.check_len(pk_sender.internal_representation, len(snd_children))
    py_typecheck.check_len(sk_receiver.internal_representation, 1)
    vals = val.internal_representation
    sk = sk_receiver.internal_representation[0]
    decryptor_fn = await rcv_child.create_value(decryptor_proto, decryptor_type)
    decryptor_args = await asyncio.gather(*[
        rcv_child.create_tuple([v, pk, sk])
        for v, pk in zip(vals, pk_sender.internal_representation)])
    decrypted_values = await asyncio.gather(*[
        rcv_child.create_call(decryptor_fn, arg)
        for arg in decryptor_args])
    decrypted_value_types = [decryptor_type.result] * len(decrypted_values)
    return federating_executor.FederatingExecutorValue(
        anonymous_tuple.from_container(decrypted_values),
        tff.NamedTupleType([
            tff.FederatedType(dvt, receiver, all_equal=True)
            for dvt in decrypted_value_types]))

  async def _generate_keys(self, key_owner):
    py_typecheck.check_type(key_owner, placement_literals.PlacementLiteral)
    executors = self.strategy._get_child_executors(key_owner)
    if self._key_generator is None:
      self._key_generator = _generate_keys()
    keygen, keygen_type = utils.lift_to_computation_spec(self._key_generator)
    pk_vals, sk_vals = [], []
    async def keygen_call(child):
      return await child.create_call(await child.create_value(
          keygen, keygen_type))
      
    key_generators = await asyncio.gather(*[
        keygen_call(executor) for executor in executors])
    public_keys = asyncio.gather(*[
        executor.create_selection(key_generator, 0) 
        for executor, key_generator in zip(executors, key_generators)])
    secret_keys = asyncio.gather(*[
        executor.create_selection(key_generator, 1) 
        for executor, key_generator in zip(executors, key_generators)])
    pk_vals, sk_vals = await asyncio.gather(public_keys, secret_keys)
    pk_type = pk_vals[0].type_signature
    sk_type = sk_vals[0].type_signature
    # all_equal whenever owner is non-CLIENTS singleton placement
    val_all_equal = len(executors) == 1 and key_owner != tff.CLIENTS
    pk_fed_val = federating_executor.FederatingExecutorValue(
        pk_vals, tff.FederatedType(pk_type, key_owner, val_all_equal))
    sk_fed_val = federating_executor.FederatingExecutorValue(
        sk_vals, tff.FederatedType(sk_type, key_owner, val_all_equal))
    self.key_references.update_keys(
        key_owner, public_key=pk_fed_val, secret_key=sk_fed_val)

  async def _share_public_key(self, key_owner, key_receiver):
    public_key = self.key_references.get_public_key(key_owner)
    children = self.strategy._get_child_executors(key_receiver)
    val = await public_key.compute()
    key_type = public_key.type_signature.member
    # we currently only support sharing n keys with 1 executor,
    # or sharing 1 key with n executors
    if isinstance(val, list):
      # sharing n keys with 1 executor
      py_typecheck.check_len(children, 1)
      executor = children[0]
      vals = [executor.create_value(v, key_type) for v in val]
      vals_type = tff.FederatedType(type_conversions.infer_type(val), key_receiver)
    else:
      # sharing 1 key with n executors
      # val is a single tensor
      vals = [c.create_value(val, key_type) for c in children]
      vals_type = tff.FederatedType(key_type, key_receiver, all_equal=True)
    public_key_rcv = federating_executor.FederatingExecutorValue(
        await asyncio.gather(*vals), vals_type)
    self.key_references.update_keys(key_owner, public_key=public_key_rcv)


def _materialize_function_with_cache(factory_func, cache, arg_spec, **factory_kwargs):
  hashable_arg_spec = tuple(
        x.compact_representation() for x in arg_spec)
  func = cache.get(hashable_arg_spec)
  if func is None:
    func = factory_func(*arg_spec, **factory_kwargs)
    cache[hashable_arg_spec] = func
  fn_proto, fn_type = utils.lift_to_computation_spec(
      func, input_arg_type=tff.NamedTupleType(arg_spec))
  return fn_proto, fn_type


def _encrypt_tensor(plaintext_type, pk_rcv_type, sk_snd_type):
  @tff.tf_computation
  def encrypt_tensor(plaintext, pk_rcv, sk_snd):
    pk_rcv = easy_box.PublicKey(pk_rcv)
    sk_snd = easy_box.SecretKey(sk_snd)
    nonce = easy_box.gen_nonce()
    ciphertext, mac = easy_box.seal_detached(plaintext, nonce, pk_rcv, sk_snd)
    return ciphertext.raw, mac.raw, nonce.raw

  return encrypt_tensor


def _decrypt_tensor(sender_values_type, pk_snd_type, sk_rcv_snd,
    orig_tensor_dtype):
  @tff.tf_computation(sender_values_type, pk_snd_type, sk_rcv_snd)
  def decrypt_tensor(sender_values, pk_snd, sk_rcv):
    ciphertext = easy_box.Ciphertext(sender_values[0])
    mac = easy_box.Mac(sender_values[1])
    nonce = easy_box.Nonce(sender_values[2])
    pk_snd = easy_box.PublicKey(pk_snd)
    sk_rcv = easy_box.SecretKey(sk_rcv)
    plaintext_recovered = easy_box.open_detached(
        ciphertext, mac, nonce, pk_snd, sk_rcv, orig_tensor_dtype)
    return plaintext_recovered

  return decrypt_tensor


def _generate_keys():
  @computations.tf_computation()
  def key_generator():
    pk, sk = easy_box.gen_keypair()
    return pk.raw, sk.raw

  return key_generator


def _get_other_placement(this_placement, both_placements):
  for p in both_placements:
    if p != this_placement:
      return p


def _check_value_placement(arg, placements):
  py_typecheck.check_type(arg, federating_executor.FederatingExecutorValue)
  py_typecheck.check_type(arg.type_signature, (tff.FederatedType, tff.NamedTupleType))
  value_type = arg.type_signature
  sender_placement = arg.type_signature.placement
  if sender_placement not in placements:
    raise ValueError(
        'Tried to send a value with placement {} through channel for '
        'placements ({},{}).'.format(
            str(sender_placement), *(str(p) for p in placements)))
