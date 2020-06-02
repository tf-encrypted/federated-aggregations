import abc
import asyncio

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
  async def send(self, value, sender=None, receiver=None):
    pass

  @abc.abstractmethod
  async def receive(self, value, sender=None, receiver=None):
    pass

  @abc.abstractmethod
  async def setup(self):
    pass


class PlaintextChannel(Channel):
  def __init__(
      self,
      strategy,
      *placements: utils.PlacementPair):
    self.strategy = strategy
    self.placements = placements

  async def setup(self):
    pass

  async def send(self, arg):
    _check_value_for_send(arg, self.placements)
    return arg

  async def receive(self, arg):
    sender_placement = arg.type_signature.placement
    receiver_placement = _get_other_placement(
        sender_placement, self.placements)
    rcv_children = self.strategy._get_child_executors(receiver_placement)
    val = await arg.compute()
    val_type = type_conversions.infer_type(val)
    return federating_executor.FederatingExecutorValue(
        await asyncio.gather(
            *[c.create_value(val, val_type) for c in rcv_children]),
        tff.FederatedType(val_type, receiver_placement, all_equal=True))


class EasyBoxChannel(Channel):
  def __init__(
      self,
      strategy,
      *placements: utils.PlacementPair):
    self.strategy = strategy
    self.placements = placements
    self.key_references = key_store.KeyStore()
    self._requires_setup = True
    self._key_generator = None  # lazy key generation

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

  async def send(self, value):
    _check_value_for_send(value, self.placements)
    sender_placement = value.type_signature.placement
    receiver_placement = _get_other_placement(
        sender_placement, self.placements)
    return await self._encrypt_values_on_sender(
        value, sender_placement, receiver_placement)

  async def receive(self, value, receiver=None, sender=None):
    return await self._decrypt_values_on_receiver(value, sender, receiver)

  async def _generate_keys(self, key_owner):
    py_typecheck.check_type(key_owner, placement_literals.PlacementLiteral)
    executors = self.strategy._get_child_executors(key_owner)
    if self._key_generator is None:
      self._key_generator = _generate_keys()
    fn, fn_type = utils.lift_to_computation_spec(self._key_generator)
    sk_vals = []
    pk_vals = []
    for executor in executors:
      key_generator = await executor.create_call(await executor.create_value(
          fn, fn_type))
      public_key = await executor.create_selection(key_generator, 0)
      secret_key = await executor.create_selection(key_generator, 1)
      pk_vals.append(public_key)
      sk_vals.append(secret_key)
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

  async def _encrypt_values_on_sender(self, val, sender=None, receiver=None):

    nb_senders = len(
        self.strategy._get_child_executors(self.sender_placement))

    if nb_senders == 1:
      input_tensor_type = val.type_signature
      self.orig_sender_tensor_dtype = input_tensor_type.dtype
    else:
      input_tensor_type = val[0].type_signature
      self.orig_sender_tensor_dtype = input_tensor_type.dtype

    pk_receiver = self.key_references.get_public_key(
        self.receiver_placement.name)
    sk_sender = self.key_references.get_secret_key(self.sender_placement.name)
    pk_rcv_type = pk_receiver.type_signature.member
    sk_snd_type = sk_sender.type_signature.member

    if not self._encrypt_tensor_fn:
      self._encrypt_tensor_fn = _encrypt_tensor(input_tensor_type,
                                                     pk_rcv_type, sk_snd_type)

    fn_type = self._encrypt_tensor_fn.type_signature
    fn = self._encrypt_tensor_fn._computation_proto

    if nb_senders == 1:
      tensor_type = val.type_signature
    else:
      tensor_type = val[0].type_signature

    val_type = tff.FederatedType(
        tensor_type, self.sender_placement, all_equal=False)

    val_key_zipped = await self._zip_val_key(
        self.sender_placement,
        val,
        pk_receiver,
        sk_sender,
        pk_index=receiver,
        sk_index=sender)

    val_encrypted = await self.strategy.federated_map(
        federating_executor.FederatingExecutorValue(
            anonymous_tuple.AnonymousTuple([(None, fn),
                                            (None, val_key_zipped)]),
            tff.NamedTupleType((fn_type, val_type))))

    if sender != None or receiver != None:
      return val_encrypted.internal_representation[0]
    else:
      return val_encrypted.internal_representation

  async def _decrypt_values_on_receiver(self, val, sender=None, receiver=None):

    pk_sender = self.key_references.get_public_key(self.sender_placement.name)
    sk_receiver = self.key_references.get_secret_key(
        self.receiver_placement.name)

    val = await self._zip_val_key(
        self.receiver_placement,
        val,
        pk_sender,
        sk_receiver,
        pk_index=sender,
        sk_index=receiver)

    sender_values_type = val[0].type_signature[0]
    pk_snd_type = val[0].type_signature[1]
    sk_snd_type = val[0].type_signature[2]

    if not self._decrypt_tensor_fn:
      self._decrypt_tensor_fn = _decrypt_tensor(
          sender_values_type, pk_snd_type, sk_snd_type,
          self.orig_sender_tensor_dtype)

    fn_type = self._decrypt_tensor_fn.type_signature
    fn = self._decrypt_tensor_fn._computation_proto

    val_type = tff.FederatedType(
        tff.TensorType(self.orig_sender_tensor_dtype),
        self.receiver_placement,
        all_equal=False)

    strat = self.strategy

    val_decrypted = await strat.federated_map(
        federating_executor.FederatingExecutorValue(
            anonymous_tuple.AnonymousTuple([(None, fn), (None, val)]),
            tff.NamedTupleType((fn_type, val_type))))

    if sender != None or receiver != None:
      return val_decrypted.internal_representation[0]
    else:
      return val_decrypted.internal_representation

  async def _zip_val_key(self,
                         placement,
                         vals,
                         pk_key,
                         sk_key,
                         pk_index=None,
                         sk_index=None):

    if isinstance(vals, list):
      val_type = tff.FederatedType(
          vals[0].type_signature, placement, all_equal=False)
    else:
      val_type = tff.FederatedType(
          vals.type_signature, placement, all_equal=False)
      vals = [vals]

    pk_key_vals = pk_key.internal_representation
    sk_key_vals = sk_key.internal_representation

    if pk_index != None:
      pk_key_vals = [pk_key_vals[pk_index]]

    if sk_index != None:
      sk_key_vals = [sk_key_vals[sk_index]]

    vals_key = federating_executor.FederatingExecutorValue(
        anonymous_tuple.AnonymousTuple([(None, vals), (None, pk_key_vals),
                                        (None, sk_key_vals)]),
        tff.NamedTupleType(
            (val_type, pk_key.type_signature, sk_key.type_signature)))

    vals_key_zipped = await self.strategy._zip(
        vals_key, placement, all_equal=False)

    return vals_key_zipped.internal_representation


def _encrypt_tensor(plaintext_type, pk_rcv_type, sk_snd_type):
  @tff.tf_computation
  def encrypt_tensor(plaintext, pk_rcv, sk_snd):
    pk_rcv = easy_box.PublicKey(pk_rcv)
    sk_snd = easy_box.PublicKey(sk_snd)
    nonce = easy_box.gen_nonce()
    ciphertext, mac = easy_box.seal_detached(plaintext, nonce, pk_rcv, sk_snd)
    return ciphertext.raw, mac.raw, nonce.raw

  return encrypt_tensor


def _decrypt_tensor(sender_values_type, pk_snd_type, sk_rcv_snd,
                    orig_sender_tensor_dtype):
  @tff.tf_computation(sender_values_type, pk_snd_type, sk_rcv_snd)
  def decrypt_tensor(sender_values, pk_snd, sk_rcv):
    ciphertext = easy_box.Ciphertext(sender_values[0])
    mac = easy_box.Mac(sender_values[1])
    nonce = easy_box.Nonce(sender_values[2])
    sk_rcv = easy_box.SecretKey(sk_rcv)
    pk_snd = easy_box.PublicKey(pk_snd)
    plaintext_recovered = easy_box.open_detached(
        ciphertext, mac, nonce, pk_snd, sk_rcv, orig_sender_tensor_dtype)
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


def _check_value_for_send(arg, placements):
  py_typecheck.check_type(arg, federating_executor.FederatingExecutorValue)
  py_typecheck.check_type(arg.type_signature, (tff.FederatedType, tff.NamedTupleType))
  value_type = arg.type_signature
  sender_placement = arg.type_signature.placement
  if sender_placement not in placements:
    raise ValueError(
        'Tried to send a value with placement {} through channel for '
        'placements ({},{}).'.format(
            str(sender_placement), *(str(p) for p in placements)))
