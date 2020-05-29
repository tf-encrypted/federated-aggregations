import abc
import asyncio

import tensorflow_federated as tff
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.executors import federating_executor
from tensorflow_federated.python.core.impl.types import placement_literals
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
  requires_setup: bool = False

  def __init__(
      self,
      strategy,
      sender: tff.framework.Placement,
      receiver: tff.framework.Placement):
    self.strategy = strategy
    self.sender_placement = sender
    self.receiver_placement = receiver

  async def setup(self, placements): pass

  async def send(self, value):
    return value

  async def receive(self, value):
    if (self.sender_placement == tff.SERVER
        and self.receiver_placement == tff.CLIENTS):
      ex = self.strategy.executor
      return await ex._compute_intrinsic_federated_broadcast(value)
    return await self.strategy._place(value, self.receiver_placement)


class EasyBoxChannel(Channel):

  def __init__(
      self,
      strategy,
      sender: tff.framework.Placement,
      receiver: tff.framework.Placement):
    self.strategy = strategy
    self.sender_placement = sender
    self.receiver_placement = receiver

    self.key_references = key_store.KeyStore()
    self.requires_setup = True

    self._encrypt_tensor_fn = None
    self._decrypt_tensor_fn = None

  async def setup(self):
    await asyncio.gather(*[
        self._generate_keys(self.sender_placement),
        self._generate_keys(self.receiver_placement)
    ])
    await asyncio.gather(*[
        self._share_public_key(
            self.sender_placement, self.receiver_placement),
        self._share_public_key(
            self.receiver_placement, self.sender_placement)])

  async def send(self, value, sender=None, receiver=None):
    return await self._encrypt_values_on_sender(value, sender, receiver)

  async def receive(self, value, receiver=None, sender=None):
    return await self._decrypt_values_on_receiver(value, sender, receiver)

  async def _generate_keys(self, key_owner):
    py_typecheck.check_type(key_owner, placement_literals.PlacementLiteral)
    executors = self.strategy._get_child_executors(key_owner)

    @computations.tf_computation()
    def generate_keys():
      pk, sk = easy_box.gen_keypair()
      return pk.raw, sk.raw

    fn_type = generate_keys.type_signature
    fn = generate_keys._computation_proto

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
    fed_key_type = tff.FederatedType(key_type, key_receiver, all_equal=True)

    # we currently only support sharing n keys with 1 executor, 
    # or sharing 1 key with n executors
    if isinstance(val, list):
      # sharing n keys with 1 executor
      py_typecheck.check_len(children, 1)
      executor = children[0]
      vals = [executor.create_value(v, key_type) for v in val]
      vals_type = tff.NamedTupleType(
          [(None, fed_key_type) for _ in range(len(val))])
    else:
      # sharing 1 key with n executors
      # val is a single tensor
      vals = [c.create_value(val, key_type) for c in children]
      vals_type = fed_key_type
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

  @computations.tf_computation(plaintext_type, pk_rcv_type, sk_snd_type)
  def encrypt_tensor(plaintext, pk_rcv, sk_snd):

    pk_rcv = easy_box.PublicKey(pk_rcv)
    sk_snd = easy_box.PublicKey(sk_snd)

    nonce = easy_box.gen_nonce()
    ciphertext, mac = easy_box.seal_detached(plaintext, nonce, pk_rcv, sk_snd)

    return ciphertext.raw, mac.raw, nonce.raw

  return encrypt_tensor

def _decrypt_tensor(sender_values_type, pk_snd_type, sk_rcv_snd,
                    orig_sender_tensor_dtype):

  @computations.tf_computation(sender_values_type, pk_snd_type, sk_rcv_snd)
  def decrypt_tensor(sender_values, pk_snd, sk_rcv):

    ciphertext = easy_box.Ciphertext(sender_values[0])
    mac = easy_box.Mac(sender_values[1])
    nonce = easy_box.Nonce(sender_values[2])
    sk_rcv = easy_box.SecretKey(sk_rcv)
    pk_snd = easy_box.PublicKey(pk_snd)

    plaintext_recovered = easy_box.open_detached(ciphertext, mac, nonce,
                                                  pk_snd, sk_rcv,
                                                  orig_sender_tensor_dtype)

    return plaintext_recovered

  return decrypt_tensor
