import abc
from dataclasses import dataclass
from typing import Tuple, Dict

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl.compiler import placement_literals
from tensorflow_federated.python.core.impl.executors import federating_executor


PlacementPair = Tuple[placement_literals.PlacementLiteral, placement_literals.PlacementLiteral]


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


class EasyBoxChannel(channel_base.Channel):

  def __init__(self, parent_executor, sender_placement, receiver_placement):

    self.parent_executor = parent_executor
    self.sender_placement = sender_placement
    self.receiver_placement = receiver_placement

    self.key_references = KeyStore()
    self._is_channel_setup = False

    self._encrypt_tensor_fn = None
    self._decrypt_tensor_fn = None

  async def setup(self):

    if not self._is_channel_setup:
      await asyncio.gather(*[
          self._generate_keys(self.sender_placement),
          self._generate_keys(self.receiver_placement)
      ])
      await asyncio.gather(*[
          self._share_public_keys(self.sender_placement,
                                  self.receiver_placement),
          self._share_public_keys(self.receiver_placement,
                                  self.sender_placement)
      ])

      self._is_channel_setup = True

  async def send(self, value, sender=None, receiver=None):
    return await self._encrypt_values_on_sender(value, sender, receiver)

  async def receive(self, value, receiver=None, sender=None):
    return await self._decrypt_values_on_receiver(value, sender, receiver)

  async def _generate_keys(self, key_owner):

    @computations.tf_computation()
    def generate_keys():
      pk, sk = easy_box.gen_keypair()
      return pk.raw, sk.raw

    fn_type = generate_keys.type_signature
    fn = generate_keys._computation_proto

    executors = self.parent_executor._get_child_executors(key_owner)

    nb_executors = len(executors)
    sk_vals = []
    pk_vals = []

    for executor in executors:
      key_generator = await executor.create_call(await executor.create_value(
          fn, fn_type))

      pk = await executor.create_selection(key_generator, 0)
      sk = await executor.create_selection(key_generator, 1)

      pk_vals.append(pk)
      sk_vals.append(sk)

    # Store list of EagerValue created by executor.create_call
    # in a FederatingExecutorValue with the key onwer placement
    sk_fed_vals = await self._place_keys(sk_vals, key_owner)

    self.key_references.add_keys(key_owner.name, pk_vals, sk_fed_vals)

  async def _share_public_keys(self, key_owner, send_pks_to):
    pk = self.key_references.get_public_key(key_owner.name)
    pk_fed_vals = await self._place_keys(pk, send_pks_to)
    self.key_references.update_keys(key_owner.name, pk_fed_vals)

  async def _encrypt_values_on_sender(self, val, sender=None, receiver=None):

    nb_senders = len(
        self.parent_executor._get_child_executors(self.sender_placement))

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
      self._encrypt_tensor_fn = self._encrypt_tensor(input_tensor_type,
                                                     pk_rcv_type, sk_snd_type)

    fn_type = self._encrypt_tensor_fn.type_signature
    fn = self._encrypt_tensor_fn._computation_proto

    if nb_senders == 1:
      tensor_type = val.type_signature
    else:
      tensor_type = val[0].type_signature

    val_type = computation_types.FederatedType(
        tensor_type, self.sender_placement, all_equal=False)

    val_key_zipped = await self._zip_val_key(
        self.sender_placement,
        val,
        pk_receiver,
        sk_sender,
        pk_index=receiver,
        sk_index=sender)

    # NOTE probably won't always be fed_ex in future design
    fed_ex = self.parent_executor.federating_executor

    val_encrypted = await fed_ex._compute_intrinsic_federated_map(
        FederatingExecutorValue(
            anonymous_tuple.AnonymousTuple([(None, fn),
                                            (None, val_key_zipped)]),
            computation_types.NamedTupleType((fn_type, val_type))))

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
      self._decrypt_tensor_fn = self._decrypt_tensor(
          sender_values_type, pk_snd_type, sk_snd_type,
          self.orig_sender_tensor_dtype)

    fn_type = self._decrypt_tensor_fn.type_signature
    fn = self._decrypt_tensor_fn._computation_proto

    val_type = computation_types.FederatedType(
        computation_types.TensorType(self.orig_sender_tensor_dtype),
        self.receiver_placement,
        all_equal=False)

    # NOTE probably won't always be fed_ex in future design
    fed_ex = self.parent_executor.federating_executor

    val_decrypted = await fed_ex._compute_intrinsic_federated_map(
        FederatingExecutorValue(
            anonymous_tuple.AnonymousTuple([(None, fn), (None, val)]),
            computation_types.NamedTupleType((fn_type, val_type))))

    if sender != None or receiver != None:
      return val_decrypted.internal_representation[0]
    else:
      return val_decrypted.internal_representation

  def _encrypt_tensor(self, plaintext_type, pk_rcv_type, sk_snd_type):

    @computations.tf_computation(plaintext_type, pk_rcv_type, sk_snd_type)
    def encrypt_tensor(plaintext, pk_rcv, sk_snd):

      pk_rcv = easy_box.PublicKey(pk_rcv)
      sk_snd = easy_box.PublicKey(sk_snd)

      nonce = easy_box.gen_nonce()
      ciphertext, mac = easy_box.seal_detached(plaintext, nonce, pk_rcv, sk_snd)

      return ciphertext.raw, mac.raw, nonce.raw

    return encrypt_tensor

  def _decrypt_tensor(self, sender_values_type, pk_snd_type, sk_rcv_snd,
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

  async def _zip_val_key(self,
                         placement,
                         vals,
                         pk_key,
                         sk_key,
                         pk_index=None,
                         sk_index=None):

    if isinstance(vals, list):
      val_type = computation_types.FederatedType(
          vals[0].type_signature, placement, all_equal=False)
    else:
      val_type = computation_types.FederatedType(
          vals.type_signature, placement, all_equal=False)
      vals = [vals]

    pk_key_vals = pk_key.internal_representation
    sk_key_vals = sk_key.internal_representation

    if pk_index != None:
      pk_key_vals = [pk_key_vals[pk_index]]

    if sk_index != None:
      sk_key_vals = [sk_key_vals[sk_index]]

    vals_key = FederatingExecutorValue(
        anonymous_tuple.AnonymousTuple([(None, vals), (None, pk_key_vals),
                                        (None, sk_key_vals)]),
        computation_types.NamedTupleType(
            (val_type, pk_key.type_signature, sk_key.type_signature)))

    vals_key_zipped = await self.parent_executor._zip(
        vals_key, placement, all_equal=False)

    return vals_key_zipped.internal_representation

  async def _place_keys(self, keys, placement):

    py_typecheck.check_type(placement, placement_literals.PlacementLiteral)
    children = self.parent_executor._get_child_executors(placement)

    # Scenario: there are as many keys as exectutors. For example
    # there are 3 clients and each should have a secret key
    if len(keys) == len(children):
      keys_type_signature = keys[0].type_signature
      return FederatingExecutorValue(
          await asyncio.gather(*[
              c.create_value(await keys[i].compute(), keys_type_signature)
              for (i, c) in enumerate(children)
          ]),
          computation_types.FederatedType(
              keys_type_signature, placement, all_equal=False))
    # Scenario: there are more keys than exectutors. For example
    # there are 3 clients and each have a public key. Each client wants
    # to share its key to the same aggregator.
    elif (len(children) == 1) & (len(children) < len(keys)):
      keys_type_signature = keys[0].type_signature
      child = children[0]
      return FederatingExecutorValue(
          await asyncio.gather(*[
              child.create_value(await k.compute(), keys_type_signature)
              for k in keys
          ]),
          computation_types.FederatedType(
              keys_type_signature, placement, all_equal=False))
    # Scenario: there are more exectutors than keys. For example
    # there is an aggregator with one public key. The aggregator
    # wants to share the samer public key to 3 different clients.
    elif (len(keys) == 1) & (len(children) > len(keys)):
      keys_type_signature = keys[0].type_signature
      return FederatingExecutorValue(
          await asyncio.gather(*[
              c.create_value(await keys[0].compute(), keys_type_signature)
              for c in children
          ]),
          computation_types.FederatedType(
              keys_type_signature, placement, all_equal=True))


@dataclass
class ChannelGrid:
  channel_dict: Dict[PlacementPair, Channel]

  def __getitem__(self, placements: PlacementPair):
    py_typecheck.check_type(placements, tuple)
    py_typecheck.check_len(placements, 2)
    sorted_placements = sorted(placements, key=lambda p: p.uri)
    return self.channel_dict.get(tuple(sorted_placements))


class KeyStore:
  def __init__(self):
    self.key_store = {}

  def add_keys(self, key_owner, pk, sk):
    self.key_store[key_owner] = {'pk': pk, 'sk': sk}

  def get_public_key(self, key_owner):
    return self.key_store[key_owner]['pk']

  def get_secret_key(self, key_owner):
    return self.key_store[key_owner]['sk']

  def update_keys(self, key_owner, pk=None, sk=None):
    if pk:
      self.key_store[key_owner]['pk'] = pk
    if sk:
      self.key_store[key_owner]['sk'] = sk