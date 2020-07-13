import collections

import tensorflow_federated as tff
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.executors import federated_resolving_strategy
from tensorflow_federated.python.core.impl.types import placement_literals


class KeyStore:
  _default_store = lambda k: {'pk': None, 'sk': None}

  def __init__(self):
    self._key_store = collections.defaultdict(self._default_store)

  def get_key_pair(self, key_owner):
    key_owner_cache = self._get_keys(key_owner)
    return key_owner_cache['pk'], key_owner_cache['sk']

  def get_public_key(self, key_owner):
    return self._get_keys(key_owner)['pk']

  def get_secret_key(self, key_owner):
    return self._get_keys(key_owner)['sk']

  def _get_keys(self, key_owner):
    py_typecheck.check_type(key_owner, placement_literals.PlacementLiteral)
    return self._key_store[key_owner.name]

  def update_keys(self, key_owner, public_key=None, secret_key=None):
    key_owner_cache = self._get_keys(key_owner)
    if public_key is not None:
      self._check_key_type(public_key)
      key_owner_cache['pk'] = public_key
    if secret_key is not None:
      self._check_key_type(secret_key)
      key_owner_cache['sk'] = secret_key

  def _check_key_type(self, key):
    py_typecheck.check_type(key,
        federated_resolving_strategy.FederatedResolvingStrategyValue)
    py_typecheck.check_type(key.type_signature,
        (tff.NamedTupleType, tff.FederatedType))