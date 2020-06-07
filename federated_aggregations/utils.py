from typing import Tuple

import tensorflow_federated as tff
from tensorflow_federated.python.core.impl.types import placement_literals

PlacementPair = Tuple[
    placement_literals.PlacementLiteral,
    placement_literals.PlacementLiteral]

def materialize_computation_from_cache(
    factory_func, cache, arg_spec, **factory_kwargs):
  """Materializes a tf_computation generated by factory_func.
  
  If the cache contains a function with arguments of type arg_spec has already been generated by factory_func, it will"""
  
  hashable_arg_spec = tuple((
        *(repr(arg) for _, arg in factory_kwargs.items()),
        *(x.compact_representation() for x in arg_spec)))
  fn_proto, fn_type = cache.get(hashable_arg_spec, (None, None))
  if fn_proto is None:
    func = factory_func(*arg_spec, **factory_kwargs)
    fn_proto, fn_type = lift_to_computation_spec(
        func, input_arg_type=tff.NamedTupleType(arg_spec))
    cache[hashable_arg_spec] = (fn_proto, fn_type)
  return fn_proto, fn_type


def lift_to_computation_spec(tf_func, input_arg_type=None):
  """Determine computation definition & type spec from a tf_computation.
  
  If tf_func is polymorphic, first make it concrete with input_arg_type.
  """
  if not hasattr(tf_func, '_computation_proto'):
    if input_arg_type is None:
      raise ValueError('Polymorphic tf_computation requires arg_type to '
                       'become concrete.')
    tf_func = tf_func.fn_for_argument_type(input_arg_type)
  return tf_func._computation_proto, tf_func.type_signature
