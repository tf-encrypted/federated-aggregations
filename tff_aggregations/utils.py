from tensorflow_federated.python.core.impl.types import placement_literals

PlacementPair = Tuple[
    placement_literals.PlacementLiteral,
    placement_literals.PlacementLiteral]

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
