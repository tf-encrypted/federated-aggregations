from tensorflow_federated.python.core.impl.compiler import placement_literals


PAILLIER = placement_literals.PlacementLiteral(
    'PAILLIER',
    'paillier',
    default_all_equal=True,
    description='An "unplacement" for Paillier computation.')
