from tensorflow_federated.python.core.impl.types import placement_literals

AGGREGATOR = placement_literals.PlacementLiteral(
    'AGGREGATOR',
    'aggregator',
    default_all_equal=True,
    description='An "unplacement" for aggregations.')
