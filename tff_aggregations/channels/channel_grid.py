from dataclasses import dataclass
from typing import Dict, Tuple

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.types import placement_literals

from tff_aggregations.channels import channel

PlacementPair = Tuple[
    placement_literals.PlacementLiteral,
    placement_literals.PlacementLiteral]

@dataclass
class ChannelGrid:
  _channel_dict: Dict[PlacementPair, channel.Channel]
  requires_setup: bool = True

  async def setup_channels(self, strategy):
    if self.requires_setup:
      for placement_pair in self._channel_dict:
        channel_cls = self._channel_dict[placement_pair]
        channel = channel_cls(strategy, *placement_pair)
        if channel.requires_setup:
          await channel.setup()
          channel.requires_setup = False
        self._channel_dict[placement_pair] = channel
      self.requires_setup = False

  def __getitem__(self, placements: PlacementPair):
    py_typecheck.check_type(placements, tuple)
    py_typecheck.check_len(placements, 2)
    sorted_placements = sorted(placements, key=lambda p: p.uri)
    return self._channel_dict.get(tuple(sorted_placements))
