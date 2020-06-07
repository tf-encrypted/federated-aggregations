import asyncio
from dataclasses import dataclass
from typing import Dict

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.types import placement_literals

from federated_aggregations.channels import channel
from federated_aggregations import utils

@dataclass
class ChannelGrid:
  _channel_dict: Dict[utils.PlacementPair, channel.Channel]
  requires_setup: bool = True

  async def setup_channels(self, strategy):
    if self.requires_setup:
      setup_steps = []
      tmp_channel_dict = {}
      for placement_pair, channel_factory in self._channel_dict.items():
        channel = channel_factory(strategy, *placement_pair)
        setup_steps.append(channel.setup())
        tmp_channel_dict[placement_pair] = channel
      await asyncio.gather(*setup_steps)
      self._channel_dict = tmp_channel_dict
      self.requires_setup = False

  def __getitem__(self, placements: utils.PlacementPair):
    py_typecheck.check_type(placements, tuple)
    py_typecheck.check_len(placements, 2)
    sorted_placements = sorted(placements, key=lambda p: p.uri)
    return self._channel_dict.get(tuple(sorted_placements))
