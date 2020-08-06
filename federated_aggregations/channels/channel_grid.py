import asyncio
from dataclasses import dataclass
from typing import Dict

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.types import placement_literals

from federated_aggregations.channels import channel
from federated_aggregations import utils

@dataclass
class ChannelGrid:
  """A container fully characterizing the network topology between placements.

  FederatingStrategy implementations can use this class to keep track of
  Channels between any pair of placements. It assumes that each Channel is
  two-way, i.e. it describes an unordered pair of placements.

  Attributes:
    requires_setup: Tracks whether the underlying channels of the grid have
        been set up; only some Channels require this setup phase.
  """
  _channel_dict: Dict[utils.PlacementPair, channel.Channel]
  requires_setup: bool = True

  async def setup_channels(self, strategy):
    if self.requires_setup:
      setup_steps = []
      tmp_channel_dict = {}
      for placement_pair, channel_factory in self._channel_dict.items():
        pair = tuple(sorted(placement_pair, key=lambda p: p.uri))
        channel = channel_factory(strategy, *pair)
        setup_steps.append(channel.setup())
        tmp_channel_dict[pair] = channel
      await asyncio.gather(*setup_steps)
      self._channel_dict = tmp_channel_dict
      self.requires_setup = False

  def __getitem__(self, placements: utils.PlacementPair):
    py_typecheck.check_type(placements, tuple)
    py_typecheck.check_len(placements, 2)
    sorted_placements = sorted(placements, key=lambda p: p.uri)
    return self._channel_dict.get(tuple(sorted_placements))
