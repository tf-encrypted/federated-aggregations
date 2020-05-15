import abc
from dataclasses import dataclass
from typing import Tuple, Dict

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.compiler import placement_literals


PlacementPair = Tuple[placement_literals.PlacementLiteral, placement_literals.PlacementLiteral]

class Channel(metaclass=abc.ABCMeta):
  @abc.abstractmethod
  async def send(self, value): pass

  @abc.abstractmethod
  async def receive(self, value): pass

  @abc.abstractmethod
  def setup(self, placements): pass

  @abc.abstractmethod
  def _keygen(self): pass

  @abc.abstractmethod
  def _keyexchange(self): pass

class StubChannel(Channel):
  def setup(self, placements): pass
  def _keygen(self): pass
  def _keyexchange(self): pass

  async def send(self, value):
    return value

  async def receive(self, value):
    return value


@dataclass
class ChannelGrid:
  channel_dict: Dict[PlacementPair, Channel]

  def __getitem__(self, placements: PlacementPair):
    py_typecheck.check_type(placements, tuple)
    py_typecheck.check_len(placements, 2)
    sorted_placements = tuple(sorted(placements, key=lambda p: p.uri))
    return self.channel_dict.get(sorted_placements)
