import tensorflow_federated as tff

from tff_aggregations.channels import channel as ch
from tff_aggregations.channels import channel_grid as grid
from tff_aggregations.channels import channel_test_utils as utils

class ChannelGridTest(utils.AsyncTestCase):
  def test_channel_grid_setup(self):
    channel_grid = grid.ChannelGrid(
        {(tff.CLIENTS, tff.SERVER): ch.PlaintextChannel})
    ex = utils.create_test_executor(channel_grid=channel_grid)
    self.run_sync(channel_grid.setup_channels(ex.intrinsic_strategy))

    channel = channel_grid[(tff.CLIENTS, tff.SERVER)]

    assert isinstance(channel, ch.PlaintextChannel)
