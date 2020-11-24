import autokeras as ak
from tensorflow.python.util import nest
from tf2cv.models.resnet import ResNet


class CustomResnetBlock(ak.Block):

    def __init__(self, in_size=(224, 224), in_channels=3, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.in_size = in_size
        self.layers_options = [[1, 1, 1, 1], [2, 1, 1, 1], [2, 2, 1, 1], [2, 2, 2, 1], [2, 2, 2, 2], [3, 3, 3, 3],
                          [3, 4, 6, 3]]

    def build(self, hp, inputs=None):
        input_node = nest.flatten(inputs)[0]

        # Get HP Params for network
        bottleneck = hp.Boolean('hp_bottleneck', default=False)

        layers_option_idx = list(range(len(self.layers_options)))
        layers_sel = hp.Choice('idx_layers', values=layers_option_idx)
        layers = self.layers_options[layers_sel]

        init_block_channels = 64
        channels_per_layers = [64, 128, 256, 512]

        if bottleneck:
            bottleneck_factor = 4
            channels_per_layers = [ci * bottleneck_factor for ci in channels_per_layers]

        channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

        width_scale = hp.Float('width_scale', min_value=0.5, max_value=1.5, step=0.1)

        if width_scale != 1.0:
            # it should not change the last block of last layer
            channels = [[int(cij * width_scale) if (i != len(channels) - 1) or (j != len(ci) - 1) else cij
                         for j, cij in enumerate(ci)] for i, ci in enumerate(channels)]
            init_block_channels = int(init_block_channels * width_scale)

        # Create layers
        net = ResNet(
            channels=channels,
            init_block_channels=init_block_channels,
            bottleneck=bottleneck,
            conv1_stride=True,
            in_channels=self.in_channels,
            in_size=self.in_size,
            use_with_ak_classification=True).features

        output_node = net(input_node)
        return output_node

