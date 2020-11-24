import autokeras as ak
from tensorflow.python.util import nest
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D


class CustomGlobalPool(ak.Block):

    def build(self, hp, inputs=None):
        input_node = nest.flatten(inputs)[0]

        avg_type = hp.Choice('global_avg_type', values=[1, 2])
        if avg_type == 1:
            net = GlobalAveragePooling2D()
        else:
            net = GlobalMaxPooling2D()

        output_node = net(input_node)
        return output_node