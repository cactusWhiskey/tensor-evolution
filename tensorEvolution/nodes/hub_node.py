"""Implements hub layers and nodes"""

import tensorflow_hub as hub
from keras.engine.keras_tensor import KerasTensor

from tensorEvolution.nodes.tensor_node import TensorNode


class HubNode(TensorNode):

    def __init__(self, url: str, input_shape=None, dtype=None, trainable=False):
        super().__init__()
        self.url = url
        self.dtype = dtype
        self.trainable = trainable
        self.input_shape = input_shape

    def _build(self, layers_so_far) -> KerasTensor:
        hub_layer = hub.KerasLayer(self.url, input_shape=self.input_shape,
                                   dtype=self.dtype, trainable=self.trainable)
        return hub_layer(layers_so_far)

    @staticmethod
    def create_random():
        raise NotImplementedError("Random creation of hub nodes not supported")

    def _clone(self):
        return HubNode(self.url)
