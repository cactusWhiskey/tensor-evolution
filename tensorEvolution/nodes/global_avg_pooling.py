"""Implements global avg pooling layers as nodes"""
import tensorflow as tf
from keras.engine.keras_tensor import KerasTensor

from tensorEvolution.nodes import node_utils
from tensorEvolution.nodes.tensor_node import TensorNode


class GlobalAveragePooling1DNode(TensorNode):
    """Implements global average pooling 1D layer as a genome node"""

    def __init__(self):
        super().__init__()
        self.accepts_variable_length_input = True

    def _clone(self):
        """Creates a new node."""
        return GlobalAveragePooling1DNode()

    def _build(self, layers_so_far: KerasTensor) -> KerasTensor:
        self.keras_tensor_input_name = layers_so_far.name
        pooling = tf.keras.layers.GlobalAveragePooling1D()
        layers_so_far = pooling(layers_so_far)
        return layers_so_far

    @staticmethod
    def create_random():
        """Creates a new node"""
        return tf.keras.layers.GlobalAveragePooling1D()

    def fix_input(self, layers_so_far: KerasTensor) -> KerasTensor:
        """Reshapes the input (if needed) so that it's valid for
        a 1D pooling layer (batches, steps, features)"""
        reshaped_input = node_utils.reshape_input(layers_so_far, 3)
        return reshaped_input

    def set_variable_input(self, is_variable_length: bool):
        """Override in subclasses as required. If the input to this node is of variable length,
         should the output be flagged as also being variable length?

         :param is_variable_length: is the input to this node of variable shape?"""

        self.has_variable_length_input = is_variable_length
