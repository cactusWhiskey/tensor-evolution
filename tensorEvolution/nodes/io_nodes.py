"""Module which contains the input node, output node implementation"""
import tensorflow as tf
from keras.engine.keras_tensor import KerasTensor

from tensorEvolution.nodes.tensor_node import TensorNode


class InputNode(TensorNode):
    """Node which represents nn inputs in the genome"""

    def __init__(self, input_shape: tuple):
        super().__init__()
        self.input_shape = input_shape
        self.is_branch_root = True

    def clone(self):
        """
        Clones this input node
        :return: a new input node
        """
        return InputNode(self.input_shape)

    def _build(self, layers_so_far):
        """Note that by default, the build method sends layers so far as a parameter,
        but for an input node, it's just equal to None.
        We redefine it below, since it's a convenient variable name to use in intermediate steps. """

        layers_so_far = tf.keras.Input(shape=self.input_shape)
        return layers_so_far

    def deserialize_cleanup(self):
        """
        converts the input shape back intp a tuple
        (it's a list whn read from json)
        :return:
        """
        self.input_shape = tuple(self.input_shape)

    @staticmethod
    def create_random():
        """Raises an error, it's not valid to randomly create inputs"""
        raise NotImplementedError("Input Node can't be created randomly")


class OutputNode(TensorNode):
    """Node which represents NN output layer in the genome. Ultimately this is
    a dense layer with a number of units specified by the problem being studied.
    Always has linear activation. Always flattens before the final dense layer."""

    def __init__(self, num_outputs):
        super().__init__()
        self.num_outputs = num_outputs
        self.node_allows_cache_training = True

    def clone(self):
        """Copies this node"""
        clone = OutputNode(self.num_outputs)
        clone.weights = self.weights
        return clone

    def _build(self, layers_so_far) -> KerasTensor:
        layers_so_far = tf.keras.layers.Flatten()(layers_so_far)
        self.keras_tensor_input_name = layers_so_far.name
        return tf.keras.layers.Dense(self.num_outputs,
                                     activation=None)(layers_so_far)

    @staticmethod
    def create_random():
        """Raises an error, not valid to create random output nodes"""
        raise NotImplementedError("Output Node can't be created randomly")
