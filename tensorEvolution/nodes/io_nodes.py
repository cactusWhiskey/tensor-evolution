"""Module which contains the input node, output node implementation"""
import tensorflow as tf
from keras.engine.keras_tensor import KerasTensor

from tensorEvolution import evo_config
from tensorEvolution.nodes.tensor_node import TensorNode


class InputNode(TensorNode):
    """Node which represents nn inputs in the genome"""

    def __init__(self, input_shape):
        super().__init__()

        if input_shape is None:
            self.input_shape = None
        else:
            for index, dim in enumerate(list(input_shape)):
                # When the population is first created from yaml config,
                # it's possible we have a "None" string
                if dim == "None":
                    input_shape[index] = None
            self.input_shape = tuple(input_shape)

        self.is_branch_root = True
        self.dtype = evo_config.master_config.input_dtype

        # have to do this check here (instead of in the loop above) to catch
        # the cases where an input node is crated via cloning or deserialization
        if self.input_shape is not None:
            for dim in self.input_shape:
                if dim is None:
                    self.variable_output_size = True

    def _clone(self):
        """
        Clones this input node
        :return: a new input node
        """
        return InputNode(self.input_shape)

    def _build(self, layers_so_far):
        """Note that by default, the build method sends layers so far as a parameter,
        but for an input node, it's just equal to None.
        We redefine it below, since it's a convenient variable
        name to use in intermediate steps. """

        layers_so_far = tf.keras.Input(shape=self.input_shape, dtype=self.dtype)
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

    def _clone(self):
        """Copies this node"""
        clone = OutputNode(self.num_outputs)
        # clone.weights = self.weights
        return clone

    def _build(self, layers_so_far) -> KerasTensor:
        layers_so_far = tf.keras.layers.Flatten()(layers_so_far)
        self.keras_tensor_input_name = layers_so_far.name
        outputs = tf.keras.layers.Dense(self.num_outputs,
                                        activation=None)
        self.name = outputs.name
        return outputs(layers_so_far)

    @staticmethod
    def create_random():
        """Raises an error, not valid to create random output nodes"""
        raise NotImplementedError("Output Node can't be created randomly")
