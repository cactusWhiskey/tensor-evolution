"""Implements RNN layers as nodes"""
import random
import tensorflow as tf
from keras.engine.keras_tensor import KerasTensor

from tensorEvolution import evo_config
from tensorEvolution.nodes import node_utils
from tensorEvolution.nodes.tensor_node import TensorNode


class LstmNode(TensorNode):
    """Implements LSTM layer as a genome node"""

    def __init__(self, num_units: int, activation='relu'):
        super().__init__()
        self.num_units = num_units
        self.activation = activation
        self.return_sequences = False
        self.can_mutate = True
        self.node_allows_cache_training = False
        self.accepts_variable_length_input = True

    def _clone(self):
        """Crates a new node with the same number of units and
        the same activation as this node."""
        clone = LstmNode(self.num_units)
        return clone

    def _build(self, layers_so_far: KerasTensor) -> KerasTensor:
        self.keras_tensor_input_name = layers_so_far.name

        return tf.keras.layers.LSTM(self.num_units,
                                    return_sequences=self.return_sequences)(layers_so_far)

    @staticmethod
    def create_random():
        """
        Creates a new node with randomized number of
        units
        :return: new node
        """
        random_power = random.randint(0, evo_config.master_config.config['lstm_power_two'])
        units = 2 ** random_power
        return LstmNode(units)

    def mutate(self):
        """
        Mutates this node's number of units
        :return:
        """
        random_power = random.randint(0, evo_config.master_config.config['lstm_power_two'])
        units = 2 ** random_power
        self.num_units = units

    def fix_input(self, layers_so_far):
        return node_utils.reshape_input(layers_so_far, 3, simple_channel_addition=False)

    def set_variable_input(self, is_variable_length: bool):
        """Override in subclasses as required. If the input to this node is of variable length,
         should the output be flagged as also being variable length?
         :param is_variable_length: is the input to this node of variable shape?"""

        self.has_variable_length_input = is_variable_length
        if is_variable_length:
            if self.return_sequences:
                self.variable_output_size = True
            else:
                self.variable_output_size = False
        else:
            self.variable_output_size = False
