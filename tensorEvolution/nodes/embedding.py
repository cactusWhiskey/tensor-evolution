"""Module which defines an embedding node in the genome"""
import random

import tensorflow as tf
from keras.engine.keras_tensor import KerasTensor

from tensorEvolution.nodes.tensor_node import TensorNode


class EmbeddingNode(TensorNode):
    """Implements embedding layer as a genome node"""

    def __init__(self, input_dim: int, output_dim: int, input_length=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_length = input_length
        self.variable_output_size = True
        self.accepts_variable_length_input = True
        self.has_variable_length_input = True
        self.can_mutate = True

    def _clone(self):
        """Creates a new embedding node with the same parameters as this node."""
        clone = EmbeddingNode(self.input_dim, self.output_dim, self.input_length)
        # clone.weights = copy.deepcopy(self.weights)
        return clone

    def _build(self, layers_so_far: KerasTensor) -> KerasTensor:
        self.keras_tensor_input_name = layers_so_far.name
        embedding = tf.keras.layers.Embedding(self.input_dim, self.output_dim,
                                              input_length=self.input_length)
        self.name = embedding.name
        return embedding(layers_so_far)

    @staticmethod
    def create_random():
        """Raises an error, not valid to create random nodes"""
        raise NotImplementedError("Embedding Node can't be created randomly")

    def mutate(self):
        """Mutates the embedding dimension"""
        output_dim = random.choice([2, 4, 8, 16, 32, 64])
        self.output_dim = output_dim
