"""Module which implements flatten, dense, addition, relu, batchnorm, dropout"""
import copy
import random

import tensorflow as tf
from keras.engine.keras_tensor import KerasTensor
from networkx import MultiDiGraph

from tensorEvolution import evo_config
from tensorEvolution.nodes import node_utils
from tensorEvolution.nodes.tensor_node import TensorNode


class FlattenNode(TensorNode):
    """Implements flatten layer as a genome node"""

    def _build(self, layers_so_far: KerasTensor):
        return tf.keras.layers.Flatten()(layers_so_far)

    def _clone(self):
        """Returns a new instance"""
        return FlattenNode()

    @staticmethod
    def create_random():
        """Returns a new instance"""
        return FlattenNode()


class AdditionNode(TensorNode):
    """Implements addition layer as a genome node"""

    def __init__(self):
        super().__init__()
        self.is_branch_termination = True

    def _clone(self):
        """Builds a new addition node"""
        clone = AdditionNode()
        # clone.saved_layers = copy.deepcopy(self.saved_layers)
        return clone

    def _call_parents(self, all_nodes: dict, graph: MultiDiGraph):
        parents = self.get_parents(all_nodes, graph)
        return [parent(all_nodes, graph) for parent in parents]

    def _build(self, layers_so_far: list) -> KerasTensor:
        return tf.keras.layers.Add()(layers_so_far)

    @staticmethod
    def create_random():
        """Just creates a new node since addition node has no
        internal state to randomize"""
        return AdditionNode()

    def fix_input(self, layers_so_far: list) -> list:
        """
        The inputs to the addition node can be tensors of any shape.
        This method adds flatten, dense, and reshape layers
        as required to force the two inputs to have
        identical shapes.
        :param layers_so_far: either a list of the
        different layers feeding into this addition node,
        or possibly just a single input (because it's
        possible that a single parent node feeds into both inputs
        of this addition node).
        :return: list of (minimum of) two tensors of identical shape        """

        # if there is only one branch return it twice
        if len(layers_so_far) == 1:
            return [layers_so_far[0], layers_so_far[0]]

        return node_utils.make_shapes_same(layers_so_far)


class DenseNode(TensorNode):
    """Implements dense layer as a genome node"""

    def __init__(self, num_units: int, activation='relu'):
        super().__init__()
        self.num_units = num_units
        self.activation = activation
        self.can_mutate = True
        self.node_allows_cache_training = True
        self.kernel_regularizer = evo_config.master_config.random_regularizer()

    def _clone(self):
        """Creates a new dense node with the same number of units and
        the same activation as this node."""
        clone = DenseNode(self.num_units, self.activation)
        # clone.weights = copy.deepcopy(self.weights)
        return clone

    def _build(self, layers_so_far: KerasTensor) -> KerasTensor:
        self.keras_tensor_input_name = layers_so_far.name
        regularizer = evo_config.EvoConfig.build_regularizer(self.kernel_regularizer)
        dense = tf.keras.layers.Dense(self.num_units,
                                      activation=self.activation,
                                      kernel_regularizer=regularizer)
        self.name = dense.name
        return dense(layers_so_far)

    @staticmethod
    def create_random():
        """
        Creates a new dense node with randomized number of
        units
        :return: new dense node
        """
        random_power = random.randint(0, evo_config.master_config.config['dense_max_power_two'])
        units = 2 ** random_power
        return DenseNode(units)

    def mutate(self):
        """
        Mutates this node's number of units and regularizer
        :return:
        """
        random_power = random.randint(0, evo_config.master_config.config['dense_max_power_two'])
        units = 2 ** random_power
        self.num_units = units
        self.weights = None
        self.kernel_regularizer = evo_config.master_config.random_regularizer()


class ReluNode(TensorNode):
    """Implements relu layer as genome node"""

    def _build(self, layers_so_far: KerasTensor) -> KerasTensor:
        return tf.keras.layers.ReLU()(layers_so_far)

    def _clone(self):
        """Crates new relu node"""
        return ReluNode()

    @staticmethod
    def create_random():
        """Crates new relu node"""
        return ReluNode()


class BatchNormNode(TensorNode):
    """Implements batch norm node type"""

    def _build(self, layers_so_far: KerasTensor) -> KerasTensor:
        return tf.keras.layers.BatchNormalization()(layers_so_far)

    def _clone(self):
        """Creates and returns new batch norm node"""
        return BatchNormNode()

    @staticmethod
    def create_random():
        """Creates and returns new batch norm node"""
        return BatchNormNode()


class DropoutNode(TensorNode):
    """Implements dropout layer"""

    def __init__(self):
        super().__init__()
        self.rate = random.uniform(0.0, 0.9)
        self.can_mutate = True
        self.accepts_variable_length_input = True

    @staticmethod
    def create_random():
        """Creates and returns a new DropoutNode"""
        return DropoutNode()

    def _clone(self):
        """Creates a new DropoutNode and sets its rate equal to this node's rate"""
        new_node = DropoutNode()
        # new_node.rate = copy.copy(self.rate)
        return new_node

    def _build(self, layers_so_far) -> KerasTensor:
        return tf.keras.layers.Dropout(self.rate)(layers_so_far)

    def mutate(self):
        self.rate = random.uniform(0.0, 0.9)


class PreprocessingNode(TensorNode):
    """Node which holds preprocessing layers"""

    def __init__(self, preprocessing_layers: list, save_index: int):
        super().__init__()
        self.preprocessing_layers = preprocessing_layers
        self.save_index = save_index

    def _build(self, layers_so_far) -> KerasTensor:
        if self.preprocessing_layers is not None:
            for pre_layer in self.preprocessing_layers:
                layers_so_far = pre_layer(layers_so_far)

        return layers_so_far

    @staticmethod
    def create_random():
        """Raises an error, it's not valid to randomly create this node"""
        raise NotImplementedError("Preprocessing Node can't be created randomly")

    def _clone(self):
        """Clones this node"""
        return PreprocessingNode(self.preprocessing_layers, save_index=self.save_index)

    def _serialize(self) -> dict:
        preprocessing_layers = self.preprocessing_layers
        self.preprocessing_layers = None
        serial_dict = copy.deepcopy(self.__dict__)
        self.preprocessing_layers = preprocessing_layers
        return serial_dict

    def load_layers(self):
        """Load this node's preprocessing layers from file"""
        preprocessing_save_paths = evo_config.master_config.config['preprocessing_save_path']
        model = tf.keras.models.load_model(preprocessing_save_paths[self.save_index])

        prelayers = []
        for layer in model.layers:
            prelayers.append(layer)
        self.preprocessing_layers = prelayers

    def deserialize_cleanup(self):
        self.load_layers()


class ConcatNode(TensorNode):
    """Implements concat layer as a node"""

    def _call_parents(self, all_nodes: dict, graph: MultiDiGraph):
        parents = self.get_parents(all_nodes, graph)
        return [parent(all_nodes, graph) for parent in parents]

    def fix_input(self, layers_so_far: list) -> list:
        if len(layers_so_far) == 1:
            return layers_so_far
        else:
            return node_utils.make_shapes_same(layers_so_far)

    def _build(self, layers_so_far: list) -> KerasTensor:
        return tf.keras.layers.Concatenate()(layers_so_far)

    @staticmethod
    def create_random():
        return ConcatNode()

    def _clone(self):
        return ConcatNode()
