"""Module which implements flatten, dense, addition, relu, batchnorm, dropout"""
import copy
import random

import tensorflow as tf
from keras.engine.keras_tensor import KerasTensor
from networkx import MultiDiGraph

from tensorEvolution import evo_config
from tensorEvolution.nodes.tensor_node import TensorNode


class FlattenNode(TensorNode):
    """Implements flatten layer as a genome node"""

    def _build(self, layers_so_far: KerasTensor):
        return tf.keras.layers.Flatten()(layers_so_far)

    def clone(self):
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

    def clone(self):
        """Builds a new addition node"""
        clone = AdditionNode()
        clone.saved_layers = copy.deepcopy(self.saved_layers)
        return clone

    def _call_parents(self, all_nodes: dict, graph: MultiDiGraph):
        parents = self.get_parents(all_nodes, graph)
        if len(parents) == 1:  # same parent hooked into both sides of the addition layer
            return parents[0](all_nodes, graph)
        elif len(parents) > 2:
            raise ValueError("Too many parents: " + str(len(parents)))
        return parents[0](all_nodes, graph), parents[1](all_nodes, graph)

    def _build(self, layers_so_far: tuple) -> KerasTensor:
        main_branch, adder_branch = layers_so_far
        return tf.keras.layers.add([main_branch, adder_branch])

    @staticmethod
    def create_random():
        """Just creates a new node since addition node has no
        internal state to randomize"""
        return AdditionNode()

    def fix_input(self, layers_so_far) -> tuple:
        """
        The inputs to the addition node can be tensors of any shape.
        This method adds flatten, dense, and reshape layers
        as required to force the two inputs to have
        identical shapes.
        :param layers_so_far: either a tuple of the two
        different layers feeding into this addition node,
        or possibly just a single input (because it's
        possible that a single parent node feeds into both inputs
        of this addition node).
        :return: tuple of two tensors of identical shape
        """
        if isinstance(layers_so_far, tuple):
            main_branch, adder_branch = layers_so_far
        else:
            return layers_so_far, layers_so_far

        if main_branch.shape[1:] != adder_branch.shape[1:]:
            adder_branch = tf.keras.layers.Flatten()(adder_branch)
            if main_branch.shape.rank == 2:  # dense shape
                units = main_branch.shape[1]  # main_shape[0] will be None
                adder_branch = tf.keras.layers.Dense(units)(adder_branch)
            elif main_branch.shape.rank in (4, 5):  # Conv2D, 3D shapes
                units = 1
                for i in range(1, main_branch.shape.rank):
                    units *= main_branch.shape[i]  # total length
                adder_branch = tf.keras.layers.Dense(units)(adder_branch)
                adder_branch = tf.keras.layers.Reshape(main_branch.shape[1:])(adder_branch)
            else:
                main_branch = tf.keras.layers.Flatten()(main_branch)
                adder_branch = tf.keras.layers.Flatten()(adder_branch)
                main_shape = main_branch.shape
                units = main_shape[1]
                adder_branch = tf.keras.layers.Dense(units)(adder_branch)

        return main_branch, adder_branch


class DenseNode(TensorNode):
    """Implements dense layer as a genome node"""

    def __init__(self, num_units: int, activation='relu'):
        super().__init__()
        self.num_units = num_units
        self.activation = activation
        self.can_mutate = True
        self.node_allows_cache_training = True
        self.kernel_regularizer = evo_config.master_config.random_regularizer()

    def clone(self):
        """Crates a new dense node with the same number of units and
        the same activation as this node."""
        clone = DenseNode(self.num_units, self.activation)
        clone.weights = copy.deepcopy(self.weights)
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
        Mutates this node's number of units
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

    def clone(self):
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

    def clone(self):
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

    @staticmethod
    def create_random():
        """Creates and returns a new DropoutNode"""
        return DropoutNode()

    def clone(self):
        """Creates a new DropoutNode and sets its rate equal to this node's rate"""
        new_node = DropoutNode()
        new_node.rate = copy.copy(self.rate)
        return new_node

    def _build(self, layers_so_far) -> KerasTensor:
        return tf.keras.layers.Dropout(self.rate)(layers_so_far)

    def mutate(self):
        self.rate = random.uniform(0.0, 0.9)


class PreprocessingNode(TensorNode):
    """Node which holds preprocessing layers"""

    def __init__(self, preprocessing_layers):
        super().__init__()
        self.preprocessing_layers = preprocessing_layers
        self.num_pre_layers = len(preprocessing_layers)

    def _build(self, layers_so_far) -> KerasTensor:
        if self.preprocessing_layers is not None:
            for pre_layer in self.preprocessing_layers:
                layers_so_far = pre_layer(layers_so_far)

        return layers_so_far

    @staticmethod
    def create_random():
        """Raises an error, it's not valid to randomly create this node"""
        raise NotImplementedError("Preprocessing Node can't be created randomly")

    def clone(self):
        """Clones this node"""
        return PreprocessingNode(self.preprocessing_layers)

    def _serialize(self) -> dict:
        preprocessing_layers = self.preprocessing_layers
        self.preprocessing_layers = None
        serial_dict = copy.deepcopy(self.__dict__)
        self.preprocessing_layers = preprocessing_layers
        return serial_dict

    def load_layers(self):
        preprocessing_save_path = evo_config.master_config.config['preprocessing_save_path']
        model = tf.keras.models.load_model(preprocessing_save_path)

        prelayers = []
        for i in range(1, (self.num_pre_layers + 1)):
            prelayers.append(model.layers[i])

        self.preprocessing_layers = prelayers
