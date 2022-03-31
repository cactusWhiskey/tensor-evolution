"""Base class as well as subclasses for genome nodes"""
import copy
import itertools
import random
import math
from abc import ABC, abstractmethod
import sympy.ntheory as sym
import tensorflow as tf
from keras.engine.keras_tensor import KerasTensor
from networkx import MultiDiGraph
import evo_config


class TensorNode(ABC):
    """''Base class for nodes in a tensor network"""
    id_iter = itertools.count()

    def __init__(self):
        self.id = next(TensorNode.id_iter)
        self.label = self.get_label()
        self.is_branch_root = False
        self.saved_layers = None
        self.can_mutate = False
        self.is_branch_termination = False
        self.node_allows_cache_training = False
        self.weights = None
        self.keras_tensor_input_name = None
        self.required_num_weights = 2

    def serialize(self) -> dict:
        """Converts this class to serial form
        :return dict that can be serialized
        """
        saved_layers = self.saved_layers
        self.saved_layers= None
        serial_dict = copy.deepcopy(self.__dict__)
        self.saved_layers = saved_layers
        return serial_dict

    @staticmethod
    def deserialize(node_dict: dict):
        """
        Creates a new node from a serial instance
        :param node_dict: serial dict to create new node from
        :return: new node
        """
        node = create(node_dict['label'])
        node.__dict__ = node_dict
        node.deserialize_cleanup()
        return node

    def deserialize_cleanup(self):
        """Do cleanup after deserialization if required,
        should be implemented in subclasses as needed."""
        pass

    def __call__(self, all_nodes: dict, graph: MultiDiGraph) -> KerasTensor:
        if self.saved_layers is not None:
            return self.saved_layers
        else:
            self.reset()
            layers_so_far = self._call_parents(all_nodes, graph)
            layers_so_far = self.fix_input(layers_so_far)

            layers_so_far = self._build(layers_so_far)

            if self.is_branch_root:
                self.saved_layers = layers_so_far

            return layers_so_far

    def get_parents(self, all_nodes: dict, graph: MultiDiGraph) -> list:
        """
        Get this node's parents
        :param all_nodes: node dict from the tensor network this node is part of
        :param graph: graph object this node is part of
        :return: list of immediate predecessors of the given node
        """
        parents_ids = graph.predecessors(self.id)
        parents = []
        for p_id in parents_ids:
            parents.append(all_nodes[p_id])

        return parents

    def reset(self):
        """Reset this node's state"""
        self.saved_layers = None

    @abstractmethod
    def _build(self, layers_so_far) -> KerasTensor:
        raise NotImplementedError

    def _call_parents(self, all_nodes: dict, graph: MultiDiGraph) -> KerasTensor:
        parents = self.get_parents(all_nodes, graph)
        if len(parents) > 0:
            return (parents[0])(all_nodes, graph)
        else:
            return None

    def mutate(self):
        """Mutate this node. Should be implemented in subclasses if
        the subclass can mutate in a valid way"""
        pass

    def fix_input(self, layers_so_far):
        """Implement in subclasses if required. A node must be able to handle input of any shape."""
        return layers_so_far

    @staticmethod
    @abstractmethod
    def create_random():
        """Each subclass must either implement creation of a random instance
        of its node type, or pass if random creation isn't valid"""
        raise NotImplementedError

    def get_label(self) -> str:
        """
        Gets the node's type. No need to implement in a subclass.
        :return: node's type as a string
        """
        label = str(type(self)).split('.')[-1]
        label = label.split('\'')[0]
        return label

    @abstractmethod
    def clone(self):
        """Subclasses must implement a clone method which returns
        a deep copy of the given node"""
        raise NotImplementedError


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
        return tf.keras.Input(shape=self.input_shape)

    def deserialize_cleanup(self):
        """
        converts the input shape back intp a tuple
        (it's a list whn read from json)
        :return:
        """
        self.input_shape = tuple(self.input_shape)

    @staticmethod
    def create_random():
        """Raises an error, it's not valid to randomly crate inputs"""
        raise NotImplementedError("Input Node can't be created randomly")


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
        clone.saved_layers = self.saved_layers
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
            elif main_branch.shape.rank == 4:  # Conv2D shape
                units = 1
                for i in range(1, 4):
                    units *= main_branch.shape[i]  # total length
                adder_branch = tf.keras.layers.Dense(units)(adder_branch)
                adder_branch = tf.keras.layers.Reshape(main_branch.shape[1:])(adder_branch)
            else:
                main_branch = tf.keras.layers.Flatten()(main_branch)
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

    def clone(self):
        """Crates a new dense node with the same number of units and
        the same activation as this node."""
        clone = DenseNode(self.num_units, self.activation)
        clone.weights = self.weights
        return clone

    def _build(self, layers_so_far: KerasTensor) -> KerasTensor:
        self.keras_tensor_input_name = layers_so_far.name

        return tf.keras.layers.Dense(self.num_units,
                                     activation=self.activation)(layers_so_far)

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


class Conv2dNode(TensorNode):
    """Implements convolution 2d layer as genome node"""
    def __init__(self, filters, kernel_size, padding='same'):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = 'relu'
        self.padding = padding
        self.can_mutate = True
        self.node_allows_cache_training = True

    def clone(self):
        """Creates a new node with the same parameters as this node"""
        clone = Conv2dNode(self.filters, self.kernel_size, self.padding)
        clone.weights = self.weights
        return clone

    def _build(self, layers_so_far: KerasTensor) -> KerasTensor:
        self.keras_tensor_input_name = layers_so_far.name

        return tf.keras.layers.Conv2D(self.filters, self.kernel_size,
                                      activation=self.activation,
                                      padding=self.padding)(layers_so_far)

    @staticmethod
    def create_random():
        """Creates a new node with random parameters"""
        random_power = random.randint(0, evo_config.master_config.config['max_conv2d_power'])
        filters = 2 ** random_power
        kernel_size = tuple(random.choice(evo_config.master_config.config['conv2d_kernel_sizes']))
        return Conv2dNode(filters, kernel_size)

    def mutate(self):
        """Mutates this nod's parameters"""
        random_power = random.randint(0, evo_config.master_config.config['max_conv2d_power'])
        filters = 2 ** random_power
        kernel_size = tuple(random.choice(evo_config.master_config.config['conv2d_kernel_sizes']))

        self.filters = filters
        self.kernel_size = kernel_size
        self.weights = None

    def fix_input(self, layers_so_far: KerasTensor) -> KerasTensor:
        """Reshapes the input (if needed) so that it's valid for
        a 2D convolution layer (batches, rows, cols, channels)"""
        return _reshape_1D_to_2D(layers_so_far)


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


class MaxPool2DNode(TensorNode):
    """Implements max pooling as genome node"""
    def __init__(self, pool_size=(2, 2), padding="valid"):
        super().__init__()
        self.pool_size = pool_size
        self.padding = padding
        self.can_mutate = True

    def clone(self):
        """Copies this node"""
        return MaxPool2DNode(self.pool_size, self.padding)

    def _build(self, layers_so_far: KerasTensor) -> KerasTensor:
        return tf.keras.layers.MaxPooling2D(self.pool_size,
                                            padding=self.padding)(layers_so_far)

    @staticmethod
    def create_random():
        """Creates a new node with random pool size"""
        pool_size = tuple(random.choice(evo_config.master_config.config['max_pooling_size']))
        return MaxPool2DNode(pool_size=pool_size)

    def mutate(self):
        """Mutates this node's pool size"""
        pool_size = tuple(random.choice(evo_config.master_config.config['max_pooling_size']))
        self.pool_size = pool_size

    def fix_input(self, layers_so_far: KerasTensor) -> KerasTensor:
        """Reshapes the input (if needed) so that it's valid for
               a 2D max pool layer (batches, rows, cols, channels)"""
        return _reshape_1D_to_2D(layers_so_far)


def create(node_type: str) -> TensorNode:
    """Static method to create nodes. The string for each node's
    type must be the same as the node's class name"""
    if node_type == "Conv2dNode":
        return Conv2dNode.create_random()
    elif node_type == "DenseNode":
        return DenseNode.create_random()
    elif node_type == "FlattenNode":
        return FlattenNode()
    elif node_type == "MaxPool2DNode":
        return MaxPool2DNode.create_random()
    elif node_type == "ReluNode":
        return ReluNode.create_random()
    elif node_type == "AdditionNode":
        return AdditionNode.create_random()
    elif node_type == "BatchNormNode":
        return BatchNormNode.create_random()
    elif node_type == "InputNode":
        return InputNode(None)
    elif node_type == "OutputNode":
        return OutputNode(None)
    else:
        raise ValueError("Unsupported node type: " + str(node_type))


def _evaluate_square_shapes(n) -> tuple:
    sqrt = int(math.sqrt(n)) - 1

    while sqrt >= 5:
        square = sqrt ** 2
        channels = n // square

        if (square * channels) == n:
            return sqrt, sqrt, channels

        sqrt -= 1
    return None


def _is_square_rgb(n):
    if (n % 3) == 0:
        n = n // 3
        return sym.primetest.is_square(n)
    else:
        return False


def _shape_from_primes(n) -> tuple:
    prime_dict = sym.factorint(n)
    prime_list = []

    for prime, repeat in prime_dict.items():
        for _ in range(repeat):
            prime_list.append(prime)

    if len(prime_list) == 2:
        return prime_list[0], prime_list[1], 1

    while len(prime_list) > 3:
        prime_list.sort(reverse=True)
        composite = prime_list[-1] * prime_list[-2]
        prime_list.pop(-1)
        prime_list.pop(-1)
        prime_list.append(composite)

    return prime_list[0], prime_list[1], prime_list[2]


def _reshape_1D_to_2D(layers_so_far: KerasTensor) -> KerasTensor:
    add_channel = False

    if layers_so_far.shape.rank == 4:
        return layers_so_far
    elif layers_so_far.shape.rank == 3:
        add_channel = True
    elif layers_so_far.shape.rank != 2:
        raise ValueError("Invalid rank tensor to shape. Got: " +
                         str(layers_so_far.shape))

    n = layers_so_far.shape[1]

    if add_channel:
        n2 = layers_so_far.shape[2]
        target_shape = (n, n2, 1)

    elif sym.isprime(n):
        target_shape = (1, 1, n)

    elif sym.primetest.is_square(n):
        sqrt = int(math.sqrt(n))
        target_shape = (sqrt, sqrt, 1)

    elif _is_square_rgb(n):
        n = n / 3
        sqrt = int(math.sqrt(n))
        target_shape = (sqrt, sqrt, 3)

    else:
        target_shape = _evaluate_square_shapes(n)
        if target_shape is None:
            target_shape = _shape_from_primes(n)

    return tf.keras.layers.Reshape(target_shape)(layers_so_far)
