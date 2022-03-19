import copy
import itertools
import random
import math
from abc import ABC, abstractmethod
import sympy.ntheory as sym
import tensorflow as tf
from keras.engine.keras_tensor import KerasTensor
from networkx import DiGraph
import config_utils


class TensorNode(ABC):
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
        return copy.deepcopy(self.__dict__)

    @staticmethod
    def deserialize(node_dict: dict):
        node = create(node_dict['label'])
        node.__dict__ = node_dict
        node.deserialize_cleanup()
        return node

    def deserialize_cleanup(self):
        pass

    def __call__(self, all_nodes: dict, graph: DiGraph, level=0) -> KerasTensor:
        if level > 10:
            print(str(level))
        if self.saved_layers is not None:
            return self.saved_layers
        else:
            self.reset()
            layers_so_far = self._call_parents(all_nodes, graph, level)
            layers_so_far = self.fix_input(layers_so_far)
            try:
                layers_so_far = self._build(layers_so_far)
            except:
                print("debug")
            if self.is_branch_root:
                self.saved_layers = layers_so_far

            return layers_so_far

    def get_parents(self, all_nodes: dict, graph: DiGraph) -> list:
        try:
            parents_ids = graph.predecessors(self.id)
            parents = []
            for p_id in parents_ids:
                parents.append(all_nodes[p_id])
        except:
            print("debug")
        return parents

    def reset(self):
        self.saved_layers = None

    @abstractmethod
    def _build(self, layers_so_far) -> KerasTensor:
        raise NotImplementedError

    def _call_parents(self, all_nodes: dict, graph: DiGraph, level) -> KerasTensor:
        parents = self.get_parents(all_nodes, graph)
        if len(parents) > 0:
            return (parents[0])(all_nodes, graph, level + 1)
        else:
            return None

    def mutate(self):
        pass

    def fix_input(self, layers_so_far):
        return layers_so_far

    @staticmethod
    @abstractmethod
    def create_random():
        raise NotImplementedError

    def get_label(self) -> str:
        label = str(type(self)).split('.')[-1]
        label = label.split('\'')[0]
        return label

    @abstractmethod
    def clone(self):
        raise NotImplementedError


class InputNode(TensorNode):
    def __init__(self, input_shape: tuple):
        super().__init__()
        self.input_shape = input_shape
        self.is_branch_root = True

    def clone(self):
        return InputNode(self.input_shape)

    def _build(self, layers_so_far):
        return tf.keras.Input(shape=self.input_shape)

    def reset(self):
        self.saved_layers = None

    def deserialize_cleanup(self):
        self.input_shape = tuple(self.input_shape)

    @staticmethod
    def create_random():
        raise NotImplementedError("Input Node can't be created randomly")


class FlattenNode(TensorNode):
    def _build(self, layers_so_far: KerasTensor):
        return tf.keras.layers.Flatten()(layers_so_far)

    def clone(self):
        return FlattenNode()

    @staticmethod
    def create_random():
        return FlattenNode()


class AdditionNode(TensorNode):
    def __init__(self):
        super().__init__()
        self.is_branch_termination = True

    def clone(self):
        clone = AdditionNode()
        clone.saved_layers = self.saved_layers
        return clone

    def _call_parents(self, all_nodes: dict, graph: DiGraph, level):
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
        return AdditionNode()

    def fix_input(self, layers_so_far) -> tuple:
        if type(layers_so_far) is tuple:
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
    def __init__(self, num_units: int, activation='relu'):
        super().__init__()
        self.num_units = num_units
        self.activation = activation
        self.can_mutate = True
        self.node_allows_cache_training = True

    def clone(self):
        clone = DenseNode(self.num_units, self.activation)
        clone.weights = self.weights
        return clone

    def _build(self, layers_so_far: KerasTensor) -> KerasTensor:
        self.keras_tensor_input_name = layers_so_far.name

        return tf.keras.layers.Dense(self.num_units,
                                     activation=self.activation)(layers_so_far)

    @staticmethod
    def create_random():
        random_power = random.randint(0, config_utils.config['dense_max_power_two'])
        units = 2 ** random_power
        return DenseNode(units)

    def mutate(self):
        random_power = random.randint(0, config_utils.config['dense_max_power_two'])
        units = 2 ** random_power
        self.num_units = units
        self.weights = None


class Conv2dNode(TensorNode):
    def __init__(self, filters, kernel_size, padding='same'):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = 'relu'
        self.padding = padding
        self.can_mutate = True
        self.node_allows_cache_training = True

    def clone(self):
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
        random_power = random.randint(0, config_utils.config['max_conv2d_power'])
        filters = 2 ** random_power
        kernel_size = tuple(random.choice(config_utils.config['conv2d_kernel_sizes']))
        return Conv2dNode(filters, kernel_size)

    def mutate(self):
        random_power = random.randint(0, config_utils.config['max_conv2d_power'])
        filters = 2 ** random_power
        kernel_size = tuple(random.choice(config_utils.config['conv2d_kernel_sizes']))

        self.filters = filters
        self.kernel_size = kernel_size
        self.weights = None

    def fix_input(self, layers_so_far: KerasTensor) -> KerasTensor:
        return reshape_1D_to_2D(layers_so_far)


class ReluNode(TensorNode):
    def _build(self, layers_so_far: KerasTensor) -> KerasTensor:
        return tf.keras.layers.ReLU()(layers_so_far)

    def clone(self):
        return ReluNode()

    @staticmethod
    def create_random():
        return ReluNode()


class BatchNormNode(TensorNode):
    def _build(self, layers_so_far: KerasTensor) -> KerasTensor:
        return tf.keras.layers.BatchNormalization()(layers_so_far)

    def clone(self):
        return BatchNormNode()

    @staticmethod
    def create_random():
        return BatchNormNode()


class OutputNode(TensorNode):
    def __init__(self, num_outputs):
        super().__init__()
        self.num_outputs = num_outputs
        self.node_allows_cache_training = True

    def clone(self):
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
        raise NotImplementedError("Output Node can't be created randomly")


class MaxPool2DNode(TensorNode):
    def __init__(self, pool_size=(2, 2), padding="valid"):
        super().__init__()
        self.pool_size = pool_size
        self.padding = padding
        self.can_mutate = True

    def clone(self):
        return MaxPool2DNode(self.pool_size, self.padding)

    def _build(self, layers_so_far: KerasTensor) -> KerasTensor:
        return tf.keras.layers.MaxPooling2D(self.pool_size,
                                            padding=self.padding)(layers_so_far)

    @staticmethod
    def create_random():
        pool_size = tuple(random.choice(config_utils.config['max_pooling_size']))
        return MaxPool2DNode(pool_size=pool_size)

    def mutate(self):
        pool_size = tuple(random.choice(config_utils.config['max_pooling_size']))
        self.pool_size = pool_size

    def fix_input(self, layers_so_far: KerasTensor) -> KerasTensor:
        return reshape_1D_to_2D(layers_so_far)


def create(node_type: str) -> TensorNode:
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


def evaluate_square_shapes(n) -> tuple:
    sqrt = int(math.sqrt(n)) - 1

    while sqrt >= 5:
        square = sqrt ** 2
        channels = n // square

        if (square * channels) == n:
            return sqrt, sqrt, channels

        sqrt -= 1
    return None


def is_square_rgb(n):
    if (n % 3) == 0:
        n = n // 3
        return sym.primetest.is_square(n)
    else:
        return False


def shape_from_primes(n) -> tuple:
    prime_dict = sym.factorint(n)
    prime_list = []

    for prime, repeat in prime_dict.items():
        for rep in range(repeat):
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


def reshape_1D_to_2D(layers_so_far: KerasTensor) -> KerasTensor:
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

    elif is_square_rgb(n):
        n = n / 3
        sqrt = int(math.sqrt(n))
        target_shape = (sqrt, sqrt, 3)

    else:
        target_shape = evaluate_square_shapes(n)
        if target_shape is None:
            target_shape = shape_from_primes(n)

    return tf.keras.layers.Reshape(target_shape)(layers_so_far)
