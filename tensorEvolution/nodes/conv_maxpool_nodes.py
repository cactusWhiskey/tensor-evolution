"""Module implementing Conv2D, MaxPool2D, Conv3D, MaxPool3D"""
import random

import tensorflow as tf
from keras.engine.keras_tensor import KerasTensor

from tensorEvolution import evo_config
from tensorEvolution.nodes import node_utils
from tensorEvolution.nodes.tensor_node import TensorNode


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
        self.kernel_regularizer = evo_config.master_config.random_regularizer()

    def _clone(self):
        """Creates a new node with the same parameters as this node"""
        clone = Conv2dNode(self.filters, self.kernel_size, self.padding)
        # clone.weights = copy.deepcopy(self.weights)
        return clone

    def deserialize_cleanup(self):
        """Do cleanup after deserialization if required,
        should be implemented in subclasses as needed."""
        self.kernel_size = tuple(self.kernel_size)

    def _build(self, layers_so_far: KerasTensor) -> KerasTensor:
        self.keras_tensor_input_name = layers_so_far.name
        regularizer = evo_config.EvoConfig.build_regularizer(self.kernel_regularizer)
        conv2d = tf.keras.layers.Conv2D(self.filters, self.kernel_size,
                                        activation=self.activation,
                                        padding=self.padding,
                                        kernel_regularizer=regularizer)
        self.name = conv2d.name
        return conv2d(layers_so_far)

    @staticmethod
    def create_random():
        """Creates a new node with random parameters"""
        random_power = random.randint(0, evo_config.master_config.config['max_conv_power'])
        filters = 2 ** random_power
        kernel_size = tuple(random.choice(evo_config.master_config.config['conv2d_kernel_sizes']))
        return Conv2dNode(filters, kernel_size)

    def mutate(self):
        """Mutates this nod's parameters"""
        random_power = random.randint(0, evo_config.master_config.config['max_conv_power'])
        filters = 2 ** random_power
        kernel_size = tuple(random.choice(evo_config.master_config.config['conv2d_kernel_sizes']))

        self.filters = filters
        self.kernel_size = kernel_size
        self.weights = None
        self.kernel_regularizer = evo_config.master_config.random_regularizer()

    def fix_input(self, layers_so_far: KerasTensor) -> KerasTensor:
        """Reshapes the input (if needed) so that it's valid for
        a 2D convolution layer (batches, rows, cols, channels)"""
        reshaped_input = node_utils.reshape_input(layers_so_far, 4)
        self.kernel_size = _fix_kernel(reshaped_input, self.kernel_size)
        return reshaped_input


class MaxPool2DNode(TensorNode):
    """Implements max pooling as genome node"""

    def __init__(self, pool_size=(2, 2), padding="valid"):
        super().__init__()
        self.pool_size = pool_size
        self.padding = padding
        self.can_mutate = True

    def _clone(self):
        """Copies this node"""
        return MaxPool2DNode(self.pool_size, self.padding)

    def deserialize_cleanup(self):
        """Do cleanup after deserialization if required,
        should be implemented in subclasses as needed."""
        self.pool_size = tuple(self.pool_size)

    def _build(self, layers_so_far: KerasTensor) -> KerasTensor:
        return tf.keras.layers.MaxPooling2D(self.pool_size,
                                            padding=self.padding)(layers_so_far)

    @staticmethod
    def create_random():
        """Creates a new node with random pool size"""
        pool_size = tuple(random.choice(evo_config.master_config.config['max_pooling_size2D']))
        return MaxPool2DNode(pool_size=pool_size)

    def mutate(self):
        """Mutates this node's pool size"""
        pool_size = tuple(random.choice(evo_config.master_config.config['max_pooling_size2D']))
        self.pool_size = pool_size

    def fix_input(self, layers_so_far: KerasTensor) -> KerasTensor:
        """Reshapes the input (if needed) so that it's valid for
               a 2D max pool layer (batches, rows, cols, channels)"""
        reshaped_input = node_utils.reshape_input(layers_so_far, 4)
        self.pool_size = _fix_kernel(reshaped_input, self.pool_size)
        return reshaped_input


class Conv3dNode(TensorNode):
    """Implements convolution 3d layer as genome node"""

    def __init__(self, filters, kernel_size, padding='same'):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = 'relu'
        self.padding = padding
        self.can_mutate = True
        self.node_allows_cache_training = True
        self.kernel_regularizer = evo_config.master_config.random_regularizer()

    def _clone(self):
        """Creates a new node with the same parameters as this node"""
        clone = Conv3dNode(self.filters, self.kernel_size, self.padding)
        # clone.weights = copy.deepcopy(self.weights)
        return clone

    def deserialize_cleanup(self):
        """Do cleanup after deserialization if required,
        should be implemented in subclasses as needed."""
        self.kernel_size = tuple(self.kernel_size)

    def _build(self, layers_so_far: KerasTensor) -> KerasTensor:
        self.keras_tensor_input_name = layers_so_far.name
        regularizer = evo_config.EvoConfig.build_regularizer(self.kernel_regularizer)
        conv3d = tf.keras.layers.Conv3D(self.filters, self.kernel_size,
                                        activation=self.activation,
                                        padding=self.padding,
                                        kernel_regularizer=regularizer)
        self.name = conv3d.name
        return conv3d(layers_so_far)

    @staticmethod
    def create_random():
        """Creates a new node with random parameters"""
        random_power = random.randint(0, evo_config.master_config.config['max_conv_power'])
        filters = 2 ** random_power
        kernel_size = tuple(random.choice(evo_config.master_config.config['conv3d_kernel_sizes']))
        return Conv3dNode(filters, kernel_size)

    def mutate(self):
        """Mutates this nod's parameters"""
        random_power = random.randint(0, evo_config.master_config.config['max_conv_power'])
        filters = 2 ** random_power
        kernel_size = tuple(random.choice(evo_config.master_config.config['conv3d_kernel_sizes']))

        self.filters = filters
        self.kernel_size = kernel_size
        self.weights = None
        self.kernel_regularizer = evo_config.master_config.random_regularizer()

    def fix_input(self, layers_so_far: KerasTensor) -> KerasTensor:
        """Reshapes the input (if needed) so that it's valid for
        a 3D convolution layer (batches, rows, cols, width, channels)"""
        reshaped_input = node_utils.reshape_input(layers_so_far, 5)
        self.kernel_size = _fix_kernel(reshaped_input, self.kernel_size)
        return reshaped_input


class MaxPool3DNode(TensorNode):
    """Implements max pooling as genome node"""

    def __init__(self, pool_size=(2, 2, 2), padding="valid"):
        super().__init__()
        self.pool_size = pool_size
        self.padding = padding
        self.can_mutate = True

    def _clone(self):
        """Copies this node"""
        return MaxPool3DNode(self.pool_size, self.padding)

    def deserialize_cleanup(self):
        """Do cleanup after deserialization if required,
        should be implemented in subclasses as needed."""
        self.pool_size = tuple(self.pool_size)

    def _build(self, layers_so_far: KerasTensor) -> KerasTensor:
        return tf.keras.layers.MaxPooling3D(self.pool_size,
                                            padding=self.padding)(layers_so_far)

    @staticmethod
    def create_random():
        """Creates a new node with random pool size"""
        pool_size = tuple(random.choice(evo_config.master_config.config['max_pooling_size3D']))
        return MaxPool3DNode(pool_size=pool_size)

    def mutate(self):
        """Mutates this node's pool size"""
        pool_size = tuple(random.choice(evo_config.master_config.config['max_pooling_size3D']))
        self.pool_size = pool_size

    def fix_input(self, layers_so_far: KerasTensor) -> KerasTensor:
        """Reshapes the input (if needed) so that it's valid for
               a 3D max pool layer (batches, rows, cols, width channels)"""
        reshaped_input = node_utils.reshape_input(layers_so_far, 5)
        self.pool_size = _fix_kernel(reshaped_input, self.pool_size)
        return reshaped_input


def _fix_kernel(reshaped_input: KerasTensor, kernel: tuple) -> tuple:
    new_kernel = ()
    for index, _ in enumerate(kernel):
        size = kernel[index]
        while reshaped_input.shape[index + 1] < size:
            size -= 1
        new_kernel += (size,)
    return new_kernel
