"""Utilities and creator function for tensorNodes"""
import copy
import math
import tensorflow as tf
import sympy.ntheory as sym
import numpy as np
from keras.engine.keras_tensor import KerasTensor
from tensorEvolution.nodes import conv_maxpool_nodes, basic_nodes, io_nodes, \
    rnn_nodes, global_avg_pooling, embedding
from tensorEvolution.nodes.tensor_node import TensorNode


def create(node_type: str) -> TensorNode:
    """Static method to create nodes. The string for each node's
    type must be the same as the node's class name"""
    if node_type == "Conv2dNode":
        return conv_maxpool_nodes.Conv2dNode.create_random()
    elif node_type == "Conv3dNode":
        return conv_maxpool_nodes.Conv3dNode.create_random()
    elif node_type == "DenseNode":
        return basic_nodes.DenseNode.create_random()
    elif node_type == "FlattenNode":
        return basic_nodes.FlattenNode()
    elif node_type == "MaxPool2DNode":
        return conv_maxpool_nodes.MaxPool2DNode.create_random()
    elif node_type == "MaxPool3DNode":
        return conv_maxpool_nodes.MaxPool3DNode.create_random()
    elif node_type == "ReluNode":
        return basic_nodes.ReluNode.create_random()
    elif node_type == "AdditionNode":
        return basic_nodes.AdditionNode.create_random()
    elif node_type == "BatchNormNode":
        return basic_nodes.BatchNormNode.create_random()
    elif node_type == "InputNode":
        return io_nodes.InputNode(None)
    elif node_type == "OutputNode":
        return io_nodes.OutputNode(None)
    elif node_type == "DropoutNode":
        return basic_nodes.DropoutNode()
    elif node_type == "LstmNode":
        return rnn_nodes.LstmNode.create_random()
    elif node_type == "GlobalAveragePooling1DNode":
        return global_avg_pooling.GlobalAveragePooling1DNode()
    elif node_type == "EmbeddingNode":
        return embedding.EmbeddingNode(None, None)
    elif node_type == "ConcatNode":
        return basic_nodes.ConcatNode()
    else:
        raise ValueError("Unsupported node type: " + str(node_type))


def deserialize_node(node_dict: dict):
    """
        Creates a new node from a serial instance
        :param node_dict: serial dict to create new node from
        :return: new node
        """
    node = create(node_dict['label'])
    node.__dict__ = node_dict
    node.id = int(node.id)
    if node.weights == "None" or node.weights is None:
        node.weights = None
    else:
        # will be a list of ndarray.tolist()
        # need to convert back to list of ndarrays
        converted_weights = []
        for list_weights in node.weights:
            converted_weights.append(np.array(list_weights))
        node.weights = converted_weights
    # do cleanup specific to a particular node type (gets implemented in subclasses)
    node.deserialize_cleanup()
    return node


def _evaluate_nice_shapes(n, power: int) -> tuple:
    # The case where n is a perfect root is dealt with elsewhere,
    # so start with the next smallest integer root
    root = int(n ** (1. / power)) - 1

    while root >= 2:
        number = root ** power
        channels = n // number

        # if everything works out to exactly n, then this is a good shape
        if (number * channels) == n:
            shape = ()
            for _ in range(power):
                shape += (root,)
            shape += (channels,)
            return shape

        # didn't work out, so try the next smallest integer square root
        root -= 1

    # nothing worked
    return None


def _is_square_rgb(n):
    """Test if the integer input is flattened square rgb data.
    First checks to see if n is divisible by 3, if so, checks if the what's left over is a square"""
    if (n % 3) == 0:
        n = n // 3
        return sym.primetest.is_square(n)
    else:
        return False


def _shape_from_primes(n, desired_length) -> tuple:
    """Decomposes n into it's prime factors. Checks if the number of factors is
    longer than the desired shape rank, if so then multiplies the two smallest factors
    together to shorten the list by one. Continues in this fashion until the rank
    is the desired size.
    Assumes n is not prime, that should be checked somewhere else
    Args:
        n: number to factor
        desired_length: desired length of final tuple. Should be 3 for 2D and 4 for 3D.
        Should be 2 for time series.
        """
    prime_dict = sym.factorint(n)
    prime_list = []

    for prime, repeat in prime_dict.items():
        for _ in range(repeat):
            prime_list.append(prime)

    # shorten list until it is desired length.
    # Note that if there weren't enough primes to make
    # a list of the desired length, then this loop is
    # just skipped
    while len(prime_list) > desired_length:
        prime_list.sort(reverse=True)
        composite = prime_list[-1] * prime_list[-2]
        prime_list.pop(-1)
        prime_list.pop(-1)
        prime_list.append(composite)

    shape = ()
    for prime in prime_list:
        shape += (prime,)
    # if the list is too short, execute this loop
    while len(shape) < desired_length:
        shape += (1,)
    return shape


def make_shapes_same(layers_so_far: list) -> list:
    """
        The inputs to the addition node can be tensors of any shape.
        This method adds flatten, dense, and reshape layers
        as required to force the two inputs to have
        identical shapes.
        :param layers_so_far: list of multiple kerasTensors
        :return: list of (minimum of) two tensors of identical shape        """

    # arbitrarily set main branch to first branch in the list of layers
    main_branch = layers_so_far[0]
    return_list = copy.deepcopy(layers_so_far)
    # iterate through the remaining branches
    done = False
    while not done:
        for index, branch in enumerate(layers_so_far[1:], start=1):
            if main_branch.shape[1:] != branch.shape[1:]:
                if branch.shape.rank != 2:
                    branch = tf.keras.layers.Flatten()(branch)
                if main_branch.shape.rank == 2:  # dense shape/flat shape
                    units = main_branch.shape[1]  # main_shape[0] will be None
                    branch = tf.keras.layers.Dense(units)(branch)
                    done = True
                elif main_branch.shape.rank in (4, 5):  # Conv2D, 3D shapes
                    units = 1
                    for i in range(1, main_branch.shape.rank):
                        units *= main_branch.shape[i]  # total length
                    branch = tf.keras.layers.Dense(units)(branch)
                    branch = tf.keras.layers.Reshape(main_branch.shape[1:])(branch)
                    done = True
                else:
                    main_branch = tf.keras.layers.Flatten()(main_branch)
                    done = False
                    break
            else:
                done = True
            return_list[0] = main_branch
            return_list[index] = branch

    main_shape = return_list[0].shape
    for branch in return_list[1:]:
        if main_shape[1:] != branch.shape[1:]:
            raise ValueError("Debug shapes")

    return return_list


def reshape_input(layers_so_far: KerasTensor, target_rank: int,
                  simple_channel_addition=True) -> KerasTensor:
    """function to reshape input to target_rank input.
    Currently, covers the following cases:
    - rank 2 (i.e. 1D) to rank 4 (i.e. 2D)
    - rank 2 to rank 5 (3D)
    - rank 3 to rank 4 (i.e. 2D no channels to 2D w/ channels)
    - rank 4 to rank 5 (i.e. 2D to 3D)
    Args:
        layers_so_far: KerasTensor that (possibly) needs reshaping
        target_rank: rank we want to reshape to. Note that 1D input is rank 2 (batch, data).
                        2D input is rank 4 (batch, rows, cols, channels).
                        3D input is rank 5 (batch, rows, cols, depth, channels).
                        time series data is rank 3 (batch, steps, features)
        simple_channel_addition: flag which, if true allows a channel to be added
                        when the given shape is 1 rank away from the desired shape.
                        Set to false if you want to force the input to be reshaped, even when
                        the given rank and target rank only differ by one.

    :return KerasTensor
    """

    # rank is what we want, so just return
    if layers_so_far.shape.rank == target_rank:
        return layers_so_far

    # just add a channel to increase rank by 1
    # covers rank 2 to rank 3 and rank 3 to rank 4
    elif (layers_so_far.shape.rank == (target_rank - 1)) \
            and (simple_channel_addition is True):

        # add a single channel and return.
        target_shape = layers_so_far.shape + (1,)
        target_shape = target_shape[1:]  # drop the batch size
        return tf.keras.layers.Reshape(target_shape)(layers_so_far)

    # covers rank 2 to rank 4 or 5
    elif layers_so_far.shape.rank == 2:
        return _reshape_rank2_target(layers_so_far, target_rank)

    # should never get a tensor of rank 1 (i.e. only batch size)
    elif layers_so_far.shape.rank == 1:
        raise ValueError("Invalid rank tensor to shape. Got: " +
                         str(layers_so_far.shape))

    else:
        flat_shape = 1
        for dim in layers_so_far.shape[1:]:
            flat_shape *= dim
        flat_shape = (1, flat_shape)  # create rank 2 tuple as shape
        return _reshape_rank2_target(layers_so_far, target_rank, flat_shape)


def _reshape_rank2_target(layers_so_far: KerasTensor,
                          target_rank: int, flat_shape=None) -> KerasTensor:
    if target_rank not in (3, 4, 5):
        raise ValueError(f"Attempting to reshape to unsupported rank. "
                         f"Expected target rank of 3, 4 or 5, but got {target_rank}")

    # if flat shape is None then the input is rank 2
    # if flat shape is not None, then the input was some other rank
    # and we've computed it's equivalent rank 2 shape
    if flat_shape is None:
        existing_shape = layers_so_far.shape
    else:
        existing_shape = flat_shape

    row_size = existing_shape[1]  # length of rows

    # check if row_size is a prime number
    if sym.isprime(row_size):
        # nothing else to be done if it's prime
        if target_rank == 3:
            target_shape = (1, row_size)
            return tf.keras.layers.Reshape(target_shape)(layers_so_far)
        elif target_rank == 4:
            target_shape = (1, 1, row_size)
            return tf.keras.layers.Reshape(target_shape)(layers_so_far)
        else:
            # target must be rank 5
            target_shape = (1, 1, 1, row_size)
            return tf.keras.layers.Reshape(target_shape)(layers_so_far)

    # row_size isn't prime, so we can factor it,let's try.
    elif target_rank == 3:
        return _reshape_rank2_rank3(layers_so_far, existing_shape)
    elif target_rank == 4:
        # Handle the 1D to 2D conversion
        return _reshape_rank2_rank4(layers_so_far, existing_shape)
    else:
        # Handle 1D to 3D
        return _reshape_rank2_rank5(layers_so_far, existing_shape)


def _reshape_rank2_rank3(layers_so_far, existing_shape) -> KerasTensor:
    row_size = existing_shape[1]  # length of rows
    target_shape = _shape_from_primes(row_size, 2)
    return tf.keras.layers.Reshape(target_shape)(layers_so_far)


def _reshape_rank2_rank4(layers_so_far, existing_shape) -> KerasTensor:
    row_size = existing_shape[1]  # length of rows

    # check if the row size corresponds to a flattened perfect square
    if sym.primetest.is_square(row_size):
        sqrt = int(math.sqrt(row_size))
        # since the shape is square, just add a channel to make a valid rank 4 tensor
        target_shape = (sqrt, sqrt, 1)

    # check to see if the row_size corresponds to some flattened square with 3 channels
    elif _is_square_rgb(row_size):
        row_size = row_size / 3
        sqrt = int(math.sqrt(row_size))
        target_shape = (sqrt, sqrt, 3)

    else:
        # check if the row_size is a flattened square with more than 3 channels
        target_shape = _evaluate_nice_shapes(row_size, 2)
        if target_shape is None:
            # nothing has worked so far. Factor row_size into primes
            # and then build up a valid shape from the factors
            target_shape = _shape_from_primes(row_size, 3)

    return tf.keras.layers.Reshape(target_shape)(layers_so_far)


def _reshape_rank2_rank5(layers_so_far, existing_shape) -> KerasTensor:
    row_size = existing_shape[1]  # length of rows

    # check if the row size corresponds to a flattened perfect cube
    cube_root = round(row_size ** (1. / 3.))
    cube_root_cubed = cube_root ** 3
    if cube_root_cubed == row_size:
        # since the shape is a cube, just add a channel to make a valid rank 5 tensor
        target_shape = (cube_root, cube_root, cube_root, 1)

    # check to see if the row_size corresponds to some
    # flattened cube with some number of channels
    else:
        target_shape = _evaluate_nice_shapes(row_size, 3)
        if target_shape is None:
            # nothing has worked so far. Factor row_size into primes
            # and then build up a valid shape from the factors
            target_shape = _shape_from_primes(row_size, 4)

    return tf.keras.layers.Reshape(target_shape)(layers_so_far)
