"""Mutation operators for tensor networks and for hyperparams"""
import random

from tensorEvolution.nodes import node_utils


def mutate_insert(individual: list):
    """inserts a node into a tensor network"""
    if _is_too_big(individual):
        # noinspection PyRedundantParentheses
        return (individual,)

    config = individual[0]
    tensor_net = individual[1]

    node_type = random.choice(config.config['valid_node_types'])
    node = node_utils.create(node_type)
    tensor_net.insert_node_before(node)
    # noinspection PyRedundantParentheses
    return (individual,)


def mutate_mutate(individual: list):
    """Mutates an existing node in a tensor network"""
    tensor_net = individual[1]
    length = len(tensor_net.get_mutatable_nodes())

    if length == 0:  # nothing to mutate
        # noinspection PyRedundantParentheses
        return (individual,)

    position = random.randint(0, length - 1)
    tensor_net.mutate_node(position)
    # noinspection PyRedundantParentheses
    return (individual,)


def mutate_delete(individual: list):
    """Deletes a node from a tensor network"""
    tensor_net = individual[1]
    length = len(tensor_net.get_middle_nodes())

    if length == 0:  # nothing to delete
        # noinspection PyRedundantParentheses
        return (individual,)

    position = random.randint(0, length - 1)
    node_id = list(tensor_net.get_middle_nodes().keys())[position]
    tensor_net.delete_node(node_id=node_id)
    # noinspection PyRedundantParentheses
    return (individual,)


def mutate_hyper(individual: list):
    """Mutates an individuals personal set of hyperparams"""
    ind_hyper_params = individual[0]
    ind_hyper_params.mutate()
    # noinspection PyRedundantParentheses
    return (individual,)


def _is_too_big(individual) -> bool:
    config = individual[0]
    tensor_net = individual[1]

    size = len(tensor_net.get_middle_nodes())
    max_size = config.config['max_network_size']

    if size >= max_size:
        return True
    return False
