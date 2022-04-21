"""Crossover operators for tensor networks and for hyperparams"""
from tensorEvolution import evo_config, tensor_network


def cx_hyper(ind1, ind2):
    """Crossover for hyperparams"""
    hyper1 = ind1[0]
    hyper2 = ind2[0]
    evo_config.cross_over(hyper1, hyper2)
    return ind1, ind2


def cx_single_node(ind1, ind2):
    """Crossover on a tensor net, only swaps one node between each individual"""
    tensor_net = ind1[1]
    other_net = ind2[1]
    tensor_network.cx_single_node(tensor_net, other_net)
    return ind1, ind2

# def cx_chain(ind1, ind2):
#     tn = ind1[1]
#     other_tn = ind2[1]
#     tensor_network.cx_chain(tn, other_tn)
#     return ind1, ind2
