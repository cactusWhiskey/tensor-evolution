"""Base class for genome nodes"""
import copy
import itertools
from abc import ABC, abstractmethod

from keras.engine.keras_tensor import KerasTensor
from networkx import MultiDiGraph


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
        self.name = None
        self.keras_tensor_input_name = None
        self.required_num_weights = 2
        self.variable_output_size = False
        self.accepts_variable_length_input = False
        self.has_variable_length_input = False
        self.is_initial_node = False
        self.is_end_initial_chain = False

    def serialize(self) -> dict:
        """Converts this class to serial form
        :return dict that can be serialized
        """
        # KerasTensor doesn't serialize well
        saved_layers = self.saved_layers
        # Easiest to just not save it, it's not critical anyway
        # (can be rebuilt first time genome is built)
        self.saved_layers = None

        # weights are a list of ndarrays, which will cause an error with default json dump
        # first store existing weights
        weights = self.weights
        # None check
        if self.weights is not None:
            converted_weights = []
            for ndarray_weights in self.weights:
                converted_weights.append(ndarray_weights.tolist())
            self.weights = converted_weights

        serial_dict = self._serialize()
        self.saved_layers = saved_layers
        self.weights = weights
        return serial_dict

    def _serialize(self) -> dict:
        return copy.deepcopy(self.__dict__)

    def deserialize_cleanup(self):
        """Do cleanup after deserialization if required,
        should be implemented in subclasses as needed."""
        pass

    def __call__(self, all_nodes: dict, graph: MultiDiGraph) -> KerasTensor:
        """Recursively calls this node's parents until it hits a base case,
        then starts building the network moving forward from the base case.
        Input nodes are always base cases, the __call__ method on an InputNode will always call
        tf.keras.Input() and return a KerasTensor.
        Other nodes in the network will sometimes act as base cases as well. For example,
        if a node has multiple child nodes, it will cache the keras tensor
        it's _build method returns.
        This way, the recursive calls from the different child branches don't all need to
        go back to the input node to hit a base case."""

        # check if this node has saved its built KerasTensor
        # (i.e. this node is acting as a base case for the recursion).
        if self.saved_layers is not None:
            return self.saved_layers
        else:
            self.reset()

            # recursive call to this node's parents
            layers_so_far = self._call_parents(all_nodes, graph)

            # recursive call has returned with the network built up until this point.
            # Now deal with fixing the input shape if required, so that this node has valid input.
            layers_so_far = self.fix_input(layers_so_far)

            # build this node, returning a KerasTensor
            layers_so_far = self._build(layers_so_far)

            # check if this node needs to save its built KerasTensor (i.e. act as a base case)
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

    def set_variable_input(self, is_variable_length: bool):
        """Override in subclasses as required. If the input to this node is of variable length,
         should the output be flagged as also being variable length?

         Default behavior is to set variable length output flag as true
         :param is_variable_length: is the input to this node of variable shape?"""

        self.has_variable_length_input = is_variable_length
        if is_variable_length:
            self.variable_output_size = True
        else:
            self.variable_output_size = False

    @staticmethod
    @abstractmethod
    def create_random():
        """Each subclass must either implement creation of a random instance
        of its node type, or raise an error if random creation isn't valid"""
        raise NotImplementedError

    def get_label(self) -> str:
        """
        Gets the node's type. No need to implement in a subclass.
        :return: node's type as a string
        """
        label = str(type(self)).split('.')[-1]
        label = label.split('\'')[0]
        return label

    def clone(self):
        """Deal with copying over the state of the cloned node"""
        new_node = self._clone()
        node_id = new_node.id
        new_node.__dict__ = copy.deepcopy(self.__dict__)
        new_node.id = node_id
        return new_node

    @abstractmethod
    def _clone(self):
        """Subclasses must implement a clone method which returns
                a new copy of the given node"""
        raise NotImplementedError()

    def __str__(self):
        return f"Node ID: {self.id}, Label: {self.label}"
