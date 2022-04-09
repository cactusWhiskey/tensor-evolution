"""defines a Tensor Network, which is the class that holds most of our genome information"""
import copy
import itertools
import json
import time
import random
import tensorflow as tf
import networkx as nx
import pydot
from matplotlib import pyplot as plt
from tensorEvolution import tensor_encoder
from tensorEvolution.nodes import tensor_node, io_nodes, node_utils


class TensorNetwork:
    """Holds all the information that defines the NN gnome"""
    id_iter = itertools.count(100)

    def __init__(self, input_shapes: list, output_units: list, connected=True):
        self.graph = nx.MultiDiGraph()
        self.net_id = next(TensorNetwork.id_iter)
        self.all_nodes = {}
        self.input_ids = []
        self.output_ids = []
        self.input_shapes = input_shapes
        self.output_units = output_units

        if connected:
            self._create_inputs(input_shapes)
            self._create_outputs(output_units)

    def serialize(self) -> dict:
        """Creates serializable version of this class"""
        serial_dict = copy.deepcopy(self.__dict__)
        serial_dict['graph'] = nx.node_link_data(self.graph)

        for node_id, node in self.all_nodes.items():
            serial_node = node.serialize()
            (serial_dict['all_nodes'])[node_id] = serial_node

        return serial_dict

    @staticmethod
    def deserialize(tn_dict: dict):
        """Builds a new tensor network from  serialized instance
        Args:
            tn_dict: Serialized Tensor Network
            """
        tensor_net = TensorNetwork(None, None, False)
        tensor_net.__dict__ = tn_dict
        tensor_net.net_id = int(tensor_net.net_id)
        tensor_net.all_nodes = {int(k): v for (k, v) in tensor_net.all_nodes.items()}
        tensor_net.graph = nx.node_link_graph(tn_dict['graph'])

        for node_id, serial_node in tensor_net.all_nodes.items():
            tensor_net.all_nodes[node_id] = node_utils.deserialize_node(serial_node)

        for index, shape in enumerate(tensor_net.input_shapes):
            tensor_net.input_shapes[index] = tuple(shape)

        return tensor_net

    def __deepcopy__(self, memodict={}):
        return self._clone()

    def _clone(self):
        clone_tn = TensorNetwork(self.input_shapes, self.output_units, connected=False)

        node_cross_ref = {}
        for node_id, node in self.all_nodes.items():
            cloned_node = node.clone()
            clone_tn.register_node(cloned_node)
            node_cross_ref[node_id] = cloned_node.id

        for edge in self.graph.edges:
            old_start_node = edge[0]
            old_end_node = edge[1]
            new_start_node = node_cross_ref[old_start_node]
            new_end_node = node_cross_ref[old_end_node]
            clone_tn.graph.add_edge(new_start_node, new_end_node)

        return clone_tn

    def _create_inputs(self, input_shapes: list):
        if len(input_shapes) > 1:
            raise ValueError("Multiple inputs not yet supported")

        for shape in input_shapes:
            node = io_nodes.InputNode(shape)
            self.register_node(node)

    def _create_outputs(self, output_units: list):
        for units in output_units:
            node = io_nodes.OutputNode(units)
            self.register_node(node)

            for input_id in self.input_ids:
                self.graph.add_edge(input_id, node.id)

    def insert_node(self, new_node: tensor_node.TensorNode, position: int):
        """inserts node before the given position.
        Positions refer to the index in the "non_input_nodes" list,
        which is kept in no particular order
        Args:
            new_node: node to be inserted
            position: position to insert node before
        """

        nodes = self.get_valid_insert_positions()
        if position > (len(nodes) - 1):
            raise ValueError("Invalid request to insert node: Length = " +
                             str(len(nodes)) + " position given as: " + str(position))

        child_node = list(nodes.values())[position]  # node at position becomes child

        # valid insert positions only have a single parent
        parent = get_parents(self, child_node)[0]

        self.register_node(new_node)

        while self.graph.has_edge(parent.id, child_node.id):
            # remove all edges between parent and child
            self.graph.remove_edge(parent.id, child_node.id)

        # add a new edge between parent and new node
        self.graph.add_edge(parent.id, new_node.id)
        # add new edge between new node and child
        self.graph.add_edge(new_node.id, child_node.id)

        if new_node.is_branch_termination:
            all_parents_ids = self.get_parent_chain_ids(new_node)
            branch_origin = random.choice(all_parents_ids)
            self.graph.add_edge(branch_origin, new_node.id)

    def delete_node(self, node_id, replacement_node=None):
        """
        deletes a node from the network
        :param node_id: node to delete
        :param replacement_node: None, unless you intend to replace the node instead of delete it

        """
        replace = False
        if replacement_node is not None:
            replace = True

        node_to_remove = self.get_a_middle_node(node_id=node_id)

        if node_to_remove.is_branch_termination:
            return  # deletion of branch endpoints not currently supported

        parents = get_parents(self, node_to_remove)
        children = self.get_children(node_to_remove)

        while self.graph.has_node(node_to_remove.id):
            self.graph.remove_node(node_to_remove.id)  # also removes adjacent edges

        self.all_nodes.pop(node_to_remove.id)

        if replace:
            self.register_node(replacement_node)
            for parent in parents:
                self.graph.add_edge(parent.id, replacement_node.id)
            for child in children:
                self.graph.add_edge(replacement_node.id, child.id)
        else:
            for parent in parents:
                for child in children:
                    self.graph.add_edge(parent.id, child.id)

    def register_node(self, node: tensor_node.TensorNode):
        """
        registers a node with the network. Adds it to the graph which holds the
        network topology. Also adds it to the main dict of nodes
        :param node: node to register
        """
        self.all_nodes[node.id] = node
        label = node.get_label()
        self.graph.add_node(node.id, label=label)

        if label == "InputNode":
            self.input_ids.append(node.id)
        elif label == "OutputNode":
            self.output_ids.append(node.id)

    # def replace_node(self, replacement_node: tensor_node.TensorNode,
    #                  position=None, existing_node_id=None):
    #
    #     old_node = self.get_a_middle_node(position, existing_node_id)
    #
    #     if len(list(self.graph.predecessors(old_node.id))) > 1:
    #         raise ValueError("Tried to replace a node with multiple parents")
    #
    #     self.all_nodes.pop(old_node.id)
    #     self.all_nodes[replacement_node.id] = replacement_node
    #
    #     parents = self.get_parents(old_node)
    #     self.graph.remove_node()
    #     nx.relabel_nodes(self.graph, {old_node.id: replacement_node.id}, copy=False)
    #     self.graph.nodes[replacement_node.id]['label'] = replacement_node.get_label()

    def remove_chain(self, id_chain: list, heal=True, replace=False, new_chain_nodes: list = None):
        """
        Removes an entire chain of linked nodes
        :param id_chain: list of node ids to be removed
        :param heal: re-connect nodes after removal
        :param replace: replace the removed chain with a new chain
        :param new_chain_nodes: chain to replace with

        """
        start_node = self.get_a_middle_node(node_id=id_chain[0])
        end_node = self.get_a_middle_node(node_id=id_chain[-1])
        start_parents = get_parents(self, start_node)
        end_children = self.get_children(end_node)

        for node_id in id_chain:
            self.all_nodes.pop(node_id)
            self.graph.remove_nodes_from(id_chain)

        if heal:
            for parent in start_parents:
                for child in end_children:
                    self.graph.add_edge(parent.id, child.id)

        if replace:
            current_parents = start_parents
            for node in new_chain_nodes:
                self.register_node(node)
                for parent in current_parents:
                    self.graph.add_edge(parent.id, node.id)
                current_parents = [node]

            for child in end_children:
                self.graph.add_edge(new_chain_nodes[-1].id, child.id)

    def get_successor_chain_ids(self, start_id: int) -> list:
        """
        Gets the ids of all nodes that are successors of the start_id.
        Recursively gets them all the way back
        to the input node
        Args:
            start_id: node to start from
        """
        node_ids = []
        current_node_id = start_id

        while True:
            successors = list(self.graph.successors(current_node_id))
            if len(successors) != 1:
                break
            node_ids.append(current_node_id)
            current_node_id = successors[0]

        return node_ids

    def mutate_node(self, position: int):
        """
        Mutates the node at the given position
        :param position: position of the node to mutate

        """
        nodes = self.get_mutatable_nodes()
        if position > (len(nodes) - 1):
            raise ValueError("Invalid request to mutate node: Length = " +
                             str(len(nodes)) + " position given as: " + str(position))

        node_to_mutate = list(nodes.values())[position]
        node_to_mutate.mutate()

    def store_weights(self, model: tf.keras.Model, direction_into_tn: bool):
        """
        Stores model weights into the genome or gets them from the genome
        :param model: model to store weights from or load weights to
        :param direction_into_tn: True stores weights into the network,
        False gets them and loads them into the model
        :return:
        """
        layers = model.layers
        nodes_to_cache = self.get_nodes_can_cache()
        for node in nodes_to_cache.values():
            if node.weights is None:
                continue

            for layer in layers:
                if layer.input.name == node.keras_tensor_input_name:
                    if direction_into_tn:
                        weights = layer.get_weights()
                        if len(weights) == node.required_num_weights:
                            node.weights = weights
                            break
                    else:
                        node_weights = node.weights
                        layer_weights = layer.get_weights()

                        if len(node_weights) != len(layer_weights):
                            continue

                        for i in range(len(node_weights)):
                            if node_weights[i].shape != layer_weights[i].shape:
                                continue

                        layer.set_weights(node_weights)

    def build_model(self) -> tf.keras.Model:
        """
        Builds a model from this network
        :return: NN model
        """
        inputs = self.all_nodes[self.input_ids[0]](self.all_nodes, self.graph)
        outputs = self.all_nodes[self.output_ids[0]](self.all_nodes, self.graph)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def get_output_nodes(self) -> dict:
        """
        Gets output nodes
        :return: dict of output nodes
        """
        return {k: v for (k, v) in self.all_nodes.items() if k in self.output_ids}

    def get_input_nodes(self) -> dict:
        """

        :return: dict of input nodes
        """
        return {k: v for (k, v) in self.all_nodes.items() if k in self.input_ids}

    def get_not_input_nodes(self) -> dict:
        """

        :return: dict of all nodes that are not inputs
        """
        return {k: v for (k, v) in self.all_nodes.items() if k not in self.input_ids}

    def get_middle_nodes(self) -> dict:
        """

        :return: dict of all nodes that are neither input nor outputs
        """
        return {k: v for (k, v) in self.all_nodes.items()
                if (k not in self.input_ids) and (k not in self.output_ids)}

    def get_mutatable_nodes(self) -> dict:
        """

        :return: dict of nodes which can be mutated
        """
        return {k: v for (k, v) in self.all_nodes.items() if v.can_mutate}

    def get_cx_nodes(self) -> dict:
        """

        :return: dict of nodes which can be crossed over with another genome
        """
        return {k: v for (k, v) in self.all_nodes.items() if
                (len(list(self.graph.predecessors(k))) == 1) and
                (len(list(self.graph.successors(k))) == 1)}

    def get_valid_insert_positions(self) -> dict:
        """

        :return: dict of nodes which it would be valid to insert a node before.
        """
        return {k: v for (k, v) in self.get_not_input_nodes().items() if
                (len(list(self.graph.predecessors(k))) == 1)}

    def get_nodes_can_cache(self) -> dict:
        """

        :return: dict of nodes that can cache weights
        """
        return {k: v for (k, v) in self.all_nodes.items() if v.node_allows_cache_training}

    def get_nodes_from_ids(self, id_list: list) -> dict:
        """

        :param id_list: list of ids to get nodes for
        :return: dict of nodes that correspond to the given ids
        """
        return {k: v for (k, v) in self.all_nodes.items() if k in id_list}

    def get_parent_chain_ids(self, node) -> list:
        """

        :param node: node to start at
        :return: list of node ids of parents recursively back to input node
        """
        current_nodes = [node.id]
        current_parents_ids = []
        all_parents_ids = []

        while True:
            for node_id in current_nodes:
                current_parents_ids += list(self.graph.predecessors(node_id))

            if len(current_parents_ids) == 0:
                break

            all_parents_ids += current_parents_ids
            current_nodes = list(set(current_parents_ids))
            current_parents_ids = []

        return list(set(all_parents_ids))

    def get_children(self, node) -> list:
        """

        :param node: node to get children for
        :return: list of nodes that are successors of the given node
        """
        child_ids = self.graph.successors(node.id)
        children = []
        for c_id in child_ids:
            children.append(self.all_nodes[c_id])
        return children

    def get_a_middle_node(self, position=None, node_id=None) -> tensor_node.TensorNode:
        """
        Gets a middle node as specified by either its position or its node id
        :param position: position to get
        :param node_id: node id to get
        :return: the requested node
        """
        if (position is None) and (node_id is None):
            raise ValueError("Must specify either position or node_id")

        nodes = self.get_middle_nodes()
        node_selected = None
        if position is not None:
            if position > (len(nodes) - 1):
                raise ValueError("Invalid request to get node: Length = " +
                                 str(len(nodes)) + " position given as: " + str(position))
            node_selected = list(nodes.values())[position]
        else:
            try:
                node_selected = nodes[node_id]
            except KeyError:
                print("Invalid request to get node_id: " + str(node_id))
        return node_selected

    def __len__(self):
        return len(self.all_nodes)

    def draw_graphviz_svg(self, filepath=None):
        """
        Saves a graph png to file of this genome
        :return:
        """
        graph = nx.drawing.nx_pydot.to_pydot(self.graph)

        if filepath is None:
            path = f'tensor_net_{self.net_id}_{time.time()}.png'
        else:
            path = filepath
        graph.write_svg(path)

    def plot(self):
        """
        Plots a graph of this genome into a pop-up window
        :return:
        """
        pos = nx.nx_pydot.graphviz_layout(self.graph)
        nx.draw_networkx(self.graph, pos)
        plt.show()

    def save(self, filename: str):
        """
        Save this network to file
        :param filename: full path to the file
        :return:
        """
        with open(filename, 'w+', encoding='latin-1') as file:
            json.dump(self, fp=file, cls=tensor_encoder.TensorEncoder)

    @staticmethod
    def load(filename: str):
        """
        Creates a new tensor network from file
        :param filename: full path to load from
        :return:
        """
        with open(filename, 'r', encoding='latin-1') as file:
            tn_dict = json.load(file)
            tensor_net = TensorNetwork.deserialize(tn_dict)
            return tensor_net


def get_parents(tensor_net: TensorNetwork, node: tensor_node.TensorNode) -> list:
    """
    Gets a nodes parents
    :param tensor_net: the tensor network the node i in
    :param node: the node to get parents for
    :return: a list of the immediate predecessors of the given node
    """
    parents_ids = tensor_net.graph.predecessors(node.id)  # iterator
    parents = []
    for p_id in parents_ids:
        parents.append(tensor_net.all_nodes[p_id])
    return parents


def cx_single_node(tensor_net: TensorNetwork, other_tn: TensorNetwork):
    """
    Does single node (only swaps one node) crossover between two tensor networks.
    :param tensor_net: one of the two networks
    :param other_tn: the second network
    :return:
    """
    tn_cx_nodes = tensor_net.get_cx_nodes()
    other_cx_nodes = other_tn.get_cx_nodes()

    if len(tn_cx_nodes) == 0 or len(other_cx_nodes) == 0:
        return

    tn_node_id = random.choice(list(tn_cx_nodes.keys()))
    tn_node = tn_cx_nodes[tn_node_id]
    other_node_id = random.choice(list(other_cx_nodes.keys()))
    other_node = other_cx_nodes[other_node_id]

    tensor_net.delete_node(node_id=tn_node_id, replacement_node=other_node)
    other_tn.delete_node(node_id=other_node_id, replacement_node=tn_node)

# def cx_chain(tn: TensorNetwork, other_tn: TensorNetwork):
#     tn_cx_nodes = tn.get_cx_nodes()
#     other_cx_nodes = other_tn.get_cx_nodes()
#
#     if len(tn_cx_nodes) == 0 or len(other_cx_nodes) == 0:
#         return
#
#     tn_node_id = random.choice(list(tn_cx_nodes.keys()))
#     other_node_id = random.choice(list(other_cx_nodes.keys()))
#
#     tn_chain_ids = tn.get_successor_chain_ids(tn_node_id)
#     tn_nodes = list(tn.get_nodes_from_ids(tn_chain_ids).values())
#     other_chain_ids = other_tn.get_successor_chain_ids(other_node_id)
#     other_chain_nodes = list(other_tn.get_nodes_from_ids(other_chain_ids).values())
#
#     tn.remove_chain(tn_chain_ids, heal=False, replace=True, new_chain_nodes=other_chain_nodes)
#     other_tn.remove_chain(other_chain_ids, heal=False, replace=True, new_chain_nodes=tn_nodes)
