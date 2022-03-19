import copy
import itertools
import json

import tensorflow as tf
import networkx as nx
import random
from matplotlib import pyplot as plt

import tensor_encoder
import tensor_node


class TensorNetwork:
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
            self.create_inputs(input_shapes)
            self.create_outputs(output_units)

    def serialize(self) -> dict:
        serial_dict = copy.deepcopy(self.__dict__)
        serial_dict['graph'] = nx.node_link_data(self.graph)

        for node_id, node in self.all_nodes.items():
            serial_node = node.serialize()
            (serial_dict['all_nodes'])[node_id] = serial_node

        return serial_dict

    @staticmethod
    def deserialize(tn_dict: dict):
        tn = TensorNetwork(None, None, False)
        tn.__dict__ = tn_dict
        tn.graph = nx.node_link_graph(tn_dict['graph'])

        for node_id, serial_node in tn.all_nodes.items():
            tn.all_nodes[node_id] = tensor_node.TensorNode.deserialize(serial_node)

        for index, shape in enumerate(tn.input_shapes):
            tn.input_shapes[index] = tuple(shape)

        return tn

    def __deepcopy__(self, memodict={}):
        return self.clone()

    def clone(self):
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

    def create_inputs(self, input_shapes: list):
        if len(input_shapes) > 1:
            raise ValueError("Multiple inputs not yet supported")

        for shape in input_shapes:
            node = tensor_node.InputNode(shape)
            self.register_node(node)

    def create_outputs(self, output_units: list):
        for units in output_units:
            node = tensor_node.OutputNode(units)
            self.register_node(node)

            for input_id in self.input_ids:
                self.graph.add_edge(input_id, node.id)

    def insert_node(self, new_node: tensor_node.TensorNode, position: int):
        # inserts node before the given position
        # positions refer to the index in the "non_input_nodes" list, which is kept in no particular oder
        nodes = self.get_valid_insert_positions()
        if position > (len(nodes) - 1):
            raise ValueError("Invalid request to insert node: Length = " +
                             str(len(nodes)) + " position given as: " + str(position))

        child_node = list(nodes.values())[position]  # node at position becomes child
        parent = get_parents(self, child_node)[0]  # valid insert positions only have a single parent

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
        nodes = self.get_mutatable_nodes()
        if position > (len(nodes) - 1):
            raise ValueError("Invalid request to mutate node: Length = " +
                             str(len(nodes)) + " position given as: " + str(position))

        node_to_mutate = list(nodes.values())[position]
        node_to_mutate.mutate()

    def store_weights(self, model: tf.keras.Model, direction_into_tn: bool):
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
                        try:
                            node_weights = node.weights
                            layer_weights = layer.get_weights()

                            if len(node_weights) != len(layer_weights):
                                continue

                            for i in range(len(node_weights)):
                                if node_weights[i].shape != layer_weights[i].shape:
                                    continue

                            layer.set_weights(node_weights)
                        except:
                            print("debug")

    def build_model(self) -> tf.keras.Model:
        inputs = self.all_nodes[self.input_ids[0]](self.all_nodes, self.graph)
        outputs = self.all_nodes[self.output_ids[0]](self.all_nodes, self.graph)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def get_output_nodes(self) -> dict:
        return {k: v for (k, v) in self.all_nodes.items() if k in self.output_ids}

    def get_input_nodes(self) -> dict:
        return {k: v for (k, v) in self.all_nodes.items() if k in self.input_ids}

    def get_not_input_nodes(self) -> dict:
        return {k: v for (k, v) in self.all_nodes.items() if k not in self.input_ids}

    def get_middle_nodes(self) -> dict:
        return {k: v for (k, v) in self.all_nodes.items()
                if (k not in self.input_ids) and (k not in self.output_ids)}

    def get_mutatable_nodes(self) -> dict:
        return {k: v for (k, v) in self.all_nodes.items() if v.can_mutate}

    def get_cx_nodes(self) -> dict:
        return {k: v for (k, v) in self.all_nodes.items() if
                (len(list(self.graph.predecessors(k))) == 1) and
                (len(list(self.graph.successors(k))) == 1)}

    def get_valid_insert_positions(self) -> dict:
        return {k: v for (k, v) in self.get_not_input_nodes().items() if
                (len(list(self.graph.predecessors(k))) == 1)}

    def get_nodes_can_cache(self) -> dict:
        return {k: v for (k, v) in self.all_nodes.items() if v.node_allows_cache_training}

    def get_nodes_from_ids(self, id_list: list) -> dict:
        return {k: v for (k, v) in self.all_nodes.items() if k in id_list}

    def get_parent_chain_ids(self, node) -> list:
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
        child_ids = self.graph.successors(node.id)
        children = []
        for c_id in child_ids:
            children.append(self.all_nodes[c_id])
        return children

    def get_a_middle_node(self, position=None, node_id=None) -> tensor_node.TensorNode:
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

    def draw_graphviz(self):
        py_graph = nx.nx_agraph.to_agraph(self.graph)
        py_graph.layout('dot')
        py_graph.draw('tensor_net_' + str(self.net_id) + '.png')

    def plot(self):
        pos = nx.nx_pydot.graphviz_layout(self.graph)
        nx.draw_networkx(self.graph, pos)
        plt.show()

    def save(self, filename: str):
        with open(filename, 'w') as file:
            json.dump(self, fp=file, cls=tensor_encoder.TensorEncoder)

    @staticmethod
    def load(filename: str):
        with open(filename, 'r') as file:
            tn_dict = json.load(file)
            tn = TensorNetwork.deserialize(tn_dict)
            return tn


def get_parents(tn: TensorNetwork, node: tensor_node.TensorNode) -> list:
    parents_ids = tn.graph.predecessors(node.id)  # iterator
    parents = []
    for p_id in parents_ids:
        parents.append(tn.all_nodes[p_id])
    return parents


def cx_single_node(tn: TensorNetwork, other_tn: TensorNetwork):
    tn_cx_nodes = tn.get_cx_nodes()
    other_cx_nodes = other_tn.get_cx_nodes()

    if len(tn_cx_nodes) == 0 or len(other_cx_nodes) == 0:
        return

    tn_node_id = random.choice(list(tn_cx_nodes.keys()))
    tn_node = tn_cx_nodes[tn_node_id]
    other_node_id = random.choice(list(other_cx_nodes.keys()))
    other_node = other_cx_nodes[other_node_id]

    tn.delete_node(node_id=tn_node_id, replacement_node=other_node)
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
