"""JSON encoders for objects in this library"""
import json
from tensorEvolution import tensor_evolution
from tensorEvolution import tensor_network
from tensorEvolution.nodes import tensor_node


class TensorEncoder(json.JSONEncoder):
    """JSON encoders for objects in this library"""
    def default(self, obj):
        if isinstance(obj, tensor_node.TensorNode):
            return obj.serialize()
        if isinstance(obj, tensor_network.TensorNetwork):
            return obj.serialize()
        if isinstance(obj, tensor_evolution.EvolutionWorker):
            return obj.serialize()
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)
