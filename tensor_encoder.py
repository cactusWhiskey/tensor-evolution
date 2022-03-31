"""JSON encoders for objects in this library"""
import json

import evo_config
import tensor_evolution
import tensor_network
import tensor_node


class TensorEncoder(json.JSONEncoder):
    """JSON encoders for objects in this library"""
    def default(self, obj):
        if isinstance(obj, tensor_node.TensorNode):
            return obj.serialize()
        if isinstance(obj, tensor_network.TensorNetwork):
            return obj.serialize()
        if isinstance(obj, evo_config.EvoConfig):
            return obj.serialize()
        if isinstance(obj, tensor_evolution.EvolutionWorker):
            return obj.serialize()
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)
