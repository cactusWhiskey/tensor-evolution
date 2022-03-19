import json

import tensor_network
import tensor_node


class TensorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, tensor_node.TensorNode):
            return obj.serialize()
        elif isinstance(obj, tensor_network.TensorNetwork):
            return obj.serialize()
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)
