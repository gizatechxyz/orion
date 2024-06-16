import onnx
from onnx import TensorProto
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info)
from onnx.checker import check_model

## Generates logsoftmax two nodes ONNX model

x = make_tensor_value_info('x', TensorProto.FLOAT, [None, None, None])
y = make_tensor_value_info('y', TensorProto.FLOAT, [None, None, None])

node1 = make_node('ReduceL2', ['x'], ['intermediate'])
node2 = make_node('ReduceL2', ['intermediate'], ['y'])

graph = make_graph([node1, node2],  
                    'reduce_l2_two_nodes', 
                    [x], 
                    [y]) 

onnx_model = make_model(graph)

del onnx_model.opset_import[:]

op_set = onnx_model.opset_import.add()
op_set.domain = ''
op_set.version = 19
onnx_model.ir_version = 9

check_model(onnx_model)
onnx.save(onnx_model, 'reduce_l2_two_nodes.onnx')




