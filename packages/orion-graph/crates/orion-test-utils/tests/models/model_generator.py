import onnx
from onnx import TensorProto, numpy_helper
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info)
from onnx.checker import check_model
import numpy as np

def model_generator(graph, test_name):
    onnx_model = make_model(graph)
    del onnx_model.opset_import[:]

    op_set = onnx_model.opset_import.add()
    op_set.domain = ''
    op_set.version = 19
    onnx_model.ir_version = 9

    check_model(onnx_model)
    onnx.save(onnx_model, test_name)


def model_generator_test_sin():
    x = make_tensor_value_info('x', TensorProto.DOUBLE, [2, 2])
    y = make_tensor_value_info('y', TensorProto.DOUBLE, [2, 2])

    node1 = make_node('Sin', ['x'], ['y'])

    graph = make_graph([node1],  
                        'sin',
                        [x], 
                        [y]) 
    model_generator(graph, 'test_sin.onnx')

def model_generator_test_add():
    a = make_tensor_value_info('a', TensorProto.DOUBLE, [2, 2])
    b = make_tensor_value_info('b', TensorProto.DOUBLE, [2, 2])
    y = make_tensor_value_info('y', TensorProto.DOUBLE, [2, 2])

    node1 = make_node('Add', ['a', 'b'], ['y'])

    graph = make_graph([node1],  
                        'add',
                        [a, b], 
                        [y]) 
    model_generator(graph, 'test_add.onnx')
    
def model_generator_test_add_broadcast():
    a = make_tensor_value_info('a', TensorProto.DOUBLE, [2, 2])
    b = make_tensor_value_info('b', TensorProto.DOUBLE, [2])
    y = make_tensor_value_info('y', TensorProto.DOUBLE, [2, 2])

    node1 = make_node('Add', ['a', 'b'], ['y'])

    graph = make_graph([node1],  
                        'add',
                        [a, b], 
                        [y]) 
    model_generator(graph, 'test_add_broadcast.onnx')
    
def model_generator_test_add_broadcast_scalar():
    a = make_tensor_value_info('a', TensorProto.DOUBLE, [2, 2])
    b = make_tensor_value_info('b', TensorProto.DOUBLE, [1])
    y = make_tensor_value_info('y', TensorProto.DOUBLE, [2, 2])

    node1 = make_node('Add', ['a', 'b'], ['y'])

    graph = make_graph([node1],  
                        'add',
                        [a, b], 
                        [y]) 
    model_generator(graph, 'test_add_broadcast_scalar.onnx')

def model_generator_test_mul():
    a = make_tensor_value_info('a', TensorProto.DOUBLE, [2, 2])
    b = make_tensor_value_info('b', TensorProto.DOUBLE, [2, 2])
    y = make_tensor_value_info('y', TensorProto.DOUBLE, [2, 2])

    node1 = make_node('Mul', ['a', 'b'], ['y'])

    graph = make_graph([node1],  
                        'mul',
                        [a, b], 
                        [y]) 
    model_generator(graph, 'test_mul.onnx')
    
def model_generator_test_mul_broadcast():
    a = make_tensor_value_info('a', TensorProto.DOUBLE, [2, 2])
    b = make_tensor_value_info('b', TensorProto.DOUBLE, [2])
    y = make_tensor_value_info('y', TensorProto.DOUBLE, [2, 2])

    node1 = make_node('Mul', ['a', 'b'], ['y'])

    graph = make_graph([node1],  
                        'mul',
                        [a, b], 
                        [y]) 
    model_generator(graph, 'test_mul_broadcast.onnx')
    
def model_generator_test_mul_broadcast_scalar():
    a = make_tensor_value_info('a', TensorProto.DOUBLE, [2, 2])
    b = make_tensor_value_info('b', TensorProto.DOUBLE, [1])
    y = make_tensor_value_info('y', TensorProto.DOUBLE, [2, 2])

    node1 = make_node('Mul', ['a', 'b'], ['y'])

    graph = make_graph([node1],  
                        'mul',
                        [a, b], 
                        [y]) 
    model_generator(graph, 'test_mul_broadcast_scalar.onnx')
    
def model_generator_test_log():
    x = make_tensor_value_info('x', TensorProto.DOUBLE, [2, 2])
    y = make_tensor_value_info('y', TensorProto.DOUBLE, [2, 2])

    node1 = make_node('Log', ['x'], ['y'])

    graph = make_graph([node1],  
                        'log',
                        [x], 
                        [y]) 
    model_generator(graph, 'test_log.onnx')
    
def model_generator_test_exp():
    x = make_tensor_value_info('x', TensorProto.DOUBLE, [2, 2])
    y = make_tensor_value_info('y', TensorProto.DOUBLE, [2, 2])

    node1 = make_node('Exp', ['x'], ['y'])

    graph = make_graph([node1],  
                        'exp',
                        [x], 
                        [y]) 
    model_generator(graph, 'test_exp.onnx')
    
def model_generator_test_sqrt():
    x = make_tensor_value_info('x', TensorProto.DOUBLE, [2, 2])
    y = make_tensor_value_info('y', TensorProto.DOUBLE, [2, 2])

    node1 = make_node('Sqrt', ['x'], ['y'])

    graph = make_graph([node1],  
                        'sqrt',
                        [x], 
                        [y]) 
    model_generator(graph, 'test_sqrt.onnx')
    
def model_generator_test_recip():
    x = make_tensor_value_info('x', TensorProto.DOUBLE, [2, 2])
    y = make_tensor_value_info('y', TensorProto.DOUBLE, [2, 2])

    node1 = make_node('Reciprocal', ['x'], ['y'])

    graph = make_graph([node1],  
                        'reciprocal',
                        [x], 
                        [y]) 
    model_generator(graph, 'test_recip.onnx')
    
def model_generator_test_mod():
    a = make_tensor_value_info('a', TensorProto.DOUBLE, [2, 1])
    b = make_tensor_value_info('b', TensorProto.DOUBLE, [1, 3])
    y = make_tensor_value_info('y', TensorProto.DOUBLE, [2, 3])

    node1 = make_node('Mod', ['a', 'b'], ['y'], fmod = 1)

    graph = make_graph([node1],  
                        'mod',
                        [a, b], 
                        [y]) 
    model_generator(graph, 'test_mod.onnx')
    
def model_generator_test_less():
    a = make_tensor_value_info('a', TensorProto.DOUBLE, [2, 1])
    b = make_tensor_value_info('b', TensorProto.DOUBLE, [1, 3])
    y = make_tensor_value_info('y', TensorProto.BOOL, [2, 3])

    node1 = make_node('Less', ['a', 'b'], ['y'])

    graph = make_graph([node1],  
                        'less',
                        [a, b], 
                        [y]) 
    model_generator(graph, 'test_less.onnx')
    
def model_generator_test_reducesum_keepdim():
    x = make_tensor_value_info('x', TensorProto.DOUBLE, [3, 2, 2])
    y = make_tensor_value_info('y', TensorProto.DOUBLE, [1, 1, 1])

    node1 = make_node('ReduceSum', ['x'], ['y'], keepdims=1)

    graph = make_graph([node1],  
                        'reducesum',
                        [x], 
                        [y]) 
    model_generator(graph, 'test_reducesum_keepdim.onnx')
    
def model_generator_test_reducesum_not_keepdim():
    x = make_tensor_value_info('x', TensorProto.DOUBLE, [3, 2, 2])
    y = make_tensor_value_info('y', TensorProto.DOUBLE, [1])

    node1 = make_node('ReduceSum', ['x'], ['y'], keepdims=0)

    graph = make_graph([node1],  
                        'reducesum',
                        [x], 
                        [y]) 
    model_generator(graph, 'test_reducesum_not_keepdim.onnx')
    
def model_generator_test_reducemax_keepdim():
    x = make_tensor_value_info('x', TensorProto.DOUBLE, [3, 2, 2])
    y = make_tensor_value_info('y', TensorProto.DOUBLE, [1, 1, 1])

    node1 = make_node('ReduceMax', ['x'], ['y'], keepdims=1)

    graph = make_graph([node1],  
                        'reducemax',
                        [x], 
                        [y]) 
    model_generator(graph, 'test_reducemax_keepdim.onnx')
    
def model_generator_test_reducemax_not_keepdim():
    x = make_tensor_value_info('x', TensorProto.DOUBLE, [3, 2, 2])
    y = make_tensor_value_info('y', TensorProto.DOUBLE, [1])

    node1 = make_node('ReduceMax', ['x'], ['y'], keepdims=0)

    graph = make_graph([node1],  
                        'reducemax',
                        [x], 
                        [y]) 
    model_generator(graph, 'test_reducemax_not_keepdim.onnx')
    
def model_generator_test_matmul_ax_a():
    A = make_tensor_value_info('A', TensorProto.DOUBLE, [4])
    B = make_tensor_value_info('B', TensorProto.DOUBLE, [4])
    output = make_tensor_value_info('output', TensorProto.DOUBLE, [4])

    node1 = make_node('MatMul', ['A', 'B'], ['output'])

    graph = make_graph([node1],  
                        'matmul',
                        [A, B], 
                        [output]) 
    model_generator(graph, 'test_matmul_ax_a.onnx')
    
def model_generator_test_matmul_abx_b():
    A = make_tensor_value_info('A', TensorProto.DOUBLE, [3, 4])
    B = make_tensor_value_info('B', TensorProto.DOUBLE, [4])
    output = make_tensor_value_info('output', TensorProto.DOUBLE, [3])

    node1 = make_node('MatMul', ['A', 'B'], ['output'])

    graph = make_graph([node1],  
                        'matmul',
                        [A, B], 
                        [output]) 
    model_generator(graph, 'test_matmul_abx_b.onnx')
    

def model_generator_test_matmul_ax_ab():
    A = make_tensor_value_info('A', TensorProto.DOUBLE, [3])
    B = make_tensor_value_info('B', TensorProto.DOUBLE, [3, 4])
    output = make_tensor_value_info('output', TensorProto.DOUBLE, [4])

    node1 = make_node('MatMul', ['A', 'B'], ['output'])

    graph = make_graph([node1],  
                        'matmul',
                        [A, B], 
                        [output]) 
    model_generator(graph, 'test_matmul_ax_ab.onnx')
    
def model_generator_test_matmul_abcx_cd():
    A = make_tensor_value_info('A', TensorProto.DOUBLE, [3, 2, 2])
    B = make_tensor_value_info('B', TensorProto.DOUBLE, [2, 6])
    output = make_tensor_value_info('output', TensorProto.DOUBLE, [3, 2, 6])

    node1 = make_node('MatMul', ['A', 'B'], ['output'])

    graph = make_graph([node1],  
                        'matmul',
                        [A, B], 
                        [output]) 
    model_generator(graph, 'test_matmul_abcx_cd.onnx')
    
def model_generator_test_matmul_abcx_acd():
    A = make_tensor_value_info('A', TensorProto.DOUBLE, [2, 3, 4])
    B = make_tensor_value_info('B', TensorProto.DOUBLE, [2, 4, 3])
    output = make_tensor_value_info('output', TensorProto.DOUBLE, [2, 3, 3])

    node1 = make_node('MatMul', ['A', 'B'], ['output'])

    graph = make_graph([node1],  
                        'matmul',
                        [A, B], 
                        [output]) 
    model_generator(graph, 'test_matmul_abcx_acd.onnx')
    
def model_generator_test_matmul_abcdx_abde():
    A = make_tensor_value_info('A', TensorProto.DOUBLE, [2, 2, 3, 2])
    B = make_tensor_value_info('B', TensorProto.DOUBLE, [2, 2, 2, 3])
    output = make_tensor_value_info('output', TensorProto.DOUBLE, [2, 2, 3, 3])

    node1 = make_node('MatMul', ['A', 'B'], ['output'])

    graph = make_graph([node1],  
                        'matmul',
                        [A, B], 
                        [output]) 
    model_generator(graph, 'test_matmul_abcdx_abde.onnx')
    
def model_generator_test_matmul_abcdex_abcef():
    A = make_tensor_value_info('A', TensorProto.DOUBLE, [2, 2, 2, 3, 2])
    B = make_tensor_value_info('B', TensorProto.DOUBLE, [2, 2, 2, 2, 3])
    output = make_tensor_value_info('output', TensorProto.DOUBLE, [2, 2, 3, 3])

    node1 = make_node('MatMul', ['A', 'B'], ['output'])

    graph = make_graph([node1],  
                        'matmul',
                        [A, B], 
                        [output]) 
    model_generator(graph, 'test_matmul_abcdex_abcef.onnx')
    
def model_generator_test_matmul():
    A = make_tensor_value_info('A', TensorProto.DOUBLE, [3, 3])
    B = make_tensor_value_info('B', TensorProto.DOUBLE, [3, 3])
    output = make_tensor_value_info('output', TensorProto.DOUBLE, [3, 3])

    node1 = make_node('MatMul', ['A', 'B'], ['output'])

    graph = make_graph([node1],  
                        'matmul',
                        [A, B], 
                        [output]) 
    model_generator(graph, 'test_matmul.onnx')
    
def model_generator_test_matmul_identity_matrix():
    A = make_tensor_value_info('A', TensorProto.DOUBLE, [3, 3])
    B = make_tensor_value_info('B', TensorProto.DOUBLE, [3, 3])
    output = make_tensor_value_info('output', TensorProto.DOUBLE, [3, 3])

    node1 = make_node('MatMul', ['A', 'B'], ['output'])

    graph = make_graph([node1],  
                        'matmul',
                        [A, B], 
                        [output]) 
    model_generator(graph, 'test_matmul_identity_matrix.onnx')
    
def model_generator_test_matmul_zero_matrix():
    A = make_tensor_value_info('A', TensorProto.DOUBLE, [3, 3])
    B = make_tensor_value_info('B', TensorProto.DOUBLE, [3, 3])
    output = make_tensor_value_info('output', TensorProto.DOUBLE, [3, 3])

    node1 = make_node('MatMul', ['A', 'B'], ['output'])

    graph = make_graph([node1],  
                        'matmul',
                        [A, B], 
                        [output]) 
    model_generator(graph, 'test_matmul_zero_matrix.onnx')
    
def model_generator_test_matmul_non_square_matrix():
    A = make_tensor_value_info('A', TensorProto.DOUBLE, [2, 3])
    B = make_tensor_value_info('B', TensorProto.DOUBLE, [3, 2])
    output = make_tensor_value_info('output', TensorProto.DOUBLE, [2, 2])

    node1 = make_node('MatMul', ['A', 'B'], ['output'])

    graph = make_graph([node1],  
                        'matmul',
                        [A, B], 
                        [output]) 
    model_generator(graph, 'test_matmul_non_square_matrix.onnx')

#Test Initializers
def model_generator_test_matmul_initializer_A():
    value = np.array([0.5, -0.6])
    
    A = numpy_helper.from_array(value, name='A')
    B = make_tensor_value_info('B', TensorProto.DOUBLE, [2, 3])
    
    output = make_tensor_value_info('output', TensorProto.DOUBLE, [2, 3])
    
    node1 = make_node('MatMul', ['A', 'B'], ['output'])

    graph = make_graph([node1],  
                        'matmul',
                        [B], 
                        [output], 
                        [A]) #initializer
    model_generator(graph, 'test_matmul_initializer_A.onnx')
    
#Test Initializers
def model_generator_test_matmul_initializer_B():
    value = np.array([0.5, -0.6])
    
    B = numpy_helper.from_array(value, name='B')
    A = make_tensor_value_info('A', TensorProto.DOUBLE, [3, 2])
    
    output = make_tensor_value_info('output', TensorProto.DOUBLE, [3, 2])
    
    node1 = make_node('MatMul', ['A', 'B'], ['output'])

    graph = make_graph([node1],  
                        'matmul',
                        [A], 
                        [output], 
                        [B]) #initializer
    model_generator(graph, 'test_matmul_initializer_B.onnx')
    
#Test multi Nodes 
def model_generator_test_matmul_multi_nodes():
    value = np.array([[0.5, 0.5, 0.5], [2, 2, 2]])
    Ax = numpy_helper.from_array(value, name='Ax')
    
    value = np.array([[1, 1], [2, 2], [0.5, 0.5]])
    Bx = numpy_helper.from_array(value, name='Bx')
    
    A = make_tensor_value_info('A', TensorProto.DOUBLE, [2, 3])
    B = make_tensor_value_info('B', TensorProto.DOUBLE, [3, 2])

    output = make_tensor_value_info('output', TensorProto.DOUBLE, [2, 2])
    
    node1 = make_node('Add', ['A', 'Ax'], ['add_A_Ax'])
    node2 = make_node('Add', ['B', 'Bx'], ['add_B_Bx'])
    node3 = make_node('MatMul', ['add_A_Ax', 'add_B_Bx'], ['output'])

    graph = make_graph([node1, node2, node3],  
                        'matmul',
                        [A, B], 
                        [output], 
                        [Ax, Bx]) #initializer
    model_generator(graph, 'test_matmul_multi_nodes.onnx')

    

if __name__ == "__main__":
    model_generator_test_sin()
    model_generator_test_add()
    model_generator_test_add_broadcast()
    model_generator_test_add_broadcast_scalar()
    model_generator_test_mul()
    model_generator_test_mul_broadcast()
    model_generator_test_mul_broadcast_scalar()
    model_generator_test_log()
    model_generator_test_exp()
    model_generator_test_sqrt()
    model_generator_test_recip()
    model_generator_test_mod()
    model_generator_test_less()
    model_generator_test_reducesum_keepdim()
    model_generator_test_reducesum_not_keepdim()
    model_generator_test_reducemax_keepdim()
    model_generator_test_reducemax_not_keepdim()
    model_generator_test_matmul_ax_a()
    model_generator_test_matmul_abx_b()
    model_generator_test_matmul_ax_ab()
    model_generator_test_matmul_abcx_cd()
    model_generator_test_matmul_abcx_acd()
    model_generator_test_matmul_abcdx_abde()
    model_generator_test_matmul_abcdex_abcef()
    model_generator_test_matmul()
    model_generator_test_matmul_identity_matrix()
    model_generator_test_matmul_zero_matrix()
    model_generator_test_matmul_non_square_matrix()
    model_generator_test_matmul_initializer_A()
    model_generator_test_matmul_initializer_B()
    model_generator_test_matmul_multi_nodes()