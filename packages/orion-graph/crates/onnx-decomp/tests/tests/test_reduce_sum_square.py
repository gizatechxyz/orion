import numpy as np
from utils_onnx_decomp import run_test_onnx_decomp
    

# ONNX Tests

def test_reduce_sum_square_keepdims_random():
    name = 'reduce_sum_square_keepdims_random.onnx'
    axes = np.array([1], dtype=np.int64)
    data = np.random.uniform(-10, 10, [3, 2, 2]).astype(np.float32)
    run_test_onnx_decomp(name,  {'data': data, 'axes': axes}, 'reduce_sum_square')
    
def test_reduce_sum_square_keepdims_example():
    name = 'reduce_sum_square_keepdims_example.onnx'
    axes = np.array([1], dtype=np.int64)
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
    run_test_onnx_decomp(name,  {'data': data, 'axes': axes}, 'reduce_sum_square')
       
def test_reduce_sum_square_do_not_keepdims_random():
    name = 'reduce_sum_square_do_not_keepdims_random.onnx'
    axes = np.array([1], dtype=np.int64)
    data = np.random.uniform(-10, 10, [3, 2, 2]).astype(np.float32)
    run_test_onnx_decomp(name,  {'data': data, 'axes': axes}, 'reduce_sum_square')
    
def test_reduce_sum_square_do_not_keepdims_example():
    name = 'reduce_sum_square_do_not_keepdims_example.onnx'
    axes = np.array([1], dtype=np.int64)
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
    run_test_onnx_decomp(name,  {'data': data, 'axes': axes}, 'reduce_sum_square')
    
def test_reduce_sum_square_negative_axes_keepdims_random():
    name = 'reduce_sum_square_negative_axes_keepdims_random.onnx'
    axes = np.array([-2], dtype=np.int64)
    data = np.random.uniform(-10, 10, [3, 2, 2]).astype(np.float32)
    run_test_onnx_decomp(name,  {'data': data, 'axes': axes}, 'reduce_sum_square')
    
def test_reduce_sum_square_negative_axes_keepdims_example():
    name = 'reduce_sum_square_negative_axes_keepdims_example.onnx'
    axes = np.array([-2], dtype=np.int64)
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
    run_test_onnx_decomp(name,  {'data': data, 'axes': axes}, 'reduce_sum_square')
    
def test_reduce_sum_square_default_axes_keepdims_random():
    name = 'reduce_sum_square_default_axes_keepdims_random.onnx'
    axes = np.array([], dtype=np.int64)
    data = np.random.uniform(-10, 10, [3, 2, 2]).astype(np.float32)
    run_test_onnx_decomp(name,  {'data': data, 'axes': axes}, 'reduce_sum_square')
    
def test_reduce_sum_square_default_axes_keepdims_example():
    name = 'reduce_sum_square_default_axes_keepdims_example.onnx'
    axes = np.array([], dtype=np.int64)
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
    run_test_onnx_decomp(name,  {'data': data, 'axes': axes}, 'reduce_sum_square')
    
def test_reduce_sum_square_empty_set():
    name = 'reduce_sum_square_empty_set.onnx'
    data = np.array([], dtype=np.float32).reshape([2, 0, 4])
    axes = np.array([1], dtype=np.int64)
    run_test_onnx_decomp(name,  {'data': data, 'axes': axes}, 'reduce_sum_square')


# Graph Tests

def test_reduce_sum_square_two_nodes():
    name = 'reduce_sum_square_two_nodes.onnx'
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
    run_test_onnx_decomp(name,  {'x': data}, 'reduce_sum_square')
