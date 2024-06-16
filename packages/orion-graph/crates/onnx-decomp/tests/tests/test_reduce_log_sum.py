import numpy as np
from utils_onnx_decomp import run_test_onnx_decomp


# ONNX Tests

def test_reduce_log_sum_desc_axes():
    name = 'reduce_log_sum_desc_axes.onnx'
    axes = np.array([2, 1], dtype=np.int64)
    data = np.random.ranf([3, 4, 5]).astype(np.float32)
    run_test_onnx_decomp(name,  {'data': data, 'axes': axes}, 'reduce_log_sum')
    
def test_reduce_log_sum_asc_axes():
    name = 'reduce_log_sum_asc_axes.onnx'
    axes = np.array([0, 1], dtype=np.int64)
    data = np.random.ranf([3, 4, 5]).astype(np.float32)
    run_test_onnx_decomp(name,  {'data': data, 'axes': axes}, 'reduce_log_sum')
       
def test_reduce_log_sum_negative_axes_keepdims_example():
    name = 'reduce_log_sum_negative_axes.onnx'
    axes = np.array([-2], dtype=np.int64)
    data = np.random.ranf([3, 4, 5]).astype(np.float32)
    run_test_onnx_decomp(name,  {'data': data, 'axes': axes}, 'reduce_log_sum')
    
def test_reduce_log_sum_default_axes_keepdims_random():
    name = 'reduce_log_sum_default.onnx'
    axes = np.array([], dtype=np.int64)
    data = np.random.ranf([3, 4, 5]).astype(np.float32)
    run_test_onnx_decomp(name,  {'data': data, 'axes': axes}, 'reduce_log_sum')
    

# Graph Tests

def test_reduce_log_sum_two_nodes():
    name = 'reduce_log_sum_two_nodes.onnx'
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
    run_test_onnx_decomp(name,  {'x': data}, 'reduce_log_sum')

