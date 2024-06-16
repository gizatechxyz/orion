import numpy as np
from utils_onnx_decomp import run_test_onnx_decomp

# ONNX Tests

def test_reduce_log_sum_exp_keepdims_random():
    name = 'reduce_log_sum_exp_keepdims_random.onnx'
    axes = np.array([1], dtype=np.int64)
    data = np.random.uniform(-10, 10, [3, 2, 2]).astype(np.double)
    run_test_onnx_decomp(name,  {'data': data, 'axes': axes}, 'reduce_log_sum_exp')
    
def test_reduce_log_sum_exp_keepdims_example():
    name = 'reduce_log_sum_exp_keepdims_example.onnx'
    axes = np.array([1], dtype=np.int64)
    data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.double)
    run_test_onnx_decomp(name,  {'data': data, 'axes': axes}, 'reduce_log_sum_exp')   
    
def test_reduce_log_sum_exp_do_not_keepdims_random():
    name = 'reduce_log_sum_exp_do_not_keepdims_random.onnx'
    axes = np.array([1], dtype=np.int64)
    data = np.random.uniform(-10, 10, [3, 2, 2]).astype(np.double)
    run_test_onnx_decomp(name,  {'data': data, 'axes': axes}, 'reduce_log_sum_exp')
    
def test_reduce_log_sum_exp_do_not_keepdims_example():
    name = 'reduce_log_sum_exp_do_not_keepdims_example.onnx'
    axes = np.array([1], dtype=np.int64)
    data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.double)
    run_test_onnx_decomp(name,  {'data': data, 'axes': axes}, 'reduce_log_sum_exp')
    
def test_reduce_log_sum_exp_negative_axes_keepdims_random():
    name = 'reduce_log_sum_exp_negative_axes_keepdims_random.onnx'
    axes = np.array([-2], dtype=np.int64)
    data = np.random.uniform(-10, 10, [3, 2, 2]).astype(np.double)
    run_test_onnx_decomp(name,  {'data': data, 'axes': axes}, 'reduce_log_sum_exp')
    
def test_reduce_log_sum_exp_negative_axes_keepdims_example():
    name = 'reduce_log_sum_exp_negative_axes_keepdims_example.onnx'
    axes = np.array([-2], dtype=np.int64)
    data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.double)
    run_test_onnx_decomp(name,  {'data': data, 'axes': axes}, 'reduce_log_sum_exp')
    
def test_reduce_log_sum_exp_default_axes_keepdims_random():
    name = 'reduce_log_sum_exp_default_axes_keepdims_random.onnx'
    axes = np.array([], dtype=np.int64)
    data = np.random.uniform(-10, 10, [3, 2, 2]).astype(np.double)
    run_test_onnx_decomp(name,  {'data': data, 'axes': axes}, 'reduce_log_sum_exp')
    
def test_reduce_log_sum_exp_default_axes_keepdims_example():
    name = 'reduce_log_sum_exp_default_axes_keepdims_example.onnx'
    axes = np.array([], dtype=np.int64)
    data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.double)
    run_test_onnx_decomp(name,  {'data': data, 'axes': axes}, 'reduce_log_sum_exp')

# Graph Tests

def test_reduce_log_sum_exp_two_nodes():
    name = 'reduce_log_sum_exp_two_nodes.onnx'
    data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)
    run_test_onnx_decomp(name,  {'x': data}, 'reduce_log_sum_exp')
