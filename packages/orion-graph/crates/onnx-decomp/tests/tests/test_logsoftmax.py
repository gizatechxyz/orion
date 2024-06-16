import numpy as np
from onnx import load
from onnxruntime import InferenceSession
import subprocess

from utils_onnx_decomp import run_test_onnx_decomp


def _test_logsoftmax(name): 
    subprocess.run(['cargo', 'run', 'logsoftmax/' + name, 'logsoftmax/decomp_' + name], cwd="..")
    with open("models/logsoftmax/" + name, "rb") as f:
        logsoftmax = load(f)
        
    with open("models/logsoftmax/decomp_" + name, "rb") as f:
        new_logsoftmax = load(f)
     
    sess = InferenceSession(logsoftmax.SerializeToString(),
                        providers=["CPUExecutionProvider"])
    
    sess1 = InferenceSession(new_logsoftmax.SerializeToString(),
                        providers=["CPUExecutionProvider"])
   
    x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
    
    expected_res = sess.run(None, {'x': x})
    actual_res = sess1.run(None, {'x': x})
    
    assert(expected_res[0].all() == actual_res[0].all())

# ONNX Tests

def test_logsoftmax_axis_0():
    name = 'logsoftmax_axis_0.onnx'
    x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
    run_test_onnx_decomp(name,  {'x': x}, 'logsoftmax')

def test_logsoftmax_axis_1():
    name = 'logsoftmax_axis_1.onnx'
    x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
    run_test_onnx_decomp(name,  {'x': x}, 'logsoftmax')
    
def test_logsoftmax_axis_2():
    name = 'logsoftmax_axis_2.onnx'
    x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
    run_test_onnx_decomp(name,  {'x': x}, 'logsoftmax')
      
def test_logsoftmax_default_axis():
    name = 'logsoftmax_default_axis.onnx'
    x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
    run_test_onnx_decomp(name,  {'x': x}, 'logsoftmax')
       
def test_logsoftmax_negative_axis():
    name = 'logsoftmax_negative_axis.onnx'
    x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
    run_test_onnx_decomp(name,  {'x': x}, 'logsoftmax')
    

# Graph test 

def test_two_nodes():
    name = 'logsoftmax_two_nodes.onnx'
    x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
    run_test_onnx_decomp(name,  {'x': x}, 'logsoftmax')