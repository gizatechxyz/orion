from onnxruntime import InferenceSession
import subprocess
from onnx import load
import numpy as np
import os


def run_test(name, input_feed): 
    cwd = "crates/orion-test-utils"
    
    subprocess.run(['cargo', 'run', name + '.onnx', 'output_' + name + '.txt', name], cwd=cwd)
    
    with open(f'{cwd}/tests/models/' + name + '.onnx', 'rb') as f:
        logsoftmax = load(f)
    with open(f'{cwd}/tests/models/output_' + name + '.txt', 'r') as f:
        lines = f.readlines()

    # Result from rust execution of Primgraph
    dimensions = tuple(map(int, lines[0].strip().split()))
    if name == "test_less" : 
        data = list(map(lambda x: x.lower() == 'true', lines[1].strip().split()))
    else : 
        data = list(map(float, lines[1].strip().split()))
    actual_res = np.array(data).reshape(dimensions)
    
    # Result from python execution of ONNX Graph
    sess = InferenceSession(logsoftmax.SerializeToString(),
                        providers=["CPUExecutionProvider"])
    expected_res = sess.run(None, input_feed)[0]
    
    print(f"actual_res: {actual_res}")
    print(f"expected_res: {expected_res}")

    tolerance = 1e-5
    assert np.allclose(actual_res, expected_res, atol=tolerance)