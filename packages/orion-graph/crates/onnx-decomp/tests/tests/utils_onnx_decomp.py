from onnxruntime import InferenceSession
import subprocess
from onnx import load


def run_test_onnx_decomp(name, input_feed, model_dir): 
    cwd = "crates/onnx-decomp"
    
    subprocess.run(['cargo', 'run', f'{model_dir}/{name}', f'{model_dir}/decomp_{name}'], cwd=cwd)
    
    with open(f'{cwd}/tests/models/{model_dir}/{name}', 'rb') as f:
        reduce_log_sum = load(f)
    with open(f'{cwd}/tests/models/{model_dir}/decomp_' + name, 'rb') as f:
        new_reduce_log_sum = load(f)

    sess = InferenceSession(reduce_log_sum.SerializeToString(),
                        providers=["CPUExecutionProvider"])
    
    sess1 = InferenceSession(new_reduce_log_sum.SerializeToString(),
                        providers=["CPUExecutionProvider"])

    expected_res = sess.run(None, input_feed)
    actual_res = sess1.run(None, input_feed)
    
    assert(expected_res[0].all() == actual_res[0].all())