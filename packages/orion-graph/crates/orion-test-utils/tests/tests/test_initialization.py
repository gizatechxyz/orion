import os
import glob
import subprocess

def test_initialization_model_generator():
    cwd = "crates/orion-test-utils"
  
    # Suppress all .txt and .onnx files in the models/ directory
    for file in glob.glob("models/*.txt") + glob.glob(f"{cwd}/tests/models/*.onnx"):
        os.remove(file)

    subprocess.run(['python3', 'model_generator.py'], cwd=f"{cwd}/tests/models/")