import os
import glob

# Directory path where Python files/modules are located
directory_path = 'nodegen/node/'

# Get all files in the directory
all_files = os.listdir(directory_path)

# Filter Python files using glob and '*.py' pattern
python_files = [file[:-3] for file in all_files if file.endswith('.py')]


fixed = [
    'abs',
    'argmax',
    'argmin',
    'concat',
    'cumsum',
    'div',
    'equal',
    'less_equal',
    'greater',
    'linear',
    'matmul',
    'mul',
    'or',
    'reduce_sum',
    'sub',
    'transpose',
    'xor',
    'less',
    'greater_equal',
    'slice',
    'gather',
    'nonzero',
    'squeeze',
    'unsqueeze',
    'sign',
    'clip',
    '__init__',
    'running'
]
for node in python_files:
    if node not in fixed:
        current_dir = os.getcwd()
        command = f'python nodegen/node/__init__.py {node}'
        os.system(command)