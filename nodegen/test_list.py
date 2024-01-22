import os
import glob
import subprocess

# Directory path where Python files/modules are located
directory_path = 'nodegen/node/'

# Get all files in the directory
all_files = os.listdir(directory_path)

# Filter Python files using glob and '*.py' pattern
python_files = [file[:-3] for file in all_files if file.endswith('.py')]

# Print the names of Python files/modules
command = 'python --version'
os.system(command)