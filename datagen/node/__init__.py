import argparse
import importlib
import os
import sys


class RunAll:
    @classmethod
    def run_all(cls):
        for method_name in dir(cls):
            if method_name.startswith('__') or method_name == 'run_all':
                continue
            method = getattr(cls, method_name)
            if callable(method):
                method()


# Add the path to the 'orion' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


def main():
    parser = argparse.ArgumentParser(description="Generate nodes.")
    parser.add_argument('node_class', help="The class of node to run.")
    args = parser.parse_args()

    class_name = args.node_class.capitalize()

    # Verify that the specified Python file exists
    filename = os.path.join('datagen/node', args.node_class + '.py')
    if not os.path.exists(filename):
        print(f"Error: {filename} does not exist.")
        return

    # Import the module dynamically
    module = importlib.import_module('datagen.node.' + args.node_class)

    # Get the class from the module
    node_class = getattr(module, class_name)

    # Instantiate the class and call the run_all method
    node_instance = node_class()
    node_instance.run_all()


if __name__ == "__main__":
    main()
