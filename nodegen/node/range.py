import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl, Trait, get_data_statement



class Range(RunAll):
     
    @staticmethod
    # We test here with fp8x23 implementation.
    def fp8x23():
        args = [1, 5, 0.3]
        args_str = get_data_statement(to_fp(np.array(args).flatten(), FixedImpl.FP8x23), Dtype.FP8x23)
        y = np.arange(*args)
        print(y)
        # Convert the floats values in `y` to fixed points with `to_fp` method:
        y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))
        
        # Define the name of the generated folder. 
        name = "range_fp8x23"
        # Invoke `make_test` method to generate corresponding Cairo tests:
        make_test(
            [], # List of input tensors.
            y, # The expected output result.
            f"TensorTrait::range({','.join(args_str)})", # The code signature.
            name, # The name of the generated folder.
        )
     
    @staticmethod
    # We test here with fp16x16 implementation.
    def fp16x16():
        args = [1, 25, 3]
        args_str = get_data_statement(to_fp(np.array(args).flatten(), FixedImpl.FP16x16), Dtype.FP16x16)
        y = np.arange(*args)
        print(y)
        # Convert the floats values in `y` to fixed points with `to_fp` method:
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
        
        # Define the name of the generated folder. 
        name = "range_fp16x16"
        # Invoke `make_test` method to generate corresponding Cairo tests:
        make_test(
            [], # List of input tensors.
            y, # The expected output result.
            f"TensorTrait::range({','.join(args_str)})", # The code signature.
            name, # The name of the generated folder.
        )
     
    @staticmethod
    # We test here with i8 implementation.
    def i8():
        args = [-1, 25, 3]
        args_str = get_data_statement(np.array(args).flatten(), Dtype.I8)
        y = np.arange(*args)
        print(y)
        # Convert the floats values in `y` to fixed points with `to_fp` method:
        y = Tensor(Dtype.I8, y.shape, y.flatten())
        
        # Define the name of the generated folder. 
        name = "range_i8"
        # Invoke `make_test` method to generate corresponding Cairo tests:
        make_test(
            [], # List of input tensors.
            y, # The expected output result.
            f"TensorTrait::range({','.join(args_str)})", # The code signature.
            name, # The name of the generated folder.
        )
     
    @staticmethod
    # We test here with i32 implementation.
    def i32():
        args = [21, 2, -3]
        args_str = get_data_statement(np.array(args).flatten(), Dtype.I32)
        y = np.arange(*args)
        print(y)
        # Convert the floats values in `y` to fixed points with `to_fp` method:
        y = Tensor(Dtype.I32, y.shape, y.flatten())
        
        # Define the name of the generated folder. 
        name = "range_i32"
        # Invoke `make_test` method to generate corresponding Cairo tests:
        make_test(
            [], # List of input tensors.
            y, # The expected output result.
            f"TensorTrait::range({','.join(args_str)})", # The code signature.
            name, # The name of the generated folder.
        )
     
    @staticmethod
    # We test here with u32 implementation.
    def u32():
        args = [1, 25, 3]
        args_str = get_data_statement(np.array(args).flatten(), Dtype.U32)
        y = np.arange(*args)
        print(y)
        # Convert the floats values in `y` to fixed points with `to_fp` method:
        y = Tensor(Dtype.U32, y.shape, y.flatten())
        
        # Define the name of the generated folder. 
        name = "range_u32"
        # Invoke `make_test` method to generate corresponding Cairo tests:
        make_test(
            [], # List of input tensors.
            y, # The expected output result.
            f"TensorTrait::range({','.join(args_str)})", # The code signature.
            name, # The name of the generated folder.
        )
        