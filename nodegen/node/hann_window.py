import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl, Trait, get_data_statement

def hann_window(size, output_datatype=None, periodic=None) -> np.ndarray:  # type: ignore
    if periodic == 1:
        N_1 = size
    else:
        N_1 = size - 1
    ni = np.arange(size, dtype=output_datatype)
    res = np.sin((ni * np.float64(np.pi).astype(output_datatype) / N_1).astype(output_datatype)) ** 2
    return res.astype(output_datatype)

class Hann_window(RunAll):
     
    @staticmethod
    # We test here with fp8x23 implementation.
    def fp8x23():
        print(get_data_statement(to_fp(np.array([np.pi]).flatten(), FixedImpl.FP8x23), Dtype.FP8x23))
        args = [4]
        # x = np.float64(4)
        args_str = get_data_statement(to_fp(np.array(args).flatten(), FixedImpl.FP8x23), Dtype.FP8x23)
        y = hann_window(*args, np.float64)
        print(y)
        
        # Convert the floats values in `y` to fixed points with `to_fp` method:
        y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))
        
        # Define the name of the generated folder. 
        name = "hann_window_fp8x23"
        # Invoke `make_test` method to generate corresponding Cairo tests:
        make_test(
            [], # List of input tensors.
            y, # The expected output result.
            f"TensorTrait::hann_window({','.join(args_str)}, Option::Some(0))", # The code signature.
            name # The name of the generated folder.
        )
     
    @staticmethod
    # We test here with fp16x16 implementation.
    def fp16x16():
        print(get_data_statement(to_fp(np.array([np.pi]).flatten(), FixedImpl.FP16x16), Dtype.FP16x16))
        args = [10]
        # x = np.float64(4)
        args_str = get_data_statement(to_fp(np.array(args).flatten(), FixedImpl.FP16x16), Dtype.FP16x16)
        y = hann_window(*args, np.float16)
        print(y)
        
        # Convert the floats values in `y` to fixed points with `to_fp` method:
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
        
        # Define the name of the generated folder. 
        name = "hann_window_fp16x16"
        # Invoke `make_test` method to generate corresponding Cairo tests:
        make_test(
            [], # List of input tensors.
            y, # The expected output result.
            f"TensorTrait::hann_window({','.join(args_str)}, Option::Some(0))", # The code signature.
            name # The name of the generated folder.
        )
     
    # @staticmethod
    # # We test here with i8 implementation.
    # def i8():
    #     print(get_data_statement(np.array([np.pi]).flatten(), Dtype.I8))
    #     args = [5]
    #     # x = np.float64(4)
    #     args_str = get_data_statement(np.array(args).flatten(), Dtype.I8)
    #     y = hann_window(*args, np.int8)
    #     print(y)
        
    #     # Convert the floats values in `y` to fixed points with `to_fp` method:
    #     y = Tensor(Dtype.I8, y.shape, y.flatten())
        
    #     # Define the name of the generated folder. 
    #     name = "hann_window_i8"
    #     # Invoke `make_test` method to generate corresponding Cairo tests:
    #     make_test(
    #         [], # List of input tensors.
    #         y, # The expected output result.
    #         f"TensorTrait::hann_window({','.join(args_str)}, Option::Some(1))", # The code signature.
    #         name # The name of the generated folder.
    #     )
    
    # @staticmethod
    # # We test here with i32 implementation.
    # def i32():
    #     print(get_data_statement(np.array([np.pi]).flatten(), Dtype.I32))
    #     args = [4]
    #     # x = np.float64(4)
    #     args_str = get_data_statement(np.array(args).flatten(), Dtype.I32)
    #     y = hann_window(*args, np.int32)
    #     print(y)
        
    #     # Convert the floats values in `y` to fixed points with `to_fp` method:
    #     y = Tensor(Dtype.I32, y.shape, y.flatten())
        
    #     # Define the name of the generated folder. 
    #     name = "hann_window_i32"
    #     # Invoke `make_test` method to generate corresponding Cairo tests:
    #     make_test(
    #         [], # List of input tensors.
    #         y, # The expected output result.
    #         f"TensorTrait::hann_window({','.join(args_str)}, Option::Some(0))", # The code signature.
    #         name # The name of the generated folder.
    #     )
     
    # @staticmethod
    # # We test here with u32 implementation.
    # def u32():
    #     print(get_data_statement(np.array([np.pi]).flatten(), Dtype.U32))
    #     args = [4]
    #     # x = np.float64(4)
    #     args_str = get_data_statement(np.array(args).flatten(), Dtype.U32)
    #     y = hann_window(*args, np.uint32)
    #     print(y)
        
    #     # Convert the floats values in `y` to fixed points with `to_fp` method:
    #     y = Tensor(Dtype.U32, y.shape, y.flatten())
        
    #     # Define the name of the generated folder. 
    #     name = "hann_window_u32"
    #     # Invoke `make_test` method to generate corresponding Cairo tests:
    #     make_test(
    #         [], # List of input tensors.
    #         y, # The expected output result.
    #         f"TensorTrait::hann_window({','.join(args_str)}, Option::Some(0))", # The code signature.
    #         name # The name of the generated folder.
    #     )
        