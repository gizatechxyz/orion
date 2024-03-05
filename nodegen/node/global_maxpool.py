import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl, Trait

def global_maxpool(data: np.ndarray) -> np.ndarray:
    spatial_shape = np.ndim(data) - 2

    result = np.max(data, axis=tuple(range(spatial_shape, spatial_shape + 2)))

    # Add singleton dimensions
    for _ in range(spatial_shape):
        result = np.expand_dims(result, -1)

    return result

class Global_maxpool(RunAll):
     
    @staticmethod
    # We test here with fp8x23 implementation.
    def fp8x23():
        # Create a random numpy array:
        x = np.random.randint(-3, 3, (2, 2, 4, 4)).astype(np.float64)
        # Ddefine the expected result:
        y = global_maxpool(x)
        # Convert the input and output to Tensor class, similar to Orion's Tensor struct:
        x = Tensor(Dtype.FP8x23, x.shape, to_fp(x.flatten(), FixedImpl.FP8x23))
        # Convert the floats values in `y` to fixed points with `to_fp` method:
        y = Tensor(Dtype.FP8x23, y.shape, to_fp(y.flatten(), FixedImpl.FP8x23))
        
        # Define the name of the generated folder. 
        name = "global_maxpool_fp8x23"
        # Invoke `make_test` method to generate corresponding Cairo tests:
        make_test(
            [x], # List of input tensors.
            y, # The expected output result.
            "NNTrait::global_maxpool(@input_0)", # The code signature.
            name, # The name of the generated folder.
            Trait.NN # The trait, if the function is present in either the TensorTrait or NNTrait.
        )
        
    # We test here with fp16x16 implementation.    
    @staticmethod
    def fp16x16():
        x = np.random.uniform(-3, 3, (2, 2, 4, 4)).astype(np.float64)
        y = global_maxpool(x)

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(
            x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(
            y.flatten(), FixedImpl.FP16x16))

        name = "global_maxpool_fp16x16"
        make_test([x], y, "NNTrait::global_maxpool(@input_0)",
                    name, Trait.NN)
        
