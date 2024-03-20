import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl, Trait

class Expand(RunAll):
    @staticmethod
    def expand_with_broadcast() -> None:
        shape = [1, 3, 1]
        x = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
        
        new_shape = [2, 1, 6]
        y = x * np.ones(new_shape, dtype=np.float32)

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16)) 

        name = "expand_with_broadcast"
        make_test([x], y, "input_0.expand(TensorTrait::new(array![3].span(),array![2, 1, 6].span()))", name)
        
    @staticmethod
    def expand_without_broadcast() -> None:
        shape = [3, 1]
        new_shape = [3, 4]
        
        x = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
        y = x * np.ones(new_shape, dtype=np.float32)
        
        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))

        name = "expand_without_broadcast"
        make_test([x], y, "input_0.expand(TensorTrait::new(array![2].span(),array![3, 4].span()))", name)
