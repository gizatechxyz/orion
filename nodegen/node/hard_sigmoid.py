import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl, Trait

class Hard_sigmoid(RunAll):

    @staticmethod
    def fp8x23():
        alpha = 0.2
        beta = 0.5
        x = np.random.uniform(-3, 3, (2, 2)).astype(np.float32)
        y = np.maximum(0, np.minimum(1, alpha * x + beta))

        x = Tensor(Dtype.FP8x23, x.shape, to_fp(
            x.flatten(), FixedImpl.FP8x23))
        y = Tensor(Dtype.FP8x23, y.shape, to_fp(
            y.flatten(), FixedImpl.FP8x23))

        name = "hard_sigmoid_fp8x23"
        make_test([x], y, "NNTrait::hard_sigmoid(@input_0, @FixedTrait::new(1677721, false), @FixedTrait::new(4194304, false))",
                    name, Trait.NN)

    @staticmethod
    def fp16x16():
        alpha = 0.2
        beta = 0.5
        x = np.random.uniform(-3, 3, (2, 2)).astype(np.float32)
        y = np.maximum(0, np.minimum(1, alpha * x + beta)) 

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(
            x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(
            y.flatten(), FixedImpl.FP16x16))

        name = "hard_sigmoid_fp16x16"
        make_test([x], y, "NNTrait::hard_sigmoid(@input_0, @FixedTrait::new(13107, false), @FixedTrait::new(32768, false))",
                    name, Trait.NN)


