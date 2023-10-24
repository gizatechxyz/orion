import numpy as np
from nodegen.node import RunAll
from ..helpers import make_node, make_test, to_fp, Tensor, Dtype, FixedImpl, Trait


class Thresholded_relu(RunAll):

    @staticmethod
    def thresholded_relu_fp8x23():

        alpha = 1.0

        x = np.random.uniform(-5, 7, (2, 2)).astype(np.float64)
        y = np.clip(x, alpha, np.inf)
        y[y == alpha] = 0

        x = Tensor(Dtype.FP8x23, x.shape, to_fp(
            x.flatten(), FixedImpl.FP8x23))
        y = Tensor(Dtype.FP8x23, y.shape, to_fp(
            y.flatten(), FixedImpl.FP8x23))

        name = "thresholded_relu_fp8x23"
        make_node([x], [y], name)
        make_test([x], y, "NNTrait::thresholded_relu(@input_0, @FixedTrait::new(256, false))",
                  name, Trait.NN)

    @staticmethod
    def thresholded_relu_fp16x16():

        alpha = 1.0

        x = np.random.uniform(-5, 7, (2, 2)).astype(np.float64)
        y = np.clip(x, alpha, np.inf)
        y[y == alpha] = 0

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(
            x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(
            y.flatten(), FixedImpl.FP16x16))

        name = "thresholded_relu_fp16x16"
        make_node([x], [y], name)
        make_test([x], y, "NNTrait::thresholded_relu(@input_0, @FixedTrait::new(65536, false))",
                  name, Trait.NN)
