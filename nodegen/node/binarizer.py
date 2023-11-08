import numpy as np
from nodegen.node import RunAll
from ..helpers import make_node, make_test, to_fp, Tensor, Dtype, FixedImpl


class Binarizer(RunAll):

    @staticmethod
    def binarizer_i32():
        x = np.random.randint(-3, 3, (3, 3, 3)).astype(np.int32)
        threshold = 1
        y = (x > threshold).astype(np.uint32)

        x = Tensor(Dtype.I32, x.shape, x.flatten())
        y = Tensor(Dtype.U32, y.shape, y.flatten())

        name = "binarizer_i32"
        make_node([x], [y], name)
        make_test([x], y, "TensorTrait::binarizer(@input_0, @IntegerTrait::new(1, false));", name)


    @staticmethod
    def binarizer_fp8x23():
        x = np.random.uniform(-3, 3, (3, 3, 3)).astype(np.float64)
        threshold = 1.0
        y = (x > threshold).astype(np.uint32)

        x = Tensor(Dtype.FP8x23, x.shape, to_fp(
            x.flatten(), FixedImpl.FP8x23))
        y = Tensor(Dtype.U32, y.shape, y.flatten())

        name = "binarizer_fp8x23"
        make_node([x], [y], name)
        make_test([x], y, "TensorTrait::binarizer(@input_0, @FixedTrait::new(8388608, false));", name)

    
    @staticmethod
    def binarizer_fp16x16():
        x = np.random.uniform(-3, 3, (3, 3, 3)).astype(np.float64)
        threshold = 1.0
        y = (x > threshold).astype(np.uint32)

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(
            x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.U32, y.shape, y.flatten())

        name = "binarizer_fp16x16"
        make_node([x], [y], name)
        make_test([x], y, "TensorTrait::binarizer(@input_0, @FixedTrait::new(65536, false));", name)
