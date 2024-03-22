import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl


class And(RunAll):
    @staticmethod
    def and_bool():
        def default():
            x = (np.random.randn(3, 4) > 0).astype(bool)
            y = (np.random.randn(3, 4) > 0).astype(bool)
            z = np.logical_and(x, y)

            x = Tensor(Dtype.BOOL, x.shape, x.flatten())
            y = Tensor(Dtype.BOOL, y.shape, y.flatten())
            z = Tensor(Dtype.I32, z.shape, z.flatten())

            name = "and_bool"
            make_test([x, y], z, "BoolTensor::and(@input_0, @input_1)", name)

        def broadcast():
            x = (np.random.randn(3, 4, 5) > 0).astype(bool)
            y = (np.random.randn(3, 4, 5) > 0).astype(bool)
            z = np.logical_and(x, y)

            x = Tensor(Dtype.BOOL, x.shape, x.flatten())
            y = Tensor(Dtype.BOOL, y.shape, y.flatten())
            z = Tensor(Dtype.I32, z.shape, z.flatten())

            name = "and_bool_broadcast"
            make_test([x, y], z, "BoolTensor::and(@input_0, @input_1)", name)

        default()
        broadcast()
