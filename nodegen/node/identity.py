import numpy as np 
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl


class Identity(RunAll):

    @staticmethod
    def identity_fP8x23():
        def identity():
            x = np.array([[1, 2], [3, 4]])
            y = x
           
            x = Tensor(Dtype.FP8x23, x.shape, x.flatten())
            y = Tensor(Dtype.FP8x23, y.shape, y.flatten())

            name = "identity_fP8x23"

            make_test(
                [x], y, "input_0.identity()", name)
        identity()

    @staticmethod
    def identity_fP16x16():
        def identity():
            x = np.array([[1, 2], [3, 4]])
            y = x
           
            x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            y = Tensor(Dtype.FP16x16, y.shape, y.flatten())

            name = "identity_fP16x16"

            make_test(
                [x], y, "input_0.identity()", name)
        identity()

    @staticmethod
    def identity_i8():
        def identity():
            x = np.array([[1, 2], [3, 4]])
            y = x
           
            x = Tensor(Dtype.I8, x.shape, x.flatten())
            y = Tensor(Dtype.I8, y.shape, y.flatten())

            name = "identity_i8"

            make_test(
                [x], y, "input_0.identity()", name)
        identity()

    @staticmethod
    def identity_i32():
        def identity():
            x = np.array([[1, 2], [3, 4]])
            y = x
           
            x = Tensor(Dtype.I32, x.shape, x.flatten())
            y = Tensor(Dtype.I32, y.shape, y.flatten())

            name = "identity_i32"

            make_test(
                [x], y, "input_0.identity()", name)
        identity()

    @staticmethod
    def identity_u32():
        def identity():
            x = np.array([[1, 2], [3, 4]])
            y = x
           
            x = Tensor(Dtype.U32, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())

            name = "identity_u32"

            make_test(
                [x], y, "input_0.identity()", name)
        identity()
