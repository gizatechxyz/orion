import numpy as np
from nodegen.node import RunAll
from ..helpers import make_node, make_test, to_fp, Tensor, Dtype, FixedImpl

class Sign(RunAll):
    @staticmethod
    def sign_i8():
        def sign():
            x = np.array(range(-5, 6)).astype(np.int8)
            y = np.array([-1, -1, -1, -1, -1,  0,  1,  1,  1,  1,  1]).astype(np.int8)
            
            x = Tensor(Dtype.I8, x.shape, x.flatten())
            y = Tensor(Dtype.I8, y.shape, y.flatten())

            name = "sign_i8"
            make_node([x], [y], name)
            make_test(
                [x], y, "input_0.sign()", name)
        sign()

    @staticmethod
    def sign_i32():
        def sign():
            x = np.array(range(-5, 6)).astype(np.int32)
            y = np.array([-1, -1, -1, -1, -1,  0,  1,  1,  1,  1,  1]).astype(np.int32)

            x = Tensor(Dtype.I32, x.shape, x.flatten())
            y = Tensor(Dtype.I32, y.shape, y.flatten())

            name = "sign_i32"
            make_node([x], [y], name)
            make_test(
                [x], y, "input_0.sign()", name)
        sign()

    @staticmethod
    def sign_fail():
        def sign():

            x = np.array(range(-5, 6)).astype(np.int32)
            y = np.array([1, -1, -1, -1, -1,  0,  1,  1,  1,  1,  -1]).astype(np.int32)

            x = Tensor(Dtype.I32, x.shape, x.flatten())
            y = Tensor(Dtype.I32, y.shape, y.flatten())

            name = "sign_fail"
            make_node([x], [y], name)
            make_test(
                [x], y, "input_0.sign()", name)
        sign()
    
    @staticmethod
    def sign_fP16x16():
        def sign():

            x = to_fp (np.array(range(-5, 6)).astype(np.int64), FixedImpl.FP16x16)
            y = to_fp (np.array([-1, -1, -1, -1, -1,  0,  1,  1,  1,  1,  1]).astype(np.int64), FixedImpl.FP16x16)
            
            x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            y = Tensor(Dtype.FP16x16, y.shape, y.flatten())

            name = "sign_fP16x16"
            make_node([x], [y], name)
            make_test(
                [x], y, "input_0.sign()", name)
        sign()

    @staticmethod
    def sign_fP8x23():
        def sign():

            x = to_fp (np.array(range(-5, 6)).astype(np.int64), FixedImpl.FP8x23)
            y = to_fp (np.array([-1, -1, -1, -1, -1,  0,  1,  1,  1,  1,  1]).astype(np.int64), FixedImpl.FP8x23)            

            x = Tensor(Dtype.FP8x23, x.shape, x.flatten())
            y = Tensor(Dtype.FP8x23, y.shape, y.flatten())

            name = "sign_fP8x23"
            make_node([x], [y], name)
            make_test(
                [x], y, "input_0.sign()", name)
        sign()