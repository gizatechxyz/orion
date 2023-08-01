import numpy as np
from datagen.node import RunAll
from ..helpers import make_node, to_fp, Tensor, Dtype, FixedImpl


class Acosh(RunAll):

    @staticmethod
    def acosh_i32():

        def fp8x23():
            x = np.random.randint(1, 127, (2, 2)).astype(np.int32)
            y = np.arccosh(x)

            x = Tensor(Dtype.I32, x.shape, x.flatten(), FixedImpl.FP8x23)
            y = Tensor(Dtype.FP8x23, y.shape, to_fp(
                y.flatten(), FixedImpl.FP8x23), FixedImpl.FP8x23)
            make_node([x], [y], "acosh_i32_fp8x23")

        def fp16x16():
            x = np.random.randint(1, 127, (2, 2)).astype(np.int32)
            y = np.arccosh(x)

            x = Tensor(Dtype.I32, x.shape, x.flatten(), FixedImpl.FP16x16)
            y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16), FixedImpl.FP16x16)
            make_node([x], [y], "acosh_i32_fp16x16")

        fp8x23()
        fp16x16()
    
    @staticmethod
    def acosh_i8():

        def fp8x23():
            x = np.random.randint(1, 5, (2, 2)).astype(np.int8)
            y = np.arccosh(x)

            x = Tensor(Dtype.I8, x.shape, x.flatten(), FixedImpl.FP8x23)
            y = Tensor(Dtype.FP8x23, y.shape, to_fp(
                y.flatten(), FixedImpl.FP8x23), FixedImpl.FP8x23)
            make_node([x], [y], "acosh_i8_fp8x23")

        def fp16x16():
            x = np.random.randint(1, 127, (2, 2)).astype(np.int32)
            y = np.arccosh(x)

            x = Tensor(Dtype.I8, x.shape, x.flatten(), FixedImpl.FP16x16)
            y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16), FixedImpl.FP16x16)
            make_node([x], [y], "acosh_i8_fp16x16")

        fp8x23()
        fp16x16()
    
    @staticmethod
    def acosh_u32():

        def fp8x23():
            x = np.random.randint(1, 127, (2, 2)).astype(np.uint32)
            y = np.arccosh(x)

            x = Tensor(Dtype.U32, x.shape, x.flatten(), FixedImpl.FP8x23)
            y = Tensor(Dtype.FP8x23, y.shape, to_fp(
                y.flatten(), FixedImpl.FP8x23), FixedImpl.FP8x23)
            make_node([x], [y], "acosh_u32_fp8x23")

        def fp16x16():
            x = np.random.randint(1, 127, (2, 2)).astype(np.uint32)
            y = np.arccosh(x)

            x = Tensor(Dtype.U32, x.shape, x.flatten(), FixedImpl.FP16x16)
            y = Tensor(Dtype.FP16x16, y.shape, to_fp(
                y.flatten(), FixedImpl.FP16x16), FixedImpl.FP16x16)
            make_node([x], [y], "acosh_u32_fp16x16")

        fp8x23()
        fp16x16()
    

    @staticmethod
    def acosh_fp8x23():
            x = np.random.uniform(1, 127, (2, 2)).astype(np.float64)
            y = np.arccosh(x)

            x = Tensor(Dtype.FP8x23, x.shape, to_fp(x.flatten(), FixedImpl.FP8x23), FixedImpl.FP8x23)
            y = Tensor(Dtype.FP8x23, y.shape, to_fp(
                y.flatten(), FixedImpl.FP8x23), FixedImpl.FP8x23)
            make_node([x], [y], "acosh_fp8x23")

    @staticmethod
    def acosh_fp16x16():
        x = np.random.uniform(1, 127, (2, 2)).astype(np.float64)
        y = np.arccosh(x)

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(x.flatten(), FixedImpl.FP16x16), FixedImpl.FP16x16)
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(
            y.flatten(), FixedImpl.FP16x16), FixedImpl.FP16x16)
        make_node([x], [y], "acosh_fp16x16")
    
