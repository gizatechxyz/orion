import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl


class Squeeze(RunAll):
    @staticmethod
    def squeeze_i8():
        def squeeze():
            x = np.ones((1, 2, 1, 2, 1), dtype=np.int8)
            y = np.ones((2, 2, 1), dtype=np.int8)
            
            x = Tensor(Dtype.I8, x.shape, x.flatten())
            y = Tensor(Dtype.I8, y.shape, y.flatten())

            name = "squeeze_i8"
            make_test(
                [x], y, "input_0.squeeze(Option::Some(array![i32 { mag: 0, sign: false }, i32 { mag: 2, sign: false }].span()))", name)
        squeeze()

    @staticmethod
    def squeeze_i32():
        def squeeze():
            x = np.ones((1, 2, 1, 2, 1), dtype=np.int32)
            y = np.ones((2, 2, 1), dtype=np.int32)

            x = Tensor(Dtype.I32, x.shape, x.flatten())
            y = Tensor(Dtype.I32, y.shape, y.flatten())

            name = "squeeze_i32"
            make_test(
                [x], y, "input_0.squeeze(Option::Some(array![i32 { mag: 0, sign: false }, i32 { mag: 2, sign: false }].span()))", name)
        squeeze()

    @staticmethod
    def squeeze_u32():
        def squeeze():
            x = np.ones((1, 2, 1, 2, 1), dtype=np.uint32)
            y = np.ones((2, 2, 1), dtype=np.uint32)

            x = Tensor(Dtype.U32, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())

            name = "squeeze_u32"
            make_test(
                [x], y, "input_0.squeeze(Option::Some(array![i32 { mag: 0, sign: false }, i32 { mag: 2, sign: false }].span()))", name)
        squeeze()
    
    @staticmethod
    def squeeze_fP16x16():
        def squeeze():
            x = to_fp(np.random.randint(0, 255, (1, 2, 1, 2, 1)
                                        ).astype(np.int64), FixedImpl.FP16x16)
            y = to_fp(np.random.randint(0, 255, (2, 2, 1)
                                        ).astype(np.int64), FixedImpl.FP16x16)

            x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            y = Tensor(Dtype.FP16x16, y.shape, y.flatten())

            name = "squeeze_fP16x16"
            make_test(
                [x], y, "input_0.squeeze(Option::Some(array![i32 { mag: 0, sign: false }, i32 { mag: 2, sign: false }].span()))", name)
        squeeze()

    @staticmethod
    def squeeze_fP8x23():
        def squeeze():
            x = to_fp(np.random.randint(0, 255, (1, 2, 1, 2, 1)
                                        ).astype(np.int64), FixedImpl.FP8x23)
            y = to_fp(np.random.randint(0, 255, (2, 2, 1)
                                        ).astype(np.int64), FixedImpl.FP8x23)

            x = Tensor(Dtype.FP8x23, x.shape, x.flatten())
            y = Tensor(Dtype.FP8x23, y.shape, y.flatten())

            name = "squeeze_fP8x23"
            make_test(
                [x], y, "input_0.squeeze(Option::Some(array![i32 { mag: 0, sign: false }, i32 { mag: 2, sign: false }].span()))", name)
        squeeze()
