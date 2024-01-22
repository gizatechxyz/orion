import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl


class Clip(RunAll):
    @staticmethod
    def clip_u32():
        def clip_2D():
            x = np.random.randint(0, 255, (2, 4)).astype(np.uint32)
            y = np.clip(x, np.uint32(10), np.uint32(20))

            x = Tensor(Dtype.U32, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())

            name = "clip_u32_2d"
            make_test(
                [x], y, "input_0.clip(Option::Some(10_u32), Option::Some(20_u32))", name)

        def clip_3D():
            x = np.random.randint(0, 255, (20, 10, 5)).astype(np.uint32)
            y = np.clip(x, np.uint32(10), np.uint32(20))

            x = Tensor(Dtype.U32, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())

            name = "clip_u32_3d"
            make_test(
                [x], y, "input_0.clip(Option::Some(10_u32), Option::Some(20_u32))", name)

        clip_2D()
        clip_3D()

    @staticmethod
    def clip_i32():
        def clip_2D():
            x = np.random.randint(-127, 127, (2, 4)).astype(np.int32)
            y = np.clip(x, np.int32(-10), np.int32(20))

            x = Tensor(Dtype.I32, x.shape, x.flatten())
            y = Tensor(Dtype.I32, y.shape, y.flatten())

            name = "clip_i32_2d"
            make_test(
                [x], y, "input_0.clip(Option::Some(-10_i32), Option::Some(20_i32))", name)

        def clip_3D():
            x = np.random.randint(-127, 127, (20, 10, 5)).astype(np.int32)
            y = np.clip(x, np.int32(-10), np.int32(20))

            x = Tensor(Dtype.I32, x.shape, x.flatten())
            y = Tensor(Dtype.I32, y.shape, y.flatten())

            name = "clip_i32_3d"
            make_test(
                [x], y, "input_0.clip(Option::Some(-10_i32), Option::Some(20_i32))", name)


        clip_2D()
        clip_3D()

    @staticmethod
    def clip_i8():
        def clip_2D():
            x = np.random.randint(-127, 127, (2, 4)).astype(np.int8)
            y = np.clip(x, np.int8(-10), np.int8(20))

            x = Tensor(Dtype.I8, x.shape, x.flatten())
            y = Tensor(Dtype.I8, y.shape, y.flatten())

            name = "clip_i8_2d"
            make_test(
                [x], y, "input_0.clip(Option::Some(-10_i8), Option::Some(20_i8))", name)

        def clip_3D():
            x = np.random.randint(-127, 127, (20, 10, 5)).astype(np.int8)
            y = np.clip(x, np.int8(-10), np.int8(20))

            x = Tensor(Dtype.I8, x.shape, x.flatten())
            y = Tensor(Dtype.I8, y.shape, y.flatten())

            name = "clip_i8_3d"
            make_test(
                [x], y, "input_0.clip(Option::Some(-10_i8), Option::Some(20_i8))", name)

        clip_2D()
        clip_3D()

    @staticmethod
    def clip_fp8x23():
        def clip_2D():
            x = to_fp(np.random.randint(-127, 127, (2, 4)
                                        ).astype(np.int64), FixedImpl.FP8x23)
            y = np.clip(x, to_fp(np.int64(-10), FixedImpl.FP8x23), to_fp(np.int64(20), FixedImpl.FP8x23))

            x = Tensor(Dtype.FP8x23, x.shape, x.flatten())
            y = Tensor(Dtype.FP8x23, y.shape, y.flatten())
            
            name = "clip_fp8x23_2d"
            make_test(
                [x], y, "input_0.clip(Option::Some(FP8x23 { mag: 83886080, sign: true }), Option::Some(FP8x23 { mag: 167772160, sign: false }))", name)

        def clip_3D():
            x = to_fp(np.random.randint(-127, 127, (20, 10, 5)
                                        ).astype(np.int64), FixedImpl.FP8x23)
            y = np.clip(x, to_fp(np.int64(-10), FixedImpl.FP8x23), to_fp(np.int64(20), FixedImpl.FP8x23))

            x = Tensor(Dtype.FP8x23, x.shape, x.flatten())
            y = Tensor(Dtype.FP8x23, y.shape, y.flatten())
            
            name = "clip_fp8x23_3d"
            make_test(
                [x], y, "input_0.clip(Option::Some(FP8x23 { mag: 83886080, sign: true }), Option::Some(FP8x23 { mag: 167772160, sign: false }))", name)

        clip_2D()
        clip_3D()

    @staticmethod
    def clip_fp16x16():
        def clip_2D():
            x = to_fp(np.random.randint(-127, 127, (2, 4)
                                        ).astype(np.int64), FixedImpl.FP16x16)
            y = np.clip(x, to_fp(np.int64(-10), FixedImpl.FP16x16), to_fp(np.int64(20), FixedImpl.FP16x16))

            x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            y = Tensor(Dtype.FP16x16, y.shape, y.flatten())

            name = "clip_fp16x16_2d"
            make_test(
                [x], y, "input_0.clip(Option::Some(FP16x16 { mag: 655360, sign: true }), Option::Some(FP16x16 { mag: 1310720, sign: false }))", name)

        def clip_3D():
            x = to_fp(np.random.randint(-127, 127, (20, 10, 5)
                                        ).astype(np.int64), FixedImpl.FP16x16)
            y = np.clip(x, to_fp(np.int64(-10), FixedImpl.FP16x16), to_fp(np.int64(20), FixedImpl.FP16x16))

            x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            y = Tensor(Dtype.FP16x16, y.shape, y.flatten())

            name = "clip_fp16x16_3d"
            make_test(
                [x], y, "input_0.clip(Option::Some(FP16x16 { mag: 655360, sign: true }), Option::Some(FP16x16 { mag: 1310720, sign: false }))", name)

        clip_2D()
        clip_3D()
