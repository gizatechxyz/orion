import numpy as np
from nodegen.node import RunAll
from ..helpers import make_node, make_test, to_fp, Tensor, Dtype, FixedImpl


class Nonzero(RunAll):
    @staticmethod
    def nonzero_u32():
        def nonzero_2D():
            x = np.random.randint(0, 255, (2, 4)).astype(np.uint32)
            y = np.array(np.nonzero(x), dtype=np.int64)

            x = Tensor(Dtype.U32, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())

            name = "nonzero_u32_2d"
            make_node([x], [y], name)
            make_test(
                [x], y, "input_0.nonzero()", name)

        def nonzero_3D():
            x = np.random.randint(0, 255, (20, 10, 5)).astype(np.uint32)
            y = np.array(np.nonzero(x), dtype=np.int64)

            x = Tensor(Dtype.U32, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())

            name = "nonzero_u32_3d"
            make_node([x], [y], name)
            make_test(
                [x], y, "input_0.nonzero()", name)

        nonzero_2D()
        nonzero_3D()

    @staticmethod
    def nonzero_i32():
        def nonzero_2D():
            x = np.random.randint(-127, 127, (2, 4)).astype(np.int32)
            y = np.array(np.nonzero(x), dtype=np.int64)

            x = Tensor(Dtype.I32, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())

            name = "nonzero_i32_2d"
            make_node([x], [y], name)
            make_test(
                [x], y, "input_0.nonzero()", name)

        def nonzero_3D():
            x = np.random.randint(-127, 127, (20, 10, 5)).astype(np.int32)
            y = np.array(np.nonzero(x), dtype=np.int64)

            x = Tensor(Dtype.I32, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())

            name = "nonzero_i32_3d"
            make_node([x], [y], name)
            make_test(
                [x], y, "input_0.nonzero()", name)


        nonzero_2D()
        nonzero_3D()

    @staticmethod
    def nonzero_i8():
        def nonzero_2D():
            x = np.random.randint(-127, 127, (2, 4)).astype(np.int8)
            y = np.array(np.nonzero(x), dtype=np.int64)

            x = Tensor(Dtype.I8, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())

            name = "nonzero_i8_2d"
            make_node([x], [y], name)
            make_test(
                [x], y, "input_0.nonzero()", name)

        def nonzero_3D():
            x = np.random.randint(-127, 127, (20, 10, 5)).astype(np.int8)
            y = np.array(np.nonzero(x), dtype=np.int64)

            x = Tensor(Dtype.I8, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())

            name = "nonzero_i8_3d"
            make_node([x], [y], name)
            make_test(
                [x], y, "input_0.nonzero()", name)

        nonzero_2D()
        nonzero_3D()

    @staticmethod
    def nonzero_fp8x23():
        def nonzero_2D():
            x = to_fp(np.random.randint(-127, 127, (2, 4)
                                        ).astype(np.int64), FixedImpl.FP8x23)
            y = np.array(np.nonzero(x), dtype=np.int64)

            x = Tensor(Dtype.FP8x23, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())
            
            name = "nonzero_fp8x23_2d"
            make_node([x], [y], name)
            make_test(
                [x], y, "input_0.nonzero()", name)

        def nonzero_3D():
            x = to_fp(np.random.randint(-127, 127, (20, 10, 5)
                                        ).astype(np.int64), FixedImpl.FP8x23)
            y = np.array(np.nonzero(x), dtype=np.int64)

            x = Tensor(Dtype.FP8x23, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())
            
            name = "nonzero_fp8x23_3d"
            make_node([x], [y], name)
            make_test(
                [x], y, "input_0.nonzero()", name)

        nonzero_2D()
        nonzero_3D()

    @staticmethod
    def nonzero_fp16x16():
        def nonzero_2D():
            x = to_fp(np.random.randint(-127, 127, (2, 4)
                                        ).astype(np.int64), FixedImpl.FP16x16)
            y = np.array(np.nonzero(x), dtype=np.int64)

            x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())

            name = "nonzero_fp16x16_2d"
            make_node([x], [y], name)
            make_test(
                [x], y, "input_0.nonzero()", name)

        def nonzero_3D():
            x = to_fp(np.random.randint(-127, 127, (20, 10, 5)
                                        ).astype(np.int64), FixedImpl.FP16x16)
            y = np.array(np.nonzero(x), dtype=np.int64)

            x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())

            name = "nonzero_fp16x16_3d"
            make_node([x], [y], name)
            make_test(
                [x], y, "input_0.nonzero()", name)

        nonzero_2D()
        nonzero_3D()
