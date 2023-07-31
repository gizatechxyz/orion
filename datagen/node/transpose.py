import numpy as np
from datagen.node import RunAll
from ..helpers import make_node, to_fp, Tensor, Dtype, FixedImpl


class Transpose(RunAll):
    @staticmethod
    def transpose_u32():
        def transpose_2D():
            x = np.random.randint(0, 255, (2, 2)).astype(np.uint32)
            y = np.transpose(x, [1, 0])

            x = Tensor(Dtype.U32, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())
            make_node([x], [y], "transpose_u32_2d")

        def transpose_3D():
            x = np.random.randint(0, 255, (2, 2, 2)).astype(np.uint32)
            y = np.transpose(x, [1, 2, 0])

            x = Tensor(Dtype.U32, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())
            make_node([x], [y], "transpose_u32_3d")

        transpose_2D()
        transpose_3D()
    
    @staticmethod
    def transpose_i32():
        def transpose_2D():
            x = np.random.randint(-127, 127, (2, 2)).astype(np.int32)
            y = np.transpose(x, [1, 0])

            x = Tensor(Dtype.I32, x.shape, x.flatten())
            y = Tensor(Dtype.I32, y.shape, y.flatten())
            make_node([x], [y], "transpose_i32_2d")

        def transpose_3D():
            x = np.random.randint(-127, 127, (2, 2, 2)).astype(np.int32)
            y = np.transpose(x, [1, 2, 0])

            x = Tensor(Dtype.I32, x.shape, x.flatten())
            y = Tensor(Dtype.I32, y.shape, y.flatten())
            make_node([x], [y], "transpose_i32_3d")

        transpose_2D()
        transpose_3D()
    
    @staticmethod
    def transpose_i8():
        def transpose_2D():
            x = np.random.randint(-127, 127, (2, 2)).astype(np.int8)
            y = np.transpose(x, [1, 0])

            x = Tensor(Dtype.I8, x.shape, x.flatten())
            y = Tensor(Dtype.I8, y.shape, y.flatten())
            make_node([x], [y], "transpose_i8_2d")

        def transpose_3D():
            x = np.random.randint(-127, 127, (2, 2, 2)).astype(np.int8)
            y = np.transpose(x, [1, 2, 0])

            x = Tensor(Dtype.I8, x.shape, x.flatten())
            y = Tensor(Dtype.I8, y.shape, y.flatten())
            make_node([x], [y], "transpose_i8_3d")

        transpose_2D()
        transpose_3D()
    
    @staticmethod
    def transpose_fp8x23():
        def transpose_2D():
            x = to_fp(np.random.randint(-127, 127, (2, 2)).astype(np.int64), FixedImpl.FP8x23)
            y = np.transpose(x, [1, 0])

            x = Tensor(Dtype.FP8x23, x.shape, x.flatten())
            y = Tensor(Dtype.FP8x23, y.shape, y.flatten())
            make_node([x], [y], "transpose_fp8x23_2d")

        def transpose_3D():
            x = to_fp(np.random.randint(-127, 127, (2, 2, 2)).astype(np.int64), FixedImpl.FP8x23)
            y = np.transpose(x, [1, 2, 0])

            x = Tensor(Dtype.FP8x23, x.shape, x.flatten())
            y = Tensor(Dtype.FP8x23, y.shape, y.flatten())
            make_node([x], [y], "transpose_fp8x23_3d")

        transpose_2D()
        transpose_3D()

    @staticmethod
    def transpose_fp16x16():
        def transpose_2D():
            x = to_fp(np.random.randint(-127, 127, (2, 2)).astype(np.int64), FixedImpl.FP16x16)
            y = np.transpose(x, [1, 0])

            x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            y = Tensor(Dtype.FP16x16, y.shape, y.flatten())
            make_node([x], [y], "transpose_fp16x16_2d")

        def transpose_3D():
            x = to_fp(np.random.randint(-127, 127, (2, 2, 2)).astype(np.int64), FixedImpl.FP16x16)
            y = np.transpose(x, [1, 2, 0])

            x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            y = Tensor(Dtype.FP16x16, y.shape, y.flatten())
            make_node([x], [y], "transpose_fp16x16_3d")

        transpose_2D()
        transpose_3D()
