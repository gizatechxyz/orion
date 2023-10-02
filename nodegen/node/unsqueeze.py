import numpy as np
from nodegen.node import RunAll
from ..helpers import make_node, make_test, to_fp, Tensor, Dtype, FixedImpl


class Unsqueeze(RunAll):
    @staticmethod
    def unsqueeze_u32():
        def unsqueeze_2D():
            x = np.random.randint(0, 255, (2, 4)).astype(np.uint32)
            y = np.expand_dims(x, axis=0)
            y = np.expand_dims(y, axis=1)
            y = np.expand_dims(y, axis=4)

            x = Tensor(Dtype.U32, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())

            name = "unsqueeze_u32_2d"
            make_node([x], [y], name)
            make_test(
                [x], y, "input_0.unsqueeze(array![1, 4, 0].span())", name)

        def unsqueeze_3D():
            x = np.random.randint(0, 255, (20, 10, 5)).astype(np.uint32)
            y = np.expand_dims(x, axis=2)
            y = np.expand_dims(y, axis=4)
            y = np.expand_dims(y, axis=5)

            x = Tensor(Dtype.U32, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())

            name = "unsqueeze_u32_3d"
            make_node([x], [y], name)
            make_test(
                [x], y, "input_0.unsqueeze(array![5, 4, 2].span())", name)

        unsqueeze_2D()
        unsqueeze_3D()

    @staticmethod
    def unsqueeze_i32():
        def unsqueeze_2D():
            x = np.random.randint(-127, 127, (2, 4)).astype(np.int32)
            y = np.expand_dims(x, axis=0)
            y = np.expand_dims(y, axis=1)
            y = np.expand_dims(y, axis=4)

            x = Tensor(Dtype.I32, x.shape, x.flatten())
            y = Tensor(Dtype.I32, y.shape, y.flatten())

            name = "unsqueeze_i32_2d"
            make_node([x], [y], name)
            make_test(
                [x], y, "input_0.unsqueeze(array![1, 4, 0].span())", name)

        def unsqueeze_3D():
            x = np.random.randint(-127, 127, (20, 10, 5)).astype(np.int32)
            y = np.expand_dims(x, axis=2)
            y = np.expand_dims(y, axis=4)
            y = np.expand_dims(y, axis=5)

            x = Tensor(Dtype.I32, x.shape, x.flatten())
            y = Tensor(Dtype.I32, y.shape, y.flatten())

            name = "unsqueeze_i32_3d"
            make_node([x], [y], name)
            make_test(
                [x], y, "input_0.unsqueeze(array![5, 4, 2].span())", name)


        unsqueeze_2D()
        unsqueeze_3D()

    @staticmethod
    def unsqueeze_i8():
        def unsqueeze_2D():
            x = np.random.randint(-127, 127, (2, 4)).astype(np.int8)
            y = np.expand_dims(x, axis=0)
            y = np.expand_dims(y, axis=1)
            y = np.expand_dims(y, axis=4)

            x = Tensor(Dtype.I8, x.shape, x.flatten())
            y = Tensor(Dtype.I8, y.shape, y.flatten())

            name = "unsqueeze_i8_2d"
            make_node([x], [y], name)
            make_test(
                [x], y, "input_0.unsqueeze(array![1, 4, 0].span())", name)

        def unsqueeze_3D():
            x = np.random.randint(-127, 127, (20, 10, 5)).astype(np.int8)
            y = np.expand_dims(x, axis=2)
            y = np.expand_dims(y, axis=4)
            y = np.expand_dims(y, axis=5)

            x = Tensor(Dtype.I8, x.shape, x.flatten())
            y = Tensor(Dtype.I8, y.shape, y.flatten())

            name = "unsqueeze_i8_3d"
            make_node([x], [y], name)
            make_test(
                [x], y, "input_0.unsqueeze(array![5, 4, 2].span())", name)

        unsqueeze_2D()
        unsqueeze_3D()

    @staticmethod
    def unsqueeze_fp8x23():
        def unsqueeze_2D():
            x = to_fp(np.random.randint(-127, 127, (2, 4)
                                        ).astype(np.int64), FixedImpl.FP8x23)
            y = np.expand_dims(x, axis=0)
            y = np.expand_dims(y, axis=1)
            y = np.expand_dims(y, axis=4)

            x = Tensor(Dtype.FP8x23, x.shape, x.flatten())
            y = Tensor(Dtype.FP8x23, y.shape, y.flatten())
            
            name = "unsqueeze_fp8x23_2d"
            make_node([x], [y], name)
            make_test(
                [x], y, "input_0.unsqueeze(array![1, 4, 0].span())", name)

        def unsqueeze_3D():
            x = to_fp(np.random.randint(-127, 127, (20, 10, 5)
                                        ).astype(np.int64), FixedImpl.FP8x23)
            y = np.expand_dims(x, axis=2)
            y = np.expand_dims(y, axis=4)
            y = np.expand_dims(y, axis=5)

            x = Tensor(Dtype.FP8x23, x.shape, x.flatten())
            y = Tensor(Dtype.FP8x23, y.shape, y.flatten())
            
            name = "unsqueeze_fp8x23_3d"
            make_node([x], [y], name)
            make_test(
                [x], y, "input_0.unsqueeze(array![5, 4, 2].span())", name)

        unsqueeze_2D()
        unsqueeze_3D()

    @staticmethod
    def unsqueeze_fp16x16():
        def unsqueeze_2D():
            x = to_fp(np.random.randint(-127, 127, (2, 4)
                                        ).astype(np.int64), FixedImpl.FP16x16)
            y = np.expand_dims(x, axis=0)
            y = np.expand_dims(y, axis=1)
            y = np.expand_dims(y, axis=4)

            x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            y = Tensor(Dtype.FP16x16, y.shape, y.flatten())

            name = "unsqueeze_fp16x16_2d"
            make_node([x], [y], name)
            make_test(
                [x], y, "input_0.unsqueeze(array![1, 4, 0].span())", name)

        def unsqueeze_3D():
            x = to_fp(np.random.randint(-127, 127, (20, 10, 5)
                                        ).astype(np.int64), FixedImpl.FP16x16)
            y = np.expand_dims(x, axis=2)
            y = np.expand_dims(y, axis=4)
            y = np.expand_dims(y, axis=5)

            x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            y = Tensor(Dtype.FP16x16, y.shape, y.flatten())

            name = "unsqueeze_fp16x16_3d"
            make_node([x], [y], name)
            make_test(
                [x], y, "input_0.unsqueeze(array![5, 4, 2].span())", name)

        unsqueeze_2D()
        unsqueeze_3D()
