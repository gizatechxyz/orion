import numpy as np
from nodegen.node import RunAll
from ..helpers import make_node, make_test, to_fp, Tensor, Dtype, FixedImpl


class Slice(RunAll):
    @staticmethod
    def slice_u32():
        def slice_2D():
            x = np.random.randint(0, 255, (2, 4)).astype(np.uint32)
            y = x[0:2, 2:4]

            x = Tensor(Dtype.U32, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())

            name = "slice_u32_2d"
            make_node([x], [y], name)
            make_test(
                [x], y, "input_0.slice(array![0, 2].span(), array![2, 4].span(), Option::Some(array![0, 1].span()), Option::Some(array![1, 1].span()))", name)

        def slice_3D():
            x = np.random.randint(0, 255, (20, 10, 5)).astype(np.uint32)
            y = x[0:3, 0:10:3]

            x = Tensor(Dtype.U32, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())

            name = "slice_u32_3d"
            make_node([x], [y], name)
            make_test(
                [x], y, "input_0.slice(array![0, 0].span(), array![3, 10].span(), Option::Some(array![0, 1].span()), Option::Some(array![1, 3].span()))", name)

        slice_2D()
        slice_3D()

    @staticmethod
    def slice_i32():
        def slice_2D():
            x = np.random.randint(-127, 127, (2, 4)).astype(np.int32)
            y = x[0:2, 2:4]

            x = Tensor(Dtype.I32, x.shape, x.flatten())
            y = Tensor(Dtype.I32, y.shape, y.flatten())

            name = "slice_i32_2d"
            make_node([x], [y], name)
            make_test(
                [x], y, "input_0.slice(array![0, 2].span(), array![2, 4].span(), Option::Some(array![0, 1].span()), Option::Some(array![1, 1].span()))", name)

        def slice_3D():
            x = np.random.randint(-127, 127, (20, 10, 5)).astype(np.int32)
            y = x[0:3, 0:10:3]

            x = Tensor(Dtype.I32, x.shape, x.flatten())
            y = Tensor(Dtype.I32, y.shape, y.flatten())

            name = "slice_i32_3d"
            make_node([x], [y], name)
            make_test(
                [x], y, "input_0.slice(array![0, 0].span(), array![3, 10].span(), Option::Some(array![0, 1].span()), Option::Some(array![1, 3].span()))", name)


        slice_2D()
        slice_3D()

    @staticmethod
    def slice_i8():
        def slice_2D():
            x = np.random.randint(-127, 127, (2, 4)).astype(np.int8)
            y = x[0:2, 2:4]

            x = Tensor(Dtype.I8, x.shape, x.flatten())
            y = Tensor(Dtype.I8, y.shape, y.flatten())

            name = "slice_i8_2d"
            make_node([x], [y], name)
            make_test(
                [x], y, "input_0.slice(array![0, 2].span(), array![2, 4].span(), Option::Some(array![0, 1].span()), Option::Some(array![1, 1].span()))", name)

        def slice_3D():
            x = np.random.randint(-127, 127, (20, 10, 5)).astype(np.int8)
            y = x[0:3, 0:10:3]

            x = Tensor(Dtype.I8, x.shape, x.flatten())
            y = Tensor(Dtype.I8, y.shape, y.flatten())

            name = "slice_i8_3d"
            make_node([x], [y], name)
            make_test(
                [x], y, "input_0.slice(array![0, 0].span(), array![3, 10].span(), Option::Some(array![0, 1].span()), Option::Some(array![1, 3].span()))", name)

        slice_2D()
        slice_3D()

    @staticmethod
    def slice_fp8x23():
        def slice_2D():
            x = to_fp(np.random.randint(-127, 127, (2, 4)
                                        ).astype(np.int64), FixedImpl.FP8x23)
            y = x[0:2, 2:4]

            x = Tensor(Dtype.FP8x23, x.shape, x.flatten())
            y = Tensor(Dtype.FP8x23, y.shape, y.flatten())
            
            name = "slice_fp8x23_2d"
            make_node([x], [y], name)
            make_test(
                [x], y, "input_0.slice(array![0, 2].span(), array![2, 4].span(), Option::Some(array![0, 1].span()), Option::Some(array![1, 1].span()))", name)

        def slice_3D():
            x = to_fp(np.random.randint(-127, 127, (20, 10, 5)
                                        ).astype(np.int64), FixedImpl.FP8x23)
            y = x[0:3, 0:10:3]

            x = Tensor(Dtype.FP8x23, x.shape, x.flatten())
            y = Tensor(Dtype.FP8x23, y.shape, y.flatten())
            
            name = "slice_fp8x23_3d"
            make_node([x], [y], name)
            make_test(
                [x], y, "input_0.slice(array![0, 0].span(), array![3, 10].span(), Option::Some(array![0, 1].span()), Option::Some(array![1, 3].span()))", name)

        slice_2D()
        slice_3D()

    @staticmethod
    def slice_fp16x16():
        def slice_2D():
            x = to_fp(np.random.randint(-127, 127, (2, 4)
                                        ).astype(np.int64), FixedImpl.FP16x16)
            y = x[0:2, 2:4]

            x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            y = Tensor(Dtype.FP16x16, y.shape, y.flatten())

            name = "slice_fp16x16_2d"
            make_node([x], [y], name)
            make_test(
                [x], y, "input_0.slice(array![0, 2].span(), array![2, 4].span(), Option::Some(array![0, 1].span()), Option::Some(array![1, 1].span()))", name)

        def slice_3D():
            x = to_fp(np.random.randint(-127, 127, (20, 10, 5)
                                        ).astype(np.int64), FixedImpl.FP16x16)
            y = x[0:3, 0:10:3]

            x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            y = Tensor(Dtype.FP16x16, y.shape, y.flatten())

            name = "slice_fp16x16_3d"
            make_node([x], [y], name)
            make_test(
                [x], y, "input_0.slice(array![0, 0].span(), array![3, 10].span(), Option::Some(array![0, 1].span()), Option::Some(array![1, 3].span()))", name)

        slice_2D()
        slice_3D()
