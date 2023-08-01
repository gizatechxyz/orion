import numpy as np
from datagen.node import RunAll
from ..helpers import make_node, to_fp, Tensor, Dtype, FixedImpl


def argmax_use_numpy(data: np.ndarray, axis: int = 0, keepdims: int = 1, dtype=np.int64) -> np.ndarray:
    result = np.argmax(data, axis=axis)
    if keepdims == 1:
        result = np.expand_dims(result, axis)
    return result.astype(dtype)


def argmax_use_numpy_select_last_index(
    data: np.ndarray, axis: int = 0, keepdims: int = True, dtype=np.int64
) -> np.ndarray:
    data = np.flip(data, axis)
    result = np.argmax(data, axis=axis)
    result = data.shape[axis] - result - 1
    if keepdims:
        result = np.expand_dims(result, axis)
    return result.astype(dtype)


class Argmax(RunAll):

    @staticmethod
    def argmax_u32():
        def argmax_1D():
            def default_params():
                x = np.random.randint(0, 255, (3)).astype(np.uint32)
                y = argmax_use_numpy(x, dtype=np.uint32).reshape((1))

                x = Tensor(Dtype.U32, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())
                make_node([x], [y], "argmax_u32_1D_default")

            def keepdims_false():
                x = np.random.randint(0, 255, (3)).astype(np.uint32)
                y = argmax_use_numpy(
                    x, keepdims=0, dtype=np.uint32).reshape((1))

                x = Tensor(Dtype.U32, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())
                make_node([x], [y], "argmax_u32_1D_keepdims_false")

            def last_index():
                x = np.random.randint(0, 255, (3)).astype(np.uint32)
                y = argmax_use_numpy_select_last_index(
                    x, dtype=np.uint32).reshape((1))

                x = Tensor(Dtype.U32, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())
                make_node([x], [y], "argmax_u32_1D_last_index")

            default_params()
            keepdims_false()
            last_index()
        argmax_1D()

        def argmax_2D():
            def default_params():
                x = np.random.randint(0, 255, (2, 2)).astype(np.uint32)
                y = argmax_use_numpy(x, dtype=np.uint32)

                x = Tensor(Dtype.U32, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())
                make_node([x], [y], "argmax_u32_2D_default")

            def keepdims_false():
                x = np.random.randint(0, 255, (2, 2)).astype(np.uint32)
                y = argmax_use_numpy(
                    x, keepdims=0, dtype=np.uint32)

                x = Tensor(Dtype.U32, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())
                make_node([x], [y], "argmax_u32_2D_keepdims_false")

            def last_index():
                x = np.random.randint(0, 255, (2, 2)).astype(np.uint32)
                y = argmax_use_numpy_select_last_index(
                    x, dtype=np.uint32)

                x = Tensor(Dtype.U32, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())
                make_node([x], [y], "argmax_u32_2D_last_index")

            default_params()
            keepdims_false()
            last_index()
        argmax_2D()

        def argmax_3D():
            def default_params():
                x = np.random.randint(0, 255, (2, 2, 2)).astype(np.uint32)
                y = argmax_use_numpy(x, dtype=np.uint32)

                x = Tensor(Dtype.U32, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())
                make_node([x], [y], "argmax_u32_3D_default")

            def keepdims_false():
                x = np.random.randint(0, 255, (2, 2, 2)).astype(np.uint32)
                y = argmax_use_numpy(
                    x, keepdims=0, dtype=np.uint32)

                x = Tensor(Dtype.U32, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())
                make_node([x], [y], "argmax_u32_3D_keepdims_false")

            def last_index():
                x = np.random.randint(0, 255, (2, 2, 2)).astype(np.uint32)
                y = argmax_use_numpy_select_last_index(
                    x, dtype=np.uint32)

                x = Tensor(Dtype.U32, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())
                make_node([x], [y], "argmax_u32_3D_last_index")

            default_params()
            keepdims_false()
            last_index()
        argmax_3D()

    @staticmethod
    def argmax_i32():
        def argmax_1D():
            def default_params():
                x = np.random.randint(-127, 127, (3)).astype(np.int32)
                y = argmax_use_numpy(x, dtype=np.uint32).reshape((1))

                x = Tensor(Dtype.I32, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())
                make_node([x], [y], "argmax_i32_1D_default")

            def keepdims_false():
                x = np.random.randint(-127, 127, (3)).astype(np.int32)
                y = argmax_use_numpy(
                    x, keepdims=0, dtype=np.uint32).reshape((1))

                x = Tensor(Dtype.I32, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())
                make_node([x], [y], "argmax_i32_1D_keepdims_false")

            def last_index():
                x = np.random.randint(0, 255, (3)).astype(np.int32)
                y = argmax_use_numpy_select_last_index(
                    x, dtype=np.uint32).reshape((1))

                x = Tensor(Dtype.I32, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())
                make_node([x], [y], "argmax_i32_1D_last_index")

            default_params()
            keepdims_false()
            last_index()
        argmax_1D()

        def argmax_2D():
            def default_params():
                x = np.random.randint(-127, 127, (2, 2)).astype(np.int32)
                y = argmax_use_numpy(x, dtype=np.uint32)

                x = Tensor(Dtype.I32, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())
                make_node([x], [y], "argmax_i32_2D_default")

            def keepdims_false():
                x = np.random.randint(-127, 127, (2, 2)).astype(np.int32)
                y = argmax_use_numpy(
                    x, keepdims=0, dtype=np.uint32)

                x = Tensor(Dtype.I32, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())
                make_node([x], [y], "argmax_i32_2D_keepdims_false")

            def last_index():
                x = np.random.randint(-127, 127, (2, 2)).astype(np.int32)
                y = argmax_use_numpy_select_last_index(
                    x, dtype=np.int32)

                x = Tensor(Dtype.I32, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())
                make_node([x], [y], "argmax_i32_2D_last_index")

            default_params()
            keepdims_false()
            last_index()
        argmax_2D()

        def argmax_3D():
            def default_params():
                x = np.random.randint(-127, 127, (2, 2, 2)).astype(np.int32)
                y = argmax_use_numpy(x, dtype=np.uint32)

                x = Tensor(Dtype.I32, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())
                make_node([x], [y], "argmax_i32_3D_default")

            def keepdims_false():
                x = np.random.randint(-127, 127, (2, 2, 2)).astype(np.int32)
                y = argmax_use_numpy(
                    x, keepdims=0, dtype=np.uint32)

                x = Tensor(Dtype.I32, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())
                make_node([x], [y], "argmax_i32_3D_keepdims_false")

            def last_index():
                x = np.random.randint(-127, 127, (2, 2, 2)).astype(np.int32)
                y = argmax_use_numpy_select_last_index(
                    x, dtype=np.uint32)

                x = Tensor(Dtype.I32, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())
                make_node([x], [y], "argmax_i32_3D_last_index")

            default_params()
            keepdims_false()
            last_index()
        argmax_3D()

    @staticmethod
    def argmax_i8():
        def argmax_1D():
            def default_params():
                x = np.random.randint(-127, 127, (3)).astype(np.int8)
                y = argmax_use_numpy(x, dtype=np.uint32).reshape((1))

                x = Tensor(Dtype.I8, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())
                make_node([x], [y], "argmax_i8_1D_default")

            def keepdims_false():
                x = np.random.randint(-127, 127, (3)).astype(np.int8)
                y = argmax_use_numpy(
                    x, keepdims=0, dtype=np.uint32).reshape((1))

                x = Tensor(Dtype.I8, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())
                make_node([x], [y], "argmax_i8_1D_keepdims_false")

            def last_index():
                x = np.random.randint(0, 255, (3)).astype(np.int8)
                y = argmax_use_numpy_select_last_index(
                    x, dtype=np.uint32).reshape((1))

                x = Tensor(Dtype.I8, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())
                make_node([x], [y], "argmax_i8_1D_last_index")

            default_params()
            keepdims_false()
            last_index()
        argmax_1D()

        def argmax_2D():
            def default_params():
                x = np.random.randint(-127, 127, (2, 2)).astype(np.int8)
                y = argmax_use_numpy(x, dtype=np.uint32)

                x = Tensor(Dtype.I8, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())
                make_node([x], [y], "argmax_i8_2D_default")

            def keepdims_false():
                x = np.random.randint(-127, 127, (2, 2)).astype(np.int8)
                y = argmax_use_numpy(
                    x, keepdims=0, dtype=np.uint32)

                x = Tensor(Dtype.I8, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())
                make_node([x], [y], "argmax_i8_2D_keepdims_false")

            def last_index():
                x = np.random.randint(-127, 127, (2, 2)).astype(np.int8)
                y = argmax_use_numpy_select_last_index(
                    x, dtype=np.int8)

                x = Tensor(Dtype.I8, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())
                make_node([x], [y], "argmax_i8_2D_last_index")

            default_params()
            keepdims_false()
            last_index()
        argmax_2D()

        def argmax_3D():
            def default_params():
                x = np.random.randint(-127, 127, (2, 2, 2)).astype(np.int8)
                y = argmax_use_numpy(x, dtype=np.uint32)

                x = Tensor(Dtype.I8, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())
                make_node([x], [y], "argmax_i8_3D_default")

            def keepdims_false():
                x = np.random.randint(-127, 127, (2, 2, 2)).astype(np.int8)
                y = argmax_use_numpy(
                    x, keepdims=0, dtype=np.uint32)

                x = Tensor(Dtype.I8, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())
                make_node([x], [y], "argmax_i8_3D_keepdims_false")

            def last_index():
                x = np.random.randint(-127, 127, (2, 2, 2)).astype(np.int8)
                y = argmax_use_numpy_select_last_index(
                    x, dtype=np.uint32)

                x = Tensor(Dtype.I8, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())
                make_node([x], [y], "argmax_i8_3D_last_index")

            default_params()
            keepdims_false()
            last_index()
        argmax_3D()

    @staticmethod
    def argmax_fp16x16():
        def argmax_1D():
            def default_params():
                x = to_fp(np.random.randint(-127, 127, (3)
                                            ).astype(np.int8), FixedImpl.FP16x16)
                y = argmax_use_numpy(x, dtype=np.uint32).reshape((1))

                x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())
                make_node([x], [y], "argmax_fp16x16_1D_default")

            def keepdims_false():
                x = to_fp(np.random.randint(-127, 127, (3)
                                            ).astype(np.int8), FixedImpl.FP16x16)
                y = argmax_use_numpy(
                    x, keepdims=0, dtype=np.uint32).reshape((1))

                x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())
                make_node([x], [y], "argmax_fp16x16_1D_keepdims_false")

            def last_index():
                x = to_fp(np.random.randint(0, 255, (3)).astype(
                    np.int8), FixedImpl.FP16x16)
                y = argmax_use_numpy_select_last_index(
                    x, dtype=np.uint32).reshape((1))

                x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())
                make_node([x], [y], "argmax_fp16x16_1D_last_index")

            default_params()
            keepdims_false()
            last_index()
        argmax_1D()

        def argmax_2D():
            def default_params():
                x = to_fp(np.random.randint(-127, 127, (2, 2)
                                            ).astype(np.int8), FixedImpl.FP16x16)
                y = argmax_use_numpy(x, dtype=np.uint32)

                x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())
                make_node([x], [y], "argmax_fp16x16_2D_default")

            def keepdims_false():
                x = to_fp(np.random.randint(-127, 127, (2, 2)
                                            ).astype(np.int8), FixedImpl.FP16x16)
                y = argmax_use_numpy(
                    x, keepdims=0, dtype=np.uint32)

                x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())
                make_node([x], [y], "argmax_fp16x16_2D_keepdims_false")

            def last_index():
                x = to_fp(np.random.randint(-127, 127, (2, 2)
                                            ).astype(np.int8), FixedImpl.FP16x16)
                y = argmax_use_numpy_select_last_index(
                    x, dtype=np.int8)

                x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())
                make_node([x], [y], "argmax_fp16x16_2D_last_index")

            default_params()
            keepdims_false()
            last_index()
        argmax_2D()

        def argmax_3D():
            def default_params():
                x = to_fp(np.random.randint(-127, 127, (2, 2, 2)
                                            ).astype(np.int8), FixedImpl.FP16x16)
                y = argmax_use_numpy(x, dtype=np.uint32)

                x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())
                make_node([x], [y], "argmax_fp16x16_3D_default")

            def keepdims_false():
                x = to_fp(np.random.randint(-127, 127, (2, 2, 2)
                                            ).astype(np.int8), FixedImpl.FP16x16)
                y = argmax_use_numpy(
                    x, keepdims=0, dtype=np.uint32)

                x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())
                make_node([x], [y], "argmax_fp16x16_3D_keepdims_false")

            def last_index():
                x = to_fp(np.random.randint(-127, 127, (2, 2, 2)
                                            ).astype(np.int8), FixedImpl.FP16x16)
                y = argmax_use_numpy_select_last_index(
                    x, dtype=np.uint32)

                x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())
                make_node([x], [y], "argmax_fp16x16_3D_last_index")

            default_params()
            keepdims_false()
            last_index()
        argmax_3D()

    @staticmethod
    def argmax_fp8x23():
        def argmax_1D():
            def default_params():
                x = to_fp(np.random.randint(-127, 127, (3)
                                            ).astype(np.int8), FixedImpl.FP8x23)
                y = argmax_use_numpy(x, dtype=np.uint32).reshape((1))

                x = Tensor(Dtype.FP8x23, x.shape, x.flatten(), FixedImpl.FP8x23)
                y = Tensor(Dtype.U32, y.shape, y.flatten())
                make_node([x], [y], "argmax_fp8x23_1D_default")

            def keepdims_false():
                x = to_fp(np.random.randint(-127, 127, (3)
                                            ).astype(np.int8), FixedImpl.FP8x23)
                y = argmax_use_numpy(
                    x, keepdims=0, dtype=np.uint32).reshape((1))

                x = Tensor(Dtype.FP8x23, x.shape, x.flatten(), FixedImpl.FP8x23)
                y = Tensor(Dtype.U32, y.shape, y.flatten())
                make_node([x], [y], "argmax_fp8x23_1D_keepdims_false")

            def last_index():
                x = to_fp(np.random.randint(0, 255, (3)).astype(
                    np.int8), FixedImpl.FP8x23)
                y = argmax_use_numpy_select_last_index(
                    x, dtype=np.uint32).reshape((1))

                x = Tensor(Dtype.FP8x23, x.shape, x.flatten(), FixedImpl.FP8x23)
                y = Tensor(Dtype.U32, y.shape, y.flatten())
                make_node([x], [y], "argmax_fp8x23_1D_last_index")

            default_params()
            keepdims_false()
            last_index()
        argmax_1D()

        def argmax_2D():
            def default_params():
                x = to_fp(np.random.randint(-127, 127, (2, 2)
                                            ).astype(np.int8), FixedImpl.FP8x23)
                y = argmax_use_numpy(x, dtype=np.uint32)

                x = Tensor(Dtype.FP8x23, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())
                make_node([x], [y], "argmax_fp8x23_2D_default")

            def keepdims_false():
                x = to_fp(np.random.randint(-127, 127, (2, 2)
                                            ).astype(np.int8), FixedImpl.FP8x23)
                y = argmax_use_numpy(
                    x, keepdims=0, dtype=np.uint32)

                x = Tensor(Dtype.FP8x23, x.shape, x.flatten(), FixedImpl.FP8x23)
                y = Tensor(Dtype.U32, y.shape, y.flatten())
                make_node([x], [y], "argmax_fp8x23_2D_keepdims_false")

            def last_index():
                x = to_fp(np.random.randint(-127, 127, (2, 2)
                                            ).astype(np.int8), FixedImpl.FP8x23)
                y = argmax_use_numpy_select_last_index(
                    x, dtype=np.int8)

                x = Tensor(Dtype.FP8x23, x.shape, x.flatten(), FixedImpl.FP8x23)
                y = Tensor(Dtype.U32, y.shape, y.flatten())
                make_node([x], [y], "argmax_fp8x23_2D_last_index")

            default_params()
            keepdims_false()
            last_index()
        argmax_2D()

        def argmax_3D():
            def default_params():
                x = to_fp(np.random.randint(-127, 127, (2, 2, 2)
                                            ).astype(np.int8), FixedImpl.FP8x23)
                y = argmax_use_numpy(x, dtype=np.uint32)

                x = Tensor(Dtype.FP8x23, x.shape, x.flatten(), FixedImpl.FP8x23)
                y = Tensor(Dtype.U32, y.shape, y.flatten())
                make_node([x], [y], "argmax_fp8x23_3D_default")

            def keepdims_false():
                x = to_fp(np.random.randint(-127, 127, (2, 2, 2)
                                            ).astype(np.int8), FixedImpl.FP8x23)
                y = argmax_use_numpy(
                    x, keepdims=0, dtype=np.uint32)

                x = Tensor(Dtype.FP8x23, x.shape, x.flatten(), FixedImpl.FP8x23)
                y = Tensor(Dtype.U32, y.shape, y.flatten())
                make_node([x], [y], "argmax_fp8x23_3D_keepdims_false")

            def last_index():
                x = to_fp(np.random.randint(-127, 127, (2, 2, 2)
                                            ).astype(np.int8), FixedImpl.FP8x23)
                y = argmax_use_numpy_select_last_index(
                    x, dtype=np.uint32)

                x = Tensor(Dtype.FP8x23, x.shape, x.flatten(), FixedImpl.FP8x23)
                y = Tensor(Dtype.U32, y.shape, y.flatten())
                make_node([x], [y], "argmax_fp8x23_3D_last_index")

            default_params()
            keepdims_false()
            last_index()
        argmax_3D()
