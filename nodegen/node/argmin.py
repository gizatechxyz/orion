import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl


def argmin_use_numpy(data: np.ndarray, axis: int = 0, keepdims: int = 1, dtype=np.int64) -> np.ndarray:
    result = np.argmin(data, axis=axis)
    if keepdims == 1:
        result = np.expand_dims(result, axis)
    return result.astype(dtype)


def argmin_use_numpy_select_last_index(
    data: np.ndarray, axis: int = 0, keepdims: int = True, dtype=np.int64
) -> np.ndarray:
    data = np.flip(data, axis)
    result = np.argmin(data, axis=axis)
    result = data.shape[axis] - result - 1
    if keepdims:
        result = np.expand_dims(result, axis)
    return result.astype(dtype)


class Argmin(RunAll):

    @staticmethod
    def argmin_u32():
        def argmin_1D():
            def default_params():
                x = np.random.randint(0, 255, (3)).astype(np.uint32)
                y = argmin_use_numpy(x, dtype=np.uint32).reshape((1))

                x = Tensor(Dtype.U32, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "argmin_u32_1D_default"
                make_test(
                    [x], y, "input_0.argmin(0, Option::None(()), Option::None(()))", name)

            def keepdims_false():
                x = np.random.randint(0, 255, (3)).astype(np.uint32)
                y = argmin_use_numpy(
                    x, keepdims=0, dtype=np.uint32).reshape((1))

                x = Tensor(Dtype.U32, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "argmin_u32_1D_keepdims_false"
                make_test(
                    [x], y, "input_0.argmin(0, Option::Some(false), Option::None(()))", name)

            def last_index():
                x = np.random.randint(0, 255, (3)).astype(np.uint32)
                y = argmin_use_numpy_select_last_index(
                    x, dtype=np.uint32).reshape((1))

                x = Tensor(Dtype.U32, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "argmin_u32_1D_last_index"
                make_test(
                    [x], y, "input_0.argmin(0, Option::None(()), Option::Some(true))", name)

            default_params()
            keepdims_false()
            last_index()
        argmin_1D()

        def argmin_2D():
            def default_params():
                x = np.random.randint(0, 255, (2, 2)).astype(np.uint32)
                y = argmin_use_numpy(x, dtype=np.uint32)

                x = Tensor(Dtype.U32, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "argmin_u32_2D_default"
                make_test(
                    [x], y, "input_0.argmin(0, Option::None(()), Option::None(()))", name)

            def keepdims_false():
                x = np.random.randint(0, 255, (2, 2)).astype(np.uint32)
                y = argmin_use_numpy(
                    x, keepdims=0, dtype=np.uint32)

                x = Tensor(Dtype.U32, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "argmin_u32_2D_keepdims_false"
                make_test(
                    [x], y, "input_0.argmin(0, Option::Some(false), Option::None(()))", name)

            def last_index():
                x = np.random.randint(0, 255, (2, 2)).astype(np.uint32)
                y = argmin_use_numpy_select_last_index(
                    x, dtype=np.uint32)

                x = Tensor(Dtype.U32, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "argmin_u32_2D_last_index"
                make_test(
                    [x], y, "input_0.argmin(0, Option::None(()), Option::Some(true))", name)

            default_params()
            keepdims_false()
            last_index()
        argmin_2D()

        def argmin_3D():
            def default_params():
                x = np.random.randint(0, 255, (2, 2, 2)).astype(np.uint32)
                y = argmin_use_numpy(x, dtype=np.uint32)

                x = Tensor(Dtype.U32, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "argmin_u32_3D_default"
                make_test(
                    [x], y, "input_0.argmin(0, Option::None(()), Option::None(()))", name)

            def keepdims_false():
                x = np.random.randint(0, 255, (2, 2, 2)).astype(np.uint32)
                y = argmin_use_numpy(
                    x, keepdims=0, dtype=np.uint32)

                x = Tensor(Dtype.U32, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "argmin_u32_3D_keepdims_false"
                make_test(
                    [x], y, "input_0.argmin(0, Option::Some(false), Option::None(()))", name)

            def last_index():
                x = np.random.randint(0, 255, (2, 2, 2)).astype(np.uint32)
                y = argmin_use_numpy_select_last_index(
                    x, dtype=np.uint32)

                x = Tensor(Dtype.U32, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "argmin_u32_3D_last_index"
                make_test(
                    [x], y, "input_0.argmin(0, Option::None(()), Option::Some(true))", name)

            default_params()
            keepdims_false()
            last_index()
        argmin_3D()

    @staticmethod
    def argmin_i32():
        def argmin_1D():
            def default_params():
                x = np.random.randint(-127, 127, (3)).astype(np.int32)
                y = argmin_use_numpy(x, dtype=np.uint32).reshape((1))

                x = Tensor(Dtype.I32, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "argmin_i32_1D_default"
                make_test(
                    [x], y, "input_0.argmin(0, Option::None(()), Option::None(()))", name)

            def keepdims_false():
                x = np.random.randint(-127, 127, (3)).astype(np.int32)
                y = argmin_use_numpy(
                    x, keepdims=0, dtype=np.uint32).reshape((1))

                x = Tensor(Dtype.I32, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "argmin_i32_1D_keepdims_false"
                make_test(
                    [x], y, "input_0.argmin(0, Option::Some(false), Option::None(()))", name)

            def last_index():
                x = np.random.randint(0, 255, (3)).astype(np.int32)
                y = argmin_use_numpy_select_last_index(
                    x, dtype=np.uint32).reshape((1))

                x = Tensor(Dtype.I32, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "argmin_i32_1D_last_index"
                make_test(
                    [x], y, "input_0.argmin(0, Option::None(()), Option::Some(true))", name)

            default_params()
            keepdims_false()
            last_index()
        argmin_1D()

        def argmin_2D():
            def default_params():
                x = np.random.randint(-127, 127, (2, 2)).astype(np.int32)
                y = argmin_use_numpy(x, dtype=np.uint32)

                x = Tensor(Dtype.I32, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "argmin_i32_2D_default"
                make_test(
                    [x], y, "input_0.argmin(0, Option::None(()), Option::None(()))", name)

            def keepdims_false():
                x = np.random.randint(-127, 127, (2, 2)).astype(np.int32)
                y = argmin_use_numpy(
                    x, keepdims=0, dtype=np.uint32)

                x = Tensor(Dtype.I32, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "argmin_i32_2D_keepdims_false"
                make_test(
                    [x], y, "input_0.argmin(0, Option::Some(false), Option::None(()))", name)

            def last_index():
                x = np.random.randint(-127, 127, (2, 2)).astype(np.int32)
                y = argmin_use_numpy_select_last_index(
                    x, dtype=np.int32)

                x = Tensor(Dtype.I32, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "argmin_i32_2D_last_index"
                make_test(
                    [x], y, "input_0.argmin(0, Option::None(()), Option::Some(true))", name)

            default_params()
            keepdims_false()
            last_index()
        argmin_2D()

        def argmin_3D():
            def default_params():
                x = np.random.randint(-127, 127, (2, 2, 2)).astype(np.int32)
                y = argmin_use_numpy(x, dtype=np.uint32)

                x = Tensor(Dtype.I32, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "argmin_i32_3D_default"
                make_test(
                    [x], y, "input_0.argmin(0, Option::None(()), Option::None(()))", name)

            def keepdims_false():
                x = np.random.randint(-127, 127, (2, 2, 2)).astype(np.int32)
                y = argmin_use_numpy(
                    x, keepdims=0, dtype=np.uint32)

                x = Tensor(Dtype.I32, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "argmin_i32_3D_keepdims_false"
                make_test(
                    [x], y, "input_0.argmin(0, Option::Some(false), Option::None(()))", name)

            def last_index():
                x = np.random.randint(-127, 127, (2, 2, 2)).astype(np.int32)
                y = argmin_use_numpy_select_last_index(
                    x, dtype=np.uint32)

                x = Tensor(Dtype.I32, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "argmin_i32_3D_last_index"
                make_test(
                    [x], y, "input_0.argmin(0, Option::None(()), Option::Some(true))", name)

            default_params()
            keepdims_false()
            last_index()
        argmin_3D()

    @staticmethod
    def argmin_i8():
        def argmin_1D():
            def default_params():
                x = np.random.randint(-127, 127, (3)).astype(np.int8)
                y = argmin_use_numpy(x, dtype=np.uint32).reshape((1))

                x = Tensor(Dtype.I8, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "argmin_i8_1D_default"
                make_test(
                    [x], y, "input_0.argmin(0, Option::None(()), Option::None(()))", name)

            def keepdims_false():
                x = np.random.randint(-127, 127, (3)).astype(np.int8)
                y = argmin_use_numpy(
                    x, keepdims=0, dtype=np.uint32).reshape((1))

                x = Tensor(Dtype.I8, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "argmin_i8_1D_keepdims_false"
                make_test(
                    [x], y, "input_0.argmin(0, Option::Some(false), Option::None(()))", name)

            def last_index():
                x = np.random.randint(0, 255, (3)).astype(np.int8)
                y = argmin_use_numpy_select_last_index(
                    x, dtype=np.uint32).reshape((1))

                x = Tensor(Dtype.I8, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "argmin_i8_1D_last_index"
                make_test(
                    [x], y, "input_0.argmin(0, Option::None(()), Option::Some(true))", name)

            default_params()
            keepdims_false()
            last_index()
        argmin_1D()

        def argmin_2D():
            def default_params():
                x = np.random.randint(-127, 127, (2, 2)).astype(np.int8)
                y = argmin_use_numpy(x, dtype=np.uint32)

                x = Tensor(Dtype.I8, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "argmin_i8_2D_default"
                make_test(
                    [x], y, "input_0.argmin(0, Option::None(()), Option::None(()))", name)

            def keepdims_false():
                x = np.random.randint(-127, 127, (2, 2)).astype(np.int8)
                y = argmin_use_numpy(
                    x, keepdims=0, dtype=np.uint32)

                x = Tensor(Dtype.I8, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "argmin_i8_2D_keepdims_false"
                make_test(
                    [x], y, "input_0.argmin(0, Option::Some(false), Option::None(()))", name)

            def last_index():
                x = np.random.randint(-127, 127, (2, 2)).astype(np.int8)
                y = argmin_use_numpy_select_last_index(
                    x, dtype=np.int8)

                x = Tensor(Dtype.I8, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "argmin_i8_2D_last_index"
                make_test(
                    [x], y, "input_0.argmin(0, Option::None(()), Option::Some(true))", name)

            default_params()
            keepdims_false()
            last_index()
        argmin_2D()

        def argmin_3D():
            def default_params():
                x = np.random.randint(-127, 127, (2, 2, 2)).astype(np.int8)
                y = argmin_use_numpy(x, dtype=np.uint32)

                x = Tensor(Dtype.I8, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "argmin_i8_3D_default"
                make_test(
                    [x], y, "input_0.argmin(0, Option::None(()), Option::None(()))", name)

            def keepdims_false():
                x = np.random.randint(-127, 127, (2, 2, 2)).astype(np.int8)
                y = argmin_use_numpy(
                    x, keepdims=0, dtype=np.uint32)

                x = Tensor(Dtype.I8, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "argmin_i8_3D_keepdims_false"
                make_test(
                    [x], y, "input_0.argmin(0, Option::Some(false), Option::None(()))", name)

            def last_index():
                x = np.random.randint(-127, 127, (2, 2, 2)).astype(np.int8)
                y = argmin_use_numpy_select_last_index(
                    x, dtype=np.uint32)

                x = Tensor(Dtype.I8, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "argmin_i8_3D_last_index"
                make_test(
                    [x], y, "input_0.argmin(0, Option::None(()), Option::Some(true))", name)

            default_params()
            keepdims_false()
            last_index()
        argmin_3D()

    @staticmethod
    def argmin_fp16x16():
        def argmin_1D():
            def default_params():
                x = to_fp(np.random.randint(-127, 127, (3)
                                            ).astype(np.int8), FixedImpl.FP16x16)
                y = argmin_use_numpy(x, dtype=np.uint32).reshape((1))

                x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "argmin_fp16x16_1D_default"
                make_test(
                    [x], y, "input_0.argmin(0, Option::None(()), Option::None(()))", name)

            def keepdims_false():
                x = to_fp(np.random.randint(-127, 127, (3)
                                            ).astype(np.int8), FixedImpl.FP16x16)
                y = argmin_use_numpy(
                    x, keepdims=0, dtype=np.uint32).reshape((1))

                x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "argmin_fp16x16_1D_keepdims_false"
                make_test(
                    [x], y, "input_0.argmin(0, Option::Some(false), Option::None(()))", name)

            def last_index():
                x = to_fp(np.random.randint(0, 255, (3)).astype(
                    np.int8), FixedImpl.FP16x16)
                y = argmin_use_numpy_select_last_index(
                    x, dtype=np.uint32).reshape((1))

                x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "argmin_fp16x16_1D_last_index"
                make_test(
                    [x], y, "input_0.argmin(0, Option::None(()), Option::Some(true))", name)

            default_params()
            keepdims_false()
            last_index()
        argmin_1D()

        def argmin_2D():
            def default_params():
                x = to_fp(np.random.randint(-127, 127, (2, 2)
                                            ).astype(np.int8), FixedImpl.FP16x16)
                y = argmin_use_numpy(x, dtype=np.uint32)

                x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "argmin_fp16x16_2D_default"
                make_test(
                    [x], y, "input_0.argmin(0, Option::None(()), Option::None(()))", name)

            def keepdims_false():
                x = to_fp(np.random.randint(-127, 127, (2, 2)
                                            ).astype(np.int8), FixedImpl.FP16x16)
                y = argmin_use_numpy(
                    x, keepdims=0, dtype=np.uint32)

                x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "argmin_fp16x16_2D_keepdims_false"
                make_test(
                    [x], y, "input_0.argmin(0, Option::Some(false), Option::None(()))", name)

            def last_index():
                x = to_fp(np.random.randint(-127, 127, (2, 2)
                                            ).astype(np.int8), FixedImpl.FP16x16)
                y = argmin_use_numpy_select_last_index(
                    x, dtype=np.int8)

                x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "argmin_fp16x16_2D_last_index"
                make_test(
                    [x], y, "input_0.argmin(0, Option::None(()), Option::Some(true))", name)

            default_params()
            keepdims_false()
            last_index()
        argmin_2D()

        def argmin_3D():
            def default_params():
                x = to_fp(np.random.randint(-127, 127, (2, 2, 2)
                                            ).astype(np.int8), FixedImpl.FP16x16)
                y = argmin_use_numpy(x, dtype=np.uint32)

                x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "argmin_fp16x16_3D_default"
                make_test(
                    [x], y, "input_0.argmin(0, Option::None(()), Option::None(()))", name)

            def keepdims_false():
                x = to_fp(np.random.randint(-127, 127, (2, 2, 2)
                                            ).astype(np.int8), FixedImpl.FP16x16)
                y = argmin_use_numpy(
                    x, keepdims=0, dtype=np.uint32)

                x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "argmin_fp16x16_3D_keepdims_false"
                make_test(
                    [x], y, "input_0.argmin(0, Option::Some(false), Option::None(()))", name)

            def last_index():
                x = to_fp(np.random.randint(-127, 127, (2, 2, 2)
                                            ).astype(np.int8), FixedImpl.FP16x16)
                y = argmin_use_numpy_select_last_index(
                    x, dtype=np.uint32)

                x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "argmin_fp16x16_3D_last_index"
                make_test(
                    [x], y, "input_0.argmin(0, Option::None(()), Option::Some(true))", name)

            default_params()
            keepdims_false()
            last_index()
        argmin_3D()

    @staticmethod
    def argmin_fp8x23():
        def argmin_1D():
            def default_params():
                x = to_fp(np.random.randint(-127, 127, (3)
                                            ).astype(np.int8), FixedImpl.FP8x23)
                y = argmin_use_numpy(x, dtype=np.uint32).reshape((1))

                x = Tensor(Dtype.FP8x23, x.shape,
                           x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "argmin_fp8x23_1D_default"
                make_test(
                    [x], y, "input_0.argmin(0, Option::None(()), Option::None(()))", name)

            def keepdims_false():
                x = to_fp(np.random.randint(-127, 127, (3)
                                            ).astype(np.int8), FixedImpl.FP8x23)
                y = argmin_use_numpy(
                    x, keepdims=0, dtype=np.uint32).reshape((1))

                x = Tensor(Dtype.FP8x23, x.shape,
                           x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "argmin_fp8x23_1D_keepdims_false"
                make_test(
                    [x], y, "input_0.argmin(0, Option::Some(false), Option::None(()))", name)

            def last_index():
                x = to_fp(np.random.randint(0, 255, (3)).astype(
                    np.int8), FixedImpl.FP8x23)
                y = argmin_use_numpy_select_last_index(
                    x, dtype=np.uint32).reshape((1))

                x = Tensor(Dtype.FP8x23, x.shape,
                           x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "argmin_fp8x23_1D_last_index"
                make_test(
                    [x], y, "input_0.argmin(0, Option::None(()), Option::Some(true))", name)

            default_params()
            keepdims_false()
            last_index()
        argmin_1D()

        def argmin_2D():
            def default_params():
                x = to_fp(np.random.randint(-127, 127, (2, 2)
                                            ).astype(np.int8), FixedImpl.FP8x23)
                y = argmin_use_numpy(x, dtype=np.uint32)

                x = Tensor(Dtype.FP8x23, x.shape, x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "argmin_fp8x23_2D_default"
                make_test(
                    [x], y, "input_0.argmin(0, Option::None(()), Option::None(()))", name)

            def keepdims_false():
                x = to_fp(np.random.randint(-127, 127, (2, 2)
                                            ).astype(np.int8), FixedImpl.FP8x23)
                y = argmin_use_numpy(
                    x, keepdims=0, dtype=np.uint32)

                x = Tensor(Dtype.FP8x23, x.shape,
                           x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "argmin_fp8x23_2D_keepdims_false"
                make_test(
                    [x], y, "input_0.argmin(0, Option::Some(false), Option::None(()))", name)

            def last_index():
                x = to_fp(np.random.randint(-127, 127, (2, 2)
                                            ).astype(np.int8), FixedImpl.FP8x23)
                y = argmin_use_numpy_select_last_index(
                    x, dtype=np.int8)

                x = Tensor(Dtype.FP8x23, x.shape,
                           x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "argmin_fp8x23_2D_last_index"
                make_test(
                    [x], y, "input_0.argmin(0, Option::None(()), Option::Some(true))", name)

            default_params()
            keepdims_false()
            last_index()
        argmin_2D()

        def argmin_3D():
            def default_params():
                x = to_fp(np.random.randint(-127, 127, (2, 2, 2)
                                            ).astype(np.int8), FixedImpl.FP8x23)
                y = argmin_use_numpy(x, dtype=np.uint32)

                x = Tensor(Dtype.FP8x23, x.shape,
                           x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "argmin_fp8x23_3D_default"
                make_test(
                    [x], y, "input_0.argmin(0, Option::None(()), Option::None(()))", name)

            def keepdims_false():
                x = to_fp(np.random.randint(-127, 127, (2, 2, 2)
                                            ).astype(np.int8), FixedImpl.FP8x23)
                y = argmin_use_numpy(
                    x, keepdims=0, dtype=np.uint32)

                x = Tensor(Dtype.FP8x23, x.shape,
                           x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "argmin_fp8x23_3D_keepdims_false"
                make_test(
                    [x], y, "input_0.argmin(0, Option::Some(false), Option::None(()))", name)

            def last_index():
                x = to_fp(np.random.randint(-127, 127, (2, 2, 2)
                                            ).astype(np.int8), FixedImpl.FP8x23)
                y = argmin_use_numpy_select_last_index(
                    x, dtype=np.uint32)

                x = Tensor(Dtype.FP8x23, x.shape,
                           x.flatten())
                y = Tensor(Dtype.U32, y.shape, y.flatten())

                name = "argmin_fp8x23_3D_last_index"
                make_test(
                    [x], y, "input_0.argmin(0, Option::None(()), Option::Some(true))", name)

            default_params()
            keepdims_false()
            last_index()
        argmin_3D()
