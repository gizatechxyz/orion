import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl


def argmax_use_numpy(data: np.ndarray, axis: int = 0, keepdims: int = 1) -> np.ndarray:
    result = np.argmax(data, axis=axis)
    if keepdims == 1:
        result = np.expand_dims(result, axis)
    return result.astype(np.int64)


def argmax_use_numpy_select_last_index(
    data: np.ndarray, axis: int = 0, keepdims: int = True
) -> np.ndarray:
    data = np.flip(data, axis)
    result = np.argmax(data, axis=axis)
    result = data.shape[axis] - result - 1
    if keepdims:
        result = np.expand_dims(result, axis)
    return result.astype(np.int64)


class Argmax(RunAll):

    @staticmethod
    def no_keepdims():
        data = np.array([[2, 1], [3, 10]], dtype=np.float32)
        axis = 1
        keepdims = 0
        result = argmax_use_numpy(data, axis=axis, keepdims=keepdims)

        x = Tensor(Dtype.FP16x16, data.shape, data.flatten())
        y = Tensor(Dtype.I32, result.shape, result.flatten())

        name = "argmax_no_keepdims"
        make_test(
            [x], y, "input_0.argmax(1, Option::Some(false), Option::None(()))", name)

    @staticmethod
    def keepdims():
        data = np.array([[2, 1], [3, 10]], dtype=np.float32)
        axis = 1
        keepdims = 1
        result = argmax_use_numpy(data, axis=axis, keepdims=keepdims)

        x = Tensor(Dtype.FP16x16, data.shape, data.flatten())
        y = Tensor(Dtype.I32, result.shape, result.flatten())

        name = "argmax_keepdims"
        make_test(
            [x], y, "input_0.argmax(1, Option::Some(true), Option::None(()))", name)

    @staticmethod
    def default_axes_keepdims():
        data = np.array([[2, 1], [3, 10]], dtype=np.float32)
        keepdims = 1
        result = argmax_use_numpy(data, keepdims=keepdims)

        x = Tensor(Dtype.FP16x16, data.shape, data.flatten())
        y = Tensor(Dtype.I32, result.shape, result.flatten())

        name = "argmax_default_axes_keepdims"
        make_test(
            [x], y, "input_0.argmax(0, Option::Some(true), Option::None(()))", name)

    @staticmethod
    def negative_axis_keepdims():
        data = np.array([[2, 1], [3, 10]], dtype=np.float32)
        axis = -1
        keepdims = 1
        result = argmax_use_numpy(data, axis=axis, keepdims=keepdims)

        x = Tensor(Dtype.FP16x16, data.shape, data.flatten())
        y = Tensor(Dtype.I32, result.shape, result.flatten())

        name = "argmax_negative_axis_keepdims"
        make_test(
            [x], y, "input_0.argmax(-1, Option::Some(true), Option::None(()))", name)

    @staticmethod
    def no_keepdims_select_last_index():
        data = np.array([[2, 2], [3, 10]], dtype=np.float32)
        axis = 1
        keepdims = 0
        result = argmax_use_numpy_select_last_index(
            data, axis=axis, keepdims=keepdims)

        x = Tensor(Dtype.FP16x16, data.shape, data.flatten())
        y = Tensor(Dtype.I32, result.shape, result.flatten())

        name = "argmax_no_keepdims_select_last_index"
        make_test(
            [x], y, "input_0.argmax(1, Option::Some(false), Option::Some(true))", name)

    @staticmethod
    def keepdims_select_last_index():
        data = np.array([[2, 2], [3, 10]], dtype=np.float32)
        axis = 1
        keepdims = 1
        result = argmax_use_numpy_select_last_index(
            data, axis=axis, keepdims=keepdims)

        x = Tensor(Dtype.FP16x16, data.shape, data.flatten())
        y = Tensor(Dtype.I32, result.shape, result.flatten())

        name = "argmax_keepdims_select_last_index"
        make_test(
            [x], y, "input_0.argmax(1, Option::Some(true), Option::Some(true))", name)

    @staticmethod
    def default_axes_keepdims_select_last_index():
        data = np.array([[2, 2], [3, 10]], dtype=np.float32)
        keepdims = 1
        result = argmax_use_numpy_select_last_index(data, keepdims=keepdims)

        x = Tensor(Dtype.FP16x16, data.shape, data.flatten())
        y = Tensor(Dtype.I32, result.shape, result.flatten())

        name = "argmax_default_axes_keepdims_select_last_index"
        make_test(
            [x], y, "input_0.argmax(0, Option::Some(true), Option::Some(true))", name)

    @staticmethod
    def negative_axis_keepdims_select_last_index():
        data = np.array([[2, 2], [3, 10]], dtype=np.float32)
        axis = -1
        keepdims = 1
        result = argmax_use_numpy_select_last_index(data, axis=axis, keepdims=keepdims)

        x = Tensor(Dtype.FP16x16, data.shape, data.flatten())
        y = Tensor(Dtype.I32, result.shape, result.flatten())

        name = "argmax_negative_axis_keepdims_select_last_index"
        make_test(
            [x], y, "input_0.argmax(-1, Option::Some(true), Option::Some(true))", name)
