import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, Tensor, Dtype

original_shape = [2, 3, 4]
data = np.random.random_sample(original_shape).astype(np.int32)


def reshape_reference_implementation(
    data: np.ndarray, shape: np.ndarray, allowzero: int = 0
) -> np.ndarray:
    # replace zeros with corresponding dim size
    # we need to do this because np.reshape doesn't support 0 by default unless 'allowzero' is set
    new_shape = np.copy(shape)
    if allowzero == 0:
        zeros_index = np.where(shape == 0)
        new_shape[zeros_index] = np.array(data.shape)[zeros_index]
    reshaped = np.reshape(data, new_shape)
    return reshaped


class Reshape(RunAll):
    @staticmethod
    def reshape_reordered_all_dims():
        y = reshape_reference_implementation(
            data, np.array([4, 2, 3], dtype=np.int64))

        x = Tensor(Dtype.I32, data.shape, data.flatten())
        y = Tensor(Dtype.I32, y.shape, y.flatten())

        name = "reshape_reordered_all_dims"
        make_test([x], y, "input_0.reshape(array![4,2,3].span())", name)

    @staticmethod
    def reshape_reordered_last_dims():
        y = reshape_reference_implementation(
            data, np.array([2, 4, 3], dtype=np.int64))

        x = Tensor(Dtype.I32, data.shape, data.flatten())
        y = Tensor(Dtype.I32, y.shape, y.flatten())

        name = "reshape_reordered_last_dims"
        make_test([x], y, "input_0.reshape(array![2,4,3].span())", name)

    @staticmethod
    def reshape_reduced_dims():
        y = reshape_reference_implementation(
            data, np.array([2, 12], dtype=np.int64))

        x = Tensor(Dtype.I32, data.shape, data.flatten())
        y = Tensor(Dtype.I32, y.shape, y.flatten())

        name = "reshape_reduced_dims"
        make_test([x], y, "input_0.reshape(array![2,12].span())", name)

    @staticmethod
    def reshape_extended_dims():
        y = reshape_reference_implementation(
            data, np.array([2, 3, 2, 2], dtype=np.int64))

        x = Tensor(Dtype.I32, data.shape, data.flatten())
        y = Tensor(Dtype.I32, y.shape, y.flatten())

        name = "reshape_extended_dims"
        make_test([x], y, "input_0.reshape(array![2, 3, 2, 2].span())", name)

    @staticmethod
    def reshape_one_dim():
        y = reshape_reference_implementation(
            data, np.array([24], dtype=np.int64))

        x = Tensor(Dtype.I32, data.shape, data.flatten())
        y = Tensor(Dtype.I32, y.shape, y.flatten())

        name = "reshape_one_dim"
        make_test([x], y, "input_0.reshape(array![24].span())", name)

    @staticmethod
    def reshape_negative_dim():
        y = reshape_reference_implementation(
            data, np.array([2, -1, 2], dtype=np.int64))

        x = Tensor(Dtype.I32, data.shape, data.flatten())
        y = Tensor(Dtype.I32, y.shape, y.flatten())

        name = "reshape_negative_dim"
        make_test([x], y, "input_0.reshape(array![2, -1, 2].span())", name)

    @staticmethod
    def reshape_negative_extended_dims():
        y = reshape_reference_implementation(
            data, np.array([-1, 2, 3, 4], dtype=np.int64))

        x = Tensor(Dtype.I32, data.shape, data.flatten())
        y = Tensor(Dtype.I32, y.shape, y.flatten())

        name = "reshape_negative_extended_dims"
        make_test([x], y, "input_0.reshape(array![-1, 2, 3, 4].span())", name)

    @staticmethod
    def reshape_zero_dim():
        y = reshape_reference_implementation(
            data, np.array([2, 0, 4, 1], dtype=np.int64))

        x = Tensor(Dtype.I32, data.shape, data.flatten())
        y = Tensor(Dtype.I32, y.shape, y.flatten())

        name = "reshape_zero_dim"
        make_test([x], y, "input_0.reshape(array![2, 0, 4, 1].span())", name)

    @staticmethod
    def reshape_zero_and_negative_dim():
        y = reshape_reference_implementation(
            data, np.array([2, 0, 1, -1], dtype=np.int64))

        x = Tensor(Dtype.I32, data.shape, data.flatten())
        y = Tensor(Dtype.I32, y.shape, y.flatten())

        name = "reshape_zero_and_negative_dim"
        make_test([x], y, "input_0.reshape(array![2, 0, 1, -1].span())", name)
