import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl


class Reduce_sum(RunAll):

    @staticmethod
    def reduce_sum_no_keep_dims():
        axes = np.array([1], dtype=np.uint32)
        keepdims = 0

        x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [
                     [9, 10], [11, 12]]]).astype(np.uint32)
        y = np.sum(x, axis=tuple(axes.tolist()), keepdims=keepdims == 1)

        x = Tensor(Dtype.U32, x.shape, x.flatten())
        y = Tensor(Dtype.U32, y.shape, y.flatten())

        name = "reduce_sum_no_keep_dims"
        make_test(
            [x], y, "input_0.reduce_sum(Option::Some(array![1].span()), Option::Some(false), Option::None)", name)

    @staticmethod
    def reduce_sum_keep_dims():
        axes = np.array([1], dtype=np.uint32)
        keepdims = 1

        x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [
            [9, 10], [11, 12]]]).astype(np.uint32)
        y = np.sum(x, axis=tuple(axes.tolist()), keepdims=keepdims == 1)

        x = Tensor(Dtype.U32, x.shape, x.flatten())
        y = Tensor(Dtype.U32, y.shape, y.flatten())

        name = "reduce_sum_keep_dims"
        make_test(
            [x], y, "input_0.reduce_sum(Option::Some(array![1].span()), Option::Some(true), Option::None)", name)

    @staticmethod
    def reduce_sum_default_axes_keepdims():
        keepdims = 1

        x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [
            [9, 10], [11, 12]]]).astype(np.uint32)
        y = np.sum(x, axis=None, keepdims=keepdims == 1)

        x = Tensor(Dtype.U32, x.shape, x.flatten())
        y = Tensor(Dtype.U32, y.shape, y.flatten())

        name = "reduce_sum_default_axes_keepdims"
        make_test(
            [x], y, "input_0.reduce_sum(Option::Some(array![].span()), Option::Some(true), Option::None)", name)

    @staticmethod
    def reduce_sum_empty_axes_input_noop():
        x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [
            [9, 10], [11, 12]]]).astype(np.uint32)
        y = np.array(x)

        x = Tensor(Dtype.U32, x.shape, x.flatten())
        y = Tensor(Dtype.U32, y.shape, y.flatten())

        name = "reduce_sum_empty_axes_input_noop"
        make_test(
            [x], y, "input_0.reduce_sum(Option::None, Option::Some(true), Option::Some(true))", name)