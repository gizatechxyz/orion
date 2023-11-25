import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl, Trait


def _unsort_outputs(
    y: np.ndarray, indices: np.ndarray, inverse_indices: np.ndarray, counts: np.ndarray
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """Unsort the result of np.unique().

    This is done because numpy unique does not retain original order (it sorts
    the output unique values).
    # https://github.com/numpy/numpy/issues/8621
    """
    original_positions = np.arange(y.size)
    unsorted_positions = original_positions[np.argsort(indices)]
    y_unsorted = y[unsorted_positions]
    inverse_indices_unsorted = unsorted_positions[inverse_indices]
    indices_unsorted = np.arange(y.size)
    counts_unsorted = counts[unsorted_positions]
    return y_unsorted, indices_unsorted, inverse_indices_unsorted, counts_unsorted


class Unique(RunAll):
    @staticmethod
    def unique_u32():
        def without_axis_sorted():
            x = np.random.randint(0, 15, (3, 3, 3)).astype(np.uint32)
            axis = None

            unique_values, indices, inverse_indices, counts = np.unique(
                x, axis=axis, return_index=True, return_inverse=True, return_counts=True
            )

            x = Tensor(Dtype.U32, x.shape, x.flatten())
            unique_values = Tensor(
                Dtype.U32, unique_values.shape, unique_values.flatten()
            )
            indices = Tensor(Dtype.I32, indices.shape, indices.flatten())
            inverse_indices = Tensor(
                Dtype.I32, inverse_indices.shape, inverse_indices.flatten()
            )
            counts = Tensor(Dtype.I32, counts.shape, counts.flatten())

            name = "unique_u32_without_axis_sorted"
            make_test(
                [x],
                (unique_values, indices, inverse_indices, counts),
                "input_0.unique(Option::None(()), Option::None(()))",
                name,
            )

        def without_axis_not_sorted():
            x = np.random.randint(0, 15, (3, 3, 3)).astype(np.uint32)
            axis = None

            unique_values, indices, inverse_indices, counts = np.unique(
                x, axis=axis, return_index=True, return_inverse=True, return_counts=True
            )

            unique_values, indices, inverse_indices, counts = _unsort_outputs(
                unique_values, indices, inverse_indices, counts
            )

            x = Tensor(Dtype.U32, x.shape, x.flatten())
            unique_values = Tensor(
                Dtype.U32, unique_values.shape, unique_values.flatten()
            )
            indices = Tensor(Dtype.I32, indices.shape, indices.flatten())
            inverse_indices = Tensor(
                Dtype.I32, inverse_indices.shape, inverse_indices.flatten()
            )
            counts = Tensor(Dtype.I32, counts.shape, counts.flatten())

            name = "unique_u32_without_axis_not_sorted"
            make_test(
                [x],
                (unique_values, indices, inverse_indices, counts),
                "input_0.unique(Option::None(()), Option::Some(false))",
                name,
            )

        def with_axis_zero_sorted():
            x = np.random.randint(0, 15, (3, 3, 3)).astype(np.uint32)
            axis = 0

            unique_values, indices, inverse_indices, counts = np.unique(
                x, axis=axis, return_index=True, return_inverse=True, return_counts=True
            )

            x = Tensor(Dtype.U32, x.shape, x.flatten())
            unique_values = Tensor(
                Dtype.U32, unique_values.shape, unique_values.flatten()
            )
            indices = Tensor(Dtype.I32, indices.shape, indices.flatten())
            inverse_indices = Tensor(
                Dtype.I32, inverse_indices.shape, inverse_indices.flatten()
            )
            counts = Tensor(Dtype.I32, counts.shape, counts.flatten())

            name = "unique_u32_with_axis_zero_sorted"
            make_test(
                [x],
                (unique_values, indices, inverse_indices, counts),
                "input_0.unique(Option::Some(0), Option::Some(true))",
                name,
            )

        def with_axis_zero_not_sorted():
            x = np.random.randint(0, 15, (3, 3, 3)).astype(np.uint32)
            axis = 0

            unique_values, indices, inverse_indices, counts = np.unique(
                x, axis=axis, return_index=True, return_inverse=True, return_counts=True
            )
            unique_values, indices, inverse_indices, counts = _unsort_outputs(
                unique_values, indices, inverse_indices, counts
            )

            x = Tensor(Dtype.U32, x.shape, x.flatten())
            unique_values = Tensor(
                Dtype.U32, unique_values.shape, unique_values.flatten()
            )
            indices = Tensor(Dtype.I32, indices.shape, indices.flatten())
            inverse_indices = Tensor(
                Dtype.I32, inverse_indices.shape, inverse_indices.flatten()
            )
            counts = Tensor(Dtype.I32, counts.shape, counts.flatten())

            name = "unique_u32_with_axis_zero_not_sorted"
            make_test(
                [x],
                (unique_values, indices, inverse_indices, counts),
                "input_0.unique(Option::Some(0), Option::Some(false))",
                name,
            )

        def with_axis_one_sorted():
            x = np.random.randint(0, 15, (3, 3, 3)).astype(np.uint32)
            axis = 1

            unique_values, indices, inverse_indices, counts = np.unique(
                x, axis=axis, return_index=True, return_inverse=True, return_counts=True
            )

            x = Tensor(Dtype.U32, x.shape, x.flatten())
            unique_values = Tensor(
                Dtype.U32, unique_values.shape, unique_values.flatten()
            )
            indices = Tensor(Dtype.I32, indices.shape, indices.flatten())
            inverse_indices = Tensor(
                Dtype.I32, inverse_indices.shape, inverse_indices.flatten()
            )
            counts = Tensor(Dtype.I32, counts.shape, counts.flatten())

            name = "unique_u32_with_axis_one_sorted"
            make_test(
                [x],
                (unique_values, indices, inverse_indices, counts),
                "input_0.unique(Option::Some(1), Option::Some(true))",
                name,
            )

        def with_axis_one_not_sorted():
            x = np.random.randint(0, 15, (3, 3, 3)).astype(np.uint32)
            axis = 1

            unique_values, indices, inverse_indices, counts = np.unique(
                x, axis=axis, return_index=True, return_inverse=True, return_counts=True
            )
            unique_values, indices, inverse_indices, counts = _unsort_outputs(
                unique_values, indices, inverse_indices, counts
            )

            x = Tensor(Dtype.U32, x.shape, x.flatten())
            unique_values = Tensor(
                Dtype.U32, unique_values.shape, unique_values.flatten()
            )
            indices = Tensor(Dtype.I32, indices.shape, indices.flatten())
            inverse_indices = Tensor(
                Dtype.I32, inverse_indices.shape, inverse_indices.flatten()
            )
            counts = Tensor(Dtype.I32, counts.shape, counts.flatten())

            name = "unique_u32_with_axis_one_not_sorted"
            make_test(
                [x],
                (unique_values, indices, inverse_indices, counts),
                "input_0.unique(Option::Some(1), Option::Some(false))",
                name,
            )

        without_axis_sorted()
        without_axis_not_sorted()
        with_axis_zero_sorted()
        with_axis_zero_not_sorted()
        with_axis_one_sorted()
        with_axis_one_not_sorted()
