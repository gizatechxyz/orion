import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl, Trait


def _unsort_outputs(
    x: np.ndarray, indices: np.ndarray, inverse_indices: np.ndarray
) -> (np.ndarray, np.ndarray, np.ndarray):
    """Unsort the result of np.unique().

    This is done because numpy unique does not retain original order (it sorts
    the output unique values).
    # https://github.com/numpy/numpy/issues/8621
    """
    sorted_unique_values = x
    unsorted_unique_values = sorted_unique_values[inverse_indices]
    unsorted_indices = np.argsort(indices)
    unsorted_inverse_indices = np.empty_like(inverse_indices)
    unsorted_inverse_indices[unsorted_indices] = np.arange(len(indices))
    return unsorted_unique_values, unsorted_indices, unsorted_inverse_indices


class Unique(RunAll):
    @staticmethod
    def unique_u32():
        def without_axis_sorted():
            # random X with random shape
            x = np.random.randint(0, 6, (3, 3, 3)).astype(np.uint32)
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
            x = np.random.randint(0, 6, (3, 3, 3)).astype(np.uint32)
            axis = None

            unique_values, indices, inverse_indices, counts = np.unique(
                x, axis=axis, return_index=True, return_inverse=True, return_counts=True
            )

            unique_values, indices, inverse_indices = _unsort_outputs(
                unique_values, indices, inverse_indices
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
            x = np.random.randint(0, 6, (3, 3, 3)).astype(np.uint32)
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
            x = np.random.randint(0, 6, (3, 3, 3)).astype(np.uint32)
            axis = 0

            unique_values, indices, inverse_indices, counts = np.unique(
                x, axis=axis, return_index=True, return_inverse=True, return_counts=True
            )
            unique_values, indices, inverse_indices = _unsort_outputs(
                unique_values, indices, inverse_indices
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
            x = np.random.randint(0, 6, (3, 3, 3)).astype(np.uint32)
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
            x = np.random.randint(0, 6, (3, 3, 3)).astype(np.uint32)
            axis = 1

            unique_values, indices, inverse_indices, counts = np.unique(
                x, axis=axis, return_index=True, return_inverse=True, return_counts=True
            )
            unique_values, indices, inverse_indices = _unsort_outputs(
                unique_values, indices, inverse_indices
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
