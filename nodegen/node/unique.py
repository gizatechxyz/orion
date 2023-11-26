import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl, Trait
from typing import Optional


def _unsort_outputs(
    x: np.ndarray, axis: Optional[int], unique_values: np.ndarray,
    indices: np.ndarray, inverse_indices: np.ndarray, counts: np.ndarray
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """Unsort the result of np.unique().

    This is done because numpy unique does not retain original order (it sorts
    the output unique values).
    (see: https://github.com/numpy/numpy/issues/8621)

    Code taken from onnx:
    https://github.com/onnx/onnx/blob/main/onnx/backend/test/case/node/unique.py
    """
    argsorted_indices = np.argsort(indices)
    inverse_indices_map = dict(
        zip(argsorted_indices, np.arange(len(argsorted_indices)))
    )
    indices = indices[argsorted_indices]
    unique_values = np.take(x, indices, axis=axis)
    inverse_indices = np.asarray(
        [inverse_indices_map[i] for i in inverse_indices], dtype=np.int32
    )
    counts = counts[argsorted_indices]
    return (unique_values, indices, inverse_indices, counts)


class Unique(RunAll):
    @staticmethod
    def unique_u32():
        def example():
            x = np.array([2, 1, 1, 3, 4, 3]).astype(np.uint32)
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

            name = "unique_u32_example"
            make_test(
                [x],
                (unique_values, indices, inverse_indices, counts),
                "input_0.unique(Option::None(()), Option::None(()))",
                name,
            )

        def example_two():
            x = np.array(
                [[1, 0, 0],
                 [1, 0, 0],
                 [2, 3, 4]]
            ).astype(np.uint32)
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

            name = "unique_u32_example_two"
            make_test(
                [x],
                (unique_values, indices, inverse_indices, counts),
                "input_0.unique(Option::Some(0), Option::None(()))",
                name,
            )

        def without_axis_sorted():
            x = np.random.randint(0, 5, (3, 3, 3)).astype(np.uint32)
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
            x = np.random.randint(0, 5, (3, 3, 3)).astype(np.uint32)
            axis = None

            unique_values, indices, inverse_indices, counts = np.unique(
                x, axis=axis, return_index=True, return_inverse=True, return_counts=True
            )

            unique_values, indices, inverse_indices, counts = _unsort_outputs(
                x, axis, unique_values, indices, inverse_indices, counts
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
            x = np.random.randint(0, 5, (3, 3, 3)).astype(np.uint32)
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
            x = np.random.randint(0, 5, (3, 3, 3)).astype(np.uint32)
            axis = 0

            unique_values, indices, inverse_indices, counts = np.unique(
                x, axis=axis, return_index=True, return_inverse=True, return_counts=True
            )
            unique_values, indices, inverse_indices, counts = _unsort_outputs(
                x, axis, unique_values, indices, inverse_indices, counts
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
            x = np.random.randint(0, 5, (3, 3, 3)).astype(np.uint32)
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
            x = np.random.randint(0, 5, (3, 3, 3)).astype(np.uint32)
            axis = 1

            unique_values, indices, inverse_indices, counts = np.unique(
                x, axis=axis, return_index=True, return_inverse=True, return_counts=True
            )
            unique_values, indices, inverse_indices, counts = _unsort_outputs(
                x, axis, unique_values, indices, inverse_indices, counts
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

        example()
        example_two()
        without_axis_sorted()
        without_axis_not_sorted()
        with_axis_zero_sorted()
        with_axis_zero_not_sorted()
        with_axis_one_sorted()
        with_axis_one_not_sorted()
