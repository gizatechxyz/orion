use array::ArrayTrait;
use array::SpanTrait;

use onnx_cairo::numbers::fixed_point::types::{Fixed, FixedType, MAX_u128};
use onnx_cairo::operators::tensor::implementations::impl_tensor_u32;
use onnx_cairo::operators::tensor::core::{Tensor, TensorTrait, ravel_index, unravel_index};
use onnx_cairo::operators::tensor::helpers::{reduce_output_shape, len_from_shape, combine_indices};
use onnx_cairo::utils::check_gas;

/// Returns the indices of the maximum values along the given axis of an FixedType tensor.
///
/// # Arguments
/// * `self` - The input tensor.
/// * `axis` - The axis along which to find the maximum values.
///
/// # Panics
/// * Panics if axis is not in the range of the input tensor's dimensions.
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * A `Tensor<usize>` instance representing the indices of the maximum values along the given axis.
fn argmax(self: @Tensor<FixedType>, axis: usize) -> Tensor<usize> {
    assert(axis <= (*self.shape).len(), 'axis out of dimensions');

    let mut output_data = ArrayTrait::new();

    let output_shape = reduce_output_shape(*self.shape, axis, false);
    let output_data_len = len_from_shape(output_shape);

    let mut index: usize = 0;
    loop {
        check_gas();

        let output_indices = unravel_index(index, output_shape);
        let current_argmax = find_argmax(
            self, output_indices, axis, 0, Fixed::new(MAX_u128, true), 0
        );

        output_data.append(current_argmax);

        index += 1;
        if index == output_data_len {
            break ();
        };
    };

    return TensorTrait::<usize>::new(output_shape, output_data.span());
}

/// Recursive helper function that finds the index of the maximum value along a specific axis.
///
/// # Arguments
/// * `input` - The input tensor.
/// * `output_indices` - A span of output indices.
/// * `axis` - The axis along which to find the maximum value.
/// * `axis_index` - The current index along the specified axis.
/// * `max_value` - The current maximum value found along the axis.
/// * `argmax` - The current index of the maximum value along the axis.
///
/// # Panics
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * A usize value representing the index of the maximum value along the specified axis.
fn find_argmax(
    input: @Tensor<FixedType>,
    output_indices: Span<usize>,
    axis: usize,
    axis_index: usize,
    max_value: FixedType,
    argmax: usize
) -> usize {
    check_gas();

    if axis_index == *(*input.shape).at(axis) {
        return argmax;
    }

    let input_indices = combine_indices(output_indices, axis_index, axis);
    let input_index = ravel_index(*input.shape, input_indices);
    let ele = *(*input.data).at(input_index);

    let (new_max_value, new_argmax) = if ele > max_value {
        (ele, axis_index)
    } else {
        (max_value, argmax)
    };

    return find_argmax(
        input, output_indices, axis, axis_index + 1_usize, new_max_value, new_argmax
    );
}
