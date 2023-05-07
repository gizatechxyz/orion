use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use onnx_cairo::operators::tensor::implementations::impl_tensor_u32;
use onnx_cairo::operators::tensor::core::{Tensor, TensorTrait, ravel_index, unravel_index};
use onnx_cairo::operators::tensor::helpers::{reduce_output_shape, len_from_shape, combine_indices};
use onnx_cairo::utils::check_gas;

/// Sums the elements along the given axis of an u32 tensor.
///
/// # Arguments
/// * `self` - The input tensor.
/// * `axis` - The axis along which to sum the elements.
/// * `keepdims` - If true, retains reduced dimensions with length 1.
///
/// # Panics
/// * Panics if axis is not in the range of the input tensor's dimensions.
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * A `Tensor<u32>` instance representing the result of the reduction.
fn reduce_sum(self: @Tensor<u32>, axis: usize, keepdims: bool) -> Tensor<u32> {
    assert(axis <= (*self.shape).len(), 'axis out of dimensions');
    let mut output_data = ArrayTrait::new();

    let output_shape = reduce_output_shape(*self.shape, axis, false);
    let output_data_len = len_from_shape(output_shape);

    let mut index: usize = 0;
    loop {
        check_gas();

        let output_indices = unravel_index(index, output_shape);
        let current_sum = accumulate_sum(self, output_indices, axis);

        output_data.append(current_sum);

        index += 1;
        if index == output_data_len {
            break ();
        };
    };

    if keepdims {
        let output_shape = reduce_output_shape(*self.shape, axis, true);
        return TensorTrait::<u32>::new(output_shape, output_data.span());
    } else {
        return TensorTrait::<u32>::new(output_shape, output_data.span());
    }
}


/// Helper function that accumulates the sum of elements along a specific axis.
///
/// # Arguments
/// * `input` - The input tensor.
/// * `output_indices` - A span of output indices.
/// * `axis` - The axis along which to accumulate the sum.
///
/// # Panics
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * An u32 value representing the accumulated sum along the specified axis.
use debug::print_felt252;
use traits::Into;
fn accumulate_sum(input: @Tensor<u32>, output_indices: Span<usize>, axis: usize) -> u32 {
    let axis_len = *(*input.shape).at(axis);
    let mut acc = 0;

    let mut axis_index: usize = 0;
    loop {
        check_gas();

        if axis_index == axis_len {
            break ();
        }

        let input_indices = combine_indices(output_indices, axis_index, axis);
        let input_index = ravel_index(*input.shape, input_indices);
        let ele = *(*input.data).at(input_index);
        acc += ele;
        axis_index += 1;
    };

    return acc;
}
