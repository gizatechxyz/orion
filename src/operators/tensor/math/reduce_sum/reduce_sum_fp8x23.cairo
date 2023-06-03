use array::ArrayTrait;
use array::SpanTrait;

use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
use orion::operators::tensor::implementations::impl_tensor_fp8x23;
use orion::numbers::fixed_point::implementations::impl_8x23;
use orion::numbers::fixed_point::implementations::impl_8x23::fp8x23;
use orion::operators::tensor::core::{Tensor, TensorTrait, ravel_index, unravel_index};
use orion::operators::tensor::helpers::{reduce_output_shape, len_from_shape, combine_indices};
use orion::utils::check_gas;

/// Cf: TensorTrait::reduce_sum docstring
fn reduce_sum(
    self: @Tensor<FixedType<fp8x23>>, axis: usize, keepdims: bool
) -> Tensor<FixedType<fp8x23>> {
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
        return TensorTrait::<FixedType>::new(output_shape, output_data.span());
    } else {
        return TensorTrait::<FixedType>::new(output_shape, output_data.span());
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
/// * An FixedType value representing the accumulated sum along the specified axis.
fn accumulate_sum(
    input: @Tensor<FixedType<fp8x23>>, output_indices: Span<usize>, axis: usize
) -> FixedType<fp8x23> {
    let axis_len = *(*input.shape).at(axis);
    let mut acc = FixedTrait::new_unscaled(0, false);

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
