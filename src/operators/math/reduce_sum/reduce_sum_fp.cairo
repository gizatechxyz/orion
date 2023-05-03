use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use onnx_cairo::operators::math::fixed_point::types::Fixed;
use onnx_cairo::operators::math::fixed_point::types::FixedType;
use onnx_cairo::operators::tensor::tensor_fp;
use onnx_cairo::operators::tensor::core::Tensor;
use onnx_cairo::operators::tensor::core::TensorTrait;
use onnx_cairo::operators::tensor::core::ravel_index;
use onnx_cairo::operators::tensor::core::unravel_index;
use onnx_cairo::operators::tensor::helpers::reduce_output_shape;
use onnx_cairo::operators::tensor::helpers::len_from_shape;
use onnx_cairo::operators::tensor::helpers::combine_indices;
use onnx_cairo::utils::check_gas;

/// Sums the elements along the given axis of an FixedType tensor.
///
/// # Arguments
/// * `self` - The input tensor.
/// * `axis` - The axis along which to sum the elements.
///
/// # Panics
/// * Panics if axis is not in the range of the input tensor's dimensions.
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * A `Tensor<FixedType>` instance representing the result of the reduction.
fn reduce_sum(self: @Tensor<FixedType>, axis: usize, keepdims: bool) -> Tensor<FixedType> {
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
use debug::print_felt252;
use traits::Into;
fn accumulate_sum(
    input: @Tensor<FixedType>, output_indices: Span<usize>, axis: usize
) -> FixedType {
    let axis_len = *(*input.shape).at(axis);
    let mut acc = Fixed::new_unscaled(0, false);

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
