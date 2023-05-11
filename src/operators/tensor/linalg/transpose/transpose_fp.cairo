use array::ArrayTrait;
use array::SpanTrait;

use onnx_cairo::numbers::fixed_point::types::FixedType;
use onnx_cairo::operators::tensor::core::{
    new_tensor, stride, Tensor, TensorTrait, ravel_index, unravel_index, reshape
};
use onnx_cairo::operators::tensor::helpers::{len_from_shape, find_axis, permutation_output_shape};
use onnx_cairo::operators::tensor::implementations::impl_tensor_fp;
use onnx_cairo::utils::check_gas;


/// Reorders the axes of an FixedType tensor according to the given axes permutation.
///
/// # Arguments
/// * `self` - The input tensor.
/// * `axes` -  A span containing the usize elements representing the axes permutation.
///
/// # Panics
/// * Panics if the length of the axes array is not equal to the rank of the input tensor.
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * A `Tensor<FixedType>` instance with the axes reordered according to the given permutation.
fn transpose(self: @Tensor<FixedType>, axes: Span<usize>) -> Tensor<FixedType> {
    assert(axes.len() == (*self.shape).len(), 'shape and axes length unequal');

    let output_shape = permutation_output_shape(*self.shape, axes);
    let output_data_len = len_from_shape(output_shape);

    let mut output_data = ArrayTrait::new();

    let mut output_index: usize = 0;
    loop {
        check_gas();

        if output_index == output_data_len {
            break ();
        }

        let output_indices = unravel_index(output_index, output_shape);
        let mut input_indices = ArrayTrait::new();

        let mut output_axis: usize = 0;
        loop {
            check_gas();
            if output_axis == axes.len() {
                break ();
            }

            let input_axis = find_axis(axes, output_axis);
            input_indices.append(*output_indices.at(input_axis));
            output_axis += 1;
        };

        let input_index = ravel_index(*self.shape, input_indices.span());
        output_data.append(*(*self.data).at(input_index));

        output_index += 1;
    };

    return TensorTrait::<FixedType>::new(output_shape, output_data.span());
}
