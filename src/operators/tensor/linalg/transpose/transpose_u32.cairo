use array::ArrayTrait;
use array::SpanTrait;

use orion::operators::tensor::core::{
    new_tensor, stride, Tensor, TensorTrait, ravel_index, unravel_index, reshape
};
use orion::operators::tensor::helpers::{len_from_shape, find_axis, permutation_output_shape};
use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;


/// Cf: TensorTrait::transpose docstring
fn transpose(self: @Tensor<u32>, axes: Span<usize>) -> Tensor<u32> {
    assert((*self.shape).len() > 1, 'cannot transpose a 1D tensor');
    assert(axes.len() == (*self.shape).len(), 'shape and axes length unequal');

    let output_shape = permutation_output_shape(*self.shape, axes);
    let output_data_len = len_from_shape(output_shape);

    let mut output_data = ArrayTrait::new();

    let mut output_index: usize = 0;
    loop {
        

        if output_index == output_data_len {
            break ();
        }

        let output_indices = unravel_index(output_index, output_shape);
        let mut input_indices = ArrayTrait::new();

        let mut output_axis: usize = 0;
        loop {
            
            if output_axis == axes.len() {
                break ();
            }

            let input_axis = find_axis(axes, output_axis);
            input_indices.append(*output_indices[input_axis]);
            output_axis += 1;
        };

        let input_index = ravel_index(*self.shape, input_indices.span());
        output_data.append(*(*self.data)[input_index]);

        output_index += 1;
    };

    return TensorTrait::new(output_shape, output_data.span(), *self.extra);
}
