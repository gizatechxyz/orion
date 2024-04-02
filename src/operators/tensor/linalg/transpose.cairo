use orion::operators::tensor::core::{
    new_tensor, stride, Tensor, TensorTrait, ravel_index, unravel_index, reshape
};
use orion::operators::tensor::helpers::{len_from_shape, find_axis, permutation_output_shape};
use orion::numbers::NumberTrait;

/// Cf: TensorTrait::transpose docstring
fn transpose<T, impl TTensor: TensorTrait<T>, impl TCopy: Copy<T>, impl TDrop: Drop<T>>(
    self: @Tensor<T>, axes: Span<usize>
) -> Tensor<T> {
    if (*self.shape).len() == 1 {
        return self.identity();
    }

    assert(axes.len() == (*self.shape).len(), 'shape and axes length unequal');

    if (*self.shape).len() == 2 {
        return transpose2D(@(*self));
    }

    let output_shape = permutation_output_shape(*self.shape, axes);
    let output_data_len = len_from_shape(output_shape);

    let mut output_data: Array<T> = array![];

    let mut output_index: usize = 0;
    while output_index != output_data_len {
        let output_indices = unravel_index(output_index, output_shape);
        let mut input_indices: Array<u32> = array![];

        let mut output_axis: usize = 0;
        while output_axis != axes
            .len() {
                let input_axis = find_axis(axes, output_axis);
                input_indices.append(*output_indices[input_axis]);

                output_axis += 1;
            };

        let input_index = ravel_index(*self.shape, input_indices.span());
        output_data.append(*(*self.data)[input_index]);

        output_index += 1;
    };

    TensorTrait::new(output_shape, output_data.span())
}

fn transpose2D<T, impl TTensor: TensorTrait<T>, impl TCopy: Copy<T>, impl TDrop: Drop<T>>(
    self: @Tensor<T>
) -> Tensor<T> {
    assert((*self.shape).len() == 2, 'transpose a 2D tensor');

    let mut output_data: Array<T> = array![];

    let n = *self.shape[0];
    let m = *self.shape[1];

    let mut output_shape: Array<u32> = array![m, n];

    let mut j: usize = 0;
    while j != m {
        let mut i = 0;
        while i != n {
            output_data.append(*(*self.data)[i * m + j]);

            i += 1;
        };

        j += 1;
    };

    TensorTrait::new(output_shape.span(), output_data.span())
}
