use array::ArrayTrait;
use array::SpanTrait;

use orion::operators::tensor::core::{Tensor, TensorTrait};


/// Cf: TensorTrait::flatten docstring
fn flatten<T, F, impl TTensorTrait: TensorTrait<T, F>>(self: @Tensor<T>, axis: usize) -> Tensor<T> {
    let current_shape = *self.shape;
    assert(axis < current_shape.len(), 'axis out of dimensions');

    let mut new_shape_first_axis = 1;
    let mut index = 0;
    loop {
        if index == axis {
            break;
        }

        new_shape_first_axis *= *current_shape[index];

        index += 1;
    };

    let new_shape_second_axis = (*self.data).len() / new_shape_first_axis;

    let mut new_shape = ArrayTrait::<usize>::new();
    new_shape.append(new_shape_first_axis);
    new_shape.append(new_shape_second_axis);

    return self.reshape(new_shape.span());
}
