use core::array::ArrayTrait;
use core::array::SpanTrait;

use orion::operators::tensor::core::{Tensor, TensorTrait};


/// Cf: TensorTrait::flatten docstring
fn flatten<T, impl TTensorTrait: TensorTrait<T>>(self: @Tensor<T>, axis: usize) -> Tensor<T> {
    let mut shape = *self.shape;
    assert(axis < shape.len(), 'axis out of dimensions');

    let mut new_shape_first_axis = 1;
    let mut index = 0;
    loop {
        match shape.pop_front() {
            Option::Some(val) => {
                if index == axis {
                    break;
                }

                new_shape_first_axis *= *val;
                index += 1;
            },
            Option::None(_) => { break; }
        };
    };

    let new_shape_second_axis = (*self.data).len() / new_shape_first_axis;

    return self.reshape(array![new_shape_first_axis, new_shape_second_axis].span());
}
