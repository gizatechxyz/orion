use core::clone::Clone;
use core::array::{ArrayTrait, SpanTrait};
use core::option::OptionTrait;
use core::debug::PrintTrait;
use core::traits::Into;

use orion::operators::tensor::helpers::replace_index;
use orion::operators::tensor::core::{Tensor, TensorTrait, u32Toi32, i32Tou32};
use orion::operators::tensor::math::concat::concat;


fn concat_from_sequence<
    T, impl TTensorTrait: TensorTrait<T>, impl TCopy: Copy<T>, impl TDrop: Drop<T>,
>(
    sequence: Array<Tensor<T>>, axis: i32, new_axis: Option<usize>
) -> Tensor<T> {
    let new_axis: usize = match new_axis {
        Option::Some(val) => {
            assert(val == 0 || val == 1, 'new_axis must be 0 or 1');
            val
        },
        Option::None(_) => 0
    };

    let first_tensor = *sequence.at(0);
    let r = first_tensor.shape.len();

    if new_axis == 0 {
        concat_without_new_axis(sequence, axis, r)
    } else {
        concat_with_new_axis(sequence, axis, r)
    }
}


fn concat_without_new_axis<
    T, impl TTensorTrait: TensorTrait<T>, impl TCopy: Copy<T>, impl TDrop: Drop<T>,
>(
    sequence: Array<Tensor<T>>, axis: i32, r: usize
) -> Tensor<T> {
    let axis_is_negative: bool = axis < 0;
    let mut axis_value: u32 = i32Tou32(axis);

    /// assert in range [-r, r - 1]
    assert(
        (axis_is_negative == false && axis_value <= r - 1)
            || (axis_is_negative == true && axis_value <= r),
        'Out of bounds for dimension'
    );

    if axis_is_negative == true {
        axis_value = r - axis_value
    }
    concat(sequence.span(), axis_value)
}


fn concat_with_new_axis<
    T, impl TTensorTrait: TensorTrait<T>, impl TCopy: Copy<T>, impl TDrop: Drop<T>,
>(
    sequence: Array<Tensor<T>>, axis: i32, r: usize
) -> Tensor<T> {
    let axis_is_negative: bool = axis < 0;
    let mut axis_value: u32 = i32Tou32(axis);

    /// assert in range [-r - 1, r]
    assert(
        (axis_is_negative == false && axis_value <= r)
            || (axis_is_negative == true && axis_value <= r + 1),
        'Out of bounds for dimension'
    );

    if axis_is_negative == true {
        if axis_value > r {
            axis_value = 0
        } else {
            axis_value = r - axis_value
        }
    }
    let mut input_sequence_copy = sequence;
    let mut reshaped_sequence = ArrayTrait::<Tensor<T>>::new();
    loop {
        match input_sequence_copy.pop_front() {
            Option::Some(input_sequence_value) => {
                let mut reshaped_tensor = add_new_dimension(input_sequence_value, axis_value);
                reshaped_sequence.append(reshaped_tensor);
            },
            Option::None(_) => { break; }
        };
    };
    concat(reshaped_sequence.span(), axis_value)
}


fn add_new_dimension<
    T, impl TTensorTrait: TensorTrait<T>, impl TCopy: Copy<T>, impl TDrop: Drop<T>,
>(
    mut tensor: Tensor<T>, axis: usize
) -> Tensor<T> {
    let mut tensor_shape = tensor.shape;
    let mut new_tensor_shape = ArrayTrait::<usize>::new();
    let mut tensor_shape_counter: usize = 0;
    loop {
        match tensor_shape.pop_front() {
            Option::Some(tensor_shape_value) => {
                if tensor_shape_counter == axis {
                    new_tensor_shape.append(1);
                }
                new_tensor_shape.append(*tensor_shape_value);
                tensor_shape_counter += 1;
            },
            Option::None(_) => { break; }
        };
    };
    if axis >= tensor.shape.len() {
        new_tensor_shape.append(1);
    }
    TensorTrait::<T>::new(new_tensor_shape.span(), tensor.data)
}
