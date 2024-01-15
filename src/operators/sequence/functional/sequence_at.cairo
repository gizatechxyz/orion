use core::array::{ArrayTrait, SpanTrait};
use core::option::OptionTrait;

use orion::operators::tensor::core::{Tensor, TensorTrait, u32Toi32, i32Tou32};
use orion::numbers::NumberTrait;

/// Cf: SequenceTrait::sequence_at docstring
fn sequence_at<T, impl TTensor: TensorTrait<T>, impl TCopy: Copy<T>, impl TDrop: Drop<T>>(
    sequence: Array<Tensor<T>>, position: Tensor<i32>
) -> Tensor<T> {
    assert(position.shape.len() == 0 && u32Toi32(position.data.len()) == 1, 'Position must be a scalar');

    let position_value_i32: i32 = *position.data.at(0);
    let is_negative: bool = position_value_i32 < 0;
    let position_value: u32 = i32Tou32(position_value_i32);

    assert(
        (is_negative == false && position_value <= sequence.len() - 1)
            || (is_negative == true && position_value <= sequence.len()),
        'Position out of bounds'
    );

    if is_negative == false {
        return *sequence.at(position_value);
    } else {
        let normalized_position_value = sequence.len() - position_value;
        return *sequence.at(normalized_position_value);
    }
}
