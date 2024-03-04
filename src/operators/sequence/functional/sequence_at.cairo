use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::numbers::{NumberTrait, I32IntoU32, U32IntoI32};

/// Cf: SequenceTrait::sequence_at docstring
fn sequence_at<T, impl TTensor: TensorTrait<T>, impl TCopy: Copy<T>, impl TDrop: Drop<T>>(
    sequence: Array<Tensor<T>>, position: Tensor<i32>
) -> Tensor<T> {
    assert(
        position.shape.len() == 0 && position.data.len().into() == 1, 'Position must be a scalar'
    );

    let position_value_i32: i32 = *position.data.at(0);
    let is_negative: bool = position_value_i32 < 0;
    let position_value: u32 = position_value_i32.into();

    assert(
        (!is_negative && position_value <= sequence.len() - 1)
            || (is_negative && position_value <= sequence.len()),
        'Position out of bounds'
    );

    if !is_negative {
        *sequence.at(position_value)
    } else {
        let normalized_position_value = sequence.len() - position_value;

        *sequence.at(normalized_position_value)
    }
}
