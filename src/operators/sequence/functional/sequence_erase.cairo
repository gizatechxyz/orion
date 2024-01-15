use core::array::{ArrayTrait, SpanTrait};
use core::option::OptionTrait;

use orion::operators::tensor::core::{Tensor, TensorTrait, u32Toi32, i32Tou32};
use orion::operators::tensor::I32Tensor;
use orion::numbers::NumberTrait;

/// Cf: SequenceTrait::sequence_erase docstring
fn sequence_erase<T, impl TTensor: TensorTrait<T>, impl TCopy: Copy<T>, impl TDrop: Drop<T>>(
    sequence: Array<Tensor<T>>, position: Option<Tensor<i32>>
) -> Array<Tensor<T>> {
    let position: Tensor<i32> = match position {
        Option::Some(p) => p,
        Option::None(_) => {
            let mut shape = ArrayTrait::<usize>::new();
            let mut data = ArrayTrait::<i32>::new();
            data.append(-1_i32);
            TensorTrait::<i32>::new(shape.span(), data.span())
        }
    };

    assert(position.shape.len() == 0 && position.data.len() == 1, 'Position must be a scalar');

    let position_value_i32: i32 = *position.data.at(0);
    let is_negative: bool = position_value_i32 < 0;
    let mut position_value: u32 = i32Tou32(position_value_i32);

    assert(
        (is_negative == false && position_value <= sequence.len() - 1)
            || (is_negative == true && position_value <= sequence.len()),
        'Position out of bounds'
    );

    if is_negative == true {
        position_value = sequence.len() - position_value;
    }

    let mut input_sequence_copy = sequence;
    let mut output_sequence = ArrayTrait::<Tensor<T>>::new();
    let mut tensor_counter: usize = 0;
    loop {
        match input_sequence_copy.pop_front() {
            Option::Some(input_sequence_value) => {
                if tensor_counter == position_value {
                    tensor_counter += 1;
                    continue;
                }
                output_sequence.append(input_sequence_value);

                tensor_counter += 1;
            },
            Option::None(_) => { break; }
        };
    };

    return output_sequence;
}


