use array::{ArrayTrait, SpanTrait};
use option::OptionTrait;

use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::I32Tensor;
use orion::numbers::NumberTrait;
use orion::numbers::signed_integer::i32::i32;

/// Cf: TensorTrait::sequence_erase docstring
fn sequence_erase<
    T, impl TTensor: TensorTrait<T>, impl TCopy: Copy<T>, impl TDrop: Drop<T>
>(
    sequence: Array<Tensor<T>>, position: Option<Tensor<i32>>
) -> Array<Tensor<T>> {

    let position: Tensor<i32> = if position.is_some() {
        position.unwrap()
    } else {
        let mut shape = ArrayTrait::<usize>::new();
        let mut data = ArrayTrait::<i32>::new();
        data.append(i32 { mag: 1, sign: true });
        TensorTrait::<i32>::new(shape.span(), data.span())
    };

    assert(position.shape.len() == 0 && position.data.len() == 1, 'Position must be a scalar');

    let position_value_i32: i32 = *position.data.at(0);
    let is_negative: bool = position_value_i32.sign;
    let mut position_value: u32 = position_value_i32.mag;

    assert((is_negative == false && position_value <= sequence.len() - 1) || (is_negative == true && position_value <= sequence.len()), 'Position out of bounds');

    if is_negative == true {
        position_value = sequence.len() - position_value;
    }

    let mut output_sequence = ArrayTrait::new();

    let mut tensor_counter: usize = 0;
    loop {
        if tensor_counter > sequence.len() - 1 {
            break;
        }

        if tensor_counter == position_value {
            continue;
        }

        output_sequence.append(*sequence.at(tensor_counter));

        tensor_counter += 1;
    };

    return output_sequence;
}