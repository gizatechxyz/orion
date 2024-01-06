use core::array::{ArrayTrait, SpanTrait};
use core::option::OptionTrait;

use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::I32Tensor;
use orion::numbers::NumberTrait;
use orion::numbers::signed_integer::i32::i32;

/// Cf: SequenceTrait::sequence_insert docstring
fn sequence_insert<T, impl TTensor: TensorTrait<T>, impl TCopy: Copy<T>, impl TDrop: Drop<T>>(
    self: Array<Tensor<T>>, tensor: @Tensor<T>, position: Option<Tensor<i32>>
) -> Array<Tensor<T>> {
    let position: Tensor<i32> = match position {
        Option::Some(p) => p,
        Option::None(_) => {
            let mut shape = ArrayTrait::<usize>::new();
            let mut data = ArrayTrait::<i32>::new();
            data.append(i32 { mag: 1, sign: true });
            TensorTrait::<i32>::new(shape.span(), data.span())
        },
    };

    assert(position.shape.len() == 0 && position.data.len() == 1, 'Position must be a scalar');

    let position_value_i32: i32 = *position.data.at(0);
    let is_negative: bool = position_value_i32.sign;
    let mut position_value: u32 = position_value_i32.mag;

    assert(
        (is_negative == false && position_value <= self.len() - 1)
            || (is_negative == true && position_value <= self.len()),
        'Position out of bounds'
    );

    if is_negative == true {
        position_value = self.len() - position_value;
    }

    let mut new_sequence = ArrayTrait::<Tensor<T>>::new();
    let mut inserted = false;
    let mut self_copy = self;
    loop {
        match self_copy.pop_front() {
            Option::Some(t) => {
                if position_value == 0 && inserted == false {
                    new_sequence.append(*tensor);
                    inserted = true;
                }
                new_sequence.append(t);
                if inserted == false {
                    position_value -= 1;
                }
            },
            Option::None(_) => { break; },
        };
    };

    return new_sequence;
}
