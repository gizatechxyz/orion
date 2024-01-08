mod input_0;
mod output_0;


use orion::operators::tensor::I32Tensor;
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I32TensorPartialEq;
use core::array::{ArrayTrait, SpanTrait};
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::sequence::I32Sequence;
use orion::operators::sequence::SequenceTrait;

#[test]
#[available_gas(2000000000)]
fn test_sequence_erase_i32_empty() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = SequenceTrait::sequence_erase(input_0, Option::None(()));

    assert_seq_eq(y, z);
}
