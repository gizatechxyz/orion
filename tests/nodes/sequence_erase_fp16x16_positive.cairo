mod input_0;
mod input_1;
mod output_0;


use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::FP16x16TensorPartialEq;
use orion::operators::sequence::I32Sequence;
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::sequence::SequenceTrait;
use orion::operators::sequence::FP16x16Sequence;
use orion::operators::tensor::I32TensorPartialEq;

#[test]
#[available_gas(2000000000)]
fn test_sequence_erase_fp16x16_positive() {
    let input_0 = input_0::input_0();
    let input_1 = input_1::input_1();
    let z = output_0::output_0();

    let y = SequenceTrait::sequence_erase(input_0, Option::Some(input_1));

    assert_seq_eq(y, z);
}
