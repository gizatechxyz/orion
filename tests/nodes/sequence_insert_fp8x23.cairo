mod input_0;
mod input_1;
mod input_2;
mod output_0;


use orion::operators::sequence::SequenceTrait;
use orion::operators::tensor::I32TensorPartialEq;
use orion::utils::{assert_eq, assert_seq_eq};
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::sequence::I32Sequence;
use orion::operators::tensor::FP8x23TensorPartialEq;
use orion::operators::sequence::FP8x23Sequence;

#[test]
#[available_gas(2000000000)]
fn test_sequence_insert_fp8x23() {
    let input_0 = input_0::input_0();
    let input_1 = input_1::input_1();
    let input_2 = input_2::input_2();
    let z = output_0::output_0();

    let y = input_0.sequence_insert(@input_1,Option::Some(input_2));

    assert_seq_eq(y, z);
}
