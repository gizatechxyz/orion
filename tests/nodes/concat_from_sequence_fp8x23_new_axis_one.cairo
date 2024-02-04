mod input_0;
mod output_0;


use orion::operators::sequence::FP8x23Sequence;
use orion::operators::tensor::FP8x23TensorPartialEq;
use orion::operators::sequence::SequenceTrait;
use core::array::{ArrayTrait, SpanTrait};
use orion::utils::{assert_eq, assert_seq_eq};

#[test]
#[available_gas(2000000000)]
fn test_concat_from_sequence_fp8x23_new_axis_one() {
    let input_0 = input_0::input_0();
    let z_0 = output_0::output_0();

    let y_0 = SequenceTrait::concat_from_sequence(input_0, 1_i32, Option::Some(1));

    assert_eq(y_0, z_0);
}
