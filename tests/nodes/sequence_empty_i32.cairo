mod output_0;


use orion::operators::tensor::I32TensorPartialEq;
use orion::operators::sequence::I32Sequence;
use orion::operators::sequence::SequenceTrait;
use core::array::{ArrayTrait, SpanTrait};
use orion::utils::{assert_eq, assert_seq_eq};

#[test]
#[available_gas(2000000000)]
fn test_sequence_empty_i32() {
    let z = output_0::output_0();

    let y = SequenceTrait::sequence_empty();

    assert_seq_eq(y, z);
}
