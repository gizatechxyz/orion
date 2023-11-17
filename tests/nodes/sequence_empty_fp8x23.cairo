mod output_0;


use orion::operators::tensor::FP8x23TensorPartialEq;
use orion::operators::sequence::FP8x23Sequence;
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::sequence::SequenceTrait;
use array::{ArrayTrait, SpanTrait};

#[test]
#[available_gas(2000000000)]
fn test_sequence_empty_fp8x23() {
    let z = output_0::output_0();

    let y = SequenceTrait::sequence_empty();

    assert_seq_eq(y, z);
}
