mod input_0;
mod output_0;


use orion::operators::tensor::I8TensorPartialEq;
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::sequence::I8Sequence;
use orion::operators::sequence::SequenceTrait;
use core::array::{ArrayTrait, SpanTrait};

#[test]
#[available_gas(2000000000)]
fn test_sequence_construct_i8() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = SequenceTrait::sequence_construct(input_0);

    assert_seq_eq(y, z);
}
