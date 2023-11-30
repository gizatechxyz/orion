mod output_0;


use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::I8Tensor;
use orion::operators::tensor::I8TensorPartialEq;

#[test]
#[available_gas(2000000000)]
fn test_sequence_empty_i8() {
    let z = output_0::output_0();

    let y = TensorTrait::sequence_empty();

    assert_seq_eq(y, z);
}
