mod input_0;
mod output_0;


use orion::operators::tensor::I8TensorPartialEq;
use orion::operators::tensor::{TensorTrait, Tensor};
use core::array::{ArrayTrait, SpanTrait};
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::I8Tensor;

#[test]
#[available_gas(2000000000)]
fn test_sequence_construct_i8() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = TensorTrait::sequence_construct(input_0);

    assert_seq_eq(y, z);
}
