mod input_0;
mod output_0;


use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23TensorPartialEq;
use orion::operators::tensor::FP8x23Tensor;
use orion::operators::tensor::U32TensorPartialEq;
use orion::utils::{assert_eq, assert_seq_eq};
use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::U32Tensor;

#[test]
#[available_gas(2000000000)]
fn test_sequence_length_fp8x23_broadcast() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = input_0.sequence_length();

    assert_eq(y, z);
}
