mod input_0;
mod output_0;


use orion::operators::tensor::{TensorTrait, Tensor};
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::{I32Tensor, I32TensorSub};
use orion::operators::tensor::{U32Tensor, U32TensorSub};
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::U32TensorPartialEq;
use orion::operators::tensor::I32TensorPartialEq;

#[test]
#[available_gas(2000000000)]
fn test_sequence_length_i32_broadcast() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = input_0.sequence_length();

    assert_eq(y, z);
}
