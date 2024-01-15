mod input_0;
mod output_0;


use orion::operators::tensor::{TensorTrait, Tensor};
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::{U32Tensor, U32TensorSub};
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::U32TensorPartialEq;

#[test]
#[available_gas(2000000000)]
fn test_sequence_length_u32() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = input_0.sequence_length();

    assert_eq(y, z);
}
