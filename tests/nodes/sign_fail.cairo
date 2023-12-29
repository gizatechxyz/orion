mod input_0;
mod output_0;


use core::array::{ArrayTrait, SpanTrait};
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::{I32Tensor, I32TensorSub};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I32TensorPartialEq;

#[test]
#[available_gas(2000000000)]
fn test_sign_fail() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = input_0.sign();

    assert_eq(y, z);
}
