mod input_0;
mod output_0;


use core::array::{ArrayTrait, SpanTrait};
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorSub};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23TensorPartialEq;

#[test]
#[available_gas(2000000000)]
fn test_sqrt_fp8x23() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = input_0.sqrt();

    assert_eq(y, z);
}
