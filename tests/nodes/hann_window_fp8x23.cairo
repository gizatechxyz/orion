mod output_0;


use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::FP8x23TensorPartialEq;
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::utils::{assert_eq, assert_seq_eq};
use orion::numbers::{FixedTrait, FP8x23};

#[test]
#[available_gas(2000000000)]
fn test_hann_window_fp8x23() {
    let z_0 = output_0::output_0();

    let y_0 = TensorTrait::hann_window(FP8x23 { mag: 33554432, sign: false }, Option::Some(0));

    assert_eq(y_0, z_0);
}
