mod input_0;
mod output_0;


use orion::operators::tensor::FP8x23TensorPartialEq;
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

#[test]
#[available_gas(2000000000)]
fn test_random_uniform_like_fp8x23() {
    let input_0 = input_0::input_0();
    let z_0 = output_0::output_0();

    let y_0 = TensorTrait::random_uniform_like(@input_0, Option::Some(FP8x23 { mag: 83886080, sign: false }),Option::Some(FP8x23 { mag: 8388608, sign: false }), Option::Some(354145));

    assert_eq(y_0, z_0);
}
