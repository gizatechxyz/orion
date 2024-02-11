mod input_0;
mod output_0;


use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::FP16x16TensorPartialEq;
use orion::numbers::{FixedTrait, FP16x16};

#[test]
#[available_gas(2000000000)]
fn test_random_uniform_like_fp16x16() {
    let input_0 = input_0::input_0();
    let z_0 = output_0::output_0();

    let y_0 = TensorTrait::random_uniform_like(@input_0, Option::Some(FP16x16 { mag: 655360, sign: false }),Option::Some(FP16x16 { mag: 65536, sign: false }), Option::Some(354145));

    assert_eq(y_0, z_0);
}
