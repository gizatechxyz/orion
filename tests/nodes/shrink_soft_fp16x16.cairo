mod input_0;
mod output_0;


use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::utils::{assert_eq, assert_seq_eq};
use orion::numbers::{FixedTrait, FP16x16};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::operators::tensor::FP16x16TensorPartialEq;

#[test]
#[available_gas(2000000000)]
fn test_shrink_soft_fp16x16() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = TensorTrait::shrink(input_0, Option::Some(FixedTrait::new(65536, false)), Option::Some(FixedTrait::new(65536, false)));

    assert_eq(y, z);
}
