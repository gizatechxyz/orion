mod input_0;
mod output_0;


use orion::operators::tensor::{I32Tensor, I32TensorAdd};
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::I32TensorPartialEq;
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::operators::tensor::FP16x16TensorPartialEq;

#[test]
#[available_gas(2000000000)]
fn test_argmax_negative_axis_keepdims() {
    let input_0 = input_0::input_0();
    let z_0 = output_0::output_0();

    let y_0 = input_0.argmax(-1, Option::Some(true), Option::None(()));

    assert_eq(y_0, z_0);
}
