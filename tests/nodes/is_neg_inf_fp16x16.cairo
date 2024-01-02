mod input_0;
mod output_0;


use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::operators::tensor::BoolTensorPartialEq;
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16TensorPartialEq;
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::BoolTensor;

#[test]
#[available_gas(2000000000)]
fn test_is_neg_inf_fp16x16() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = TensorTrait::is_inf(@input_0, Option::Some(1), Option::Some(0));

    assert_eq(y, z);
}
