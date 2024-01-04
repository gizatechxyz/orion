mod input_0;
mod output_0;


use orion::operators::tensor::BoolTensor;
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::I8Tensor;
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::BoolTensorPartialEq;
use orion::operators::tensor::I8TensorPartialEq;

#[test]
#[available_gas(2000000000)]
fn test_is_pos_inf_i8() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = TensorTrait::is_inf(@input_0, Option::Some(0), Option::Some(1));

    assert_eq(y, z);
}
