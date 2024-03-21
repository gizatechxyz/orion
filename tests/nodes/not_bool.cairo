mod input_0;
mod output_0;


use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::BoolTensor;
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::BoolTensorPartialEq;
use orion::utils::{assert_eq, assert_seq_eq};

#[test]
#[available_gas(2000000000)]
fn test_not_bool() {
    let input_0 = input_0::input_0();
    let z_0 = output_0::output_0();

    let y_0 = input_0.not();

    assert_eq(y_0, z_0);
}
