mod input_0;
mod input_1;
mod output_0;


use core::array::{ArrayTrait, SpanTrait};
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::BoolTensor;
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::BoolTensorPartialEq;

#[test]
#[available_gas(2000000000)]
fn test_and_bool() {
    let input_0 = input_0::input_0();
    let input_1 = input_1::input_1();
    let z = output_0::output_0();

    let y = BoolTensor::and(@input_0, @input_1);

    assert_eq(y, z);
}
