mod input_0;
mod input_1;
mod output_0;


use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I8Tensor;
use orion::operators::tensor::I8TensorPartialEq;
use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::BoolTensorPartialEq;
use orion::operators::tensor::BoolTensor;

#[test]
#[available_gas(2000000000)]
fn test_and_i8_broadcast() {
    let input_0 = input_0::input_0();
    let input_1 = input_1::input_1();
    let z = output_0::output_0();

    let y = input_0.and(@input_1);

    assert_eq(y, z);
}
