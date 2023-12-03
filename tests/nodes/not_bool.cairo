mod input_0; 
mod output_0; 


use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::TensorTrait;
use orion::operators::tensor::BoolTensor;
use orion::operators::tensor::BoolTensorPartialEq;
use orion::utils::assert_eq;

#[test]
#[available_gas(2000000000)]
fn test_not_bool() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = input_0.not();

    assert_eq(y, z);
}