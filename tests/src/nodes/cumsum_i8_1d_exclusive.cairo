mod input_0; 
mod output_0; 


use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::TensorTrait;
use orion::operators::tensor::I8Tensor;
use orion::operators::tensor::I8TensorPartialEq;
use orion::utils::assert_eq;

#[test]
#[available_gas(2000000000)]
fn test_cumsum_i8_1d_exclusive() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = input_0.cumsum(0, Option::Some(true), Option::Some(false));

    assert_eq(y, z);
}