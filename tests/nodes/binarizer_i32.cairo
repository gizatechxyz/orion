mod input_0; 
mod output_0; 


use array::{ArrayTrait, SpanTrait};
use orion::numbers::IntegerTrait;
use orion::operators::tensor::TensorTrait;
use orion::operators::tensor::I32Tensor;
use orion::operators::tensor::U32TensorPartialEq;
use orion::utils::assert_eq;

#[test]
#[available_gas(2000000000)]
fn test_binarizer_i32() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = TensorTrait::binarizer(@input_0, @IntegerTrait::new(1, false));

    assert_eq(y, z);
}