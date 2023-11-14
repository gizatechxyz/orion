mod input_0; 
mod output_0; 


use array::{ArrayTrait, SpanTrait};
use orion::numbers::FixedTrait;
use orion::operators::tensor::TensorTrait;
use orion::operators::tensor::FP16x16Tensor;
use orion::operators::tensor::FP16x16TensorPartialEq;
use orion::utils::assert_eq;

#[test]
#[available_gas(2000000000)]
fn test_binarizer_fp16x16() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = TensorTrait::binarizer(@input_0, Option::Some(FixedTrait::new(65536, false)));

    assert_eq(y, z);
}