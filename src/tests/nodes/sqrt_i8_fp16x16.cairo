mod input_0; 
mod output_0; 


use array::ArrayTrait;
use orion::operators::tensor::core::TensorTrait;
use orion::operators::tensor::implementations::impl_tensor_i8::Tensor_i8;
use orion::operators::tensor::implementations::impl_tensor_fp::FP16x16Tensor::FPTensorPartialEq;
use orion::utils::assert_eq;

#[test]
#[available_gas(2000000000)]
fn test_sqrt_i8_fp16x16() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = input_0.sqrt();

    assert_eq(y, z);
}