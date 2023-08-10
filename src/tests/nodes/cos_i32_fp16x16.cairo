mod input_0; 
mod output_0; 


use array::ArrayTrait;
use orion::operators::tensor::core::TensorTrait;
use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
use orion::operators::tensor::implementations::impl_tensor_fp::FP16x16Tensor::FPTensorPartialEq;
use orion::utils::assert_eq;

#[test]
#[available_gas(2000000000)]
fn test_cos_i32_fp16x16() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = input_0.cos();

    assert_eq(y, z);
}