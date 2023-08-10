mod input_0; 
mod output_0; 


use array::ArrayTrait;
use orion::operators::tensor::core::TensorTrait;
use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
use orion::operators::tensor::implementations::impl_tensor_fp::FP16x16Tensor::FPTensorPartialEq;
use orion::utils::assert_eq;

#[test]
#[available_gas(2000000000)]
fn test_reduce_sum_fp16x16_2D_keepdims() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = input_0.reduce_sum(0, true);

    assert_eq(y, z);
}