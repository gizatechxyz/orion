mod input_0; 
mod output_0; 


use array::ArrayTrait;
use orion::operators::tensor::core::TensorTrait;
use orion::operators::tensor::implementations::tensor_fp16x16::{
    FP16x16Tensor, FP16x16TensorPartialEq
};
use orion::utils::assert_eq;

#[test]
#[available_gas(2000000000)]
fn test_reduce_sum_fp16x16_1D() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = input_0.reduce_sum(0, false);

    assert_eq(y, z);
}