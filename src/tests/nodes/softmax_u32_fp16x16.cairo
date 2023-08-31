mod input_0; 
mod output_0; 


use orion::operators::nn::core::NNTrait;
use orion::operators::nn::implementations::nn_u32_fp16x16::NN_u32_fp16x16;
use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16Impl;
use orion::operators::tensor::implementations::tensor_fp16x16::FP16x16TensorPartialEq;
use orion::utils::assert_eq;

#[test]
#[available_gas(2000000000)]
fn test_softmax_u32_fp16x16() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = NNTrait::softmax(@input_0, 0);

    assert_eq(y, z);
}