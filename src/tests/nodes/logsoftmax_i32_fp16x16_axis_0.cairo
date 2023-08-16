mod input_0; 
mod output_0; 


use orion::operators::nn::core::NNTrait;
use orion::numbers::fixed_point::core::FixedTrait;
use orion::operators::nn::implementations::impl_nn_i32::NN_i32;
use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16Impl;
use orion::operators::tensor::implementations::impl_tensor_fp::FP16x16Tensor::FPTensorPartialEq;
use orion::utils::assert_eq;

#[test]
#[available_gas(2000000000)]
fn test_logsoftmax_i32_fp16x16_axis_0() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = NNTrait::logsoftmax(@input_0, 0);

    assert_eq(y, z);
}