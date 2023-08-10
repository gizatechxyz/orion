mod input_0; 
mod output_0; 


use orion::operators::nn::core::NNTrait;
use orion::numbers::fixed_point::core::FixedTrait;
use orion::operators::nn::implementations::impl_nn_i32::NN_i32;
use orion::numbers::fixed_point::implementations::impl_16x16::FP16x16Impl;
use orion::operators::tensor::implementations::impl_tensor_i32::i32TensorPartialEq;
use orion::utils::assert_eq;

#[test]
#[available_gas(2000000000)]
fn test_relu_i32() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = NNTrait::relu(@input_0);

    assert_eq(y, z);
}