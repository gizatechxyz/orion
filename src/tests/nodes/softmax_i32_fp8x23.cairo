mod input_0; 
mod output_0; 


use orion::operators::nn::core::NNTrait;
use orion::numbers::fixed_point::core::FixedTrait;
use orion::operators::nn::implementations::nn_i32_fp8x23::NN_i32_fp8x23;
use orion::operators::tensor::implementations::tensor_fp8x23::FP8x23TensorPartialEq;
use orion::utils::assert_eq;

#[test]
#[available_gas(2000000000)]
fn test_softmax_i32_fp8x23() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = NNTrait::softmax(@input_0, 0);

    assert_eq(y, z);
}