mod input_0; 
mod output_0; 


use orion::operators::nn::core::NNTrait;
use orion::numbers::fixed_point::core::FixedTrait;
use orion::operators::nn::implementations::nn_i32_fp16x16::NN_i32_fp16x16;
use orion::operators::tensor::implementations::tensor_fp16x16::FP16x16TensorPartialEq;
use orion::utils::assert_eq;
#[test]
#[available_gas(2000000000)]
fn test_logsoftmax_i32_fp16x16_axis_1() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = NNTrait::logsoftmax(@input_0, 1);

    assert_eq(y, z);
}