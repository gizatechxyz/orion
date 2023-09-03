mod input_0; 
mod output_0; 


use orion::operators::nn::NNTrait;
use orion::numbers::FixedTrait;
use orion::operators::nn::FP16x16NN;
use orion::operators::tensor::FP16x16TensorPartialEq;
use orion::utils::assert_eq;

#[test]
#[available_gas(2000000000)]
fn test_logsoftmax_fp16x16_axis_0() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = NNTrait::logsoftmax(@input_0, 0);

    assert_eq(y, z);
}