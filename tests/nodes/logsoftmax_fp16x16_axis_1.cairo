mod input_0;
mod output_0;


use orion::utils::{assert_eq, assert_seq_eq};
use orion::numbers::FixedTrait;
use orion::operators::tensor::FP16x16TensorPartialEq;
use orion::operators::nn::NNTrait;
use orion::operators::nn::FP16x16NN;

#[test]
#[available_gas(2000000000)]
fn test_logsoftmax_fp16x16_axis_1() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = NNTrait::logsoftmax(@input_0, 1);

    assert_eq(y, z);
}
