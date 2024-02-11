mod input_0;
mod output_0;


use orion::operators::tensor::FP16x16TensorPartialEq;
use orion::operators::nn::FP16x16NN;
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::nn::NNTrait;
use orion::numbers::FixedTrait;

#[test]
#[available_gas(2000000000)]
fn test_space_to_depth_fp16x16() {
    let input_0 = input_0::input_0();
    let z_0 = output_0::output_0();

    let y_0 = NNTrait::space_to_depth(@input_0, 2);

    assert_eq(y_0, z_0);
}
