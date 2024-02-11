mod input_0;
mod output_0;


use orion::operators::nn::FP8x23NN;
use orion::operators::nn::NNTrait;
use orion::numbers::FixedTrait;
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::FP8x23TensorPartialEq;

#[test]
#[available_gas(2000000000)]
fn test_depth_to_space_fp8x23() {
    let input_0 = input_0::input_0();
    let z_0 = output_0::output_0();

    let y_0 = NNTrait::depth_to_space(@input_0, 2, 'DCR');

    assert_eq(y_0, z_0);
}
