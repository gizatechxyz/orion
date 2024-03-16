mod input_0;
mod output_0;


use orion::operators::nn::FP8x23NN;
use orion::numbers::FixedTrait;
use orion::operators::tensor::FP8x23TensorPartialEq;
use orion::operators::nn::NNTrait;
use orion::utils::{assert_eq, assert_seq_eq};

#[test]
#[available_gas(2000000000)]
fn test_mish_fp8x23() {
    let input_0 = input_0::input_0();
    let z_0 = output_0::output_0();

    let y_0 = NNTrait::mish(@input_0);

    assert_eq(y_0, z_0);
}
