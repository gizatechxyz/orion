mod input_0;
mod output_0;


use orion::operators::nn::FP8x23NN;
use orion::utils::{assert_eq, assert_seq_eq};
use orion::numbers::FixedTrait;
use orion::operators::tensor::FP8x23TensorPartialEq;
use orion::operators::nn::NNTrait;

#[test]
#[available_gas(2000000000)]
fn test_leaky_relu_fp8x23() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = NNTrait::leaky_relu(@input_0, @FixedTrait::new(838861, false));

    assert_eq(y, z);
}
