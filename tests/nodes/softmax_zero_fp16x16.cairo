mod input_0;
mod output_0;


use orion::utils::{assert_eq, assert_seq_eq};
use orion::numbers::FixedTrait;
use orion::operators::nn::FP16x16NN;
use orion::operators::nn::NNTrait;
use orion::operators::tensor::FP16x16TensorPartialEq;

#[test]
#[available_gas(2000000000)]
fn test_softmax_zero_fp16x16() {
    let input_0 = input_0::input_0();
    let z_0 = output_0::output_0();

    let y_0 = NNTrait::softmax_zero(@input_0, 1);

    assert_eq(y_0, z_0);
}
