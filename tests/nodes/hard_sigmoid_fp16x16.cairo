mod input_0;
mod output_0;


use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::nn::NNTrait;
use orion::numbers::FixedTrait;
use orion::operators::tensor::FP16x16TensorPartialEq;
use orion::operators::nn::FP16x16NN;

#[test]
#[available_gas(2000000000)]
fn test_hard_sigmoid_fp16x16() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = NNTrait::hard_sigmoid(
        @input_0, @FixedTrait::new(13107, false), @FixedTrait::new(32768, false)
    );

    assert_eq(y, z);
}
