mod input_0;
mod output_0;


use orion::operators::nn::NNTrait;
use orion::operators::tensor::FP16x16TensorPartialEq;
use orion::operators::nn::FP16x16NN;
use orion::numbers::FixedTrait;
use orion::utils::{assert_eq, assert_seq_eq};

#[test]
#[available_gas(2000000000)]
fn test_softmax_fp16x16() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = NNTrait::softmax(@input_0, 1);

    assert_eq(y, z);
}
