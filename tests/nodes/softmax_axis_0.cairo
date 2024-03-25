mod input_0;
mod output_0;


use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::nn::FP16x16NN;
use orion::operators::nn::NNTrait;
use orion::numbers::FixedTrait;
use orion::operators::tensor::FP16x16TensorPartialEq;

#[test]
#[available_gas(2000000000)]
fn test_softmax_axis_0() {
    let input_0 = input_0::input_0();
    let z_0 = output_0::output_0();

    let y_0 = NNTrait::softmax(@input_0, Option::Some(0));

    assert_eq(y_0, z_0);
}
