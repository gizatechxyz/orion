mod input_0;
mod input_1;
mod output_0;


use orion::operators::nn::FP16x16NN;
use orion::numbers::FixedTrait;
use orion::operators::tensor::FP16x16TensorPartialEq;
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::nn::NNTrait;
use orion::operators::nn::{PADDING_MODE};

#[test]
#[available_gas(2000000000)]
fn test_grid_sample_padding_reflection() {
    let input_0 = input_0::input_0();
    let input_1 = input_1::input_1();
    let z_0 = output_0::output_0();

    let y_0 = NNTrait::grid_sample(
        @input_0, @input_1, Option::None, Option::None, Option::Some(PADDING_MODE::REFLECTION)
    );

    assert_eq(y_0, z_0);
}
