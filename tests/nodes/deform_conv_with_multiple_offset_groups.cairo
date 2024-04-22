mod input_0;
mod input_1;
mod input_2;
mod output_0;


use orion::numbers::FixedTrait;
use orion::operators::nn::NNTrait;
use orion::operators::nn::FP16x16NN;
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::FP16x16TensorPartialEq;

#[test]
#[available_gas(2000000000)]
fn test_deform_conv_with_multiple_offset_groups() {
    let input_0 = input_0::input_0();
    let input_1 = input_1::input_1();
    let input_2 = input_2::input_2();
    let z_0 = output_0::output_0();

    let y_0 = NNTrait::deform_conv(
        @input_0,
        @input_1,
        @input_2,
        Option::None,
        Option::None,
        Option::None,
        Option::None,
        Option::Some(array![2, 2].span()),
        Option::Some(2),
        Option::None,
        Option::None
    );

    assert_eq(y_0, z_0);
}
