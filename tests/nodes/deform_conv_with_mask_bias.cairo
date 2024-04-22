mod input_0;
mod input_1;
mod input_2;
mod input_3;
mod input_4;
mod output_0;


use orion::numbers::FixedTrait;
use orion::operators::nn::NNTrait;
use orion::operators::nn::FP16x16NN;
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::FP16x16TensorPartialEq;

#[test]
#[available_gas(2000000000)]
fn test_deform_conv_with_mask_bias() {
    let input_0 = input_0::input_0();
    let input_1 = input_1::input_1();
    let input_2 = input_2::input_2();
    let input_3 = input_3::input_3();
    let input_4 = input_4::input_4();
    let z_0 = output_0::output_0();

    let y_0 = NNTrait::deform_conv(
        @input_0,
        @input_1,
        @input_2,
        Option::Some(input_3.data),
        Option::Some(input_4),
        Option::None,
        Option::None,
        Option::Some(array![2, 2].span()),
        Option::None,
        Option::None,
        Option::None
    );

    assert_eq(y_0, z_0);
}
