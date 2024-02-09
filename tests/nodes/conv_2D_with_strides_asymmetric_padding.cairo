mod input_0;
mod input_1;
mod output_0;


use orion::numbers::FixedTrait;
use orion::operators::nn::NNTrait;
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::FP16x16TensorPartialEq;
use orion::operators::nn::FP16x16NN;

#[test]
#[available_gas(2000000000)]
fn test_conv_2D_with_strides_asymmetric_padding() {
    let input_0 = input_0::input_0();
    let input_1 = input_1::input_1();
    let z_0 = output_0::output_0();

    let y_0 = NNTrait::conv(
        @input_0,
        @input_1,
        Option::None,
        Option::None,
        Option::None,
        Option::None,
        Option::Some(array![3, 3].span()),
        Option::Some(array![1, 0, 1, 0].span()),
        Option::Some(array![2, 2].span())
    );

    assert_eq(y_0, z_0);
}
