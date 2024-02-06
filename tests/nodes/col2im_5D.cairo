mod input_0;
mod output_0;


use orion::operators::nn::NNTrait;
use orion::operators::tensor::FP16x16TensorPartialEq;
use orion::numbers::FixedTrait;
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::nn::FP16x16NN;

#[test]
#[available_gas(2000000000)]
fn test_col2im_5D() {
    let input_0 = input_0::input_0();
    let z_0 = output_0::output_0();

    let y_0 = NNTrait::col2im(
        @input_0,
        array![3, 4, 5].span(),
        array![1, 1, 5].span(),
        Option::None,
        Option::None,
        Option::None
    );

    assert_eq(y_0, z_0);
}
