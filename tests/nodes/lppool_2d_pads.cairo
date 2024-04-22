mod input_0;
mod output_0;


use orion::numbers::FixedTrait;
use orion::operators::nn::FP16x16NN;
use orion::operators::nn::NNTrait;
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::FP16x16TensorPartialEq;

#[test]
#[available_gas(2000000000)]
fn test_lppool_2d_pads() {
    let input_0 = input_0::input_0();
    let z_0 = output_0::output_0();

    let y_0 = NNTrait::lp_pool(
        @input_0,
        Option::None,
        Option::None,
        Option::None,
        array![3, 3].span(),
        Option::Some(3),
        Option::Some(array![2, 2, 2, 2].span()),
        Option::Some(array![1, 1].span()),
        Option::Some(1)
    );

    assert_eq(y_0, z_0);
}
