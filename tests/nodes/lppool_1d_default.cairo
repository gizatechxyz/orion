mod input_0;
mod output_0;


use orion::numbers::FixedTrait;
use orion::operators::nn::FP16x16NN;
use orion::operators::nn::NNTrait;
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::FP16x16TensorPartialEq;

#[test]
#[available_gas(2000000000)]
fn test_lppool_1d_default() {
    let input_0 = input_0::input_0();
    let z_0 = output_0::output_0();

    let y_0 = NNTrait::lp_pool(
        @input_0,
        Option::None,
        Option::None,
        Option::None,
        array![2].span(),
        Option::Some(3),
        Option::None,
        Option::Some(array![1].span()),
        Option::None
    );

    assert_eq(y_0, z_0);
}
