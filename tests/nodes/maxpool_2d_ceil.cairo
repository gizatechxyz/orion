mod input_0;
mod output_0;


use orion::operators::nn::NNTrait;
use orion::numbers::FixedTrait;
use orion::operators::tensor::FP16x16TensorPartialEq;
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::nn::FP16x16NN;

#[test]
#[available_gas(2000000000)]
fn test_maxpool_2d_ceil() {
    let input_0 = input_0::input_0();
    let z_0 = output_0::output_0();

    let (y_0, _) = NNTrait::max_pool(
        @input_0,
        Option::None,
        Option::Some(1),
        Option::None,
        array![3, 3].span(),
        Option::None,
        Option::None,
        Option::Some(array![2, 2].span()),
        1
    );

    assert_eq(y_0, z_0);
}
