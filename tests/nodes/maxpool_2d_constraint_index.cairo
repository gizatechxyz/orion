mod input_0;
mod output_0;


use orion::operators::nn::NNTrait;
use orion::operators::tensor::U32TensorPartialEq;
use orion::numbers::FixedTrait;
use orion::operators::tensor::I32TensorPartialEq;
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::nn::FP16x16NN;
use orion::operators::nn::U32NN;


#[test]
#[available_gas(2000000000)]
fn test_maxpool_2d_constraint_index() {
    let input_0 = input_0::input_0();
    let z_0 = output_0::output_0();

    let (_, y_0) = NNTrait::max_pool(
        @input_0,
        Option::None,
        Option::None,
        Option::None,
        array![2, 2].span(),
        Option::None,
        Option::Some(1),
        Option::Some(array![2, 2].span()),
        2
    );

    assert_eq(y_0.unwrap(), z_0);
}

