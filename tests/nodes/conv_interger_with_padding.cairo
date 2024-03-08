mod input_0;
mod input_1;
mod input_2;
mod output_0;


use orion::utils::{assert_eq, assert_seq_eq};
use orion::numbers::FixedTrait;
use orion::operators::nn::NNTrait;
use orion::operators::tensor::U32TensorPartialEq;
use orion::operators::tensor::I8TensorPartialEq;
use orion::operators::nn::U32NN;
use orion::operators::nn::I8NN;

#[test]
#[available_gas(2000000000)]
fn test_conv_interger_with_padding() {
    let input_0 = input_0::input_0();
    let input_1 = input_1::input_1();
    let input_2 = input_2::input_2();
    let z_0 = output_0::output_0();

    let y_0 = NNTrait::conv_integer(
        @input_0,
        @input_1,
        Option::Some(@input_2),
        Option::None,
        Option::None,
        Option::None,
        Option::None,
        Option::None,
        Option::Some(array![1, 1, 1, 1].span()),
        Option::None
    );

    assert_eq(y_0, z_0);
}
