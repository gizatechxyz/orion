mod input_0;
mod output_0;
mod output_non_axes;
mod output_negatives;

use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::TensorTrait;
use orion::operators::tensor::U32Tensor;
use orion::operators::tensor::U32TensorPartialEq;
use orion::numbers::signed_integer::i32::{i32, IntegerTrait};

// Non Axes parameters
fn non_axes() {
    let input_0 = input_0::input_0();
    let none_axes = input_0.squeeze(Option::None(()));
    let z = output_non_axes::non_axes();
    assert(none_axes.shape == z.shape, 'shapes do not match (non axes)');
}

// Negatives Axes
fn negatives() {
    let input_0 = input_0::input_0();
    let negatives = input_0
        .squeeze(
            Option::Some(array![i32 { mag: 5, sign: true }, i32 { mag: 3, sign: true }].span())
        );
    let z = output_negatives::negatives();
    assert(negatives.shape == z.shape, 'shapes do not match (negatives)');
}

#[test]
#[available_gas(2000000000)]
fn test_squeeze_u32() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = input_0
        .squeeze(
            Option::Some(array![i32 { mag: 0, sign: false }, i32 { mag: 2, sign: false }].span())
        );

    assert(y.shape == z.shape, 'shapes do not match');

    non_axes();

    negatives();
}
