mod input_0;
mod input_1;
mod output_0;


use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::FP16x16TensorPartialEq;
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::preview_training::momentum::MODE;

#[test]
#[available_gas(2000000000)]
fn test_momentum_standard() {
    let input_0 = input_0::input_0();
    let input_1 = input_1::input_1();
    let z = output_0::output_0();

    let (y0, y1) = TensorTrait::momentum(
        *input_1.data.at(0),
        *input_1.data.at(1),
        @input_0,
        *input_1.data.at(2),
        *input_1.data.at(3),
        MODE::STANDARD,
        *input_1.data.at(4)
    );

    assert_eq(y0, *z.at(0));
    assert_eq(y1, *z.at(1));
}

