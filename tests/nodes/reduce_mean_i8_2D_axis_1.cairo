mod input_0;
mod output_0;


use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23TensorPartialEq;
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::FP8x23Tensor;

#[test]
#[available_gas(2000000000)]
fn test_reduce_mean_i8_2D_axis_1() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = input_0.reduce_mean(Option::Some(array![1].span()), Option::None(()), Option::None(()));

    assert_eq(y, z);
}
