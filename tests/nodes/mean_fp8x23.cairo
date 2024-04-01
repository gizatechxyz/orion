mod input_0;
mod input_1;
mod output_0;


use orion::operators::tensor::FP8x23TensorPartialEq;
use orion::utils::{assert_eq, assert_seq_eq};
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};

#[test]
#[available_gas(2000000000)]
fn test_mean_fp8x23() {
    let input_0 = input_0::input_0();
    let input_1 = input_1::input_1();
    let z_0 = output_0::output_0();

    let y_0 = TensorTrait::mean(array![input_0, input_1].span());

    assert_eq(y_0, z_0);
}
