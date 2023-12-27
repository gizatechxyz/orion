mod input_0;
mod input_1;
mod output_0;


use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::{TensorTrait, Tensor};
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::FP8x23TensorPartialEq;

#[test]
#[available_gas(2000000000)]
fn test_concat_i8_3d_default() {
    let input_0 = input_0::input_0();
    let input_1 = input_1::input_1();
    let z = output_0::output_0();

    let y = TensorTrait::concat(array![input_0, input_1].span(), 0);

    assert_eq(y, z);
}
