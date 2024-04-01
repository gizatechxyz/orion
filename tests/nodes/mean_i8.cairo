mod input_0;
mod input_1;
mod input_2;
mod output_0;


use orion::operators::tensor::{I8Tensor, I8TensorAdd};
use orion::utils::{assert_eq, assert_seq_eq};
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::I8TensorPartialEq;
use orion::operators::tensor::{TensorTrait, Tensor};

#[test]
#[available_gas(2000000000)]
fn test_mean_i8() {
    let input_0 = input_0::input_0();
    let input_1 = input_1::input_1();
    let input_2 = input_2::input_2();
    let z_0 = output_0::output_0();

    let y_0 = TensorTrait::mean(array![input_0, input_1, input_2].span());

    assert_eq(y_0, z_0);
}
