mod input_0;
mod output_0;


use orion::operators::tensor::{I8Tensor, I8TensorAdd};
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::I8TensorPartialEq;
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::utils::{assert_eq, assert_seq_eq};

#[test]
#[available_gas(2000000000)]
fn test_reduce_max_i8_1D() {
    let input_0 = input_0::input_0();
    let z_0 = output_0::output_0();

    let (y_0) = input_0.reduce_max(Option::None(())), Option::None(()), Option::None(());

    assert_eq(y_0, z_0);
}