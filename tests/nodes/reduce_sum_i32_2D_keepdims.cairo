mod input_0;
mod output_0;


use orion::utils::{assert_eq, assert_seq_eq};
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::I32TensorPartialEq;
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I32Tensor, I32TensorMul};

#[test]
#[available_gas(2000000000)]
fn test_reduce_sum_i32_2D_keepdims() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = input_0.reduce_sum(0, true);

    assert_eq(y, z);
}
