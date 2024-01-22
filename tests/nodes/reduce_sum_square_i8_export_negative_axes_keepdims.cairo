mod input_0;
mod output_0;


use orion::operators::tensor::I8TensorPartialEq;
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::utils::{assert_eq, assert_seq_eq};
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{I8Tensor, I8TensorAdd};

#[test]
#[available_gas(2000000000)]
fn test_reduce_sum_square_i8_export_negative_axes_keepdims() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = input_0.reduce_sum_square(0, true);

    assert_eq(y, z);
}
