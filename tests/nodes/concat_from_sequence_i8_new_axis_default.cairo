mod input_0;
mod output_0;


use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::I8TensorPartialEq;
use orion::operators::tensor::{I8Tensor, I8TensorAdd};

#[test]
#[available_gas(2000000000)]
fn test_concat_from_sequence_i8_new_axis_default() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = TensorTrait::concat_from_sequence(input_0, 1_i32, Option::None(()));

    assert_eq(y, z);
}
