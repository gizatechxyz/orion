mod input_0;
mod output_0;


use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use core::array::{ArrayTrait, SpanTrait};
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::FP16x16TensorPartialEq;
use orion::operators::tensor::{TensorTrait, Tensor};

#[test]
#[available_gas(2000000000)]
fn test_concat_from_sequence_fp16x16_new_axis_one() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = TensorTrait::concat_from_sequence(input_0, 1_i32, Option::Some(1));

    assert_eq(y, z);
}
