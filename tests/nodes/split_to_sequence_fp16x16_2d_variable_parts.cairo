mod input_0;
mod output_0;


use orion::utils::{assert_eq, assert_seq_eq};
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::FP16x16TensorPartialEq;
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::operators::tensor::U32Tensor;

#[test]
#[available_gas(2000000000)]
fn test_split_to_sequence_fp16x16_2d_variable_parts() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = input_0.split_to_sequence(Option::Some(TensorTrait::new(shape: array![2].span(), data: array![1,2].span())), 0, Option::None(()));

    assert_seq_eq(y, z);
}
