mod input_0;
mod output_0;


use orion::operators::tensor::U32TensorPartialEq;
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use core::array::{ArrayTrait, SpanTrait};

#[test]
#[available_gas(2000000000)]
fn test_split_to_sequence_u32_1d_variable_parts() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = input_0.split_to_sequence(0, 1, Option::Some(TensorTrait::<u32>::new(shape: array![2].span(), data: array![2, 4].span(),)));

    assert_seq_eq(y, z);
}
