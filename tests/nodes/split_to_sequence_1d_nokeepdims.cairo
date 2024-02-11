mod input_0;
mod output_0;


use orion::operators::tensor::U32TensorPartialEq;
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use core::array::{ArrayTrait, SpanTrait};

#[test]
#[available_gas(2000000000)]
fn test_split_to_sequence_1d_nokeepdims() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = input_0.split_to_sequence(0, 0, Option::None(()));

    assert_seq_eq(y, z);
}
