mod input_0;
mod output_0;


use orion::utils::{assert_eq, assert_seq_eq};
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::FP8x23TensorPartialEq;
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorSub};
use orion::operators::tensor::{TensorTrait, Tensor};

#[test]
#[available_gas(2000000000)]
fn test_sequence_erase_fp8x23_empty() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = TensorTrait::sequence_erase(input_0, Option::None(()));

    assert_seq_eq(y, z);
}
