mod input_0;
mod input_1;
mod output_0;


use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorSub};
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::{I32Tensor, I32TensorSub};
use orion::operators::tensor::FP8x23TensorPartialEq;
use orion::operators::tensor::I32TensorPartialEq;
use orion::operators::sequence::FP8x23Sequence;
use orion::operators::sequence::SequenceTrait;

#[test]
#[available_gas(2000000000)]
fn test_sequence_at_fp8x23_positive() {
    let input_0 = input_0::input_0();
    let input_1 = input_1::input_1();
    let z = output_0::output_0();

    let y = SequenceTrait::sequence_at(input_0, input_1);

    assert_eq(y, z);
}
