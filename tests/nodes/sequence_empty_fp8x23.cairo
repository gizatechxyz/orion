mod output_0;


use orion::operators::tensor::{TensorTrait, Tensor};
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::FP8x23TensorPartialEq;
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorSub};
use core::array::{ArrayTrait, SpanTrait};

#[test]
#[available_gas(2000000000)]
fn test_sequence_empty_fp8x23() {
    let z = output_0::output_0();

    let y = TensorTrait::sequence_empty();

    assert_seq_eq(y, z);
}
