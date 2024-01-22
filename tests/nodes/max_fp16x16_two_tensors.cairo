mod input_0;
mod input_1;
mod output_0;


use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorSub};
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::{TensorTrait, Tensor};
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::FP16x16TensorPartialEq;

#[test]
#[available_gas(2000000000)]
fn test_max_fp16x16_two_tensors() {
    let input_0 = input_0::input_0();
    let input_1 = input_1::input_1();
    let z = output_0::output_0();

    let y = TensorTrait::max(array![input_0, input_1].span());

    assert_eq(y, z);
}
