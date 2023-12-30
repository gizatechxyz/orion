mod input_0;
mod output_0;


use orion::operators::tensor::I8TensorPartialEq;
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I8Tensor, I8TensorSub};
use core::array::{ArrayTrait, SpanTrait};

#[test]
#[available_gas(2000000000)]
fn test_clip_i8_2d() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = input_0.clip(Option::Some(10_i8), Option::Some(20_i8));

    assert_eq(y, z);
}
