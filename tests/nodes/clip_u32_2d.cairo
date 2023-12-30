mod input_0;
mod output_0;


use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::{U32Tensor, U32TensorSub};
use orion::operators::tensor::U32TensorPartialEq;
use orion::operators::tensor::{TensorTrait, Tensor};
use core::array::{ArrayTrait, SpanTrait};

#[test]
#[available_gas(2000000000)]
fn test_clip_u32_2d() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = input_0.clip(Option::Some(10_u32), Option::Some(20_u32));

    assert_eq(y, z);
}
