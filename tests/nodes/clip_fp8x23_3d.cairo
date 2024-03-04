mod input_0;
mod output_0;


use orion::numbers::{FixedTrait, FP8x23};
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::{TensorTrait, Tensor};
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::FP8x23TensorPartialEq;
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorSub};

#[test]
#[available_gas(2000000000)]
fn test_clip_fp8x23_3d() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = input_0
        .clip(
            Option::Some(FP8x23 { mag: 83886080, sign: true }),
            Option::Some(FP8x23 { mag: 167772160, sign: false })
        );

    assert_eq(y, z);
}
