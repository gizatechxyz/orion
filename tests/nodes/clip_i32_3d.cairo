mod input_0;
mod output_0;


use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::TensorTrait;
use orion::operators::tensor::I32Tensor;
use orion::operators::tensor::I32TensorPartialEq;
use orion::utils::assert_eq;
use orion::numbers::signed_integer::i32::{i32, IntegerTrait};

#[test]
#[available_gas(2000000000)]
fn test_clip_i32_3d() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = input_0
        .clip(
            Option::Some(i32 { mag: 10, sign: true }), Option::Some(i32 { mag: 20, sign: false })
        );

    assert_eq(y, z);
}
