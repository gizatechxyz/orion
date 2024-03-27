mod input_0;
mod output_0;


use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::TensorTrait;
use orion::operators::tensor::FP16x16Tensor;
use orion::operators::tensor::FP16x16TensorPartialEq;
use orion::utils::assert_eq;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

#[test]
#[available_gas(2000000000)]
fn test_clip_fp16x16_3d() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = input_0
        .clip(
            Option::Some(FP16x16 { mag: 655360, sign: true }),
            Option::Some(FP16x16 { mag: 1310720, sign: false })
        );

    assert_eq(y, z);
}
