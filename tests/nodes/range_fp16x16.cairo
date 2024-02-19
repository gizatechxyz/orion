mod output_0;


use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16TensorPartialEq;
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

#[test]
#[available_gas(2000000000)]
fn test_range_fp16x16() {
    let z_0 = output_0::output_0();

    let y_0 = TensorTrait::range(
        FP16x16 { mag: 65536, sign: false },
        FP16x16 { mag: 1638400, sign: false },
        FP16x16 { mag: 196608, sign: false }
    );

    assert_eq(y_0, z_0);
}
