mod output_0;


use orion::operators::tensor::FP8x23TensorPartialEq;
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::utils::{assert_eq, assert_seq_eq};
use core::array::{ArrayTrait, SpanTrait};
use orion::numbers::{FixedTrait, FP8x23};

#[test]
#[available_gas(2000000000)]
fn test_range_fp8x23() {
    let z_0 = output_0::output_0();

    let y_0 = TensorTrait::range(
        FP8x23 { mag: 8388608, sign: false },
        FP8x23 { mag: 41943040, sign: false },
        FP8x23 { mag: 2516582, sign: false }
    );

    assert_eq(y_0, z_0);
}
