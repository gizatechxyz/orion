mod output_0;


use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::FP8x23TensorPartialEq;
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::numbers::{FixedTrait, FP8x23};

#[test]
#[available_gas(2000000000)]
fn test_blackman_window_fp8x23() {
    let z_0 = output_0::output_0();

    let y_0 = TensorTrait::blackman_window(FP8x23 { mag: 25165824, sign: false }, Option::Some(0));

    assert_eq(y_0, z_0);
}
