mod input_0;
mod output_0;


use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::FP8x23Tensor;
use orion::operators::tensor::FP8x23TensorPartialEq;
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};

#[test]
#[available_gas(2000000000)]
fn test_reduce_log_sum_fp16x16_export_do_not_keepdims() {
    let input_0 = input_0::input_0();
    let z_0 = output_0::output_0();

    let y_0 = input_0.reduce_log_sum(2, false);

    assert_eq(y_0, z_0);
}
