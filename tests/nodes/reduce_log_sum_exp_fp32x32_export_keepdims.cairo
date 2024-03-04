mod input_0;
mod output_0;


use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::FP32x32Tensor;
use orion::operators::tensor::FP32x32TensorPartialEq;

#[test]
#[available_gas(2000000000)]
fn test_reduce_log_sum_exp_fp32x32_export_keepdims() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = input_0.reduce_log_sum_exp(2, true);

    assert_eq(y, z);
}
