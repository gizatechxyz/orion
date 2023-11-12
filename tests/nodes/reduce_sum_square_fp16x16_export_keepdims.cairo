mod input_0;
mod output_0;


use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::TensorTrait;
use orion::operators::tensor::FP16x16Tensor;
use orion::operators::tensor::FP16x16TensorPartialEq;
use orion::utils::assert_eq;

#[test]
#[available_gas(2000000000)]
fn test_reduce_sum_square_fp16x16_export_keepdims() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = input_0.reduce_sum_square(2, true);

    assert_eq(y, z);
}
