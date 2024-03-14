mod output_0;


use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::FP16x16TensorPartialEq;
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::numbers::{FixedTrait, FP16x16};


#[test]
#[available_gas(2000000000)]
fn test_mel_weight_matrix_fp16x16() {
    let z_0 = output_0::output_0();

    let y_0 = TensorTrait::mel_weight_matrix(8, 16, 8192, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 268435456, sign: false });

    assert_eq(y_0, z_0);
}
