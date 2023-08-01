mod input_0;
mod output_0;

use array::ArrayTrait;
use orion::operators::tensor::core::TensorTrait;
use orion::operators::tensor::implementations::impl_tensor_fp::{
    Tensor_fp, FP16x16Tensor::FPTensorPartialEq
};
use orion::utils::assert_eq;

#[test]
#[available_gas(200000000)]
fn test_abs_fp16x16() {
    let x = input_0::input_0();
    let z = output_0::output_0();

    let y = x.abs();

    assert_eq(y, z)
}
