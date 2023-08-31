mod input_0;
mod output_0;


use array::ArrayTrait;
use orion::operators::tensor::core::TensorTrait;
use orion::operators::tensor::implementations::tensor_i32_fp8x23::{
    Tensor_i32_fp8x23, i32TensorPartialEq
};
use orion::utils::assert_eq;

#[test]
#[available_gas(2000000000)]
fn test_abs_i32() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = input_0.abs();

    assert_eq(y, z);
}
