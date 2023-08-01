mod input_0; 
mod input_1; 
mod output_0; 

use array::ArrayTrait;
use orion::operators::tensor::core::TensorTrait;
use orion::operators::tensor::implementations::impl_tensor_i32::{Tensor_i32, i32TensorPartialEq};
use orion::utils::assert_eq;

#[test]
#[available_gas(200000000)]
fn test_matmul_i32_1d() {
    let a = input_0::input_0();
    let b = input_1::input_1();
    let z = output_0::output_0();

    let y = a.matmul(@b);

    assert_eq(y, z)
}
