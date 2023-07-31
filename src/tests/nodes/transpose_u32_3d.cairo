mod input_0;
mod output_0;

use array::ArrayTrait;
use orion::operators::tensor::core::TensorTrait;
use orion::operators::tensor::implementations::impl_tensor_u32::{Tensor_u32, u32TensorPartialEq};
use orion::utils::assert_eq;

#[test]
#[available_gas(200000000)]
fn test_transpose_u32_3d() {
    let x = input_0::input_0();
    let z = output_0::output_0();

    let y = x.transpose(array![1, 2, 0].span());

    assert_eq(y, z)
}
