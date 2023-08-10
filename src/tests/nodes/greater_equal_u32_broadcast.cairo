mod input_0; 
mod input_1; 
mod output_0; 


use array::ArrayTrait;
use orion::operators::tensor::core::TensorTrait;
use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
use orion::operators::tensor::implementations::impl_tensor_u32::u32TensorPartialEq;
use orion::utils::assert_eq;

#[test]
#[available_gas(2000000000)]
fn test_greater_equal_u32_broadcast() {
    let input_0 = input_0::input_0();
    let input_1 = input_1::input_1();
    let z = output_0::output_0();

    let y = input_0.greater_equal(@input_1);

    assert_eq(y, z);
}