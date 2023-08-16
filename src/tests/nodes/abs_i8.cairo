mod input_0; 
mod output_0; 


use array::ArrayTrait;
use orion::operators::tensor::core::TensorTrait;
use orion::operators::tensor::implementations::impl_tensor_i8::Tensor_i8;
use orion::operators::tensor::implementations::impl_tensor_i8::i8TensorPartialEq;
use orion::utils::assert_eq;

#[test]
#[available_gas(2000000000)]
fn test_abs_i8() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = input_0.abs();

    assert_eq(y, z);
}