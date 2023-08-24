mod input_0; 
mod output_0; 


use array::ArrayTrait;
use orion::operators::tensor::core::TensorTrait;
use orion::operators::tensor::implementations::impl_tensor_i8::Tensor_i8;
use orion::operators::tensor::implementations::impl_tensor_i8::i8TensorPartialEq;
use orion::utils::assert_eq;

#[test]
#[available_gas(2000000000)]
fn test_slice_i8_3d() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = input_0.slice(array![0, 0].span(), array![3, 10].span(), Option::Some(array![0, 1].span()), Option::Some(array![1, 3].span()));

    assert_eq(y, z);
}