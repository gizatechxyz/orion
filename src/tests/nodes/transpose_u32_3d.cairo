mod input_0; 
mod output_0; 


use array::ArrayTrait;
use orion::operators::tensor::core::TensorTrait;
use orion::operators::tensor::U32Tensor;
use orion::operators::tensor::implementations::tensor_u32_fp16x16::u32TensorPartialEq;
use orion::utils::assert_eq;

#[test]
#[available_gas(2000000000)]
fn test_transpose_u32_3d() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = input_0.transpose(array![1, 2, 0].span());

    assert_eq(y, z);
}