mod input_0; 
mod input_1; 
mod input_2; 
mod output_0; 


use array::ArrayTrait;
use orion::operators::tensor::core::TensorTrait;
use orion::operators::tensor::U32Tensor;
use orion::operators::tensor::implementations::tensor_u32_fp16x16::u32TensorPartialEq;
use orion::utils::assert_eq;

#[test]
#[available_gas(2000000000)]
fn test_concat_u32_3d_three_tensors_axis_2() {
    let input_0 = input_0::input_0();
    let input_1 = input_1::input_1();
    let input_2 = input_2::input_2();
    let z = output_0::output_0();

    let y = TensorTrait::concat(array![input_0, input_1, input_2].span(), 2);

    assert_eq(y, z);
}