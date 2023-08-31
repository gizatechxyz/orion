mod input_0; 
mod input_1; 
mod output_0; 


use array::ArrayTrait;
use orion::operators::tensor::core::TensorTrait;
use orion::operators::tensor::implementations::tensor_fp16x16::{
    Tensor_fp16x16, FP16x16TensorPartialEq
};
use orion::utils::assert_eq;

#[test]
#[available_gas(2000000000)]
fn test_concat_fp16x16_3d_axis_1() {
    let input_0 = input_0::input_0();
    let input_1 = input_1::input_1();
    let z = output_0::output_0();

    let y = TensorTrait::concat(array![input_0, input_1].span(), 1);

    assert_eq(y, z);
}