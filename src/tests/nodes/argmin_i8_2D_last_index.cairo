mod input_0; 
mod output_0; 


use array::ArrayTrait;
use orion::operators::tensor::core::TensorTrait;
use orion::operators::tensor::implementations::tensor_i8_fp16x16::Tensor_i8_fp16x16;
use orion::operators::tensor::implementations::tensor_u32_fp16x16::u32TensorPartialEq;
use orion::utils::assert_eq;

#[test]
#[available_gas(2000000000)]
fn test_argmin_i8_2D_last_index() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = input_0.argmin(0, Option::None(()), Option::Some(true));

    assert_eq(y, z);
}