mod input_0; 
mod output_0; 


use array::ArrayTrait;
use orion::operators::tensor::core::TensorTrait;
use orion::operators::tensor::I8Tensor;
use orion::operators::tensor::implementations::tensor_u32_fp16x16::u32TensorPartialEq;
use orion::utils::assert_eq;

#[test]
#[available_gas(2000000000)]
fn test_argmax_i8_2D_keepdims_false() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = input_0.argmax(0, Option::Some(false), Option::None(()));

    assert_eq(y, z);
}