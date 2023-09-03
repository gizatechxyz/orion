mod input_0; 
mod output_0; 


use array::ArrayTrait;
use orion::operators::tensor::core::TensorTrait;
use orion::operators::tensor::I8Tensor;
use orion::operators::tensor::implementations::tensor_fp8x23::FP8x23TensorPartialEq;
use orion::utils::assert_eq;


#[test]
#[available_gas(2000000000)]
fn test_asinh_i8_fp8x23() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = input_0.asinh();

    assert_eq(y, z);
}