mod input_0;
mod input_1;
mod output_0;


use orion::operators::tensor::{I32Tensor, I32TensorAdd};
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::{TensorTrait, Tensor};
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::I32TensorPartialEq;

#[test]
#[available_gas(2000000000)]
fn test_matmul_i32_1x3() {
    let input_0 = input_0::input_0();
    let input_1 = input_1::input_1();
    let z_0 = output_0::output_0();

    let y_0 = input_0.matmul(@input_1);

    assert_eq(y_0, z_0);
}
