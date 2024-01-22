mod input_0;
mod input_1;
mod output_0;


use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorDiv};
use orion::operators::tensor::FP8x23TensorPartialEq;
use orion::utils::{assert_eq, assert_seq_eq};
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};

#[test]
#[available_gas(2000000000)]
fn test_matmul_fp8x23_2x1() {
    let input_0 = input_0::input_0();
    let input_1 = input_1::input_1();
    let z = output_0::output_0();

    let y = input_0.matmul(@input_1);

    assert_eq(y, z);
}
