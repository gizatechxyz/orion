mod input_0;
mod input_1;
mod output_0;


use orion::operators::tensor::{U32Tensor, U32TensorDiv};
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorDiv};
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16TensorPartialEq;
use orion::operators::tensor::U32TensorPartialEq;

#[test]
#[available_gas(2000000000)]
fn test_greater_fp16x16() {
    let input_0 = input_0::input_0();
    let input_1 = input_1::input_1();
    let z = output_0::output_0();

    let y = input_0.greater(@input_1);

    assert_eq(y, z);
}
