mod input_0;
mod input_1;
mod output_0;


use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::{U32Tensor, U32TensorSub};
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::FP8x23TensorPartialEq;
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorSub};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32TensorPartialEq;

#[test]
#[available_gas(2000000000)]
fn test_gather_elements_fp8x23_3d_default() {
    let input_0 = input_0::input_0();
    let input_1 = input_1::input_1();
    let z = output_0::output_0();

    let y = input_0.gather_elements(indices:input_1, axis:Option::Some(0));

    assert_eq(y, z);
}
