mod input_0;
mod output_0;


use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::operators::tensor::U32TensorPartialEq;
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::I32TensorPartialEq;
use orion::operators::tensor::{I32Tensor, I32TensorAdd};
use orion::operators::tensor::{TensorTrait, Tensor};

#[test]
#[available_gas(2000000000)]
fn test_argmax_i32_1D_default() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = input_0.argmax(0, Option::None(()), Option::None(()));

    assert_eq(y, z);
}
