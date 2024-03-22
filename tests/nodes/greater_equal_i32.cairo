mod input_0;
mod input_1;
mod output_0;


use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::U32TensorPartialEq;
use orion::operators::tensor::{TensorTrait, Tensor};
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{I32Tensor, I32TensorSub};
use orion::operators::tensor::I32TensorPartialEq;
use orion::operators::tensor::{U32Tensor, U32TensorSub};

#[test]
#[available_gas(2000000000)]
fn test_greater_equal_i32() {
    let input_0 = input_0::input_0();
    let input_1 = input_1::input_1();
    let z = output_0::output_0();

    let y = input_0.greater_equal(@input_1);

    assert_eq(y, z);
}
