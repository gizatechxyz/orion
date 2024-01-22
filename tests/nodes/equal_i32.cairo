mod input_0;
mod input_1;
mod output_0;


use orion::operators::tensor::I32TensorPartialEq;
use orion::operators::tensor::U32TensorPartialEq;
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::{I32Tensor, I32TensorDiv};
use orion::operators::tensor::{U32Tensor, U32TensorDiv};

#[test]
#[available_gas(2000000000)]
fn test_equal_i32() {
    let input_0 = input_0::input_0();
    let input_1 = input_1::input_1();
    let z = output_0::output_0();

    let y = input_0.equal(@input_1);

    assert_eq(y, z);
}
