mod input_0;
mod output_0;


use orion::operators::tensor::BoolTensorPartialEq;
use orion::operators::tensor::U32TensorPartialEq;
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::{TensorTrait, Tensor};
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};
use orion::operators::tensor::BoolTensor;

#[test]
#[available_gas(2000000000)]
fn test_is_inf_u32() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = TensorTrait::is_inf(@input_0, Option::None, Option::None);

    assert_eq(y, z);
}
