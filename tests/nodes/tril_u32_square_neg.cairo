mod input_0;
mod output_0;


use orion::operators::tensor::U32TensorPartialEq;
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::utils::{assert_eq, assert_seq_eq};
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};

#[test]
#[available_gas(2000000000)]
fn test_tril_u32_square_neg() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = input_0.trilu(false, -1);

    assert_eq(y, z);
}
