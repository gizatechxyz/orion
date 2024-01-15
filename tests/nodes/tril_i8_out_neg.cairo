mod input_0;
mod output_0;


use orion::operators::tensor::I8TensorPartialEq;
use orion::operators::tensor::{I8Tensor, I8TensorAdd};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::utils::{assert_eq, assert_seq_eq};
use core::array::{ArrayTrait, SpanTrait};

#[test]
#[available_gas(2000000000)]
fn test_tril_i8_out_neg() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = input_0.trilu(false, -7);

    assert_eq(y, z);
}
