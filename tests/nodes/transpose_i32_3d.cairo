mod input_0;
mod output_0;


use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I32TensorPartialEq;
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::{I32Tensor, I32TensorSub};
use core::array::{ArrayTrait, SpanTrait};

#[test]
#[available_gas(2000000000)]
fn test_transpose_i32_3d() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = input_0.transpose(array![1, 2, 0].span());

    assert_eq(y, z);
}
