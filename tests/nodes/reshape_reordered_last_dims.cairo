mod input_0;
mod output_0;


use orion::operators::tensor::{TensorTrait, Tensor};
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::I32TensorPartialEq;
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};

#[test]
#[available_gas(2000000000)]
fn test_reshape_reordered_last_dims() {
    let input_0 = input_0::input_0();
    let z_0 = output_0::output_0();

    let y_0 = input_0.reshape(array![2,4,3].span());

    assert_eq(y_0, z_0);
}
