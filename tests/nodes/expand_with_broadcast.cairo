mod input_0;
mod output_0;


use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::operators::tensor::FP16x16TensorPartialEq;
use orion::operators::tensor::{TensorTrait, Tensor};
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{U32Tensor};

#[test]
#[available_gas(2000000000)]
fn test_expand_with_broadcast() {
    let input_0 = input_0::input_0();
    let z_0 = output_0::output_0();

    let y_0 = input_0.expand(TensorTrait::new(array![3].span(), array![2, 1, 6].span()));

    assert_eq(y_0, z_0);
}

