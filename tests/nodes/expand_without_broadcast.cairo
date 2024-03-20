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
fn test_expand_without_broadcast() {
    let input_0 = input_0::input_0();
    let z_0 = output_0::output_0();

    let y_0 = input_0.expand(TensorTrait::new(array![2].span(), array![3, 4].span()));

    assert_eq(y_0, z_0);
}
