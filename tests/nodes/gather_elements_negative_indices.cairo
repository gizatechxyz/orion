mod input_0;
mod input_1;
mod output_0;


use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::operators::tensor::FP16x16TensorPartialEq;
use orion::operators::tensor::{I32Tensor, I32TensorAdd};
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::I32TensorPartialEq;
use orion::operators::tensor::{TensorTrait, Tensor};

#[test]
#[available_gas(2000000000)]
fn test_gather_elements_negative_indices() {
    let input_0 = input_0::input_0();
    let input_1 = input_1::input_1();
    let z_0 = output_0::output_0();

    let y_0 = input_0.gather_elements(indices: input_1, axis: Option::Some(0));

    assert_eq(y_0, z_0);
}
