mod input_0;
mod output_0;


use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::FP16x16TensorPartialEq;
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorSub};
use orion::operators::tensor::{TensorTrait, Tensor};

#[test]
#[available_gas(2000000000)]
fn test_slice_fp16x16_3d() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = input_0
        .slice(
            array![0, 0].span(),
            array![3, 10].span(),
            Option::Some(array![0, 1].span()),
            Option::Some(array![1, 3].span())
        );

    assert_eq(y, z);
}
