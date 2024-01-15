mod input_0;
mod output_0;


use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23TensorPartialEq;
use orion::utils::{assert_eq, assert_seq_eq};
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorSub};

#[test]
#[available_gas(2000000000)]
fn test_unsqueeze_fp8x23_3d() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = input_0.unsqueeze(array![5, 4, 2].span());

    assert_eq(y, z);
}
