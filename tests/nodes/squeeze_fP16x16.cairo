mod input_0;
mod output_0;


use orion::operators::tensor::FP16x16TensorPartialEq;
use orion::operators::tensor::{TensorTrait, Tensor};
use core::array::{ArrayTrait, SpanTrait};
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorSub};

#[test]
#[available_gas(2000000000)]
fn test_squeeze_fP16x16() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = input_0.squeeze(Option::Some(array![0_i32, 2_i32].span()));

    assert(y.shape == z.shape, 'shapes do not match');
}
