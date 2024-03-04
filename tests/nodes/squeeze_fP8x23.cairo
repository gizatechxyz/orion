mod input_0;
mod output_0;


use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23TensorPartialEq;
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorSub};
use orion::utils::{assert_eq, assert_seq_eq};

#[test]
#[available_gas(2000000000)]
fn test_squeeze_fP8x23() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = input_0.squeeze(Option::Some(array![0, 2].span()));

    assert(y.shape == z.shape, 'shapes do not match');
}
