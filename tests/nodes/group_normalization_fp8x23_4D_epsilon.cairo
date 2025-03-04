mod input_0;
mod input_1;
mod input_2;
mod output_0;


use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23TensorPartialEq;
use core::array::{ArrayTrait, SpanTrait};
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FP8x23,FixedTrait};


#[test]
#[available_gas(2000000000)]
fn test_group_normalization_fp8x23_4D_epsilon() {
    let input_0 = input_0::input_0();
    let input_1 = input_1::input_1();
    let input_2 = input_2::input_2();
    let z_0 = output_0::output_0();

    let y_0 = input_0.group_normalization(2, @input_1 , @input_2 , Option::Some( FixedTrait::new(838860, false)) );

    assert_eq(y_0, z_0);
}
