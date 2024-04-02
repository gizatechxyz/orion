mod input_0;
mod input_1;
mod input_2;
mod output_0;


use orion::operators::tensor::FP16x16TensorPartialEq;
use orion::operators::tensor::{TensorTrait, Tensor};
use core::array::{ArrayTrait, SpanTrait};
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FP16x16,FixedTrait};


#[test]
#[available_gas(2000000000)]
fn test_group_normalization_fp16x16_3D_groups_equal_to_channels() {
    let input_0 = input_0::input_0();
    let input_1 = input_1::input_1();
    let input_2 = input_2::input_2();
    let z_0 = output_0::output_0();

    let y_0 = input_0.group_normalization(2, @input_1 , @input_2 , Option::Some( FixedTrait::new(6554, false)) );

    assert_eq(y_0, z_0);
}
