mod input_0;
mod input_1;
mod input_2;
mod output_0;


use core::array::{ArrayTrait, SpanTrait};
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16TensorPartialEq;

#[test]
#[available_gas(2000000000)]
fn test_instance_normalization_fp16x16_highdim() {
    let input_0 = input_0::input_0();
    let input_1 = input_1::input_1();
    let input_2 = input_2::input_2();
    let z_0 = output_0::output_0();

    let y_0 = input_0.instance_normalization( @input_1 , @input_2 , Option::None(()) );

    assert_eq(y_0, z_0);
}
