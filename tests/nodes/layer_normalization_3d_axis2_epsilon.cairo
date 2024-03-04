mod input_0;
mod input_1;
mod input_2;
mod output_0;


use orion::operators::tensor::{TensorTrait, Tensor};
use orion::numbers::{FixedTrait, FP16x16};
use core::array::{ArrayTrait, SpanTrait};
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::operators::tensor::FP16x16TensorPartialEq;

#[test]
#[available_gas(2000000000)]
fn test_layer_normalization_3d_axis2_epsilon() {
    let input_0 = input_0::input_0();
    let input_1 = input_1::input_1();
    let input_2 = input_2::input_2();
    let z_0 = output_0::output_0();

    let (y_0, _, _) = input_0
        .layer_normalization(
            @input_1,
            Option::Some(@input_2),
            Option::Some(2),
            Option::Some(FixedTrait::new(6554, false)),
            Option::None
        );

    assert_eq(y_0, z_0);
}
