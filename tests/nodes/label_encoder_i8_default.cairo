mod input_0;
mod input_1;
mod input_2;
mod input_3;
mod output_0;


use orion::operators::tensor::{I8Tensor, I8TensorAdd};
use orion::operators::tensor::{TensorTrait, Tensor};
use core::array::{ArrayTrait, SpanTrait};
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::I8TensorPartialEq;

#[test]
#[available_gas(2000000000)]
fn test_label_encoder_i8_default() {
    let input_0 = input_0::input_0();
    let input_1 = input_1::input_1();
    let input_2 = input_2::input_2();
    let input_3 = input_3::input_3();
    let z_0 = output_0::output_0();

    let y_0 = input_0.label_encoder(default_list:Option::None, default_tensor: Option::Some(input_1), 
                    keys:Option::None, keys_tensor: Option::Some(input_2),
                    values: Option::None, values_tensor: Option::Some(input_3));

    assert_eq(y_0, z_0);
}
