mod input_0;
mod input_1;
mod input_2;
mod output_0;


use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I8TensorPartialEq;
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::{I8Tensor, I8TensorAdd};
use orion::operators::tensor::FP16x16TensorPartialEq;
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::implementations::tensor_fp16x16::{
    TensorI8IntoTensorFP16x16, FP16x16TensorSub, FP16x16TensorDiv, FP16x16TensorMul
};
use orion::numbers::{I8IntoFP16x16};

#[test]
#[available_gas(2000000000)]
fn test_qlinear_conv() {
    let input_0 = input_0::input_0();
    let input_1 = input_1::input_1();
    let input_2 = input_2::input_2();
    let z_0 = output_0::output_0();

    let y_0 = FP16x16Tensor::qlinear_conv(
        @input_0,
        @TensorTrait::new(shape: array![1].span(), data: array![*input_2.data.at(0)].span(),),
        @TensorTrait::new(shape: array![1].span(), data: array![*input_2.data.at(1)].span(),),
        @input_1,
        @TensorTrait::new(shape: array![1].span(), data: array![*input_2.data.at(2)].span(),),
        @TensorTrait::new(shape: array![1].span(), data: array![*input_2.data.at(3)].span(),),
        Option::None,
        Option::None,
        Option::None,
        Option::None,
        Option::None,
        Option::None,
        Option::None,
        @TensorTrait::new(shape: array![1].span(), data: array![*input_2.data.at(4)].span(),),
        @TensorTrait::new(shape: array![1].span(), data: array![*input_2.data.at(5)].span(),)
    );

    assert_eq(y_0, z_0);
}
