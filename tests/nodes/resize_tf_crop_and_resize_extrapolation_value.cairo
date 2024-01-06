mod input_0;
mod input_1;
mod input_2;
mod output_0;


use orion::operators::tensor::FP16x16TensorPartialEq;
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::FP16x16Tensor;
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::U32Tensor;
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32TensorPartialEq;
use orion::operators::tensor::math::resize::{
    MODE, NEAREST_MODE, KEEP_ASPECT_RATIO_POLICY, TRANSFORMATION_MODE
};
use orion::numbers::{FP16x16, FP16x16Impl, FP32x32, FP32x32Impl, FixedTrait};

#[test]
#[available_gas(2000000000)]
fn test_resize_tf_crop_and_resize_extrapolation_value() {
    let data = input_0::input_0();
    let mut sizes = Option::Some(input_1::input_1().data);
    let roi = Option::Some(input_2::input_2());
    let z_0 = output_0::output_0();

    let y_0 = data
        .resize(
            roi,
            Option::None,
            sizes,
            Option::None,
            Option::None,
            Option::Some(TRANSFORMATION_MODE::TF_CROP_AND_RESIZE),
            Option::None,
            Option::None,
            Option::Some(FixedTrait::<FP16x16>::new(655360, false)),
            Option::None,
            Option::Some(MODE::LINEAR),
            Option::None,
        );

    assert_eq(y_0, z_0);
}
