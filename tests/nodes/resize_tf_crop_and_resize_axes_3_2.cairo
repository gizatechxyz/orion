mod input_0;
mod input_1;
mod input_2;
mod input_3;
mod output_0;


use orion::operators::tensor::U32TensorPartialEq;
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::FP16x16Tensor;
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16TensorPartialEq;
use orion::operators::tensor::U32Tensor;
use orion::operators::tensor::math::resize::{
    MODE, NEAREST_MODE, KEEP_ASPECT_RATIO_POLICY, TRANSFORMATION_MODE
};

#[test]
#[available_gas(2000000000)]
fn test_resize_tf_crop_and_resize_axes_3_2() {
    let data = input_0::input_0();
    let mut sizes = Option::Some(input_1::input_1().data);
    let roi = Option::Some(input_2::input_2());
    let axes = Option::Some(input_3::input_3().data);
    let z_0 = output_0::output_0();

    let y_0 = data
        .resize(
            roi,
            Option::None,
            sizes,
            Option::None,
            axes,
            Option::Some(TRANSFORMATION_MODE::TF_CROP_AND_RESIZE),
            Option::None,
            Option::None,
            Option::None,
            Option::None,
            Option::Some(MODE::LINEAR),
            Option::None,
        );

    assert_eq(y_0, z_0);
}
