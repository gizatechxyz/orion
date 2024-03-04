mod input_0;
mod input_1;
mod output_0;


use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::FP16x16Tensor;
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16TensorPartialEq;
use orion::operators::tensor::math::resize::{
    MODE, NEAREST_MODE, KEEP_ASPECT_RATIO_POLICY, TRANSFORMATION_MODE
};

#[test]
#[available_gas(2000000000)]
fn test_resize_downsample_scales_linear_half_pixel_symmetric() {
    let data = input_0::input_0();
    let mut scales = Option::Some(input_1::input_1().data);
    let z_0 = output_0::output_0();

    let y_0 = data
        .resize(
            Option::None,
            scales,
            Option::None,
            Option::None,
            Option::None,
            Option::Some(TRANSFORMATION_MODE::HALF_PIXEL_SYMMETRIC),
            Option::None,
            Option::None,
            Option::None,
            Option::None,
            Option::Some(MODE::LINEAR),
            Option::None,
        );

    assert_eq(y_0, z_0);
}
