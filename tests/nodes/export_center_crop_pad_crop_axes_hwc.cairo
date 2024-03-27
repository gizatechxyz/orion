mod input_0;
mod output_0;


use orion::operators::tensor::Complex64Tensor;
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::Complex64TensorPartialEq;
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};

#[test]
#[available_gas(2000000000)]
fn test_export_center_crop_pad_crop_axes_hwc() {
    let input_0 = input_0::input_0();
    let z_0 = output_0::output_0();

    let y_0 = input_0.center_crop_pad(TensorTrait::new(array![2].span(), array![10,9].span()), Option::Some(array![0,1]));

    assert_eq(y_0, z_0);
}
