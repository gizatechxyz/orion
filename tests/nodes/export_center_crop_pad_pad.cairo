mod input_0;
mod output_0;


use orion::operators::tensor::Complex64Tensor;
use orion::operators::tensor::Complex64TensorPartialEq;
use orion::operators::tensor::{TensorTrait, Tensor};
use core::array::{ArrayTrait, SpanTrait};
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};

#[test]
#[available_gas(2000000000)]
fn test_export_center_crop_pad_pad() {
    let input_0 = input_0::input_0();
    let z_0 = output_0::output_0();

    let y_0 = input_0
        .center_crop_pad(
            TensorTrait::new(array![3].span(), array![20, 10, 3].span()), Option::None(())
        );

    assert_eq(y_0, z_0);
}
