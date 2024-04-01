mod input_0;
mod input_1;
mod output_0;


use orion::operators::tensor::FP16x16TensorPartialEq;
use orion::operators::nn::FP16x16NN;
use orion::numbers::FixedTrait;
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::nn::NNTrait;
use orion::operators::nn::functional::roi_align::TRANSFORMATION_MODE;
use orion::numbers::FP16x16;
use orion::operators::tensor::{TensorTrait, U32Tensor};


#[test]
#[available_gas(2000000000)]
fn test_roi_align_aligned_true() {
    let input_0 = input_0::input_0();
    let input_1 = input_1::input_1();
    let z_0 = output_0::output_0();

    let y_0 = NNTrait::roi_align(
        @input_0,
        @input_1,
        @TensorTrait::new(array![3].span(), array![0, 0, 0].span()),
        Option::Some(TRANSFORMATION_MODE::HALF_PIXEL),
        Option::None,
        Option::Some(5),
        Option::Some(5),
        Option::Some(FP16x16 { mag: 131072, sign: false }),
        Option::Some(FP16x16 { mag: 65536, sign: false })
    );

    assert_eq(y_0, z_0);
}
