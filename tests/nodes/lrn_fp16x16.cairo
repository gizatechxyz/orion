mod input_0;
mod output_0;


use orion::operators::tensor::FP16x16TensorPartialEq;
use orion::operators::nn::NNTrait;
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::nn::FP16x16NN;
use orion::numbers::{FixedTrait, FP16x16};

#[test]
#[available_gas(2000000000)]
fn test_lrn_fp16x16() {
    let input_0 = input_0::input_0();
    let z_0 = output_0::output_0();

    let y_0 = NNTrait::lrn(@input_0, 3, Option::Some(FP16x16 { mag: 13, sign: false }),Option::Some(FP16x16 { mag: 32768, sign: false }),Option::Some(FP16x16 { mag: 131072, sign: false }));

    assert_eq(y_0, z_0);
}
