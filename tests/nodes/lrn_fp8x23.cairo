mod input_0;
mod output_0;


use orion::operators::nn::FP8x23NN;
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::nn::NNTrait;
use orion::operators::tensor::FP8x23TensorPartialEq;
use orion::numbers::{FixedTrait, FP8x23};

#[test]
#[available_gas(2000000000)]
fn test_lrn_fp8x23() {
    let input_0 = input_0::input_0();
    let z_0 = output_0::output_0();

    let y_0 = NNTrait::lrn(@input_0, 3, Option::Some(FP8x23 { mag: 1677, sign: false }),Option::Some(FP8x23 { mag: 4194304, sign: false }),Option::Some(FP8x23 { mag: 16777216, sign: false }));

    assert_eq(y_0, z_0);
}
