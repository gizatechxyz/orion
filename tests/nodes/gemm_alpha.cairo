mod input_0;
mod input_1;
mod output_0;


use orion::operators::nn::NNTrait;
use orion::numbers::FixedTrait;
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::nn::FP16x16NN;
use orion::operators::tensor::FP16x16TensorPartialEq;

#[test]
#[available_gas(2000000000)]
fn test_gemm_alpha() {
    let input_0 = input_0::input_0();
    let input_1 = input_1::input_1();
    let z = output_0::output_0();

    let y = NNTrait::gemm(input_0, input_1, Option::None(()), Option::Some(FixedTrait::new(32768, false)), Option::None(()), false, false);

    assert_eq(y, z);
}
