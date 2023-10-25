mod input_0; 
mod input_1; 
mod output_0; 


use orion::operators::nn::NNTrait;
use orion::numbers::FixedTrait;
use orion::operators::nn::FP16x16NN;
use orion::operators::tensor::FP16x16TensorPartialEq;
use orion::utils::assert_eq;

#[test]
#[available_gas(2000000000)]
fn test_gemm_default_no_bias() {
    let input_0 = input_0::input_0();
    let input_1 = input_1::input_1();
    let z = output_0::output_0();

    let y = NNTrait::gemm(input_0, input_1, Option::None(()), Option::None(()), Option::None(()), false, false);

    assert_eq(y, z);
}