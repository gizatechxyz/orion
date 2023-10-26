mod input_0; 
mod output_0; 


use orion::operators::nn::NNTrait;
use orion::numbers::FixedTrait;
use orion::operators::nn::FP8x23NN;
use orion::operators::tensor::FP8x23TensorPartialEq;
use orion::utils::assert_eq;

#[test]
#[available_gas(2000000000)]
fn test_hard_sigmoid_fp8x23() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = NNTrait::hard_sigmoid(@input_0, @FixedTrait::new(1677721, false), @FixedTrait::new(4194304, false));

    assert_eq(y, z);
}
