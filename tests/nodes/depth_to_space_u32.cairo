mod input_0;
mod output_0;


use orion::operators::nn::NNTrait;
use orion::numbers::FixedTrait;
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::U32TensorPartialEq;
use orion::operators::nn::U32NN;

#[test]
#[available_gas(2000000000)]
fn test_depth_to_space_u32() {
    let input_0 = input_0::input_0();
    let z_0 = output_0::output_0();

    let y_0 = NNTrait::depth_to_space(@input_0, 2, 'CRD');

    assert_eq(y_0, z_0);
}
