mod input_0;
mod output_0;


use orion::operators::nn::NNTrait;
use orion::numbers::FixedTrait;
use orion::operators::nn::I8NN;
use orion::operators::tensor::I8TensorPartialEq;
use orion::utils::assert_eq;

#[test]
#[available_gas(2000000000)]
fn test_relu_i8() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = NNTrait::relu(@input_0);

    assert_eq(y, z);
}
