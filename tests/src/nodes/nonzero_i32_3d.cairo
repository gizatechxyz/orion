mod input_0;
mod output_0;


use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::TensorTrait;
use orion::operators::tensor::I32Tensor;
use orion::operators::tensor::U32TensorPartialEq;
use orion::utils::assert_eq;

#[test]
#[available_gas(2000000000)]
fn test_nonzero_i32_3d() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = input_0.nonzero();

    assert_eq(y, z);
}
