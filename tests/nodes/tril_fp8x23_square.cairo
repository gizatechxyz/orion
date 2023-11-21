mod input_0;
mod output_0;


use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::TensorTrait;
use orion::operators::tensor::FP8x23Tensor;
use orion::operators::tensor::FP8x23TensorPartialEq;
use orion::utils::assert_eq;

#[test]
#[available_gas(2000000000)]
fn test_tril_fp8x23_square() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = input_0.trilu(false, 0);

    assert_eq(y, z);
}
