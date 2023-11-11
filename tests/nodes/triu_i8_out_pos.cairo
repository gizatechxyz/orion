mod input_0;
mod output_0;


use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::TensorTrait;
use orion::operators::tensor::I8Tensor;
use orion::operators::tensor::I8TensorPartialEq;
use orion::utils::assert_eq;

#[test]
#[available_gas(2000000000)]
fn test_triu_i8_out_pos() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = input_0.trilu(true, 6);

    assert_eq(y, z);
}
