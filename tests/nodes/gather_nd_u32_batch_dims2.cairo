mod input_0;
mod input_1;
mod output_0;


use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::U32Tensor;
use orion::operators::tensor::{TensorTrait, Tensor};
use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::U32TensorPartialEq;

#[test]
#[available_gas(2000000000)]
fn test_gather_nd_u32_batch_dims2() {
    let input_0 = input_0::input_0();
    let input_1 = input_1::input_1();
    let z = output_0::output_0();

    let y = input_0.gather_nd(indices:input_1, batch_dims:Option::Some(2));

    assert_eq(y, z);
}
