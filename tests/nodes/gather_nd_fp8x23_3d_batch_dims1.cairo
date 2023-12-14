mod input_0;
mod input_1;
mod output_0;


use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::U32Tensor;
use orion::operators::tensor::FP8x23Tensor;
use orion::operators::tensor::{TensorTrait, Tensor};
use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::U32TensorPartialEq;
use orion::operators::tensor::FP8x23TensorPartialEq;

#[test]
#[available_gas(2000000000)]
fn test_gather_nd_fp8x23_3d_batch_dims1() {
    let input_0 = input_0::input_0();
    let input_1 = input_1::input_1();
    let z = output_0::output_0();

    let y = input_0.gather_nd(indices:input_1, batch_dims:Option::Some(1));

    assert_eq(y, z);
}
