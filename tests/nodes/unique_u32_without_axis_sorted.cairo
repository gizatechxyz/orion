mod input_0;
mod output_0;
mod output_1;
mod output_2;
mod output_3;


use orion::operators::tensor::U32TensorPartialEq;
use orion::operators::tensor::I32Tensor;
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::I32TensorPartialEq;
use orion::operators::tensor::U32Tensor;
use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};

#[test]
#[available_gas(2000000000)]
fn test_unique_u32_without_axis_sorted() {
    let input_0 = input_0::input_0();
    let z_0 = output_0::output_0();
    let z_1 = output_1::output_1();
    let z_2 = output_2::output_2();
    let z_3 = output_3::output_3();

    let (y_0, y_1, y_2, y_3) = input_0.unique(Option::None(()), Option::None(()));

    assert_eq(y_0, z_0);
    assert_eq(y_1, z_1);
    assert_eq(y_2, z_2);
    assert_eq(y_3, z_3);
}
