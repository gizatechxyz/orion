mod input_0;
mod output_0;


use orion::operators::tensor::U32Tensor;
use orion::operators::tensor::I32Tensor;
use orion::operators::tensor::I32TensorPartialEq;
use core::array::{ArrayTrait, SpanTrait};
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::{TensorTrait, Tensor};

#[test]
#[available_gas(2000000000)]
fn test_reverse_sequence_i32_2d_batch_equal_parts() {
    let input_0 = input_0::input_0();
    let z_0 = output_0::output_0();

    let y_0 = input_0
        .reverse_sequence(
            TensorTrait::<usize>::new(array![4].span(), array![1, 2, 3, 4].span()),
            Option::Some(0),
            Option::Some(1)
        );

    assert_eq(y_0, z_0);
}
