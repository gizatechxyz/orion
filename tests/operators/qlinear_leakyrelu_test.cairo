use core::debug::PrintTrait;
use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, I8Tensor, I32Tensor, U32Tensor, FP16x16Tensor};
use orion::numbers::{FP16x16, FP16x16Impl, FP32x32, FP32x32Impl, FixedTrait};
use orion::numbers::{NumberTrait};

#[test]
#[available_gas(200000000000)]
fn qlinear_leakyrelu_test() {
    let a = TensorTrait::<
        i8
    >::new(
        shape: array![2, 3].span(),
        data: array![
            -10_i8,
            -10_i8,
            -10_i8,
            10_i8,
            10_i8,
            10_i8
        ]
            .span(),
    );

    let a_scale = TensorTrait::<
        FP16x16
    >::new(
        shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(131072, false)].span(),
    );
    let a_zero_point = TensorTrait::<
        FP16x16
    >::new(
        shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(131072, false)].span(),
    );

    let alpha = FixedTrait::<FP16x16>::new(655360, false);

    let actual_output = a.qlinear_leakyrelu(@a_scale, @a_zero_point, alpha);

    assert((*actual_output.data[0]).abs().into() == 118, '*result[0] == 118');
    assert((*actual_output.data[1]).abs().into() == 118, '*result[1] == 118');
    assert((*actual_output.data[2]).abs().into() == 118, '*result[2] == 118');
    assert((*actual_output.data[3]).into() == 10, '*result[3] == 10');
    assert((*actual_output.data[4]).into() == 10, '*result[4] == 10');
    assert((*actual_output.data[5]).into() == 10, '*result[5] == 10');
}

