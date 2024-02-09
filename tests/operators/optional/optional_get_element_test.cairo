use core::debug::PrintTrait;
use core::array::{ArrayTrait, SpanTrait};
use core::option::OptionTrait;

use orion::operators::tensor::{TensorTrait, Tensor, I8Tensor, I32Tensor, U32Tensor, FP16x16Tensor};
use orion::numbers::{FP16x16, FP16x16Impl, FP32x32, FP32x32Impl, FixedTrait};
use orion::numbers::{NumberTrait};
use orion::operators::tensor::helpers::{optional_has_element, optional_get_element};

#[test]
#[available_gas(200000000000)]
fn optional_get_element_i8_test() {
    let a = TensorTrait::<
        i8
    >::new(
        shape: array![4, 2].span(),
        data: array![
            1_i8,
            2_i8,
            3_i8,
            4_i8,
            5_i8,
            6_i8,
            7_i8,
            8_i8
        ]
            .span(),
    );
    let ele = optional_get_element(a.optional());

    assert(*(ele.data).at(0) == *(a.data).at(0), 'ele[0] == a[0]');
    assert(*(ele.data).at(1) == *(a.data).at(1), 'ele[1] == a[1]');
    assert(*(ele.data).at(2) == *(a.data).at(2), 'ele[2] == a[2]');
    assert(*(ele.data).at(3) == *(a.data).at(3), 'ele[3] == a[3]');
    assert(*(ele.data).at(4) == *(a.data).at(4), 'ele[4] == a[4]');
    assert(*(ele.data).at(5) == *(a.data).at(5), 'ele[5] == a[5]');
    assert(*(ele.data).at(6) == *(a.data).at(6), 'ele[6] == a[6]');
    assert(*(ele.data).at(7) == *(a.data).at(7), 'ele[7] == a[7]');
}

#[test]
#[available_gas(200000000000)]
fn optional_get_element_fp16x16_test() {
    let a = TensorTrait::<
        FP16x16
    >::new(
        shape: array![4, 2].span(),
        data: array![
            FixedTrait::<FP16x16>::new_unscaled(1, false),
            FixedTrait::<FP16x16>::new_unscaled(2, false),
            FixedTrait::<FP16x16>::new_unscaled(3, false),
            FixedTrait::<FP16x16>::new_unscaled(4, false),
            FixedTrait::<FP16x16>::new_unscaled(5, false),
            FixedTrait::<FP16x16>::new_unscaled(6, false),
            FixedTrait::<FP16x16>::new_unscaled(7, false),
            FixedTrait::<FP16x16>::new_unscaled(8, false)
        ]
            .span(),
    );
    let ele = optional_get_element(a.optional());

    assert(*(ele.data).at(0) == *(a.data).at(0), 'ele[0] == a[0]');
    assert(*(ele.data).at(1) == *(a.data).at(1), 'ele[1] == a[1]');
    assert(*(ele.data).at(2) == *(a.data).at(2), 'ele[2] == a[2]');
    assert(*(ele.data).at(3) == *(a.data).at(3), 'ele[3] == a[3]');
    assert(*(ele.data).at(4) == *(a.data).at(4), 'ele[4] == a[4]');
    assert(*(ele.data).at(5) == *(a.data).at(5), 'ele[5] == a[5]');
    assert(*(ele.data).at(6) == *(a.data).at(6), 'ele[6] == a[6]');
    assert(*(ele.data).at(7) == *(a.data).at(7), 'ele[7] == a[7]');
}