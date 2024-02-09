use core::debug::PrintTrait;
use core::array::{ArrayTrait, SpanTrait};
use core::option::OptionTrait;

use orion::operators::tensor::{TensorTrait, Tensor, I8Tensor, I32Tensor, U32Tensor, FP16x16Tensor};
use orion::numbers::{FP16x16, FP16x16Impl, FP32x32, FP32x32Impl, FixedTrait};
use orion::numbers::{NumberTrait};
use orion::operators::tensor::helpers::{optional_has_element, optional_get_element};

#[test]
#[available_gas(200000000000)]
fn optional_has_element_i8_test() {
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
    let a_optional = a.optional();
    let has_ele = optional_has_element(a_optional);

    assert(*(has_ele.data).at(0) == true, 'has_ele[0] == true');
}

#[test]
#[available_gas(200000000000)]
fn optional_has_element_fp16x16_test() {
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
    let a_optional = a.optional();
    let has_ele = optional_has_element(a_optional);

    assert(*(has_ele.data).at(0) == true, 'has_ele[0] == true');
}

#[test]
#[available_gas(200000000000)]
fn optional_has_element_none_test() {
    let a: Option<Tensor<u32>> = Option::None(());
    let has_ele = optional_has_element(a);

    assert(*(has_ele.data).at(0) == false, 'has_ele[0] == false');
}