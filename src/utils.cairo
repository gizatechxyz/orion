use core::traits::TryInto;
use option::OptionTrait;
use array::ArrayTrait;
use array::SpanTrait;

use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
use orion::numbers::fixed_point::implementations::impl_8x23::{FP8x23Impl, ONE as ONE_8x23};
use orion::numbers::fixed_point::implementations::impl_16x16::{FP16x16Impl, ONE as ONE_16x16};

fn u32_max(a: u32, b: u32) -> u32 {
    if a > b {
        a
    } else {
        b
    }
}


fn saturate<T, impl TCopy: Copy<T>, impl TDrop: Drop<T>, impl PartialOrdT: PartialOrd<T>>(
    min: T, max: T, x: T
) -> T {
    if x < min {
        min
    } else if x > max {
        max
    } else {
        x
    }
}

