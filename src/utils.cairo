use core::traits::TryInto;
use option::OptionTrait;
use array::ArrayTrait;

use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
use orion::numbers::fixed_point::implementations::impl_8x23::{FP8x23Impl, ONE};

fn u32_max(a: u32, b: u32) -> u32 {
    if a > b {
        a
    } else {
        b
    }
}

fn fp8x23_to_i32(x: FixedType) -> i32 {
    let unscaled_mag = x.mag / ONE;
    return IntegerTrait::new(unscaled_mag.try_into().unwrap(), x.sign);
}

fn fp8x23_to_u32(x: FixedType) -> u32 {
    let unscaled_mag = x.mag / ONE;
    return unscaled_mag.try_into().unwrap();
}
