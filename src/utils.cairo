use core::traits::TryInto;
use option::OptionTrait;
use array::ArrayTrait;

use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
use orion::numbers::fixed_point::implementations::impl_8x23::fp8x23;
use orion::numbers::fixed_point::implementations::impl_8x23;

// Fake macro to compute gas left
// TODO: Remove when automatically handled by compiler.
#[inline(always)]
fn check_gas() {
    match gas::withdraw_gas_all(get_builtin_costs()) {
        Option::Some(_) => {},
        Option::None(_) => {
            let mut data = ArrayTrait::new();
            data.append('Out of gas');
            panic(data);
        }
    }
}

fn u32_max(a: u32, b: u32) -> u32 {
    if a > b {
        a
    } else {
        b
    }
}

fn fp8x23_to_i32(x: FixedType<fp8x23>) -> i32 {
    let unscaled_mag = x.mag / impl_8x23::ONE;
    return IntegerTrait::new(unscaled_mag.try_into().unwrap(), x.sign);
}

fn fp8x23_to_u32(x: FixedType<fp8x23>) -> u32 {
    let unscaled_mag = x.mag / impl_8x23::ONE;
    return unscaled_mag.try_into().unwrap();
}
