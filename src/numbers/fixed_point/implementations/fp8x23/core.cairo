use debug::PrintTrait;

use option::OptionTrait;
use result::{ResultTrait, ResultTraitImpl};
use traits::{TryInto, Into};

use orion::numbers::signed_integer::{i32::i32, i8::i8};
use orion::numbers::fixed_point::core::{FixedTrait};
use orion::numbers::fixed_point::implementations::fp8x23::math::{core, trig, hyp};
use orion::numbers::fixed_point::utils;

/// A struct representing a fixed point number.
#[derive(Serde, Copy, Drop)]
struct FP8x23 {
    mag: u32,
    sign: bool
}

// CONSTANTS

const TWO: u32 = 16777216; // 2 ** 24
const ONE: u32 = 8388608; // 2 ** 23
const HALF: u32 = 4194304; // 2 ** 22
const MAX: u32 = 2147483648; // 2 ** 31


impl FP8x23Impl of FixedTrait<FP8x23, u32> {
    fn ZERO() -> FP8x23 {
        return FP8x23 { mag: 0, sign: false };
    }

    fn ONE() -> FP8x23 {
        return FP8x23 { mag: ONE, sign: false };
    }

    fn MAX() -> FP8x23 {
        return FP8x23 { mag: MAX, sign: false };
    }

    fn new(mag: u32, sign: bool) -> FP8x23 {
        return FP8x23 { mag: mag, sign: sign };
    }

    fn new_unscaled(mag: u32, sign: bool) -> FP8x23 {
        return FP8x23 { mag: mag * ONE, sign: sign };
    }

    fn from_felt(val: felt252) -> FP8x23 {
        let mag = integer::u32_try_from_felt252(utils::felt_abs(val)).unwrap();
        return FixedTrait::new(mag, utils::felt_sign(val));
    }

    fn abs(self: FP8x23) -> FP8x23 {
        return core::abs(self);
    }

    fn acos(self: FP8x23) -> FP8x23 {
        return trig::acos_fast(self);
    }

    fn acos_fast(self: FP8x23) -> FP8x23 {
        return trig::acos_fast(self);
    }

    fn acosh(self: FP8x23) -> FP8x23 {
        return hyp::acosh(self);
    }

    fn asin(self: FP8x23) -> FP8x23 {
        return trig::asin_fast(self);
    }

    fn asin_fast(self: FP8x23) -> FP8x23 {
        return trig::asin_fast(self);
    }

    fn asinh(self: FP8x23) -> FP8x23 {
        return hyp::asinh(self);
    }

    fn atan(self: FP8x23) -> FP8x23 {
        return trig::atan_fast(self);
    }

    fn atan_fast(self: FP8x23) -> FP8x23 {
        return trig::atan_fast(self);
    }

    fn atanh(self: FP8x23) -> FP8x23 {
        return hyp::atanh(self);
    }

    fn ceil(self: FP8x23) -> FP8x23 {
        return core::ceil(self);
    }

    fn cos(self: FP8x23) -> FP8x23 {
        return trig::cos_fast(self);
    }

    fn cos_fast(self: FP8x23) -> FP8x23 {
        return trig::cos_fast(self);
    }

    fn cosh(self: FP8x23) -> FP8x23 {
        return hyp::cosh(self);
    }

    fn floor(self: FP8x23) -> FP8x23 {
        return core::floor(self);
    }

    // Calculates the natural exponent of x: e^x
    fn exp(self: FP8x23) -> FP8x23 {
        return core::exp(self);
    }

    // Calculates the binary exponent of x: 2^x
    fn exp2(self: FP8x23) -> FP8x23 {
        return core::exp2(self);
    }

    // Calculates the natural logarithm of x: ln(x)
    // self must be greater than zero
    fn ln(self: FP8x23) -> FP8x23 {
        return core::ln(self);
    }

    // Calculates the binary logarithm of x: log2(x)
    // self must be greather than zero
    fn log2(self: FP8x23) -> FP8x23 {
        return core::log2(self);
    }

    // Calculates the base 10 log of x: log10(x)
    // self must be greater than zero
    fn log10(self: FP8x23) -> FP8x23 {
        return core::log10(self);
    }

    // Calclates the value of x^y and checks for overflow before returning
    // self is a fixed point value
    // b is a fixed point value
    fn pow(self: FP8x23, b: FP8x23) -> FP8x23 {
        return core::pow(self, b);
    }

    fn round(self: FP8x23) -> FP8x23 {
        return core::round(self);
    }

    fn sin(self: FP8x23) -> FP8x23 {
        return trig::sin_fast(self);
    }

    fn sin_fast(self: FP8x23) -> FP8x23 {
        return trig::sin_fast(self);
    }

    fn sinh(self: FP8x23) -> FP8x23 {
        return hyp::sinh(self);
    }

    // Calculates the square root of a fixed point value
    // x must be positive
    fn sqrt(self: FP8x23) -> FP8x23 {
        return core::sqrt(self);
    }

    fn tan(self: FP8x23) -> FP8x23 {
        return trig::tan_fast(self);
    }

    fn tan_fast(self: FP8x23) -> FP8x23 {
        return trig::tan_fast(self);
    }

    fn tanh(self: FP8x23) -> FP8x23 {
        return hyp::tanh(self);
    }
}


impl FP8x23Print of PrintTrait<FP8x23> {
    fn print(self: FP8x23) {
        self.sign.print();
        self.mag.print();
    }
}

// Into a raw felt without unscaling
impl FP8x23IntoFelt252 of Into<FP8x23, felt252> {
    fn into(self: FP8x23) -> felt252 {
        let mag_felt = self.mag.into();

        if self.sign {
            return mag_felt * -1;
        } else {
            return mag_felt * 1;
        }
    }
}

impl FP8x23TryIntoU128 of TryInto<FP8x23, u128> {
    fn try_into(self: FP8x23) -> Option<u128> {
        if self.sign {
            return Option::None(());
        } else {
            // Unscale the magnitude and round down
            return Option::Some((self.mag / ONE).into());
        }
    }
}

impl FP8x23TryIntoU64 of TryInto<FP8x23, u64> {
    fn try_into(self: FP8x23) -> Option<u64> {
        if self.sign {
            return Option::None(());
        } else {
            // Unscale the magnitude and round down
            return Option::Some((self.mag / ONE).into());
        }
    }
}

impl FP8x23TryIntoU32 of TryInto<FP8x23, u32> {
    fn try_into(self: FP8x23) -> Option<u32> {
        if self.sign {
            return Option::None(());
        } else {
            // Unscale the magnitude and round down
            return Option::Some(self.mag / ONE);
        }
    }
}

impl FP8x23TryIntoU16 of TryInto<FP8x23, u16> {
    fn try_into(self: FP8x23) -> Option<u16> {
        if self.sign {
            Option::None(())
        } else {
            // Unscale the magnitude and round down
            return (self.mag / ONE).try_into();
        }
    }
}

impl FP8x23TryIntoU8 of TryInto<FP8x23, u8> {
    fn try_into(self: FP8x23) -> Option<u8> {
        if self.sign {
            Option::None(())
        } else {
            // Unscale the magnitude and round down
            return (self.mag / ONE).try_into();
        }
    }
}

impl FP8x23IntoI32 of Into<FP8x23, i32> {
    fn into(self: FP8x23) -> i32 {
        _i32_into_fp(self)
    }
}

impl FP8x23TryIntoI8 of TryInto<FP8x23, i8> {
    fn try_into(self: FP8x23) -> Option<i8> {
        _i8_try_from_fp(self)
    }
}

impl FP8x23PartialEq of PartialEq<FP8x23> {
    #[inline(always)]
    fn eq(lhs: @FP8x23, rhs: @FP8x23) -> bool {
        return core::eq(lhs, rhs);
    }

    #[inline(always)]
    fn ne(lhs: @FP8x23, rhs: @FP8x23) -> bool {
        return core::ne(lhs, rhs);
    }
}

impl FP8x23Add of Add<FP8x23> {
    fn add(lhs: FP8x23, rhs: FP8x23) -> FP8x23 {
        return core::add(lhs, rhs);
    }
}

impl FP8x23AddEq of AddEq<FP8x23> {
    #[inline(always)]
    fn add_eq(ref self: FP8x23, other: FP8x23) {
        self = Add::add(self, other);
    }
}

impl FP8x23Sub of Sub<FP8x23> {
    fn sub(lhs: FP8x23, rhs: FP8x23) -> FP8x23 {
        return core::sub(lhs, rhs);
    }
}

impl FP8x23SubEq of SubEq<FP8x23> {
    #[inline(always)]
    fn sub_eq(ref self: FP8x23, other: FP8x23) {
        self = Sub::sub(self, other);
    }
}

impl FP8x23Mul of Mul<FP8x23> {
    fn mul(lhs: FP8x23, rhs: FP8x23) -> FP8x23 {
        return core::mul(lhs, rhs);
    }
}

impl FP8x23MulEq of MulEq<FP8x23> {
    #[inline(always)]
    fn mul_eq(ref self: FP8x23, other: FP8x23) {
        self = Mul::mul(self, other);
    }
}

impl FP8x23Div of Div<FP8x23> {
    fn div(lhs: FP8x23, rhs: FP8x23) -> FP8x23 {
        return core::div(lhs, rhs);
    }
}

impl FP8x23DivEq of DivEq<FP8x23> {
    #[inline(always)]
    fn div_eq(ref self: FP8x23, other: FP8x23) {
        self = Div::div(self, other);
    }
}

impl FP8x23PartialOrd of PartialOrd<FP8x23> {
    #[inline(always)]
    fn ge(lhs: FP8x23, rhs: FP8x23) -> bool {
        return core::ge(lhs, rhs);
    }

    #[inline(always)]
    fn gt(lhs: FP8x23, rhs: FP8x23) -> bool {
        return core::gt(lhs, rhs);
    }

    #[inline(always)]
    fn le(lhs: FP8x23, rhs: FP8x23) -> bool {
        return core::le(lhs, rhs);
    }

    #[inline(always)]
    fn lt(lhs: FP8x23, rhs: FP8x23) -> bool {
        return core::lt(lhs, rhs);
    }
}

impl FP8x23Neg of Neg<FP8x23> {
    #[inline(always)]
    fn neg(a: FP8x23) -> FP8x23 {
        return core::neg(a);
    }
}

impl FP8x23Rem of Rem<FP8x23> {
    #[inline(always)]
    fn rem(lhs: FP8x23, rhs: FP8x23) -> FP8x23 {
        return core::rem(lhs, rhs);
    }
}

/// INTERNAL

fn _i32_into_fp(x: FP8x23) -> i32 {
    i32 { mag: x.mag / ONE, sign: x.sign }
}

fn _i8_try_from_fp(x: FP8x23) -> Option<i8> {
    let unscaled_mag: Option<u8> = (x.mag / ONE).try_into();

    match unscaled_mag {
        Option::Some(val) => Option::Some(i8 { mag: unscaled_mag.unwrap(), sign: x.sign }),
        Option::None(_) => Option::None(())
    }
}
