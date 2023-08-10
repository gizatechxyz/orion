use debug::PrintTrait;

use option::OptionTrait;
use result::{ResultTrait, ResultTraitImpl};
use traits::{TryInto, Into};

use orion::numbers::signed_integer::{i32::i32, i8::i8};
use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
use orion::numbers::fixed_point::implementations::fp8x23::math::{core, trig, hyp};
use orion::numbers::fixed_point::utils;

// CONSTANTS

const TWO: u32 = 16777216; // 2 ** 24
const ONE: u32 = 8388608; // 2 ** 23
const HALF: u32 = 4194304; // 2 ** 22
const MAX: u32 = 2147483648; // 2 ** 31


impl FP8x23Impl of FixedTrait {
    fn ZERO() -> FixedType {
        return FixedType { mag: 0, sign: false };
    }

    fn ONE() -> FixedType {
        return FixedType { mag: ONE, sign: false };
    }

    fn new(mag: u32, sign: bool) -> FixedType {
        return FixedType { mag: mag, sign: sign };
    }

    fn new_unscaled(mag: u32, sign: bool) -> FixedType {
        return FixedType { mag: mag * ONE, sign: sign };
    }

    fn from_felt(val: felt252) -> FixedType {
        let mag = integer::u32_try_from_felt252(utils::felt_abs(val)).unwrap();
        return FixedTrait::new(mag, utils::felt_sign(val));
    }

    fn abs(self: FixedType) -> FixedType {
        return core::abs(self);
    }

    fn acos(self: FixedType) -> FixedType {
        return trig::acos(self);
    }

    fn acos_fast(self: FixedType) -> FixedType {
        return trig::acos_fast(self);
    }

    fn acosh(self: FixedType) -> FixedType {
        return hyp::acosh(self);
    }

    fn asin(self: FixedType) -> FixedType {
        return trig::asin(self);
    }

    fn asin_fast(self: FixedType) -> FixedType {
        return trig::asin_fast(self);
    }

    fn asinh(self: FixedType) -> FixedType {
        return hyp::asinh(self);
    }

    fn atan(self: FixedType) -> FixedType {
        return trig::atan(self);
    }

    fn atan_fast(self: FixedType) -> FixedType {
        return trig::atan_fast(self);
    }

    fn atanh(self: FixedType) -> FixedType {
        return hyp::atanh(self);
    }

    fn ceil(self: FixedType) -> FixedType {
        return core::ceil(self);
    }

    fn cos(self: FixedType) -> FixedType {
        return trig::cos(self);
    }

    fn cos_fast(self: FixedType) -> FixedType {
        return trig::cos_fast(self);
    }

    fn cosh(self: FixedType) -> FixedType {
        return hyp::cosh(self);
    }

    fn floor(self: FixedType) -> FixedType {
        return core::floor(self);
    }

    // Calculates the natural exponent of x: e^x
    fn exp(self: FixedType) -> FixedType {
        return core::exp(self);
    }

    // Calculates the binary exponent of x: 2^x
    fn exp2(self: FixedType) -> FixedType {
        return core::exp2(self);
    }

    // Calculates the natural logarithm of x: ln(x)
    // self must be greater than zero
    fn ln(self: FixedType) -> FixedType {
        return core::ln(self);
    }

    // Calculates the binary logarithm of x: log2(x)
    // self must be greather than zero
    fn log2(self: FixedType) -> FixedType {
        return core::log2(self);
    }

    // Calculates the base 10 log of x: log10(x)
    // self must be greater than zero
    fn log10(self: FixedType) -> FixedType {
        return core::log10(self);
    }

    // Calclates the value of x^y and checks for overflow before returning
    // self is a fixed point value
    // b is a fixed point value
    fn pow(self: FixedType, b: FixedType) -> FixedType {
        return core::pow(self, b);
    }

    fn round(self: FixedType) -> FixedType {
        return core::round(self);
    }

    fn sin(self: FixedType) -> FixedType {
        return trig::sin(self);
    }

    fn sin_fast(self: FixedType) -> FixedType {
        return trig::sin_fast(self);
    }

    fn sinh(self: FixedType) -> FixedType {
        return hyp::sinh(self);
    }

    // Calculates the square root of a fixed point value
    // x must be positive
    fn sqrt(self: FixedType) -> FixedType {
        return core::sqrt(self);
    }

    fn tan(self: FixedType) -> FixedType {
        return trig::tan(self);
    }

    fn tan_fast(self: FixedType) -> FixedType {
        return trig::tan_fast(self);
    }

    fn tanh(self: FixedType) -> FixedType {
        return hyp::tanh(self);
    }
}


impl FP8x23Print of PrintTrait<FixedType> {
    fn print(self: FixedType) {
        self.sign.print();
        self.mag.print();
    }
}

// Into a raw felt without unscaling
impl FP8x23IntoFelt252 of Into<FixedType, felt252> {
    fn into(self: FixedType) -> felt252 {
        let mag_felt = self.mag.into();

        if self.sign {
            return mag_felt * -1;
        } else {
            return mag_felt * 1;
        }
    }
}

impl FP8x23TryIntoU128 of TryInto<FixedType, u128> {
    fn try_into(self: FixedType) -> Option<u128> {
        if self.sign {
            return Option::None(());
        } else {
            // Unscale the magnitude and round down
            return Option::Some((self.mag / ONE).into());
        }
    }
}

impl FP8x23TryIntoU64 of TryInto<FixedType, u64> {
    fn try_into(self: FixedType) -> Option<u64> {
        if self.sign {
            return Option::None(());
        } else {
            // Unscale the magnitude and round down
            return Option::Some((self.mag / ONE).into());
        }
    }
}

impl FP8x23TryIntoU32 of TryInto<FixedType, u32> {
    fn try_into(self: FixedType) -> Option<u32> {
        if self.sign {
            return Option::None(());
        } else {
            // Unscale the magnitude and round down
            return Option::Some(self.mag / ONE);
        }
    }
}

impl FP8x23TryIntoU16 of TryInto<FixedType, u16> {
    fn try_into(self: FixedType) -> Option<u16> {
        if self.sign {
            Option::None(())
        } else {
            // Unscale the magnitude and round down
            return (self.mag / ONE).try_into();
        }
    }
}

impl FP8x23TryIntoU8 of TryInto<FixedType, u8> {
    fn try_into(self: FixedType) -> Option<u8> {
        if self.sign {
            Option::None(())
        } else {
            // Unscale the magnitude and round down
            return (self.mag / ONE).try_into();
        }
    }
}

impl FP8x23IntoI32 of Into<FixedType, i32> {
    fn into(self: FixedType) -> i32 {
        _i32_into_fp(self)
    }
}

impl FP8x23TryIntoI8 of TryInto<FixedType, i8> {
    fn try_into(self: FixedType) -> Option<i8> {
        _i8_try_from_fp(self)
    }
}

impl FP8x23PartialEq of PartialEq<FixedType> {
    #[inline(always)]
    fn eq(lhs: @FixedType, rhs: @FixedType) -> bool {
        return core::eq(lhs, rhs);
    }

    #[inline(always)]
    fn ne(lhs: @FixedType, rhs: @FixedType) -> bool {
        return core::ne(lhs, rhs);
    }
}

impl FP8x23Add of Add<FixedType> {
    fn add(lhs: FixedType, rhs: FixedType) -> FixedType {
        return core::add(lhs, rhs);
    }
}

impl FP8x23AddEq of AddEq<FixedType> {
    #[inline(always)]
    fn add_eq(ref self: FixedType, other: FixedType) {
        self = Add::add(self, other);
    }
}

impl FP8x23Sub of Sub<FixedType> {
    fn sub(lhs: FixedType, rhs: FixedType) -> FixedType {
        return core::sub(lhs, rhs);
    }
}

impl FP8x23SubEq of SubEq<FixedType> {
    #[inline(always)]
    fn sub_eq(ref self: FixedType, other: FixedType) {
        self = Sub::sub(self, other);
    }
}

impl FP8x23Mul of Mul<FixedType> {
    fn mul(lhs: FixedType, rhs: FixedType) -> FixedType {
        return core::mul(lhs, rhs);
    }
}

impl FP8x23MulEq of MulEq<FixedType> {
    #[inline(always)]
    fn mul_eq(ref self: FixedType, other: FixedType) {
        self = Mul::mul(self, other);
    }
}

impl FP8x23Div of Div<FixedType> {
    fn div(lhs: FixedType, rhs: FixedType) -> FixedType {
        return core::div(lhs, rhs);
    }
}

impl FP8x23DivEq of DivEq<FixedType> {
    #[inline(always)]
    fn div_eq(ref self: FixedType, other: FixedType) {
        self = Div::div(self, other);
    }
}

impl FP8x23PartialOrd of PartialOrd<FixedType> {
    #[inline(always)]
    fn ge(lhs: FixedType, rhs: FixedType) -> bool {
        return core::ge(lhs, rhs);
    }

    #[inline(always)]
    fn gt(lhs: FixedType, rhs: FixedType) -> bool {
        return core::gt(lhs, rhs);
    }

    #[inline(always)]
    fn le(lhs: FixedType, rhs: FixedType) -> bool {
        return core::le(lhs, rhs);
    }

    #[inline(always)]
    fn lt(lhs: FixedType, rhs: FixedType) -> bool {
        return core::lt(lhs, rhs);
    }
}

impl FP8x23Neg of Neg<FixedType> {
    #[inline(always)]
    fn neg(a: FixedType) -> FixedType {
        return core::neg(a);
    }
}

impl FP8x23Rem of Rem<FixedType> {
    #[inline(always)]
    fn rem(lhs: FixedType, rhs: FixedType) -> FixedType {
        return core::rem(lhs, rhs);
    }
}

/// INTERNAL

fn _i32_into_fp(x: FixedType) -> i32 {
    i32 { mag: x.mag / ONE, sign: x.sign }
}

fn _i8_try_from_fp(x: FixedType) -> Option<i8> {
    let unscaled_mag: Option<u8> = (x.mag / ONE).try_into();

    match unscaled_mag {
        Option::Some(val) => Option::Some(i8 { mag: unscaled_mag.unwrap(), sign: x.sign }),
        Option::None(_) => Option::None(())
    }
}
