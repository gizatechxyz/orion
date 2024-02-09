use core::debug::PrintTrait;

use core::option::OptionTrait;
use core::result::{ResultTrait, ResultTraitImpl};
use core::traits::{TryInto, Into};

use orion::numbers::{fixed_point::core::{FixedTrait}, FP8x23};
use orion::numbers::fixed_point::implementations::fp8x23wide::math::{
    core as core_math, trig, hyp, erf
};
use orion::numbers::fixed_point::utils;

/// A struct representing a fixed point number.
#[derive(Serde, Copy, Drop)]
struct FP8x23W {
    mag: u64,
    sign: bool
}

// CONSTANTS

const TWO: u64 = 16777216; // 2 ** 24
const ONE: u64 = 8388608; // 2 ** 23
const HALF: u64 = 4194304; // 2 ** 22
const MAX: u64 = 2147483648; // 2 ** 31


impl FP8x23WImpl of FixedTrait<FP8x23W, u64> {
    fn ZERO() -> FP8x23W {
        return FP8x23W { mag: 0, sign: false };
    }

    fn HALF() -> FP8x23W {
        return FP8x23W { mag: HALF, sign: false };
    }

    fn ONE() -> FP8x23W {
        return FP8x23W { mag: ONE, sign: false };
    }

    fn MAX() -> FP8x23W {
        return FP8x23W { mag: MAX, sign: false };
    }

    fn new(mag: u64, sign: bool) -> FP8x23W {
        return FP8x23W { mag: mag, sign: sign };
    }

    fn new_unscaled(mag: u64, sign: bool) -> FP8x23W {
        return FP8x23W { mag: mag * ONE, sign: sign };
    }

    fn from_felt(val: felt252) -> FP8x23W {
        let mag = core::integer::u64_try_from_felt252(utils::felt_abs(val)).unwrap();
        return FixedTrait::new(mag, utils::felt_sign(val));
    }

    fn abs(self: FP8x23W) -> FP8x23W {
        return core_math::abs(self);
    }

    fn acos(self: FP8x23W) -> FP8x23W {
        return trig::acos_fast(self);
    }

    fn acos_fast(self: FP8x23W) -> FP8x23W {
        return trig::acos_fast(self);
    }

    fn acosh(self: FP8x23W) -> FP8x23W {
        return hyp::acosh(self);
    }

    fn asin(self: FP8x23W) -> FP8x23W {
        return trig::asin_fast(self);
    }

    fn asin_fast(self: FP8x23W) -> FP8x23W {
        return trig::asin_fast(self);
    }

    fn asinh(self: FP8x23W) -> FP8x23W {
        return hyp::asinh(self);
    }

    fn atan(self: FP8x23W) -> FP8x23W {
        return trig::atan_fast(self);
    }

    fn atan_fast(self: FP8x23W) -> FP8x23W {
        return trig::atan_fast(self);
    }

    fn atanh(self: FP8x23W) -> FP8x23W {
        return hyp::atanh(self);
    }

    fn ceil(self: FP8x23W) -> FP8x23W {
        return core_math::ceil(self);
    }

    fn cos(self: FP8x23W) -> FP8x23W {
        return trig::cos_fast(self);
    }

    fn cos_fast(self: FP8x23W) -> FP8x23W {
        return trig::cos_fast(self);
    }

    fn cosh(self: FP8x23W) -> FP8x23W {
        return hyp::cosh(self);
    }

    fn floor(self: FP8x23W) -> FP8x23W {
        return core_math::floor(self);
    }

    // Calculates the natural exponent of x: e^x
    fn exp(self: FP8x23W) -> FP8x23W {
        return core_math::exp(self);
    }

    // Calculates the binary exponent of x: 2^x
    fn exp2(self: FP8x23W) -> FP8x23W {
        return core_math::exp2(self);
    }

    // Calculates the natural logarithm of x: ln(x)
    // self must be greater than zero
    fn ln(self: FP8x23W) -> FP8x23W {
        return core_math::ln(self);
    }

    // Calculates the binary logarithm of x: log2(x)
    // self must be greather than zero
    fn log2(self: FP8x23W) -> FP8x23W {
        return core_math::log2(self);
    }

    // Calculates the base 10 log of x: log10(x)
    // self must be greater than zero
    fn log10(self: FP8x23W) -> FP8x23W {
        return core_math::log10(self);
    }

    // Calclates the value of x^y and checks for overflow before returning
    // self is a fixed point value
    // b is a fixed point value
    fn pow(self: FP8x23W, b: FP8x23W) -> FP8x23W {
        return core_math::pow(self, b);
    }

    fn round(self: FP8x23W) -> FP8x23W {
        return core_math::round(self);
    }

    fn sin(self: FP8x23W) -> FP8x23W {
        return trig::sin_fast(self);
    }

    fn sin_fast(self: FP8x23W) -> FP8x23W {
        return trig::sin_fast(self);
    }

    fn sinh(self: FP8x23W) -> FP8x23W {
        return hyp::sinh(self);
    }

    // Calculates the square root of a fixed point value
    // x must be positive
    fn sqrt(self: FP8x23W) -> FP8x23W {
        return core_math::sqrt(self);
    }

    fn tan(self: FP8x23W) -> FP8x23W {
        return trig::tan_fast(self);
    }

    fn tan_fast(self: FP8x23W) -> FP8x23W {
        return trig::tan_fast(self);
    }

    fn tanh(self: FP8x23W) -> FP8x23W {
        return hyp::tanh(self);
    }

    fn sign(self: FP8x23W) -> FP8x23W {
        return core_math::sign(self);
    }

    fn NaN() -> FP8x23W {
        return FP8x23W { mag: 0, sign: true };
    }

    fn is_nan(self: FP8x23W) -> bool {
        self == FP8x23W { mag: 0, sign: true }
    }

    fn INF() -> FP8x23W {
        return FP8x23W { mag: 4294967295, sign: false };
    }

    fn POS_INF() -> FP8x23W {
        return FP8x23W { mag: 4294967295, sign: false };
    }

    fn NEG_INF() -> FP8x23W {
        return FP8x23W { mag: 4294967295, sign: true };
    }

    fn is_inf(self: FP8x23W) -> bool {
        self.mag == 4294967295
    }

    fn is_pos_inf(self: FP8x23W) -> bool {
        self.is_inf() && !self.sign
    }

    fn is_neg_inf(self: FP8x23W) -> bool {
        self.is_inf() && self.sign
    }

    fn erf(self: FP8x23W) -> FP8x23W {
        return erf::erf(self);
    }
}


impl FP8x23WPrint of PrintTrait<FP8x23W> {
    fn print(self: FP8x23W) {
        self.sign.print();
        self.mag.print();
    }
}

// Into a raw felt without unscaling
impl FP8x23WIntoFelt252 of Into<FP8x23W, felt252> {
    fn into(self: FP8x23W) -> felt252 {
        let mag_felt = self.mag.into();

        if self.sign {
            return mag_felt * -1;
        } else {
            return mag_felt * 1;
        }
    }
}

impl FP8x23IntoFP8x23W of Into<FP8x23, FP8x23W> {
    fn into(self: FP8x23) -> FP8x23W {
        FP8x23W { mag: self.mag.into(), sign: self.sign }
    }
}

impl FP8x23WTryIntoFP8x23 of TryInto<FP8x23W, FP8x23> {
    fn try_into(self: FP8x23W) -> Option<FP8x23> {
        match self.mag.try_into() {
            Option::Some(val) => { Option::Some(FP8x23 { mag: val, sign: self.sign }) },
            Option::None(_) => { Option::None(()) }
        }
    }
}

impl FP8x23WTryIntoU128 of TryInto<FP8x23W, u128> {
    fn try_into(self: FP8x23W) -> Option<u128> {
        if self.sign {
            return Option::None(());
        } else {
            // Unscale the magnitude and round down
            return Option::Some((self.mag / ONE).into());
        }
    }
}

impl FP8x23WTryIntoU64 of TryInto<FP8x23W, u64> {
    fn try_into(self: FP8x23W) -> Option<u64> {
        if self.sign {
            return Option::None(());
        } else {
            // Unscale the magnitude and round down
            return Option::Some((self.mag / ONE).into());
        }
    }
}


impl FP8x23WTryIntoU16 of TryInto<FP8x23W, u16> {
    fn try_into(self: FP8x23W) -> Option<u16> {
        if self.sign {
            Option::None(())
        } else {
            // Unscale the magnitude and round down
            return (self.mag / ONE).try_into();
        }
    }
}

impl FP8x23WTryIntoU8 of TryInto<FP8x23W, u8> {
    fn try_into(self: FP8x23W) -> Option<u8> {
        if self.sign {
            Option::None(())
        } else {
            // Unscale the magnitude and round down
            return (self.mag / ONE).try_into();
        }
    }
}

impl FP8x23WIntoI32 of Into<FP8x23W, i32> {
    fn into(self: FP8x23W) -> i32 {
        _i32_into_fp(self)
    }
}

impl FP8x23WTryIntoI8 of TryInto<FP8x23W, i8> {
    fn try_into(self: FP8x23W) -> Option<i8> {
        _i8_try_from_fp(self)
    }
}

impl FP8x23WPartialEq of PartialEq<FP8x23W> {
    #[inline(always)]
    fn eq(lhs: @FP8x23W, rhs: @FP8x23W) -> bool {
        return core_math::eq(lhs, rhs);
    }

    #[inline(always)]
    fn ne(lhs: @FP8x23W, rhs: @FP8x23W) -> bool {
        return core_math::ne(lhs, rhs);
    }
}

impl FP8x23WAdd of Add<FP8x23W> {
    fn add(lhs: FP8x23W, rhs: FP8x23W) -> FP8x23W {
        return core_math::add(lhs, rhs);
    }
}

impl FP8x23WAddEq of AddEq<FP8x23W> {
    #[inline(always)]
    fn add_eq(ref self: FP8x23W, other: FP8x23W) {
        self = Add::add(self, other);
    }
}

impl FP8x23WSub of Sub<FP8x23W> {
    fn sub(lhs: FP8x23W, rhs: FP8x23W) -> FP8x23W {
        return core_math::sub(lhs, rhs);
    }
}

impl FP8x23WSubEq of SubEq<FP8x23W> {
    #[inline(always)]
    fn sub_eq(ref self: FP8x23W, other: FP8x23W) {
        self = Sub::sub(self, other);
    }
}

impl FP8x23WMul of Mul<FP8x23W> {
    fn mul(lhs: FP8x23W, rhs: FP8x23W) -> FP8x23W {
        return core_math::mul(lhs, rhs);
    }
}

impl FP8x23WMulEq of MulEq<FP8x23W> {
    #[inline(always)]
    fn mul_eq(ref self: FP8x23W, other: FP8x23W) {
        self = Mul::mul(self, other);
    }
}

impl FP8x23WDiv of Div<FP8x23W> {
    fn div(lhs: FP8x23W, rhs: FP8x23W) -> FP8x23W {
        return core_math::div(lhs, rhs);
    }
}

impl FP8x23WDivEq of DivEq<FP8x23W> {
    #[inline(always)]
    fn div_eq(ref self: FP8x23W, other: FP8x23W) {
        self = Div::div(self, other);
    }
}

impl FP8x23WPartialOrd of PartialOrd<FP8x23W> {
    #[inline(always)]
    fn ge(lhs: FP8x23W, rhs: FP8x23W) -> bool {
        return core_math::ge(lhs, rhs);
    }

    #[inline(always)]
    fn gt(lhs: FP8x23W, rhs: FP8x23W) -> bool {
        return core_math::gt(lhs, rhs);
    }

    #[inline(always)]
    fn le(lhs: FP8x23W, rhs: FP8x23W) -> bool {
        return core_math::le(lhs, rhs);
    }

    #[inline(always)]
    fn lt(lhs: FP8x23W, rhs: FP8x23W) -> bool {
        return core_math::lt(lhs, rhs);
    }
}

impl FP8x23WNeg of Neg<FP8x23W> {
    #[inline(always)]
    fn neg(a: FP8x23W) -> FP8x23W {
        return core_math::neg(a);
    }
}

impl FP8x23WRem of Rem<FP8x23W> {
    #[inline(always)]
    fn rem(lhs: FP8x23W, rhs: FP8x23W) -> FP8x23W {
        return core_math::rem(lhs, rhs);
    }
}

/// INTERNAL

fn _i32_into_fp(x: FP8x23W) -> i32 {
    let number_felt: felt252 = (x.mag / ONE).into();
    let number_i32: i32 = number_felt.try_into().unwrap();
    if x.sign {
        return number_i32 * -1_i32;
    }
    number_i32
}

fn _i8_try_from_fp(x: FP8x23W) -> Option<i8> {
    let unscaled_mag: Option<u8> = (x.mag / ONE).try_into();

    match unscaled_mag {
        Option::Some(val) => {
            let number_felt: felt252 = unscaled_mag.unwrap().into();
            let mut number_i8: i8 = number_felt.try_into().unwrap();
            if x.sign {
                return Option::Some(number_i8 * -1_i8);
            }
            Option::Some(number_i8)
        },
        Option::None(_) => Option::None(())
    }
}
