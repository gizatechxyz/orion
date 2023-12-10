use core::debug::PrintTrait;

use core::option::OptionTrait;
use core::result::{ResultTrait, ResultTraitImpl};
use core::traits::{TryInto, Into};

use orion::numbers::signed_integer::{i32::i32, i8::i8};
use orion::numbers::{fixed_point::core::FixedTrait, FP16x16};
use orion::numbers::fixed_point::implementations::fp16x16wide::math::{core as core_math, trig, hyp, erf};
use orion::numbers::fixed_point::utils;

/// A struct representing a fixed point number.
#[derive(Serde, Copy, Drop)]
struct FP16x16W {
    mag: u64,
    sign: bool
}

// CONSTANTS

const TWO: u64 = 131072; // 2 ** 17
const ONE: u64 = 65536; // 2 ** 16
const HALF: u64 = 32768; // 2 ** 15
const MAX: u64 = 2147483648; // 2 ** 31


impl FP16x16WImpl of FixedTrait<FP16x16W, u64> {
    fn ZERO() -> FP16x16W {
        return FP16x16W { mag: 0, sign: false };
    }

    fn HALF() -> FP16x16W {
        return FP16x16W { mag: HALF, sign: false };
    }

    fn ONE() -> FP16x16W {
        return FP16x16W { mag: ONE, sign: false };
    }

    fn MAX() -> FP16x16W {
        return FP16x16W { mag: MAX, sign: false };
    }

    fn new(mag: u64, sign: bool) -> FP16x16W {
        return FP16x16W { mag: mag, sign: sign };
    }

    fn new_unscaled(mag: u64, sign: bool) -> FP16x16W {
        return FP16x16W { mag: mag * ONE, sign: sign };
    }

    fn from_felt(val: felt252) -> FP16x16W {
        let mag = core::integer::u64_try_from_felt252(utils::felt_abs(val)).unwrap();
        return FixedTrait::new(mag, utils::felt_sign(val));
    }

    fn abs(self: FP16x16W) -> FP16x16W {
        return core_math::abs(self);
    }

    fn acos(self: FP16x16W) -> FP16x16W {
        return trig::acos_fast(self);
    }

    fn acos_fast(self: FP16x16W) -> FP16x16W {
        return trig::acos_fast(self);
    }

    fn acosh(self: FP16x16W) -> FP16x16W {
        return hyp::acosh(self);
    }

    fn asin(self: FP16x16W) -> FP16x16W {
        return trig::asin_fast(self);
    }

    fn asin_fast(self: FP16x16W) -> FP16x16W {
        return trig::asin_fast(self);
    }

    fn asinh(self: FP16x16W) -> FP16x16W {
        return hyp::asinh(self);
    }

    fn atan(self: FP16x16W) -> FP16x16W {
        return trig::atan_fast(self);
    }

    fn atan_fast(self: FP16x16W) -> FP16x16W {
        return trig::atan_fast(self);
    }

    fn atanh(self: FP16x16W) -> FP16x16W {
        return hyp::atanh(self);
    }

    fn ceil(self: FP16x16W) -> FP16x16W {
        return core_math::ceil(self);
    }

    fn cos(self: FP16x16W) -> FP16x16W {
        return trig::cos_fast(self);
    }

    fn cos_fast(self: FP16x16W) -> FP16x16W {
        return trig::cos_fast(self);
    }

    fn cosh(self: FP16x16W) -> FP16x16W {
        return hyp::cosh(self);
    }

    fn floor(self: FP16x16W) -> FP16x16W {
        return core_math::floor(self);
    }

    // Calculates the natural exponent of x: e^x
    fn exp(self: FP16x16W) -> FP16x16W {
        return core_math::exp(self);
    }

    // Calculates the binary exponent of x: 2^x
    fn exp2(self: FP16x16W) -> FP16x16W {
        return core_math::exp2(self);
    }

    // Calculates the natural logarithm of x: ln(x)
    // self must be greater than zero
    fn ln(self: FP16x16W) -> FP16x16W {
        return core_math::ln(self);
    }

    // Calculates the binary logarithm of x: log2(x)
    // self must be greather than zero
    fn log2(self: FP16x16W) -> FP16x16W {
        return core_math::log2(self);
    }

    // Calculates the base 10 log of x: log10(x)
    // self must be greater than zero
    fn log10(self: FP16x16W) -> FP16x16W {
        return core_math::log10(self);
    }

    // Calclates the value of x^y and checks for overflow before returning
    // self is a fixed point value
    // b is a fixed point value
    fn pow(self: FP16x16W, b: FP16x16W) -> FP16x16W {
        return core_math::pow(self, b);
    }

    fn round(self: FP16x16W) -> FP16x16W {
        return core_math::round(self);
    }

    fn sin(self: FP16x16W) -> FP16x16W {
        return trig::sin_fast(self);
    }

    fn sin_fast(self: FP16x16W) -> FP16x16W {
        return trig::sin_fast(self);
    }

    fn sinh(self: FP16x16W) -> FP16x16W {
        return hyp::sinh(self);
    }

    // Calculates the square root of a fixed point value
    // x must be positive
    fn sqrt(self: FP16x16W) -> FP16x16W {
        return core_math::sqrt(self);
    }

    fn tan(self: FP16x16W) -> FP16x16W {
        return trig::tan_fast(self);
    }

    fn tan_fast(self: FP16x16W) -> FP16x16W {
        return trig::tan_fast(self);
    }

    fn tanh(self: FP16x16W) -> FP16x16W {
        return hyp::tanh(self);
    }

    fn sign(self: FP16x16W) -> FP16x16W {
        return core_math::sign(self);
    }

    fn NaN() -> FP16x16W {
        return FP16x16W { mag: 0, sign: true };
    }

    fn is_nan(self: FP16x16W) -> bool {
        self == FP16x16W { mag: 0, sign: true }
    }

    fn INF() -> FP16x16W {
        return FP16x16W { mag: 4294967295, sign: false };
    }

    fn POS_INF() -> FP16x16W {
        return FP16x16W { mag: 4294967295, sign: false };
    }

    fn NEG_INF() -> FP16x16W {
        return FP16x16W { mag: 4294967295, sign: true };
    }

    fn is_inf(self: FP16x16W) -> bool {
        self.mag == 4294967295
    }

    fn is_pos_inf(self: FP16x16W) -> bool {
        self.is_inf() && !self.sign
    }

    fn is_neg_inf(self: FP16x16W) -> bool {
        self.is_inf() && self.sign
    }

    fn erf(self: FP16x16W) -> FP16x16W {
        return erf::erf(self);
    }
}


impl FP16x16WPrint of PrintTrait<FP16x16W> {
    fn print(self: FP16x16W) {
        self.sign.print();
        self.mag.print();
    }
}

// Into a raw felt without unscaling
impl FP16x16WIntoFelt252 of Into<FP16x16W, felt252> {
    fn into(self: FP16x16W) -> felt252 {
        let mag_felt = self.mag.into();

        if self.sign {
            return mag_felt * -1;
        } else {
            return mag_felt * 1;
        }
    }
}

impl FP16x16WIntoI32 of Into<FP16x16W, i32> {
    fn into(self: FP16x16W) -> i32 {
        _i32_into_fp(self)
    }
}

impl FP16x16IntoFP16x16W of Into<FP16x16, FP16x16W> {
    fn into(self: FP16x16) -> FP16x16W {
        FP16x16W { mag: self.mag.into(), sign: self.sign }
    }
}

impl FP16x16WTryIntoFP16x16 of TryInto<FP16x16W, FP16x16> {
    fn try_into(self: FP16x16W) -> Option<FP16x16> {
        match self.mag.try_into() {
            Option::Some(val) => { Option::Some(FP16x16 { mag: val, sign: self.sign }) },
            Option::None(_) => { Option::None(()) }
        }
    }
}

impl FP16x16WTryIntoI8 of TryInto<FP16x16W, i8> {
    fn try_into(self: FP16x16W) -> Option<i8> {
        _i8_try_from_fp(self)
    }
}


impl FP16x16WTryIntoU128 of TryInto<FP16x16W, u128> {
    fn try_into(self: FP16x16W) -> Option<u128> {
        if self.sign {
            return Option::None(());
        } else {
            // Unscale the magnitude and round down
            return Option::Some((self.mag / ONE).into());
        }
    }
}

impl FP16x16WTryIntoU64 of TryInto<FP16x16W, u64> {
    fn try_into(self: FP16x16W) -> Option<u64> {
        if self.sign {
            return Option::None(());
        } else {
            // Unscale the magnitude and round down
            return Option::Some((self.mag / ONE).into());
        }
    }
}

impl FP16x16WTryIntoU32 of TryInto<FP16x16W, u64> {
    fn try_into(self: FP16x16W) -> Option<u64> {
        if self.sign {
            return Option::None(());
        } else {
            // Unscale the magnitude and round down
            return Option::Some(self.mag / ONE);
        }
    }
}

impl FP16x16WTryIntoU16 of TryInto<FP16x16W, u16> {
    fn try_into(self: FP16x16W) -> Option<u16> {
        if self.sign {
            Option::None(())
        } else {
            // Unscale the magnitude and round down
            return (self.mag / ONE).try_into();
        }
    }
}

impl FP16x16WTryIntoU8 of TryInto<FP16x16W, u8> {
    fn try_into(self: FP16x16W) -> Option<u8> {
        if self.sign {
            Option::None(())
        } else {
            // Unscale the magnitude and round down
            return (self.mag / ONE).try_into();
        }
    }
}

impl FP16x16WPartialEq of PartialEq<FP16x16W> {
    #[inline(always)]
    fn eq(lhs: @FP16x16W, rhs: @FP16x16W) -> bool {
        return core_math::eq(lhs, rhs);
    }

    #[inline(always)]
    fn ne(lhs: @FP16x16W, rhs: @FP16x16W) -> bool {
        return core_math::ne(lhs, rhs);
    }
}

impl FP16x16WAdd of Add<FP16x16W> {
    fn add(lhs: FP16x16W, rhs: FP16x16W) -> FP16x16W {
        return core_math::add(lhs, rhs);
    }
}

impl FP16x16WAddEq of AddEq<FP16x16W> {
    #[inline(always)]
    fn add_eq(ref self: FP16x16W, other: FP16x16W) {
        self = Add::add(self, other);
    }
}

impl FP16x16WSub of Sub<FP16x16W> {
    fn sub(lhs: FP16x16W, rhs: FP16x16W) -> FP16x16W {
        return core_math::sub(lhs, rhs);
    }
}

impl FP16x16WSubEq of SubEq<FP16x16W> {
    #[inline(always)]
    fn sub_eq(ref self: FP16x16W, other: FP16x16W) {
        self = Sub::sub(self, other);
    }
}

impl FP16x16WMul of Mul<FP16x16W> {
    fn mul(lhs: FP16x16W, rhs: FP16x16W) -> FP16x16W {
        return core_math::mul(lhs, rhs);
    }
}

impl FP16x16WMulEq of MulEq<FP16x16W> {
    #[inline(always)]
    fn mul_eq(ref self: FP16x16W, other: FP16x16W) {
        self = Mul::mul(self, other);
    }
}

impl FP16x16WDiv of Div<FP16x16W> {
    fn div(lhs: FP16x16W, rhs: FP16x16W) -> FP16x16W {
        return core_math::div(lhs, rhs);
    }
}

impl FP16x16WDivEq of DivEq<FP16x16W> {
    #[inline(always)]
    fn div_eq(ref self: FP16x16W, other: FP16x16W) {
        self = Div::div(self, other);
    }
}

impl FP16x16WPartialOrd of PartialOrd<FP16x16W> {
    #[inline(always)]
    fn ge(lhs: FP16x16W, rhs: FP16x16W) -> bool {
        return core_math::ge(lhs, rhs);
    }

    #[inline(always)]
    fn gt(lhs: FP16x16W, rhs: FP16x16W) -> bool {
        return core_math::gt(lhs, rhs);
    }

    #[inline(always)]
    fn le(lhs: FP16x16W, rhs: FP16x16W) -> bool {
        return core_math::le(lhs, rhs);
    }

    #[inline(always)]
    fn lt(lhs: FP16x16W, rhs: FP16x16W) -> bool {
        return core_math::lt(lhs, rhs);
    }
}

impl FP16x16WNeg of Neg<FP16x16W> {
    #[inline(always)]
    fn neg(a: FP16x16W) -> FP16x16W {
        return core_math::neg(a);
    }
}

impl FP16x16WRem of Rem<FP16x16W> {
    #[inline(always)]
    fn rem(lhs: FP16x16W, rhs: FP16x16W) -> FP16x16W {
        return core_math::rem(lhs, rhs);
    }
}


/// INTERNAL

fn _i32_into_fp(x: FP16x16W) -> i32 {
    i32 { mag: (x.mag / ONE).try_into().unwrap(), sign: x.sign }
}

fn _i8_try_from_fp(x: FP16x16W) -> Option<i8> {
    let unscaled_mag: Option<u8> = (x.mag / ONE).try_into();

    match unscaled_mag {
        Option::Some(val) => Option::Some(i8 { mag: unscaled_mag.unwrap(), sign: x.sign }),
        Option::None(_) => Option::None(())
    }
}
