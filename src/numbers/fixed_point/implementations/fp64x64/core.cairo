use core::debug::PrintTrait;

use core::option::OptionTrait;
use core::result::{ResultTrait, ResultTraitImpl};
use core::traits::{TryInto, Into};

use cubit::f128 as fp64x64;
use cubit::f128::types::Fixed as FP64x64;
use cubit::f128::ONE_u128 as ONE;
use cubit::f128::ops::MAX_u128 as MAX;

use orion::numbers::fixed_point::implementations::fp64x64::erf;
use orion::numbers::fixed_point::core::{FixedTrait};
use orion::numbers::fixed_point::utils;

const HALF: u128 = 9223372036854775808_u128; // 2 ** 63

impl FP64x64Impl of FixedTrait<FP64x64, u128> {
    fn ZERO() -> FP64x64 {
        return FP64x64 { mag: 0, sign: false };
    }

    fn HALF() -> FP64x64 {
        return FP64x64 { mag: HALF, sign: false };
    }

    fn ONE() -> FP64x64 {
        return FP64x64 { mag: ONE, sign: false };
    }

    fn MAX() -> FP64x64 {
        return FP64x64 { mag: MAX, sign: false };
    }

    fn new(mag: u128, sign: bool) -> FP64x64 {
        return FP64x64 { mag: mag, sign: sign };
    }

    fn new_unscaled(mag: u128, sign: bool) -> FP64x64 {
        return FP64x64 { mag: mag * ONE, sign: sign };
    }

    fn from_felt(val: felt252) -> FP64x64 {
        let mag = core::integer::u128_try_from_felt252(utils::felt_abs(val)).unwrap();
        return FixedTrait::new(mag, utils::felt_sign(val));
    }

    fn abs(self: FP64x64) -> FP64x64 {
        return fp64x64::ops::abs(self);
    }

    fn acos(self: FP64x64) -> FP64x64 {
        return fp64x64::trig::acos_fast(self);
    }

    fn acos_fast(self: FP64x64) -> FP64x64 {
        return fp64x64::trig::acos_fast(self);
    }

    fn acosh(self: FP64x64) -> FP64x64 {
        return fp64x64::hyp::acosh(self);
    }

    fn asin(self: FP64x64) -> FP64x64 {
        return fp64x64::trig::asin_fast(self);
    }

    fn asin_fast(self: FP64x64) -> FP64x64 {
        return fp64x64::trig::asin_fast(self);
    }

    fn asinh(self: FP64x64) -> FP64x64 {
        return fp64x64::hyp::asinh(self);
    }

    fn atan(self: FP64x64) -> FP64x64 {
        return fp64x64::trig::atan_fast(self);
    }

    fn atan_fast(self: FP64x64) -> FP64x64 {
        return fp64x64::trig::atan_fast(self);
    }

    fn atanh(self: FP64x64) -> FP64x64 {
        return fp64x64::hyp::atanh(self);
    }

    fn ceil(self: FP64x64) -> FP64x64 {
        return fp64x64::ops::ceil(self);
    }

    fn cos(self: FP64x64) -> FP64x64 {
        return fp64x64::trig::cos_fast(self);
    }

    fn cos_fast(self: FP64x64) -> FP64x64 {
        return fp64x64::trig::cos_fast(self);
    }

    fn cosh(self: FP64x64) -> FP64x64 {
        return fp64x64::hyp::cosh(self);
    }

    fn floor(self: FP64x64) -> FP64x64 {
        return fp64x64::ops::floor(self);
    }

    // Calculates the natural exponent of x: e^x
    fn exp(self: FP64x64) -> FP64x64 {
        return fp64x64::ops::exp(self);
    }

    // Calculates the binary exponent of x: 2^x
    fn exp2(self: FP64x64) -> FP64x64 {
        return fp64x64::ops::exp2(self);
    }

    // Calculates the natural logarithm of x: ln(x)
    // self must be greater than zero
    fn ln(self: FP64x64) -> FP64x64 {
        return fp64x64::ops::ln(self);
    }

    // Calculates the binary logarithm of x: log2(x)
    // self must be greather than zero
    fn log2(self: FP64x64) -> FP64x64 {
        return fp64x64::ops::log2(self);
    }

    // Calculates the base 10 log of x: log10(x)
    // self must be greater than zero
    fn log10(self: FP64x64) -> FP64x64 {
        return fp64x64::ops::log10(self);
    }

    // Calclates the value of x^y and checks for overflow before returning
    // self is a fixed point value
    // b is a fixed point value
    fn pow(self: FP64x64, b: FP64x64) -> FP64x64 {
        return fp64x64::ops::pow(self, b);
    }

    fn round(self: FP64x64) -> FP64x64 {
        return fp64x64::ops::round(self);
    }

    fn sin(self: FP64x64) -> FP64x64 {
        return fp64x64::trig::sin_fast(self);
    }

    fn sin_fast(self: FP64x64) -> FP64x64 {
        return fp64x64::trig::sin_fast(self);
    }

    fn sinh(self: FP64x64) -> FP64x64 {
        return fp64x64::hyp::sinh(self);
    }

    // Calculates the square root of a fixed point value
    // x must be positive
    fn sqrt(self: FP64x64) -> FP64x64 {
        return fp64x64::ops::sqrt(self);
    }

    fn tan(self: FP64x64) -> FP64x64 {
        return fp64x64::trig::tan_fast(self);
    }

    fn tan_fast(self: FP64x64) -> FP64x64 {
        return fp64x64::trig::tan_fast(self);
    }

    fn tanh(self: FP64x64) -> FP64x64 {
        return fp64x64::hyp::tanh(self);
    }

    fn sign(self: FP64x64) -> FP64x64 {
        panic(array!['not supported!'])
    }

    fn NaN() -> FP64x64 {
        return FP64x64 { mag: 0, sign: true };
    }

    fn is_nan(self: FP64x64) -> bool {
        self == FP64x64 { mag: 0, sign: true }
    }

    fn INF() -> FP64x64 {
        return FP64x64 { mag: 4294967295, sign: false };
    }

    fn POS_INF() -> FP64x64 {
        return FP64x64 { mag: 4294967295, sign: false };
    }

    fn NEG_INF() -> FP64x64 {
        return FP64x64 { mag: 4294967295, sign: true };
    }

    fn is_inf(self: FP64x64) -> bool {
        self.mag == 4294967295
    }

    fn is_pos_inf(self: FP64x64) -> bool {
        self.is_inf() && !self.sign
    }

    fn is_neg_inf(self: FP64x64) -> bool {
        self.is_inf() && self.sign
    }

    fn erf(self: FP64x64) -> FP64x64 {
        return erf::erf(self);
    }
}


impl FP64x64Print of PrintTrait<FP64x64> {
    fn print(self: FP64x64) {
        self.sign.print();
        self.mag.print();
    }
}

// Into a raw felt without unscaling
impl FP64x64IntoFelt252 of Into<FP64x64, felt252> {
    fn into(self: FP64x64) -> felt252 {
        let mag_felt = self.mag.into();

        if self.sign {
            return mag_felt * -1;
        } else {
            return mag_felt * 1;
        }
    }
}

impl FP64x64TryIntoU128 of TryInto<FP64x64, u128> {
    fn try_into(self: FP64x64) -> Option<u128> {
        if self.sign {
            return Option::None(());
        } else {
            // Unscale the magnitude and round down
            return Option::Some((self.mag / ONE).into());
        }
    }
}

impl FP64x64TryIntoU16 of TryInto<FP64x64, u16> {
    fn try_into(self: FP64x64) -> Option<u16> {
        if self.sign {
            Option::None(())
        } else {
            // Unscale the magnitude and round down
            return (self.mag / ONE).try_into();
        }
    }
}

impl FP64x64TryIntoU32 of TryInto<FP64x64, u32> {
    fn try_into(self: FP64x64) -> Option<u32> {
        if self.sign {
            Option::None(())
        } else {
            // Unscale the magnitude and round down
            return (self.mag / ONE).try_into();
        }
    }
}

impl FP64x64TryIntoU8 of TryInto<FP64x64, u8> {
    fn try_into(self: FP64x64) -> Option<u8> {
        if self.sign {
            Option::None(())
        } else {
            // Unscale the magnitude and round down
            return (self.mag / ONE).try_into();
        }
    }
}

impl FP64x64TryIntoI8 of TryInto<FP64x64, i8> {
    fn try_into(self: FP64x64) -> Option<i8> {
        _i8_try_from_fp(self)
    }
}

// impl FP64x64PartialEq of PartialEq<FP64x64> {
//     #[inline(always)]
//     fn eq(lhs: @FP64x64, rhs: @FP64x64) -> bool {
//         return fp64x64::ops::eq(lhs, rhs);
//     }

//     #[inline(always)]
//     fn ne(lhs: @FP64x64, rhs: @FP64x64) -> bool {
//         return fp64x64::ops::ne(lhs, rhs);
//     }
// }

impl FP64x64Add of Add<FP64x64> {
    fn add(lhs: FP64x64, rhs: FP64x64) -> FP64x64 {
        return fp64x64::ops::add(lhs, rhs);
    }
}

impl FP64x64AddEq of AddEq<FP64x64> {
    #[inline(always)]
    fn add_eq(ref self: FP64x64, other: FP64x64) {
        self = fp64x64::ops::add(self, other);
    }
}

impl FP64x64Sub of Sub<FP64x64> {
    fn sub(lhs: FP64x64, rhs: FP64x64) -> FP64x64 {
        return fp64x64::ops::sub(lhs, rhs);
    }
}

impl FP64x64SubEq of SubEq<FP64x64> {
    #[inline(always)]
    fn sub_eq(ref self: FP64x64, other: FP64x64) {
        self = fp64x64::ops::sub(self, other);
    }
}

impl FP64x64Mul of Mul<FP64x64> {
    fn mul(lhs: FP64x64, rhs: FP64x64) -> FP64x64 {
        return fp64x64::ops::mul(lhs, rhs);
    }
}

impl FP64x64MulEq of MulEq<FP64x64> {
    #[inline(always)]
    fn mul_eq(ref self: FP64x64, other: FP64x64) {
        self = fp64x64::ops::mul(self, other);
    }
}

impl FP64x64Div of Div<FP64x64> {
    fn div(lhs: FP64x64, rhs: FP64x64) -> FP64x64 {
        return fp64x64::ops::div(lhs, rhs);
    }
}

impl FP64x64DivEq of DivEq<FP64x64> {
    #[inline(always)]
    fn div_eq(ref self: FP64x64, other: FP64x64) {
        self = fp64x64::ops::div(self, other);
    }
}

impl FP64x64PartialOrd of PartialOrd<FP64x64> {
    #[inline(always)]
    fn ge(lhs: FP64x64, rhs: FP64x64) -> bool {
        return fp64x64::ops::ge(lhs, rhs);
    }

    #[inline(always)]
    fn gt(lhs: FP64x64, rhs: FP64x64) -> bool {
        return fp64x64::ops::gt(lhs, rhs);
    }

    #[inline(always)]
    fn le(lhs: FP64x64, rhs: FP64x64) -> bool {
        return fp64x64::ops::le(lhs, rhs);
    }

    #[inline(always)]
    fn lt(lhs: FP64x64, rhs: FP64x64) -> bool {
        return fp64x64::ops::lt(lhs, rhs);
    }
}

impl FP64x64Neg of Neg<FP64x64> {
    #[inline(always)]
    fn neg(a: FP64x64) -> FP64x64 {
        return fp64x64::ops::neg(a);
    }
}

impl FP64x64Rem of Rem<FP64x64> {
    #[inline(always)]
    fn rem(lhs: FP64x64, rhs: FP64x64) -> FP64x64 {
        return fp64x64::ops::rem(lhs, rhs);
    }
}

fn eq(a: @FP64x64, b: @FP64x64) -> bool {
    return (*a.mag == *b.mag) && (*a.sign == *b.sign);
}

/// INTERNAL

fn _i8_try_from_fp(x: FP64x64) -> Option<i8> {
    let unscaled_mag: Option<u8> = (x.mag / ONE).try_into();

    match unscaled_mag {
        Option::Some => {
            let number_felt: felt252 = unscaled_mag.unwrap().into();
            let mut number_i8: i8 = number_felt.try_into().unwrap();
            if x.sign {
                return Option::Some(number_i8 * -1_i8);
            }
            Option::Some(number_i8)
        },
        Option::None => Option::None(())
    }
}
