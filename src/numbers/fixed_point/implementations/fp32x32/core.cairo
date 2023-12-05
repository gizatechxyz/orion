use debug::PrintTrait;

use option::OptionTrait;
use result::{ResultTrait, ResultTraitImpl};
use traits::{TryInto, Into};

use cubit::f64 as fp32x32;
use cubit::f64::Fixed as FP32x32;
use cubit::f64::{ONE, HALF};
use cubit::f64::types::fixed;

use orion::numbers::fixed_point::implementations::fp32x32::erf;
use orion::numbers::fixed_point::core::{FixedTrait};
use orion::numbers::fixed_point::utils;
use orion::numbers::{i32, i8};

const MAX: u64 = 9223372036854775808;

impl FP32x32Impl of FixedTrait<FP32x32, u64> {
    fn ZERO() -> FP32x32 {
        return FP32x32 { mag: 0, sign: false };
    }

    fn HALF() -> FP32x32 {
        return FP32x32 { mag: HALF, sign: false };
    }

    fn ONE() -> FP32x32 {
        return FP32x32 { mag: ONE, sign: false };
    }

    fn MAX() -> FP32x32 {
        return FP32x32 { mag: MAX, sign: false };
    }

    fn new(mag: u64, sign: bool) -> FP32x32 {
        return FP32x32 { mag: mag, sign: sign };
    }

    fn new_unscaled(mag: u64, sign: bool) -> FP32x32 {
        return FP32x32 { mag: mag * ONE, sign: sign };
    }

    fn from_felt(val: felt252) -> FP32x32 {
        let mag = integer::u64_try_from_felt252(utils::felt_abs(val)).unwrap();
        return FixedTrait::new(mag, utils::felt_sign(val));
    }

    fn abs(self: FP32x32) -> FP32x32 {
        return fp32x32::core::abs(self);
    }

    fn acos(self: FP32x32) -> FP32x32 {
        return fp32x32::trig::acos_fast(self);
    }

    fn acos_fast(self: FP32x32) -> FP32x32 {
        return fp32x32::trig::acos_fast(self);
    }

    fn acosh(self: FP32x32) -> FP32x32 {
        return fp32x32::hyp::acosh(self);
    }

    fn asin(self: FP32x32) -> FP32x32 {
        return fp32x32::trig::asin_fast(self);
    }

    fn asin_fast(self: FP32x32) -> FP32x32 {
        return fp32x32::trig::asin_fast(self);
    }

    fn asinh(self: FP32x32) -> FP32x32 {
        return fp32x32::hyp::asinh(self);
    }

    fn atan(self: FP32x32) -> FP32x32 {
        return fp32x32::trig::atan_fast(self);
    }

    fn atan_fast(self: FP32x32) -> FP32x32 {
        return fp32x32::trig::atan_fast(self);
    }

    fn atanh(self: FP32x32) -> FP32x32 {
        return fp32x32::hyp::atanh(self);
    }

    fn ceil(self: FP32x32) -> FP32x32 {
        return fp32x32::core::ceil(self);
    }

    fn cos(self: FP32x32) -> FP32x32 {
        return fp32x32::trig::cos_fast(self);
    }

    fn cos_fast(self: FP32x32) -> FP32x32 {
        return fp32x32::trig::cos_fast(self);
    }

    fn cosh(self: FP32x32) -> FP32x32 {
        return fp32x32::hyp::cosh(self);
    }

    fn floor(self: FP32x32) -> FP32x32 {
        return fp32x32::core::floor(self);
    }

    // Calculates the natural exponent of x: e^x
    fn exp(self: FP32x32) -> FP32x32 {
        return fp32x32::core::exp(self);
    }

    // Calculates the binary exponent of x: 2^x
    fn exp2(self: FP32x32) -> FP32x32 {
        return fp32x32::core::exp2(self);
    }

    // Calculates the natural logarithm of x: ln(x)
    // self must be greater than zero
    fn ln(self: FP32x32) -> FP32x32 {
        return fp32x32::core::ln(self);
    }

    // Calculates the binary logarithm of x: log2(x)
    // self must be greather than zero
    fn log2(self: FP32x32) -> FP32x32 {
        return fp32x32::core::log2(self);
    }

    // Calculates the base 10 log of x: log10(x)
    // self must be greater than zero
    fn log10(self: FP32x32) -> FP32x32 {
        return fp32x32::core::log10(self);
    }

    // Calclates the value of x^y and checks for overflow before returning
    // self is a fixed point value
    // b is a fixed point value
    fn pow(self: FP32x32, b: FP32x32) -> FP32x32 {
        return fp32x32::core::pow(self, b);
    }

    fn round(self: FP32x32) -> FP32x32 {
        return fp32x32::core::round(self);
    }

    fn sin(self: FP32x32) -> FP32x32 {
        return fp32x32::trig::sin_fast(self);
    }

    fn sin_fast(self: FP32x32) -> FP32x32 {
        return fp32x32::trig::sin_fast(self);
    }

    fn sinh(self: FP32x32) -> FP32x32 {
        return fp32x32::hyp::sinh(self);
    }

    // Calculates the square root of a fixed point value
    // x must be positive
    fn sqrt(self: FP32x32) -> FP32x32 {
        return fp32x32::core::sqrt(self);
    }

    fn tan(self: FP32x32) -> FP32x32 {
        return fp32x32::trig::tan_fast(self);
    }

    fn tan_fast(self: FP32x32) -> FP32x32 {
        return fp32x32::trig::tan_fast(self);
    }

    fn tanh(self: FP32x32) -> FP32x32 {
        return fp32x32::hyp::tanh(self);
    }

    fn sign(self: FP32x32) -> FP32x32 {
        panic(array!['not supported!'])
    }

    fn NaN() -> FP32x32 {
        return FP32x32 { mag: 0, sign: true };
    }

    fn is_nan(self: FP32x32) -> bool {
        self == FP32x32 { mag: 0, sign: true }
    }

    fn erf(self: FP32x32) -> FP32x32 {
        return erf::erf(self);
    }
}


impl FP32x32Print of PrintTrait<FP32x32> {
    fn print(self: FP32x32) {
        self.sign.print();
        self.mag.print();
    }
}

// Into a raw felt without unscaling
impl FP32x32IntoFelt252 of Into<FP32x32, felt252> {
    fn into(self: FP32x32) -> felt252 {
        let mag_felt = self.mag.into();

        if self.sign {
            return mag_felt * -1;
        } else {
            return mag_felt * 1;
        }
    }
}

impl FP32x32TryIntoU64 of TryInto<FP32x32, u64> {
    fn try_into(self: FP32x32) -> Option<u64> {
        if self.sign {
            return Option::None(());
        } else {
            // Unscale the magnitude and round down
            return Option::Some((self.mag / ONE).into());
        }
    }
}

impl FP32x32TryIntoU16 of TryInto<FP32x32, u16> {
    fn try_into(self: FP32x32) -> Option<u16> {
        if self.sign {
            Option::None(())
        } else {
            // Unscale the magnitude and round down
            return (self.mag / ONE).try_into();
        }
    }
}

impl FP32x32TryIntoU32 of TryInto<FP32x32, u32> {
    fn try_into(self: FP32x32) -> Option<u32> {
        if self.sign {
            Option::None(())
        } else {
            // Unscale the magnitude and round down
            return (self.mag / ONE).try_into();
        }
    }
}

impl FP32x32TryIntoU8 of TryInto<FP32x32, u8> {
    fn try_into(self: FP32x32) -> Option<u8> {
        if self.sign {
            Option::None(())
        } else {
            // Unscale the magnitude and round down
            return (self.mag / ONE).try_into();
        }
    }
}

impl FP32x32TryIntoI8 of TryInto<FP32x32, i8> {
    fn try_into(self: FP32x32) -> Option<i8> {
        _i8_try_from_fp(self)
    }
}

// impl FP32x32PartialEq of PartialEq<FP32x32> {
//     #[inline(always)]
//     fn eq(lhs: @FP32x32, rhs: @FP32x32) -> bool {
//         return fp32x32::core::eq(lhs, rhs);
//     }

//     #[inline(always)]
//     fn ne(lhs: @FP32x32, rhs: @FP32x32) -> bool {
//         return fp32x32::core::ne(lhs, rhs);
//     }
// }

impl FP32x32Add of Add<FP32x32> {
    fn add(lhs: FP32x32, rhs: FP32x32) -> FP32x32 {
        return fp32x32::core::add(lhs, rhs);
    }
}

impl FP32x32AddEq of AddEq<FP32x32> {
    #[inline(always)]
    fn add_eq(ref self: FP32x32, other: FP32x32) {
        self = fp32x32::core::add(self, other);
    }
}

impl FP32x32Sub of Sub<FP32x32> {
    fn sub(lhs: FP32x32, rhs: FP32x32) -> FP32x32 {
        return fp32x32::core::sub(lhs, rhs);
    }
}

impl FP32x32SubEq of SubEq<FP32x32> {
    #[inline(always)]
    fn sub_eq(ref self: FP32x32, other: FP32x32) {
        self = fp32x32::core::sub(self, other);
    }
}

impl FP32x32Mul of Mul<FP32x32> {
    fn mul(lhs: FP32x32, rhs: FP32x32) -> FP32x32 {
        return fp32x32::core::mul(lhs, rhs);
    }
}

impl FP32x32MulEq of MulEq<FP32x32> {
    #[inline(always)]
    fn mul_eq(ref self: FP32x32, other: FP32x32) {
        self = fp32x32::core::mul(self, other);
    }
}

impl FP32x32Div of Div<FP32x32> {
    fn div(lhs: FP32x32, rhs: FP32x32) -> FP32x32 {
        return fp32x32::core::div(lhs, rhs);
    }
}

impl FP32x32DivEq of DivEq<FP32x32> {
    #[inline(always)]
    fn div_eq(ref self: FP32x32, other: FP32x32) {
        self = fp32x32::core::div(self, other);
    }
}

impl FP32x32PartialOrd of PartialOrd<FP32x32> {
    #[inline(always)]
    fn ge(lhs: FP32x32, rhs: FP32x32) -> bool {
        return fp32x32::core::ge(lhs, rhs);
    }

    #[inline(always)]
    fn gt(lhs: FP32x32, rhs: FP32x32) -> bool {
        return fp32x32::core::gt(lhs, rhs);
    }

    #[inline(always)]
    fn le(lhs: FP32x32, rhs: FP32x32) -> bool {
        return fp32x32::core::le(lhs, rhs);
    }

    #[inline(always)]
    fn lt(lhs: FP32x32, rhs: FP32x32) -> bool {
        return fp32x32::core::lt(lhs, rhs);
    }
}

impl FP32x32Neg of Neg<FP32x32> {
    #[inline(always)]
    fn neg(a: FP32x32) -> FP32x32 {
        return fp32x32::core::neg(a);
    }
}

impl FP32x32Rem of Rem<FP32x32> {
    #[inline(always)]
    fn rem(lhs: FP32x32, rhs: FP32x32) -> FP32x32 {
        return fp32x32::core::rem(lhs, rhs);
    }
}

fn eq(a: @FP32x32, b: @FP32x32) -> bool {
    return (*a.mag == *b.mag) && (*a.sign == *b.sign);
}

/// INTERNAL

fn _i8_try_from_fp(x: FP32x32) -> Option<i8> {
    let unscaled_mag: Option<u8> = (x.mag / ONE).try_into();

    match unscaled_mag {
        Option::Some(val) => Option::Some(i8 { mag: unscaled_mag.unwrap(), sign: x.sign }),
        Option::None(_) => Option::None(())
    }
}
