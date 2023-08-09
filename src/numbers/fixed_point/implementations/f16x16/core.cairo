use debug::PrintTrait;

use option::OptionTrait;
use result::{ResultTrait, ResultTraitImpl};
use traits::{TryInto, Into};

use orion::numbers::fixed_point::implementations::f16x16::math::{core, trig, hyp};
use orion::numbers::fixed_point::utils;

// CONSTANTS

const TWO: u32 = 131072; // 2 ** 17
const ONE: u32 = 65536; // 2 ** 16
const HALF: u32 = 32768; // 2 ** 15

// STRUCTS

#[derive(Copy, Drop, Serde)]
struct Fixed {
    mag: u32,
    sign: bool
}

// TRAITS

trait FixedTrait {
    fn ZERO() -> Fixed;
    fn ONE() -> Fixed;

    // Constructors
    fn new(mag: u32, sign: bool) -> Fixed;
    fn new_unscaled(mag: u32, sign: bool) -> Fixed;
    fn from_felt(val: felt252) -> Fixed;

    // Math
    fn abs(self: Fixed) -> Fixed;
    fn ceil(self: Fixed) -> Fixed;
    fn exp(self: Fixed) -> Fixed;
    fn exp2(self: Fixed) -> Fixed;
    fn floor(self: Fixed) -> Fixed;
    fn ln(self: Fixed) -> Fixed;
    fn log2(self: Fixed) -> Fixed;
    fn log10(self: Fixed) -> Fixed;
    fn pow(self: Fixed, b: Fixed) -> Fixed;
    fn round(self: Fixed) -> Fixed;
    fn sqrt(self: Fixed) -> Fixed;

    // Trigonometry
    fn acos(self: Fixed) -> Fixed;
    fn acos_fast(self: Fixed) -> Fixed;
    fn asin(self: Fixed) -> Fixed;
    fn asin_fast(self: Fixed) -> Fixed;
    fn atan(self: Fixed) -> Fixed;
    fn atan_fast(self: Fixed) -> Fixed;
    fn cos(self: Fixed) -> Fixed;
    fn cos_fast(self: Fixed) -> Fixed;
    fn sin(self: Fixed) -> Fixed;
    fn sin_fast(self: Fixed) -> Fixed;
    fn tan(self: Fixed) -> Fixed;
    fn tan_fast(self: Fixed) -> Fixed;

    // Hyperbolic
    fn acosh(self: Fixed) -> Fixed;
    fn asinh(self: Fixed) -> Fixed;
    fn atanh(self: Fixed) -> Fixed;
    fn cosh(self: Fixed) -> Fixed;
    fn sinh(self: Fixed) -> Fixed;
    fn tanh(self: Fixed) -> Fixed;
}

impl FixedImpl of FixedTrait {
    fn ZERO() -> Fixed {
        return Fixed { mag: 0, sign: false };
    }

    fn ONE() -> Fixed {
        return Fixed { mag: ONE, sign: false };
    }

    fn new(mag: u32, sign: bool) -> Fixed {
        return Fixed { mag: mag, sign: sign };
    }

    fn new_unscaled(mag: u32, sign: bool) -> Fixed {
        return Fixed { mag: mag * ONE, sign: sign };
    }

    fn from_felt(val: felt252) -> Fixed {
        let mag = integer::u32_try_from_felt252(utils::felt_abs(val)).unwrap();
        return FixedTrait::new(mag, utils::felt_sign(val));
    }

    fn abs(self: Fixed) -> Fixed {
        return core::abs(self);
    }

    fn acos(self: Fixed) -> Fixed {
        return trig::acos(self);
    }

    fn acos_fast(self: Fixed) -> Fixed {
        return trig::acos_fast(self);
    }

    fn acosh(self: Fixed) -> Fixed {
        return hyp::acosh(self);
    }

    fn asin(self: Fixed) -> Fixed {
        return trig::asin(self);
    }

    fn asin_fast(self: Fixed) -> Fixed {
        return trig::asin_fast(self);
    }

    fn asinh(self: Fixed) -> Fixed {
        return hyp::asinh(self);
    }

    fn atan(self: Fixed) -> Fixed {
        return trig::atan(self);
    }

    fn atan_fast(self: Fixed) -> Fixed {
        return trig::atan_fast(self);
    }

    fn atanh(self: Fixed) -> Fixed {
        return hyp::atanh(self);
    }

    fn ceil(self: Fixed) -> Fixed {
        return core::ceil(self);
    }

    fn cos(self: Fixed) -> Fixed {
        return trig::cos(self);
    }

    fn cos_fast(self: Fixed) -> Fixed {
        return trig::cos_fast(self);
    }

    fn cosh(self: Fixed) -> Fixed {
        return hyp::cosh(self);
    }

    fn floor(self: Fixed) -> Fixed {
        return core::floor(self);
    }

    // Calculates the natural exponent of x: e^x
    fn exp(self: Fixed) -> Fixed {
        return core::exp(self);
    }

    // Calculates the binary exponent of x: 2^x
    fn exp2(self: Fixed) -> Fixed {
        return core::exp2(self);
    }

    // Calculates the natural logarithm of x: ln(x)
    // self must be greater than zero
    fn ln(self: Fixed) -> Fixed {
        return core::ln(self);
    }

    // Calculates the binary logarithm of x: log2(x)
    // self must be greather than zero
    fn log2(self: Fixed) -> Fixed {
        return core::log2(self);
    }

    // Calculates the base 10 log of x: log10(x)
    // self must be greater than zero
    fn log10(self: Fixed) -> Fixed {
        return core::log10(self);
    }

    // Calclates the value of x^y and checks for overflow before returning
    // self is a fixed point value
    // b is a fixed point value
    fn pow(self: Fixed, b: Fixed) -> Fixed {
        return core::pow(self, b);
    }

    fn round(self: Fixed) -> Fixed {
        return core::round(self);
    }

    fn sin(self: Fixed) -> Fixed {
        return trig::sin(self);
    }

    fn sin_fast(self: Fixed) -> Fixed {
        return trig::sin_fast(self);
    }

    fn sinh(self: Fixed) -> Fixed {
        return hyp::sinh(self);
    }

    // Calculates the square root of a fixed point value
    // x must be positive
    fn sqrt(self: Fixed) -> Fixed {
        return core::sqrt(self);
    }

    fn tan(self: Fixed) -> Fixed {
        return trig::tan(self);
    }

    fn tan_fast(self: Fixed) -> Fixed {
        return trig::tan_fast(self);
    }

    fn tanh(self: Fixed) -> Fixed {
        return hyp::tanh(self);
    }
}


impl FixedPrint of PrintTrait<Fixed> {
    fn print(self: Fixed) {
        self.sign.print();
        self.mag.print();
    }
}

// Into a raw felt without unscaling
impl FixedIntoFelt252 of Into<Fixed, felt252> {
    fn into(self: Fixed) -> felt252 {
        let mag_felt = self.mag.into();

        if self.sign {
            return mag_felt * -1;
        } else {
            return mag_felt * 1;
        }
    }
}

impl FixedTryIntoU128 of TryInto<Fixed, u128> {
    fn try_into(self: Fixed) -> Option<u128> {
        if self.sign {
            return Option::None(());
        } else {
            // Unscale the magnitude and round down
            return Option::Some((self.mag / ONE).into());
        }
    }
}

impl FixedTryIntoU64 of TryInto<Fixed, u64> {
    fn try_into(self: Fixed) -> Option<u64> {
        if self.sign {
            return Option::None(());
        } else {
            // Unscale the magnitude and round down
            return Option::Some((self.mag / ONE).into());
        }
    }
}

impl FixedTryIntoU32 of TryInto<Fixed, u32> {
    fn try_into(self: Fixed) -> Option<u32> {
        if self.sign {
            return Option::None(());
        } else {
            // Unscale the magnitude and round down
            return Option::Some(self.mag / ONE);
        }
    }
}

impl FixedTryIntoU16 of TryInto<Fixed, u16> {
    fn try_into(self: Fixed) -> Option<u16> {
        if self.sign {
            Option::None(())
        } else {
            // Unscale the magnitude and round down
            return (self.mag / ONE).try_into();
        }
    }
}

impl FixedTryIntoU8 of TryInto<Fixed, u8> {
    fn try_into(self: Fixed) -> Option<u8> {
        if self.sign {
            Option::None(())
        } else {
            // Unscale the magnitude and round down
            return (self.mag / ONE).try_into();
        }
    }
}

impl FixedPartialEq of PartialEq<Fixed> {
    #[inline(always)]
    fn eq(lhs: @Fixed, rhs: @Fixed) -> bool {
        return core::eq(lhs, rhs);
    }

    #[inline(always)]
    fn ne(lhs: @Fixed, rhs: @Fixed) -> bool {
        return core::ne(lhs, rhs);
    }
}

impl FixedAdd of Add<Fixed> {
    fn add(lhs: Fixed, rhs: Fixed) -> Fixed {
        return core::add(lhs, rhs);
    }
}

impl FixedAddEq of AddEq<Fixed> {
    #[inline(always)]
    fn add_eq(ref self: Fixed, other: Fixed) {
        self = Add::add(self, other);
    }
}

impl FixedSub of Sub<Fixed> {
    fn sub(lhs: Fixed, rhs: Fixed) -> Fixed {
        return core::sub(lhs, rhs);
    }
}

impl FixedSubEq of SubEq<Fixed> {
    #[inline(always)]
    fn sub_eq(ref self: Fixed, other: Fixed) {
        self = Sub::sub(self, other);
    }
}

impl FixedMul of Mul<Fixed> {
    fn mul(lhs: Fixed, rhs: Fixed) -> Fixed {
        return core::mul(lhs, rhs);
    }
}

impl FixedMulEq of MulEq<Fixed> {
    #[inline(always)]
    fn mul_eq(ref self: Fixed, other: Fixed) {
        self = Mul::mul(self, other);
    }
}

impl FixedDiv of Div<Fixed> {
    fn div(lhs: Fixed, rhs: Fixed) -> Fixed {
        return core::div(lhs, rhs);
    }
}

impl FixedDivEq of DivEq<Fixed> {
    #[inline(always)]
    fn div_eq(ref self: Fixed, other: Fixed) {
        self = Div::div(self, other);
    }
}

impl FixedPartialOrd of PartialOrd<Fixed> {
    #[inline(always)]
    fn ge(lhs: Fixed, rhs: Fixed) -> bool {
        return core::ge(lhs, rhs);
    }

    #[inline(always)]
    fn gt(lhs: Fixed, rhs: Fixed) -> bool {
        return core::gt(lhs, rhs);
    }

    #[inline(always)]
    fn le(lhs: Fixed, rhs: Fixed) -> bool {
        return core::le(lhs, rhs);
    }

    #[inline(always)]
    fn lt(lhs: Fixed, rhs: Fixed) -> bool {
        return core::lt(lhs, rhs);
    }
}

impl FixedNeg of Neg<Fixed> {
    #[inline(always)]
    fn neg(a: Fixed) -> Fixed {
        return core::neg(a);
    }
}

impl FixedRem of Rem<Fixed> {
    #[inline(always)]
    fn rem(lhs: Fixed, rhs: Fixed) -> Fixed {
        return core::rem(lhs, rhs);
    }
}

