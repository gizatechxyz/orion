pub(crate) mod ops;
pub(crate) mod lut;
pub(crate) mod trig;
pub(crate) mod hyp;
pub(crate) mod comp;
pub(crate) mod erf;
pub(crate) mod helpers;

use orion_numbers::FixedTrait;

// CONSTANTS

const TWO: i64 = 8589934592; // 2 ** 33
const ONE: i64 = 4294967296; // 2 ** 32
const HALF: i64 = 2147483648; // 2 ** 31
const MAX_i128: i128 = 18_446_744_073_709_551_615; //2**64 - 1

// STRUCTS

#[derive(Copy, Drop, Serde)]
struct F64 {
    d: i64,
}


impl F64Impl of FixedTrait<F64, i64> {
    fn ZERO() -> F64 {
        return core::num::traits::Zero::zero();
    }

    fn ONE() -> F64 {
        return core::num::traits::One::one();
    }

    fn new(val: i64) -> F64 {
        return F64 { d: val };
    }

    fn new_unscaled(val: i64) -> F64 {
        return F64 { d: val * ONE };
    }

    fn from_felt(val: felt252) -> F64 {
        return FixedTrait::new(val.try_into().unwrap());
    }

    fn from_unscaled_felt(val: felt252) -> F64 {
        return FixedTrait::from_felt(val * ONE.into());
    }

    fn abs(self: F64) -> F64 {
        return ops::abs(self);
    }

    fn acos(self: F64) -> F64 {
        return trig::acos_fast(self);
    }

    fn acosh(self: F64) -> F64 {
        return hyp::acosh(self);
    }

    fn asin(self: F64) -> F64 {
        return trig::asin_fast(self);
    }


    fn asinh(self: F64) -> F64 {
        return hyp::asinh(self);
    }

    fn atan(self: F64) -> F64 {
        return trig::atan_fast(self);
    }

    fn atanh(self: F64) -> F64 {
        return hyp::atanh(self);
    }

    fn ceil(self: F64) -> F64 {
        return ops::ceil(self);
    }

    fn cos(self: F64) -> F64 {
        return trig::cos_fast(self);
    }

    fn cosh(self: F64) -> F64 {
        return hyp::cosh(self);
    }

    fn floor(self: F64) -> F64 {
        return ops::floor(self);
    }

    // Calculates the natural exponent of x: e^x
    fn exp(self: F64) -> F64 {
        return ops::exp(self);
    }

    // Calculates the binary exponent of x: 2^x
    fn exp2(self: F64) -> F64 {
        return ops::exp2(self);
    }

    // Calculates the natural logarithm of x: ln(x)
    // self must be greater than zero
    fn ln(self: F64) -> F64 {
        return ops::ln(self);
    }

    // Calculates the binary logarithm of x: log2(x)
    // self must be greather than zero
    fn log2(self: F64) -> F64 {
        return ops::log2(self);
    }

    // Calculates the base 10 log of x: log10(x)
    // self must be greater than zero
    fn log10(self: F64) -> F64 {
        return ops::log10(self);
    }

    // Calclates the value of x^y and checks for overflow before returning
    // self is a fixed point value
    // b is a fixed point value
    fn pow(self: F64, b: F64) -> F64 {
        return ops::pow(self, b);
    }

    fn round(self: F64) -> F64 {
        return ops::round(self);
    }

    fn sin(self: F64) -> F64 {
        return trig::sin_fast(self);
    }

    fn sinh(self: F64) -> F64 {
        return hyp::sinh(self);
    }

    // Calculates the square root of a fixed point value
    // x must be positive
    fn sqrt(self: F64) -> F64 {
        return ops::sqrt(self);
    }

    fn tan(self: F64) -> F64 {
        return trig::tan_fast(self);
    }

    fn tanh(self: F64) -> F64 {
        return hyp::tanh(self);
    }

    fn erf(self: F64) -> F64 {
        return erf::erf(self);
    }
}

// Into a raw felt without unscaling
impl FixedIntoFelt252 of Into<F64, felt252> {
    fn into(self: F64) -> felt252 {
        self.d.into()
    }
}

impl FixedTryIntoU128 of TryInto<F64, u128> {
    fn try_into(self: F64) -> Option<u128> {
        if self.d < 0 {
            return Option::None(());
        } else {
            // Unscale the magnitude and round down
            return Option::Some((self.d / ONE).try_into().unwrap());
        }
    }
}

impl FixedTryIntoU64 of TryInto<F64, u64> {
    fn try_into(self: F64) -> Option<u64> {
        if self.d < 0 {
            return Option::None(());
        } else {
            // Unscale the magnitude and round down
            return Option::Some((self.d / ONE).try_into().unwrap());
        }
    }
}

impl FixedTryIntoU32 of TryInto<F64, u32> {
    fn try_into(self: F64) -> Option<u32> {
        if self.d < 0 {
            return Option::None(());
        } else {
            // Unscale the magnitude and round down
            return Option::Some((self.d / ONE).try_into().unwrap());
        }
    }
}

impl FixedTryIntoU16 of TryInto<F64, u16> {
    fn try_into(self: F64) -> Option<u16> {
        if self.d < 0 {
            return Option::None(());
        } else {
            // Unscale the magnitude and round down
            return Option::Some((self.d / ONE).try_into().unwrap());
        }
    }
}

impl FixedTryIntoU8 of TryInto<F64, u8> {
    fn try_into(self: F64) -> Option<u8> {
        if self.d < 0 {
            return Option::None(());
        } else {
            // Unscale the magnitude and round down
            return Option::Some((self.d / ONE).try_into().unwrap());
        }
    }
}

impl U8IntoFixed of Into<u8, F64> {
    fn into(self: u8) -> F64 {
        FixedTrait::new_unscaled(self.into())
    }
}

impl U16IntoFixed of Into<u16, F64> {
    fn into(self: u16) -> F64 {
        FixedTrait::new_unscaled(self.into())
    }
}

impl U32IntoFixed of Into<u32, F64> {
    fn into(self: u32) -> F64 {
        FixedTrait::new_unscaled(self.into())
    }
}


impl F16PartialEq of PartialEq<F64> {
    #[inline(always)]
    fn eq(lhs: @F64, rhs: @F64) -> bool {
        return ops::eq(lhs, rhs);
    }

    #[inline(always)]
    fn ne(lhs: @F64, rhs: @F64) -> bool {
        return ops::ne(lhs, rhs);
    }
}


impl F16Add of Add<F64> {
    fn add(lhs: F64, rhs: F64) -> F64 {
        return ops::add(lhs, rhs);
    }
}

impl F16AddAssign of core::ops::AddAssign<F64, F64> {
    #[inline(always)]
    fn add_assign(ref self: F64, rhs: F64) {
        self = Add::add(self, rhs);
    }
}

impl F16Sub of Sub<F64> {
    fn sub(lhs: F64, rhs: F64) -> F64 {
        return ops::sub(lhs, rhs);
    }
}

impl F16SubAssign of core::ops::SubAssign<F64, F64> {
    #[inline(always)]
    fn sub_assign(ref self: F64, rhs: F64) {
        self = Sub::sub(self, rhs);
    }
}


impl F16Mul of Mul<F64> {
    fn mul(lhs: F64, rhs: F64) -> F64 {
        return ops::mul(lhs, rhs);
    }
}

impl F16MulAssign of core::ops::MulAssign<F64, F64> {
    #[inline(always)]
    fn mul_assign(ref self: F64, rhs: F64) {
        self = Mul::mul(self, rhs);
    }
}

impl F16Div of Div<F64> {
    fn div(lhs: F64, rhs: F64) -> F64 {
        return ops::div(lhs, rhs);
    }
}

impl F16DivAssign of core::ops::DivAssign<F64, F64> {
    #[inline(always)]
    fn div_assign(ref self: F64, rhs: F64) {
        self = Div::div(self, rhs);
    }
}

impl F16PartialOrd of PartialOrd<F64> {
    #[inline(always)]
    fn ge(lhs: F64, rhs: F64) -> bool {
        return ops::ge(lhs, rhs);
    }

    #[inline(always)]
    fn gt(lhs: F64, rhs: F64) -> bool {
        return ops::gt(lhs, rhs);
    }

    #[inline(always)]
    fn le(lhs: F64, rhs: F64) -> bool {
        return ops::le(lhs, rhs);
    }

    #[inline(always)]
    fn lt(lhs: F64, rhs: F64) -> bool {
        return ops::lt(lhs, rhs);
    }
}

impl F16Neg of Neg<F64> {
    #[inline(always)]
    fn neg(a: F64) -> F64 {
        return ops::neg(a);
    }
}

impl F16Rem of Rem<F64> {
    #[inline(always)]
    fn rem(lhs: F64, rhs: F64) -> F64 {
        return ops::rem(lhs, rhs);
    }
}

impl F16Zero of core::num::traits::Zero<F64> {
    fn zero() -> F64 {
        F64 { d: 0 }
    }
    #[inline(always)]
    fn is_zero(self: @F64) -> bool {
        *self.d == 0
    }
    #[inline(always)]
    fn is_non_zero(self: @F64) -> bool {
        !self.is_zero()
    }
}

// One trait implementations
impl F16One of core::num::traits::One<F64> {
    fn one() -> F64 {
        F64 { d: ONE }
    }
    #[inline(always)]
    fn is_one(self: @F64) -> bool {
        *self == Self::one()
    }
    #[inline(always)]
    fn is_non_one(self: @F64) -> bool {
        !self.is_one()
    }
}
