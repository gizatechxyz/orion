pub(crate) mod ops;
pub(crate) mod trig;
pub(crate) mod lut;
pub(crate) mod comp;
pub(crate) mod erf;
pub mod helpers;

use orion_numbers::FixedTrait;

// CONSTANTS

pub const TWO: i64 = 8589934592; // 2 ** 33
pub const ONE: i64 = 4294967296; // 2 ** 32
pub const HALF: i64 = 2147483648; // 2 ** 31
pub const MAX: i64 = 9223372036854775807; //2**63 - 1
const MIN: i64 = -9223372036854775808; // -2 ** 63
pub const NaN: i64 = 0x4e614e;

// STRUCTS

#[derive(Copy, Drop, Serde)]
pub struct F64 {
    pub d: i64,
}


pub impl F64Impl of FixedTrait<F64, i64> {
    fn ZERO() -> F64 {
        return core::num::traits::Zero::zero();
    }

    fn HALF() -> F64 {
        return F64 { d: HALF };
    }

    fn ONE() -> F64 {
        return core::num::traits::One::one();
    }

    fn TWO() -> F64 {
        return F64 { d: TWO };
    }

    fn MIN() -> F64 {
        return F64 { d: MIN };
    }

    fn MAX() -> F64 {
        return F64 { d: MAX };
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
        panic!("Not implemented yet!")
    }

    fn acosh(self: F64) -> F64 {
        panic!("Not implemented yet!")
    }

    fn asin(self: F64) -> F64 {
        panic!("Not implemented yet!")
    }


    fn asinh(self: F64) -> F64 {
        panic!("Not implemented yet!")
    }

    fn atan(self: F64) -> F64 {
        panic!("Not implemented yet!")
    }

    fn atanh(self: F64) -> F64 {
        panic!("Not implemented yet!")
    }

    fn ceil(self: F64) -> F64 {
        return ops::ceil(self);
    }

    fn cos(self: F64) -> F64 {
        panic!("Not implemented yet!")
    }

    fn cosh(self: F64) -> F64 {
        panic!("Not implemented yet!")
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
        panic!("Not implemented yet!")
    }

    // Calculates the square root of a fixed point value
    // x must be positive
    fn sqrt(self: F64) -> F64 {
        return ops::sqrt(self);
    }

    fn tan(self: F64) -> F64 {
        panic!("Not implemented yet!")
    }

    fn tanh(self: F64) -> F64 {
        panic!("Not implemented yet!")
    }

    fn erf(self: F64) -> F64 {
        return erf::erf(self);
    }
}

// Into a raw felt without unscaling
pub impl F64IntoFelt252 of Into<F64, felt252> {
    fn into(self: F64) -> felt252 {
        self.d.into()
    }
}

pub impl F64TryIntoU128 of TryInto<F64, u128> {
    fn try_into(self: F64) -> Option<u128> {
        if self.d < 0 {
            return Option::None(());
        } else {
            // Unscale the magnitude and round down
            return Option::Some((self.d / ONE).try_into().unwrap());
        }
    }
}

pub impl F64TryIntoU64 of TryInto<F64, u64> {
    fn try_into(self: F64) -> Option<u64> {
        if self.d < 0 {
            return Option::None(());
        } else {
            // Unscale the magnitude and round down
            return Option::Some((self.d / ONE).try_into().unwrap());
        }
    }
}

pub impl F64TryIntoU32 of TryInto<F64, u32> {
    fn try_into(self: F64) -> Option<u32> {
        if self.d < 0 {
            return Option::None(());
        } else {
            // Unscale the magnitude and round down
            return Option::Some((self.d / ONE).try_into().unwrap());
        }
    }
}

pub impl F64TryIntoU16 of TryInto<F64, u16> {
    fn try_into(self: F64) -> Option<u16> {
        if self.d < 0 {
            return Option::None(());
        } else {
            // Unscale the magnitude and round down
            return Option::Some((self.d / ONE).try_into().unwrap());
        }
    }
}

pub impl F64TryIntoU8 of TryInto<F64, u8> {
    fn try_into(self: F64) -> Option<u8> {
        if self.d < 0 {
            return Option::None(());
        } else {
            // Unscale the magnitude and round down
            return Option::Some((self.d / ONE).try_into().unwrap());
        }
    }
}

pub impl U8IntoF64 of Into<u8, F64> {
    fn into(self: u8) -> F64 {
        FixedTrait::new_unscaled(self.into())
    }
}

pub impl U16IntoF64 of Into<u16, F64> {
    fn into(self: u16) -> F64 {
        FixedTrait::new_unscaled(self.into())
    }
}

pub impl U32IntoF64 of Into<u32, F64> {
    fn into(self: u32) -> F64 {
        FixedTrait::new_unscaled(self.into())
    }
}


pub impl F64PartialEq of PartialEq<F64> {
    #[inline(always)]
    fn eq(lhs: @F64, rhs: @F64) -> bool {
        return ops::eq(lhs, rhs);
    }

    #[inline(always)]
    fn ne(lhs: @F64, rhs: @F64) -> bool {
        return ops::ne(lhs, rhs);
    }
}


pub impl F64Add of Add<F64> {
    fn add(lhs: F64, rhs: F64) -> F64 {
        return ops::add(lhs, rhs);
    }
}

pub impl F64AddAssign of core::ops::AddAssign<F64, F64> {
    #[inline(always)]
    fn add_assign(ref self: F64, rhs: F64) {
        self = Add::add(self, rhs);
    }
}

pub impl F64Sub of Sub<F64> {
    fn sub(lhs: F64, rhs: F64) -> F64 {
        return ops::sub(lhs, rhs);
    }
}

pub impl F64SubAssign of core::ops::SubAssign<F64, F64> {
    #[inline(always)]
    fn sub_assign(ref self: F64, rhs: F64) {
        self = Sub::sub(self, rhs);
    }
}


pub impl F64Mul of Mul<F64> {
    fn mul(lhs: F64, rhs: F64) -> F64 {
        return ops::mul(lhs, rhs);
    }
}

pub impl F64MulAssign of core::ops::MulAssign<F64, F64> {
    #[inline(always)]
    fn mul_assign(ref self: F64, rhs: F64) {
        self = Mul::mul(self, rhs);
    }
}

pub impl F64Div of Div<F64> {
    fn div(lhs: F64, rhs: F64) -> F64 {
        return ops::div(lhs, rhs);
    }
}

pub impl F64DivAssign of core::ops::DivAssign<F64, F64> {
    #[inline(always)]
    fn div_assign(ref self: F64, rhs: F64) {
        self = Div::div(self, rhs);
    }
}

pub impl F64PartialOrd of PartialOrd<F64> {
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

pub impl F64Neg of Neg<F64> {
    #[inline(always)]
    fn neg(a: F64) -> F64 {
        return ops::neg(a);
    }
}

pub impl F64Rem of Rem<F64> {
    #[inline(always)]
    fn rem(lhs: F64, rhs: F64) -> F64 {
        return ops::rem(lhs, rhs);
    }
}

pub impl F64Zero of core::num::traits::Zero<F64> {
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
pub impl F64One of core::num::traits::One<F64> {
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
