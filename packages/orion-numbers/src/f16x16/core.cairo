use orion_numbers::f16x16::{math, trig, erf};
use core::traits::TryInto;
use core::option::OptionTrait;
use core::traits::Mul;

pub type f16x16 = i32;

// CONSTANTS
pub const TWO: f16x16 = 131072; // 2 ** 17
pub const ONE: f16x16 = 65536; // 2 ** 16
pub const HALF: f16x16 = 32768; // 2 ** 15
pub const MAX: f16x16 = 2147483647; // 2 ** 31 -1 
pub const MIN: f16x16 = -2147483648; // 2 ** 31

#[generate_trait]
pub impl f16x16Impl of FixedTrait {
    fn ZERO() -> f16x16 {
        0
    }

    fn HALF() -> f16x16 {
        HALF
    }

    fn ONE() -> f16x16 {
        ONE
    }

    fn MAX() -> f16x16 {
        MAX
    }

    fn MIN() -> f16x16 {
        MIN
    }

    fn new_unscaled(x: i32) -> f16x16 {
        x * ONE
    }

    fn new(x: i32) -> f16x16 {
        x
    }

    fn from_felt(x: felt252) -> f16x16 {
        x.try_into().unwrap()
    }

    fn from_unscaled_felt(x: felt252) -> f16x16 {
        return Self::from_felt(x * ONE.into());
    }

    fn abs(self: f16x16) -> f16x16 {
        math::abs(self)
    }

    fn acos(self: f16x16) -> f16x16 {
        trig::acos_fast(self)
    }

    fn acosh(self: f16x16) -> f16x16 {
        trig::acosh(self)
    }

    fn asin(self: f16x16) -> f16x16 {
        trig::asin_fast(self)
    }

    fn asinh(self: f16x16) -> f16x16 {
        trig::asinh(self)
    }

    fn atan(self: f16x16) -> f16x16 {
        trig::atan_fast(self)
    }

    fn atanh(self: f16x16) -> f16x16 {
        trig::atanh(self)
    }

    fn add(lhs: f16x16, rhs: f16x16) -> f16x16 {
        math::add(lhs, rhs)
    }

    fn ceil(self: f16x16) -> f16x16 {
        math::ceil(self)
    }

    fn cos(self: f16x16) -> f16x16 {
        trig::cos_fast(self)
    }

    fn cosh(self: f16x16) -> f16x16 {
        trig::cosh(self)
    }

    fn div(lhs: f16x16, rhs: f16x16) -> f16x16 {
        math::div(lhs, rhs)
    }

    // Calculates the natural exponent of x: e^x
    fn exp(self: f16x16) -> f16x16 {
        math::exp(self)
    }

    // Calculates the binary exponent of x: 2^x
    fn exp2(self: f16x16) -> f16x16 {
        math::exp2(self)
    }

    fn floor(self: f16x16) -> f16x16 {
        math::floor(self)
    }

    // Calculates the natural logarithm of x: ln(x)
    // self must be greater than zero
    fn ln(self: f16x16) -> f16x16 {
        math::ln(self)
    }

    // Calculates the binary logarithm of x: log2(x)
    // self must be greather than zero
    fn log2(self: f16x16) -> f16x16 {
        math::log2(self)
    }

    // Calculates the base 10 log of x: log10(x)
    // self must be greater than zero
    fn log10(self: f16x16) -> f16x16 {
        math::log10(self)
    }

    fn mul(lhs: f16x16, rhs: f16x16) -> f16x16 {
        math::mul(lhs, rhs)
    }

    // Calclates the value of x^y and checks for overflow before returning
    // self is a fixed point value
    // b is a fixed point value
    fn pow(self: f16x16, b: f16x16) -> f16x16 {
        math::pow(self, b)
    }

    fn round(self: f16x16) -> f16x16 {
        math::round(self)
    }

    fn sin(self: f16x16) -> f16x16 {
        trig::sin_fast(self)
    }

    fn sinh(self: f16x16) -> f16x16 {
        trig::sinh(self)
    }

    // Calculates the square root of a fixed point value
    // x must be positive
    fn sqrt(self: f16x16) -> f16x16 {
        math::sqrt(self)
    }

    fn tan(self: f16x16) -> f16x16 {
        trig::tan_fast(self)
    }

    fn tanh(self: f16x16) -> f16x16 {
        trig::tanh(self)
    }

    fn sign(self: f16x16) -> f16x16 {
        math::sign(self)
    }

    fn sub(lhs: f16x16, rhs: f16x16) -> f16x16 {
        math::sub(lhs, rhs)
    }

    fn NaN() -> f16x16 {
        -0
    }

    fn is_nan(self: f16x16) -> bool {
        self == -0
    }

    fn INF() -> f16x16 {
        MAX
    }

    fn POS_INF() -> f16x16 {
        MAX
    }

    fn NEG_INF() -> f16x16 {
        MIN
    }

    fn is_inf(self: f16x16) -> bool {
        self == MAX
    }

    fn is_pos_inf(self: f16x16) -> bool {
        self == MAX
    }

    fn is_neg_inf(self: f16x16) -> bool {
        self == MIN
    }

    fn erf(self: f16x16) -> f16x16 {
        erf::erf(self)
    }
}
