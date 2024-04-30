use core::debug::PrintTrait;

use orion::numbers::fixed_point::core::FixedTrait;
use orion::numbers::fixed_point::implementations::fp16x16::math::{
    core as core_math, trig, hyp, erf
};
use orion::numbers::fixed_point::utils;


// CONSTANTS
const TWO: u32 = 131072; // 2 ** 17
const ONE: u32 = 65536; // 2 ** 16
const HALF: u32 = 32768; // 2 ** 15
const MAX: u32 = 2147483648; // 2 ** 31

impl fp16Impl of FixedTrait<i32> {
    fn ZERO() -> i32 {
       0
    }

    fn HALF() -> i32 {
       HALF
    }

    fn ONE() -> i32 {
       ONE
    }

    fn MAX() -> i32 {
       MAX
    }

    fn abs(self: i32) -> i32 {
        core_math::abs(self)
    }

    fn acos(self: i32) -> i32 {
        trig::acos_fast(self)
    }

    fn acos_fast(self: i32) -> i32 {
        trig::acos_fast(self)
    }

    fn acosh(self: i32) -> i32 {
        hyp::acosh(self)
    }

    fn asin(self: i32) -> i32 {
        trig::asin_fast(self)
    }

    fn asin_fast(self: i32) -> i32 {
        trig::asin_fast(self)
    }

    fn asinh(self: i32) -> i32 {
        hyp::asinh(self)
    }

    fn atan(self: i32) -> i32 {
        trig::atan_fast(self)
    }

    fn atan_fast(self: i32) -> i32 {
        trig::atan_fast(self)
    }

    fn atanh(self: i32) -> i32 {
        hyp::atanh(self)
    }

    fn ceil(self: i32) -> i32 {
        core_math::ceil(self)
    }

    fn cos(self: i32) -> i32 {
        trig::cos_fast(self)
    }

    fn cos_fast(self: i32) -> i32 {
        trig::cos_fast(self)
    }

    fn cosh(self: i32) -> i32 {
        hyp::cosh(self)
    }

    fn floor(self: i32) -> i32 {
        core_math::floor(self)
    }

    // Calculates the natural exponent of x: e^x
    fn exp(self: i32) -> i32 {
        core_math::exp(self)
    }

    // Calculates the binary exponent of x: 2^x
    fn exp2(self: i32) -> i32 {
        core_math::exp2(self)
    }

    // Calculates the natural logarithm of x: ln(x)
    // self must be greater than zero
    fn ln(self: i32) -> i32 {
        core_math::ln(self)
    }

    // Calculates the binary logarithm of x: log2(x)
    // self must be greather than zero
    fn log2(self: i32) -> i32 {
        core_math::log2(self)
    }

    // Calculates the base 10 log of x: log10(x)
    // self must be greater than zero
    fn log10(self: i32) -> i32 {
        core_math::log10(self)
    }

    // Calclates the value of x^y and checks for overflow before returning
    // self is a fixed point value
    // b is a fixed point value
    fn pow(self: i32, b: i32) -> i32 {
        core_math::pow(self, b)
    }

    fn round(self: i32) -> i32 {
        core_math::round(self)
    }

    fn sin(self: i32) -> i32 {
        trig::sin_fast(self)
    }

    fn sin_fast(self: i32) -> i32 {
        trig::sin_fast(self)
    }

    fn sinh(self: i32) -> i32 {
        hyp::sinh(self)
    }

    // Calculates the square root of a fixed point value
    // x must be positive
    fn sqrt(self: i32) -> i32 {
        core_math::sqrt(self)
    }

    fn tan(self: FP16x16) -> FP16x16 {
        trig::tan_fast(self)
    }

    fn tan_fast(self: FP16x16) -> FP16x16 {
        trig::tan_fast(self)
    }

    fn tanh(self: FP16x16) -> FP16x16 {
        hyp::tanh(self)
    }

    fn sign(self: FP16x16) -> FP16x16 {
        core_math::sign(self)
    }

   
}
