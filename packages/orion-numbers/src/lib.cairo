pub mod f16x16;
pub mod f32x32;
pub mod core_trait;

use orion_numbers::f16x16::core::F16x16Impl;
use orion_numbers::f32x32::core::F32x32Impl;


trait FixedTrait<T> {
    fn ZERO() -> T;
    fn HALF() -> T;
    fn ONE() -> T;
    fn MAX() -> T;
    fn MIN() -> T;
    fn new_unscaled(x: T) -> T;
    fn new(x: T) -> T;
    fn from_felt(x: felt252) -> T;
    fn from_unscaled_felt(x: felt252) -> T;
    fn abs(self: T) -> T;
    fn acos(self: T) -> T;
    fn acosh(self: T) -> T;
    fn asin(self: T) -> T;
    fn asinh(self: T) -> T;
    fn atan(self: T) -> T;
    fn atanh(self: T) -> T;
    fn add(lhs: T, rhs: T) -> T;
    fn ceil(self: T) -> T;
    fn cos(self: T) -> T;
    fn cosh(self: T) -> T;
    fn div(self: T, rhs: T) -> T;
    fn exp(self: T) -> T;
    fn exp2(self: T) -> T;
    fn floor(self: T) -> T;
    fn ln(self: T) -> T;
    fn log2(self: T) -> T;
    fn log10(self: T) -> T;
    fn mul(self: T, rhs: T) -> T;
    fn pow(self: T, b: T) -> T;
    fn round(self: T) -> T;
    fn sin(self: T) -> T;
    fn sinh(self: T) -> T;
    fn sqrt(self: T) -> T;
    fn tan(self: T) -> T;
    fn tanh(self: T) -> T;
    fn sign(self: T) -> T;
    fn sub(lhs: T, rhs: T) -> T;
    fn NaN() -> T;
    fn is_nan(self: T) -> bool;
    fn INF() -> T;
    fn POS_INF() -> T;
    fn NEG_INF() -> T;
    fn is_inf(self: T) -> bool;
    fn is_pos_inf(self: T) -> bool;
    fn is_neg_inf(self: T) -> bool;
    fn erf(self: T) -> T;
}
