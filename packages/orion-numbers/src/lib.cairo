pub mod f64;

pub trait FixedTrait<T, S> {
    fn ZERO() -> T;
    fn ONE() -> T;

    // Constructors
    fn new(val: S) -> T;
    fn new_unscaled(val: S) -> T;
    fn from_felt(val: felt252) -> T;
    fn from_unscaled_felt(val: felt252) -> T;

    // // Math
    fn abs(self: T) -> T;
    fn ceil(self: T) -> T;
    fn exp(self: T) -> T;
    fn exp2(self: T) -> T;
    fn floor(self: T) -> T;
    fn ln(self: T) -> T;
    fn log2(self: T) -> T;
    fn log10(self: T) -> T;
    fn pow(self: T, b: T) -> T;
    fn round(self: T) -> T;
    fn sqrt(self: T) -> T;

    // Trigonometry
    fn acos(self: T) -> T;
    fn asin(self: T) -> T;
    fn atan(self: T) -> T;
    fn cos(self: T) -> T;
    fn sin(self: T) -> T;
    fn tan(self: T) -> T;

    // Hyperbolic
    fn acosh(self: T) -> T;
    fn asinh(self: T) -> T;
    fn atanh(self: T) -> T;
    fn cosh(self: T) -> T;
    fn sinh(self: T) -> T;
    fn tanh(self: T) -> T;

    fn erf(self: T) -> T;
}

pub use f64::{
    F64, F64Add, F64AddAssign, F64Div, F64DivAssign, F64Mul, F64MulAssign, F64Neg, F64One,
    F64PartialEq, F64PartialOrd, F64Rem, F64Sub, F64SubAssign, F64Zero, F64Impl, F64IntoFelt252,
    F64TryIntoU128, F64TryIntoU16, F64TryIntoU32, F64TryIntoU64, F64TryIntoU8,
};
