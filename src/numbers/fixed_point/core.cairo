/// A struct representing a fixed point number.
#[derive(Serde, Copy, Drop)]
struct FixedType {
    mag: u32,
    sign: bool
}

/// A struct listing fixed point implementations.
#[derive(Serde, Copy, Drop)]
enum FixedImpl {
    FP8x23: (),
    FP16x16: ()
}

/// Trait
///
/// new - Constructs a new fixed point instance.
/// new_unscaled - Creates a new fixed point instance with the specified unscaled magnitude and sign.
/// from_felt - Creates a new fixed point instance from a `felt252` value.
/// from_unscaled_felt - Creates a new fixed point instance from an unscaled `felt252` value.
/// abs - Returns the absolute value of the fixed point number.
/// ceil - Returns the smallest integer greater than or equal to the fixed point number.
/// floor - Returns the largest integer less than or equal to the fixed point number.
/// exp - Returns the value of e raised to the power of the fixed point number. 
/// exp2 - Returns the value of 2 raised to the power of the fixed point number.
/// log - Returns the natural logarithm of the fixed point number.
/// log2 - Returns the base-2 logarithm of the fixed point number.
/// log10 - Returns the base-10 logarithm of the fixed point number.
/// pow - Returns the result of raising the fixed point number to the power of another fixed point number
/// round - Rounds the fixed point number to the nearest whole number.
/// sqrt - Returns the square root of the fixed point number.
/// sin - Returns the sine of the fixed point number.
/// cos - Returns the cosine of the fixed point number.
/// asin - Returns the arcsine (inverse of sine) of the fixed point number.
/// sinh - Returns the value of the hyperbolic sine of the fixed point number.
/// tanh - Returns the value of the hyperbolic tangent of the fixed point number.
/// cosh - Returns the value of the hyperbolic cosine of the fixed point number.
/// acosh - Returns the value of the inverse hyperbolic cosine of the fixed point number.
/// asinh - Returns the inverse hyperbolic sine of the input fixed point number.
/// atan - Returns the arctangent (inverse of tangent) of the input fixed point number.
/// acos - Returns the arccosine (inverse of cosine) of the fixed point number.
/// 
trait FixedTrait {
    fn ZERO() -> FixedType;
    fn ONE() -> FixedType;

    // Constructors
    fn new(mag: u32, sign: bool) -> FixedType;
    fn new_unscaled(mag: u32, sign: bool) -> FixedType;
    fn from_felt(val: felt252) -> FixedType;

    // Math
    fn abs(self: FixedType) -> FixedType;
    fn ceil(self: FixedType) -> FixedType;
    fn exp(self: FixedType) -> FixedType;
    fn exp2(self: FixedType) -> FixedType;
    fn floor(self: FixedType) -> FixedType;
    fn ln(self: FixedType) -> FixedType;
    fn log2(self: FixedType) -> FixedType;
    fn log10(self: FixedType) -> FixedType;
    fn pow(self: FixedType, b: FixedType) -> FixedType;
    fn round(self: FixedType) -> FixedType;
    fn sqrt(self: FixedType) -> FixedType;

    // Trigonometry
    fn acos(self: FixedType) -> FixedType;
    fn acos_fast(self: FixedType) -> FixedType;
    fn asin(self: FixedType) -> FixedType;
    fn asin_fast(self: FixedType) -> FixedType;
    fn atan(self: FixedType) -> FixedType;
    fn atan_fast(self: FixedType) -> FixedType;
    fn cos(self: FixedType) -> FixedType;
    fn cos_fast(self: FixedType) -> FixedType;
    fn sin(self: FixedType) -> FixedType;
    fn sin_fast(self: FixedType) -> FixedType;
    fn tan(self: FixedType) -> FixedType;
    fn tan_fast(self: FixedType) -> FixedType;

    // Hyperbolic
    fn acosh(self: FixedType) -> FixedType;
    fn asinh(self: FixedType) -> FixedType;
    fn atanh(self: FixedType) -> FixedType;
    fn cosh(self: FixedType) -> FixedType;
    fn sinh(self: FixedType) -> FixedType;
    fn tanh(self: FixedType) -> FixedType;
}
