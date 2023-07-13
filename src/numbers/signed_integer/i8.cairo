use traits::Into;

use orion::numbers::signed_integer::integer_trait::IntegerTrait;
use orion::numbers::signed_integer::i32::i32;
use orion::numbers::fixed_point::implementations::impl_8x23::ONE as ONE_fp8x23;
use orion::numbers::fixed_point::implementations::impl_16x16::ONE as ONE_fp16x16;
use orion::numbers::fixed_point::core::{FixedType, FixedTrait};


// ====================== INT 8 ======================

// i8 represents a 8-bit integer.
// The mag field holds the absolute value of the integer.
// The sign field is true for negative integers, and false for non-negative integers.
#[derive(Copy, Drop)]
struct i8 {
    mag: u8,
    sign: bool,
}

impl i8Impl of IntegerTrait<i8, u8> {
    fn new(mag: u8, sign: bool) -> i8 {
        i8_new(mag, sign)
    }

    fn div_rem(self: i8, other: i8) -> (i8, i8) {
        i8_div_rem(self, other)
    }

    fn abs(self: i8) -> i8 {
        i8_abs(self)
    }

    fn max(self: i8, other: i8) -> i8 {
        i8_max(self, other)
    }

    fn min(self: i8, other: i8) -> i8 {
        i8_min(self, other)
    }
}

// Implements the Into trait for i8.
impl i8Into of Into<i8, felt252> {
    fn into(self: i8) -> felt252 {
        let mag_felt = self.mag.into();

        if (self.sign == true) {
            return mag_felt * -1;
        } else {
            return mag_felt;
        }
    }
}

// Implements the Add trait for i8.
impl i8Add of Add<i8> {
    fn add(lhs: i8, rhs: i8) -> i8 {
        i8_add(lhs, rhs)
    }
}

// Implements the AddEq trait for i8.
impl i8AddEq of AddEq<i8> {
    #[inline(always)]
    fn add_eq(ref self: i8, other: i8) {
        self = Add::add(self, other);
    }
}

// Implements the Sub trait for i8.
impl i8Sub of Sub<i8> {
    fn sub(lhs: i8, rhs: i8) -> i8 {
        i8_sub(lhs, rhs)
    }
}

// Implements the SubEq trait for i8.
impl i8SubEq of SubEq<i8> {
    #[inline(always)]
    fn sub_eq(ref self: i8, other: i8) {
        self = Sub::sub(self, other);
    }
}

// Implements the Mul trait for i8.
impl i8Mul of Mul<i8> {
    fn mul(lhs: i8, rhs: i8) -> i8 {
        i8_mul(lhs, rhs)
    }
}

// Implements the MulEq trait for i8.
impl i8MulEq of MulEq<i8> {
    #[inline(always)]
    fn mul_eq(ref self: i8, other: i8) {
        self = Mul::mul(self, other);
    }
}

// Implements the Div trait for i8.
impl i8Div of Div<i8> {
    fn div(lhs: i8, rhs: i8) -> i8 {
        i8_div(lhs, rhs)
    }
}

// Implements the DivEq trait for i8.
impl i8DivEq of DivEq<i8> {
    #[inline(always)]
    fn div_eq(ref self: i8, other: i8) {
        self = Div::div(self, other);
    }
}

// Implements the Rem trait for i8.
impl i8Rem of Rem<i8> {
    fn rem(lhs: i8, rhs: i8) -> i8 {
        i8_rem(lhs, rhs)
    }
}

// Implements the RemEq trait for i8.
impl i8RemEq of RemEq<i8> {
    #[inline(always)]
    fn rem_eq(ref self: i8, other: i8) {
        self = Rem::rem(self, other);
    }
}

// Implements the PartialEq trait for i8.
impl i8PartialEq of PartialEq<i8> {
    fn eq(lhs: @i8, rhs: @i8) -> bool {
        i8_eq(*lhs, *rhs)
    }

    fn ne(lhs: @i8, rhs: @i8) -> bool {
        i8_ne(*lhs, *rhs)
    }
}

// Implements the PartialOrd trait for i8.
impl i8PartialOrd of PartialOrd<i8> {
    fn le(lhs: i8, rhs: i8) -> bool {
        i8_le(rhs, lhs)
    }
    fn ge(lhs: i8, rhs: i8) -> bool {
        i8_ge(lhs, rhs)
    }

    fn lt(lhs: i8, rhs: i8) -> bool {
        i8_lt(lhs, rhs)
    }
    fn gt(lhs: i8, rhs: i8) -> bool {
        i8_gt(lhs, rhs)
    }
}

// Implements the Neg trait for i8.
impl i8Neg of Neg<i8> {
    fn neg(a: i8) -> i8 {
        i8_neg(a)
    }
}

// Implements the Into trait for i8 to i32.
impl I8IntoI32 of Into<i8, i32> {
    fn into(self: i8) -> i32 {
        i8_to_i32(self)
    }
}

// Implements the Into trait for i8 to fp_8x23.
impl I8IntoFP8x23 of Into<i8, FixedType> {
    fn into(self: i8) -> FixedType {
        i8_to_fp8x23(self)
    }
}

// Implements the Into trait for i8 to fp_16x16.
impl I8IntoFP16x16 of Into<i8, FixedType> {
    fn into(self: i8) -> FixedType {
        i8_to_fp16x16(self)
    }
}

// Checks if the given i8 integer is zero and has the correct sign.
// # Arguments
// * `x` - The i8 integer to check.
// # Panics
// Panics if `x` is zero and has a sign that is not false.
fn i8_check_sign_zero(x: i8) {
    if x.mag == 0_u8 {
        assert(x.sign == false, 'sign of 0 must be false');
    }
}

/// Cf: IntegerTrait::new docstring
fn i8_new(mag: u8, sign: bool) -> i8 {
    if sign == true {
        assert(mag <= 128_u8, 'int: out of range');
    } else {
        assert(mag <= 127_u8, 'int: out of range');
    }
    i8 { mag, sign }
}

// Adds two i8 integers.
// # Arguments
// * `a` - The first i8 to add.
// * `b` - The second i8 to add.
// # Returns
// * `i8` - The sum of `a` and `b`.
fn i8_add(a: i8, b: i8) -> i8 {
    i8_check_sign_zero(a);
    i8_check_sign_zero(b);

    // If both integers have the same sign, 
    // the sum of their absolute values can be returned.
    if a.sign == b.sign {
        let sum = a.mag + b.mag;
        if (sum == 0_u8) {
            return IntegerTrait::new(sum, false);
        }
        return IntegerTrait::new(sum, a.sign);
    } else {
        // If the integers have different signs, 
        // the larger absolute value is subtracted from the smaller one.
        let (larger, smaller) = if a.mag >= b.mag {
            (a, b)
        } else {
            (b, a)
        };
        let difference = larger.mag - smaller.mag;

        if (difference == 0_u8) {
            return IntegerTrait::new(difference, false);
        }
        return IntegerTrait::new(difference, larger.sign);
    }
}

// Subtracts two i8 integers.
// # Arguments
// * `a` - The first i8 to subtract.
// * `b` - The second i8 to subtract.
// # Returns
// * `i8` - The difference of `a` and `b`.
fn i8_sub(a: i8, b: i8) -> i8 {
    i8_check_sign_zero(a);
    i8_check_sign_zero(b);

    if (b.mag == 0_u8) {
        return a;
    }

    // The subtraction of `a` to `b` is achieved by negating `b` sign and adding it to `a`.
    let neg_b = IntegerTrait::new(b.mag, !b.sign);
    return a + neg_b;
}

// Multiplies two i8 integers.
// 
// # Arguments
//
// * `a` - The first i8 to multiply.
// * `b` - The second i8 to multiply.
//
// # Returns
//
// * `i8` - The product of `a` and `b`.
fn i8_mul(a: i8, b: i8) -> i8 {
    i8_check_sign_zero(a);
    i8_check_sign_zero(b);

    // The sign of the product is the XOR of the signs of the operands.
    let sign = a.sign ^ b.sign;
    // The product is the product of the absolute values of the operands.
    let mag = a.mag * b.mag;

    if (mag == 0_u8) {
        return IntegerTrait::new(mag, false);
    }

    return IntegerTrait::new(mag, sign);
}

// Divides the first i8 by the second i8.
// # Arguments
// * `a` - The i8 dividend.
// * `b` - The i8 divisor.
// # Returns
// * `i8` - The quotient of `a` and `b`.
fn i8_div(a: i8, b: i8) -> i8 {
    i8_check_sign_zero(a);
    // Check that the divisor is not zero.
    assert(b.mag != 0_u8, 'b can not be 0');

    // The sign of the quotient is the XOR of the signs of the operands.
    let sign = a.sign ^ b.sign;

    if (sign == false) {
        // If the operands are positive, the quotient is simply their absolute value quotient.
        return IntegerTrait::new(a.mag / b.mag, sign);
    }

    // If the operands have different signs, rounding is necessary.
    // First, check if the quotient is an integer.
    if (a.mag % b.mag == 0_u8) {
        let quotient = a.mag / b.mag;
        if (quotient == 0_u8) {
            return IntegerTrait::new(quotient, false);
        }
        return IntegerTrait::new(quotient, sign);
    }

    // If the quotient is not an integer, multiply the dividend by 10 to move the decimal point over.
    let quotient = (a.mag * 10_u8) / b.mag;
    let last_digit = quotient % 10_u8;

    if (quotient == 0_u8) {
        return IntegerTrait::new(quotient, false);
    }

    // Check the last digit to determine rounding direction.
    if (last_digit <= 5_u8) {
        return IntegerTrait::new(quotient / 10_u8, sign);
    } else {
        return IntegerTrait::new((quotient / 10_u8) + 1_u8, sign);
    }
}

// Calculates the remainder of the division of a first i8 by a second i8.
// # Arguments
// * `a` - The i8 dividend.
// * `b` - The i8 divisor.
// # Returns
// * `i8` - The remainder of dividing `a` by `b`.
fn i8_rem(a: i8, b: i8) -> i8 {
    i8_check_sign_zero(a);
    // Check that the divisor is not zero.
    assert(b.mag != 0_u8, 'b can not be 0');

    return a - (b * (a / b));
}

/// Cf: IntegerTrait::div_rem docstring
fn i8_div_rem(a: i8, b: i8) -> (i8, i8) {
    let quotient = i8_div(a, b);
    let remainder = i8_rem(a, b);

    return (quotient, remainder);
}

// Compares two i8 integers for equality.
// # Arguments
// * `a` - The first i8 integer to compare.
// * `b` - The second i8 integer to compare.
// # Returns
// * `bool` - `true` if the two integers are equal, `false` otherwise.
fn i8_eq(a: i8, b: i8) -> bool {
    // Check if the two integers have the same sign and the same absolute value.
    if a.sign == b.sign && a.mag == b.mag {
        return true;
    }

    return false;
}

// Compares two i8 integers for inequality.
// # Arguments
// * `a` - The first i8 integer to compare.
// * `b` - The second i8 integer to compare.
// # Returns
// * `bool` - `true` if the two integers are not equal, `false` otherwise.
fn i8_ne(a: i8, b: i8) -> bool {
    // The result is the inverse of the equal function.
    return !i8_eq(a, b);
}

// Compares two i8 integers for greater than.
// # Arguments
// * `a` - The first i8 integer to compare.
// * `b` - The second i8 integer to compare.
// # Returns
// * `bool` - `true` if `a` is greater than `b`, `false` otherwise.
fn i8_gt(a: i8, b: i8) -> bool {
    // Check if `a` is negative and `b` is positive.
    if (a.sign & !b.sign) {
        return false;
    }
    // Check if `a` is positive and `b` is negative.
    if (!a.sign & b.sign) {
        return true;
    }
    // If `a` and `b` have the same sign, compare their absolute values.
    if (a.sign & b.sign) {
        return a.mag < b.mag;
    } else {
        return a.mag > b.mag;
    }
}

// Determines whether the first i8 is less than the second i8.
// # Arguments
// * `a` - The i8 to compare against the second i8.
// * `b` - The i8 to compare against the first i8.
// # Returns
// * `bool` - `true` if `a` is less than `b`, `false` otherwise.
fn i8_lt(a: i8, b: i8) -> bool {
    if (a.sign != b.sign) {
        return a.sign;
    } else {
        return a.mag != b.mag && (a.mag < b.mag) ^ a.sign;
    }
}

// Checks if the first i8 integer is less than or equal to the second.
// # Arguments
// * `a` - The first i8 integer to compare.
// * `b` - The second i8 integer to compare.
// # Returns
// * `bool` - `true` if `a` is less than or equal to `b`, `false` otherwise.
fn i8_le(a: i8, b: i8) -> bool {
    if (a == b || i8_lt(a, b) == true) {
        return true;
    } else {
        return false;
    }
}

// Checks if the first i8 integer is greater than or equal to the second.
// # Arguments
// * `a` - The first i8 integer to compare.
// * `b` - The second i8 integer to compare.
// # Returns
// * `bool` - `true` if `a` is greater than or equal to `b`, `false` otherwise.
fn i8_ge(a: i8, b: i8) -> bool {
    if (a == b || i8_gt(a, b) == true) {
        return true;
    } else {
        return false;
    }
}

// Negates the given i8 integer.
// # Arguments
// * `x` - The i8 integer to negate.
// # Returns
// * `i8` - The negation of `x`.
fn i8_neg(x: i8) -> i8 {
    // The negation of an integer is obtained by flipping its sign.
    return IntegerTrait::new(x.mag, !x.sign);
}

/// Cf: IntegerTrait::abs docstring
fn i8_abs(x: i8) -> i8 {
    return IntegerTrait::new(x.mag, false);
}

/// Cf: IntegerTrait::max docstring
fn i8_max(a: i8, b: i8) -> i8 {
    if (a > b) {
        return a;
    } else {
        return b;
    }
}

/// Cf: IntegerTrait::min docstring
fn i8_min(a: i8, b: i8) -> i8 {
    if (a < b) {
        return a;
    } else {
        return b;
    }
}

fn i8_to_i32(x: i8) -> i32 {
    i32 { mag: x.mag.into(), sign: x.sign }
}

use debug::PrintTrait;

fn i8_to_fp8x23(x: i8) -> FixedType {
    FixedType { mag: x.mag.into() * ONE_fp8x23, sign: x.sign }
}

fn i8_to_fp16x16(x: i8) -> FixedType {
    FixedType { mag: x.mag.into() * ONE_fp16x16, sign: x.sign }
}
