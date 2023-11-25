use core::option::OptionTrait;
use core::traits::TryInto;
use traits::Into;

use orion::numbers::signed_integer::integer_trait::IntegerTrait;
use orion::numbers::signed_integer::i8::i8;

// ====================== INT 32 ======================

// i32 represents a 32-bit integer.
// The mag field holds the absolute value of the integer.
// The sign field is true for negative integers, and false for non-negative integers.
#[derive(Serde, Copy, Drop)]
struct i32 {
    mag: u32,
    sign: bool,
}

impl i32Impl of IntegerTrait<i32, u32> {
    fn new(mag: u32, sign: bool) -> i32 {
        i32_new(mag, sign)
    }

    fn div_rem(self: i32, other: i32) -> (i32, i32) {
        i32_div_rem(self, other)
    }

    fn abs(self: i32) -> i32 {
        i32_abs(self)
    }

    fn max(self: i32, other: i32) -> i32 {
        i32_max(self, other)
    }

    fn min(self: i32, other: i32) -> i32 {
        i32_min(self, other)
    }

    fn sign(self: i32) -> i32 {
        i32_sign(self)
    }
}

// Implements the Into trait for u32 -> i32
impl u32Intoi32 of Into<u32, i32> {
    fn into(self: u32) -> i32 {
        i32 {
            mag: self.into(),
            sign: false
        }
    }
}

// Implements the Into trait for i32.
impl i32Into of Into<i32, felt252> {
    fn into(self: i32) -> felt252 {
        let mag_felt = self.mag.into();

        if (self.sign == true) {
            return mag_felt * -1;
        } else {
            return mag_felt;
        }
    }
}

// Implements the Add trait for i32.
impl i32Add of Add<i32> {
    fn add(lhs: i32, rhs: i32) -> i32 {
        i32_add(lhs, rhs)
    }
}

// Implements the AddEq trait for i32.
impl i32AddEq of AddEq<i32> {
    #[inline(always)]
    fn add_eq(ref self: i32, other: i32) {
        self = Add::add(self, other);
    }
}

// Implements the Sub trait for i32.
impl i32Sub of Sub<i32> {
    fn sub(lhs: i32, rhs: i32) -> i32 {
        i32_sub(lhs, rhs)
    }
}

// Implements the SubEq trait for i32.
impl i32SubEq of SubEq<i32> {
    #[inline(always)]
    fn sub_eq(ref self: i32, other: i32) {
        self = Sub::sub(self, other);
    }
}

// Implements the Mul trait for i32.
impl i32Mul of Mul<i32> {
    fn mul(lhs: i32, rhs: i32) -> i32 {
        i32_mul(lhs, rhs)
    }
}

// Implements the MulEq trait for i32.
impl i32MulEq of MulEq<i32> {
    #[inline(always)]
    fn mul_eq(ref self: i32, other: i32) {
        self = Mul::mul(self, other);
    }
}

// Implements the Div trait for i32.
impl i32Div of Div<i32> {
    fn div(lhs: i32, rhs: i32) -> i32 {
        i32_div(lhs, rhs)
    }
}

// Implements the DivEq trait for i32.
impl i32DivEq of DivEq<i32> {
    #[inline(always)]
    fn div_eq(ref self: i32, other: i32) {
        self = Div::div(self, other);
    }
}

// Implements the Rem trait for i32.
impl i32Rem of Rem<i32> {
    fn rem(lhs: i32, rhs: i32) -> i32 {
        i32_rem(lhs, rhs)
    }
}

// Implements the RemEq trait for i32.
impl i32RemEq of RemEq<i32> {
    #[inline(always)]
    fn rem_eq(ref self: i32, other: i32) {
        self = Rem::rem(self, other);
    }
}

// Implements the PartialEq trait for i32.
impl i32PartialEq of PartialEq<i32> {
    fn eq(lhs: @i32, rhs: @i32) -> bool {
        i32_eq(*lhs, *rhs)
    }

    fn ne(lhs: @i32, rhs: @i32) -> bool {
        i32_ne(*lhs, *rhs)
    }
}

// Implements the PartialOrd trait for i32.
impl i32PartialOrd of PartialOrd<i32> {
    fn le(lhs: i32, rhs: i32) -> bool {
        i32_le(lhs, rhs)
    }
    fn ge(lhs: i32, rhs: i32) -> bool {
        i32_ge(lhs, rhs)
    }

    fn lt(lhs: i32, rhs: i32) -> bool {
        i32_lt(lhs, rhs)
    }
    fn gt(lhs: i32, rhs: i32) -> bool {
        i32_gt(lhs, rhs)
    }
}

// Implements the Neg trait for i32.
impl i32Neg of Neg<i32> {
    fn neg(a: i32) -> i32 {
        i32_neg(a)
    }
}

impl I32TryIntoI8 of TryInto<i32, i8> {
    fn try_into(self: i32) -> Option<i8> {
        i8_try_from_i32(self)
    }
}

// Checks if the given i32 integer is zero and has the correct sign.
// # Arguments
// * `x` - The i32 integer to check.
// # Panics
// Panics if `x` is zero and has a sign that is not false.
fn i32_check_sign_zero(x: i32) {
    if x.mag == 0 {
        assert(x.sign == false, 'sign of 0 must be false');
    }
}

/// Cf: IntegerTrait::new docstring
fn i32_new(mag: u32, sign: bool) -> i32 {
    if sign == true {
        assert(mag <= 2147483648, 'int: out of range');
    } else {
        assert(mag <= 2147483647, 'int: out of range');
    }
    i32 { mag, sign }
}

// Adds two i32 integers.
// # Arguments
// * `a` - The first i32 to add.
// * `b` - The second i32 to add.
// # Returns
// * `i32` - The sum of `a` and `b`.
fn i32_add(a: i32, b: i32) -> i32 {
    i32_check_sign_zero(a);
    i32_check_sign_zero(b);

    // If both integers have the same sign, 
    // the sum of their absolute values can be returned.
    if a.sign == b.sign {
        let sum = a.mag + b.mag;
        if (sum == 0) {
            return IntegerTrait::new(sum, false);
        }
        return ensure_non_negative_zero(sum, a.sign);
    } else {
        // If the integers have different signs, 
        // the larger absolute value is subtracted from the smaller one.
        let (larger, smaller) = if a.mag >= b.mag {
            (a, b)
        } else {
            (b, a)
        };
        let difference = larger.mag - smaller.mag;

        if (difference == 0) {
            return IntegerTrait::new(difference, false);
        }
        return ensure_non_negative_zero(difference, larger.sign);
    }
}

// Subtracts two i32 integers.
// # Arguments
// * `a` - The first i32 to subtract.
// * `b` - The second i32 to subtract.
// # Returns
// * `i32` - The difference of `a` and `b`.
fn i32_sub(a: i32, b: i32) -> i32 {
    i32_check_sign_zero(a);
    i32_check_sign_zero(b);

    if (b.mag == 0) {
        return a;
    }

    // The subtraction of `a` to `b` is achieved by negating `b` sign and adding it to `a`.
    let neg_b = ensure_non_negative_zero(b.mag, !b.sign);
    return a + neg_b;
}

// Multiplies two i32 integers.
// 
// # Arguments
//
// * `a` - The first i32 to multiply.
// * `b` - The second i32 to multiply.
//
// # Returns
//
// * `i32` - The product of `a` and `b`.
fn i32_mul(a: i32, b: i32) -> i32 {
    i32_check_sign_zero(a);
    i32_check_sign_zero(b);

    // The sign of the product is the XOR of the signs of the operands.
    let sign = a.sign ^ b.sign;
    // The product is the product of the absolute values of the operands.
    let mag = a.mag * b.mag;

    if (mag == 0) {
        return IntegerTrait::new(mag, false);
    }

    return ensure_non_negative_zero(mag, sign);
}

// Divides the first i32 by the second i32.
// # Arguments
// * `a` - The i32 dividend.
// * `b` - The i32 divisor.
// # Returns
// * `i32` - The quotient of `a` and `b`.
fn i32_div(a: i32, b: i32) -> i32 {
    i32_check_sign_zero(a);
    // Check that the divisor is not zero.
    assert(b.mag != 0, 'b can not be 0');

    // The sign of the quotient is the XOR of the signs of the operands.
    let sign = a.sign ^ b.sign;

    if (sign == false) {
        // If the operands are positive, the quotient is simply their absolute value quotient.
        return ensure_non_negative_zero(a.mag / b.mag, sign);
    }

    // If the operands have different signs, rounding is necessary.
    // First, check if the quotient is an integer.
    if (a.mag % b.mag == 0) {
        let quotient = a.mag / b.mag;
        if (quotient == 0) {
            return IntegerTrait::new(quotient, false);
        }
        return ensure_non_negative_zero(quotient, sign);
    }

    // If the quotient is not an integer, multiply the dividend by 10 to move the decimal point over.
    let quotient = (a.mag * 10) / b.mag;
    let last_digit = quotient % 10;

    if (quotient == 0) {
        return ensure_non_negative_zero(quotient, false);
    }

    // Check the last digit to determine rounding direction.
    if (last_digit <= 5) {
        return ensure_non_negative_zero(quotient / 10, sign);
    } else {
        return ensure_non_negative_zero((quotient / 10) + 1, sign);
    }
}

// Calculates the remainder of the division of a first i32 by a second i32.
// # Arguments
// * `a` - The i32 dividend.
// * `b` - The i32 divisor.
// # Returns
// * `i32` - The remainder of dividing `a` by `b`.
fn i32_rem(a: i32, b: i32) -> i32 {
    i32_check_sign_zero(a);
    // Check that the divisor is not zero.
    assert(b.mag != 0, 'b can not be 0');

    return a - (b * (a / b));
}

/// Cf: IntegerTrait::div_rem docstring
fn i32_div_rem(a: i32, b: i32) -> (i32, i32) {
    let quotient = i32_div(a, b);
    let remainder = i32_rem(a, b);

    return (quotient, remainder);
}

// Compares two i32 integers for equality.
// # Arguments
// * `a` - The first i32 integer to compare.
// * `b` - The second i32 integer to compare.
// # Returns
// * `bool` - `true` if the two integers are equal, `false` otherwise.
fn i32_eq(a: i32, b: i32) -> bool {
    // Check if the two integers have the same sign and the same absolute value.
    if a.sign == b.sign && a.mag == b.mag {
        return true;
    }

    return false;
}

// Compares two i32 integers for inequality.
// # Arguments
// * `a` - The first i32 integer to compare.
// * `b` - The second i32 integer to compare.
// # Returns
// * `bool` - `true` if the two integers are not equal, `false` otherwise.
fn i32_ne(a: i32, b: i32) -> bool {
    // The result is the inverse of the equal function.
    return !i32_eq(a, b);
}

// Compares two i32 integers for greater than.
// # Arguments
// * `a` - The first i32 integer to compare.
// * `b` - The second i32 integer to compare.
// # Returns
// * `bool` - `true` if `a` is greater than `b`, `false` otherwise.
fn i32_gt(a: i32, b: i32) -> bool {
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

// Determines whether the first i32 is less than the second i32.
// # Arguments
// * `a` - The i32 to compare against the second i32.
// * `b` - The i32 to compare against the first i32.
// # Returns
// * `bool` - `true` if `a` is less than `b`, `false` otherwise.
fn i32_lt(a: i32, b: i32) -> bool {
    if (a.sign != b.sign) {
        return a.sign;
    } else {
        return a.mag != b.mag && (a.mag < b.mag) ^ a.sign;
    }
}

// Checks if the first i32 integer is less than or equal to the second.
// # Arguments
// * `a` - The first i32 integer to compare.
// * `b` - The second i32 integer to compare.
// # Returns
// * `bool` - `true` if `a` is less than or equal to `b`, `false` otherwise.
fn i32_le(a: i32, b: i32) -> bool {
    if (a == b || i32_lt(a, b) == true) {
        return true;
    } else {
        return false;
    }
}

// Checks if the first i32 integer is greater than or equal to the second.
// # Arguments
// * `a` - The first i32 integer to compare.
// * `b` - The second i32 integer to compare.
// # Returns
// * `bool` - `true` if `a` is greater than or equal to `b`, `false` otherwise.
fn i32_ge(a: i32, b: i32) -> bool {
    if (a == b || i32_gt(a, b) == true) {
        return true;
    } else {
        return false;
    }
}

// Negates the given i32 integer.
// # Arguments
// * `x` - The i32 integer to negate.
// # Returns
// * `i32` - The negation of `x`.
fn i32_neg(x: i32) -> i32 {
    // The negation of an integer is obtained by flipping its sign.
    return ensure_non_negative_zero(x.mag, !x.sign);
}

/// Cf: IntegerTrait::abs docstring
fn i32_abs(x: i32) -> i32 {
    return IntegerTrait::new(x.mag, false);
}

/// Cf: IntegerTrait::max docstring
fn i32_max(a: i32, b: i32) -> i32 {
    if (a > b) {
        return a;
    } else {
        return b;
    }
}

/// Cf: IntegerTrait::min docstring
fn i32_min(a: i32, b: i32) -> i32 {
    if (a < b) {
        return a;
    } else {
        return b;
    }
}

fn i32_xor(a: i32, b: i32) -> bool {
    if (a.mag == 0_u32 || b.mag == 0_u32) && a.mag != b.mag {
        return true;
    } else {
        return false;
    }
}

fn i32_or(a: i32, b: i32) -> bool {
    if (a.mag == 0_u32 && b.mag == 0_u32) {
        return false;
    } else {
        return true;
    }
}

fn i8_try_from_i32(x: i32) -> Option<i8> {
    match x.mag.try_into() {
        Option::Some(val) => Option::Some(i8 { mag: val, sign: x.sign }),
        Option::None(_) => Option::None(())
    }
}

fn ensure_non_negative_zero(mag: u32, sign: bool) -> i32 {
    if mag == 0 {
        IntegerTrait::<i32>::new(mag, false)
    } else {
        IntegerTrait::<i32>::new(mag, sign)
    }
}

fn i32_sign(a: i32) -> i32 {
    if a.mag == 0 {
        IntegerTrait::<i32>::new(0, false)
    } else {
        IntegerTrait::<i32>::new(1, a.sign)
    }
}

fn i32_bitwise_and(a: i32, b: i32) -> i32 {
    IntegerTrait::<i32>::new(a.mag & b.mag, a.sign & b.sign)
}
