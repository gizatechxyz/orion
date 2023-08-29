use traits::Into;

use orion::numbers::signed_integer::integer_trait::IntegerTrait;
use orion::numbers::zero::Zero;
use orion::numbers::one::One;


// ====================== INT 16 ======================

// i16 represents a 16-bit integer.
// The mag field holds the absolute value of the integer.
// The sign field is true for negative integers, and false for non-negative integers.
#[derive(Serde, Copy, Drop)]
struct i16 {
    mag: u16,
    sign: bool,
}

impl i16Impl of IntegerTrait<i16, u16> {
    fn new(mag: u16, sign: bool) -> i16 {
        i16_new(mag, sign)
    }

    fn div_rem(self: i16, other: i16) -> (i16, i16) {
        i16_div_rem(self, other)
    }

    fn abs(self: i16) -> i16 {
        i16_abs(self)
    }

    fn max(self: i16, other: i16) -> i16 {
        i16_max(self, other)
    }

    fn min(self: i16, other: i16) -> i16 {
        i16_min(self, other)
    }
}

// Implements the Into trait for i16.
impl i8Into of Into<i16, felt252> {
    fn into(self: i16) -> felt252 {
        let mag_felt = self.mag.into();

        if (self.sign == true) {
            return mag_felt * -1;
        } else {
            return mag_felt;
        }
    }
}

// Implements the Add trait for i16.
impl i16Add of Add<i16> {
    fn add(lhs: i16, rhs: i16) -> i16 {
        i16_add(lhs, rhs)
    }
}

// Implements the AddEq trait for i16.
impl i16AddEq of AddEq<i16> {
    #[inline(always)]
    fn add_eq(ref self: i16, other: i16) {
        self = Add::add(self, other);
    }
}

// Implements the Sub trait for i16.
impl i16Sub of Sub<i16> {
    fn sub(lhs: i16, rhs: i16) -> i16 {
        i16_sub(lhs, rhs)
    }
}

// Implements the SubEq trait for i16.
impl i16SubEq of SubEq<i16> {
    #[inline(always)]
    fn sub_eq(ref self: i16, other: i16) {
        self = Sub::sub(self, other);
    }
}

// Implements the Mul trait for i16.
impl i16Mul of Mul<i16> {
    fn mul(lhs: i16, rhs: i16) -> i16 {
        i16_mul(lhs, rhs)
    }
}

// Implements the MulEq trait for i16.
impl i16MulEq of MulEq<i16> {
    #[inline(always)]
    fn mul_eq(ref self: i16, other: i16) {
        self = Mul::mul(self, other);
    }
}

// Implements the Div trait for i16.
impl i16Div of Div<i16> {
    fn div(lhs: i16, rhs: i16) -> i16 {
        i16_div(lhs, rhs)
    }
}

// Implements the DivEq trait for i16.
impl i16DivEq of DivEq<i16> {
    #[inline(always)]
    fn div_eq(ref self: i16, other: i16) {
        self = Div::div(self, other);
    }
}

// Implements the Rem trait for i16.
impl i16Rem of Rem<i16> {
    fn rem(lhs: i16, rhs: i16) -> i16 {
        i16_rem(lhs, rhs)
    }
}

// Implements the RemEq trait for i16.
impl i16RemEq of RemEq<i16> {
    #[inline(always)]
    fn rem_eq(ref self: i16, other: i16) {
        self = Rem::rem(self, other);
    }
}

// Implements the PartialEq trait for i16.
impl i16PartialEq of PartialEq<i16> {
    fn eq(lhs: @i16, rhs: @i16) -> bool {
        i16_eq(*lhs, *rhs)
    }

    fn ne(lhs: @i16, rhs: @i16) -> bool {
        i16_ne(*lhs, *rhs)
    }
}

// Implements the PartialOrd trait for i16.
impl i16PartialOrd of PartialOrd<i16> {
    fn le(lhs: i16, rhs: i16) -> bool {
        i16_le(lhs, rhs)
    }
    fn ge(lhs: i16, rhs: i16) -> bool {
        i16_ge(lhs, rhs)
    }

    fn lt(lhs: i16, rhs: i16) -> bool {
        i16_lt(lhs, rhs)
    }
    fn gt(lhs: i16, rhs: i16) -> bool {
        i16_gt(lhs, rhs)
    }
}

// Implements the Neg trait for i16.
impl i16Neg of Neg<i16> {
    fn neg(a: i16) -> i16 {
        i16_neg(a)
    }
}

impl I16Zero of Zero<i16> {
    #[inline(always)]
    fn zero() -> i16 {
        return i16 { mag: 0, sign: false };
    }

    #[inline(always)]
    fn is_zero(self: i16) -> bool {
        return self == i16 { mag: 0, sign: false };
    }
}

impl I16One of One<i16> {
    #[inline(always)]
    fn one() -> i16 {
        return i16 { mag: 1, sign: false };
    }

    #[inline(always)]
    fn is_one(self: i16) -> bool {
        return self == i16 { mag: 1, sign: false };
    }
}

// Checks if the given i16 integer is zero and has the correct sign.
// # Arguments
// * `x` - The i16 integer to check.
// # Panics
// Panics if `x` is zero and has a sign that is not false.
fn i16_check_sign_zero(x: i16) {
    if x.mag == 0_u16 {
        assert(x.sign == false, 'sign of 0 must be false');
    }
}

/// Cf: IntegerTrait::new docstring
fn i16_new(mag: u16, sign: bool) -> i16 {
    if sign == true {
        assert(mag <= 32768_u16, 'int: out of range');
    } else {
        assert(mag <= 32767_u16, 'int: out of range');
    }
    i16 { mag, sign }
}

// Adds two i16 integers.
// # Arguments
// * `a` - The first i16 to add.
// * `b` - The second i16 to add.
// # Returns
// * `i16` - The sum of `a` and `b`.
fn i16_add(a: i16, b: i16) -> i16 {
    i16_check_sign_zero(a);
    i16_check_sign_zero(b);

    // If both integers have the same sign, 
    // the sum of their absolute values can be returned.
    if a.sign == b.sign {
        let sum = a.mag + b.mag;
        if (sum == 0_u16) {
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

        if (difference == 0_u16) {
            return IntegerTrait::new(difference, false);
        }
        return ensure_non_negative_zero(difference, larger.sign);
    }
}

// Subtracts two i16 integers.
// # Arguments
// * `a` - The first i16 to subtract.
// * `b` - The second i16 to subtract.
// # Returns
// * `i16` - The difference of `a` and `b`.
fn i16_sub(a: i16, b: i16) -> i16 {
    i16_check_sign_zero(a);
    i16_check_sign_zero(b);

    if (b.mag == 0_u16) {
        return a;
    }

    // The subtraction of `a` to `b` is achieved by negating `b` sign and adding it to `a`.
    let neg_b = ensure_non_negative_zero(b.mag, !b.sign);
    return a + neg_b;
}

// Multiplies two i16 integers.
// 
// # Arguments
//
// * `a` - The first i16 to multiply.
// * `b` - The second i16 to multiply.
//
// # Returns
//
// * `i16` - The product of `a` and `b`.
fn i16_mul(a: i16, b: i16) -> i16 {
    i16_check_sign_zero(a);
    i16_check_sign_zero(b);

    // The sign of the product is the XOR of the signs of the operands.
    let sign = a.sign ^ b.sign;
    // The product is the product of the absolute values of the operands.
    let mag = a.mag * b.mag;

    if (mag == 0_u16) {
        return IntegerTrait::new(mag, false);
    }

    return ensure_non_negative_zero(mag, sign);
}

// Divides the first i16 by the second i16.
// # Arguments
// * `a` - The i16 dividend.
// * `b` - The i16 divisor.
// # Returns
// * `i16` - The quotient of `a` and `b`.
fn i16_div(a: i16, b: i16) -> i16 {
    i16_check_sign_zero(a);
    // Check that the divisor is not zero.
    assert(b.mag != 0_u16, 'b can not be 0');

    // The sign of the quotient is the XOR of the signs of the operands.
    let sign = a.sign ^ b.sign;

    if (sign == false) {
        // If the operands are positive, the quotient is simply their absolute value quotient.
        return ensure_non_negative_zero(a.mag / b.mag, sign);
    }

    // If the operands have different signs, rounding is necessary.
    // First, check if the quotient is an integer.
    if (a.mag % b.mag == 0_u16) {
        let quotient = a.mag / b.mag;
        if (quotient == 0_u16) {
            return IntegerTrait::new(quotient, false);
        }
        return ensure_non_negative_zero(quotient, sign);
    }

    // If the quotient is not an integer, multiply the dividend by 10 to move the decimal point over.
    let quotient = (a.mag * 10_u16) / b.mag;
    let last_digit = quotient % 10_u16;

    if (quotient == 0_u16) {
        return IntegerTrait::new(quotient, false);
    }

    // Check the last digit to determine rounding direction.
    if (last_digit <= 5_u16) {
        return ensure_non_negative_zero(quotient / 10_u16, sign);
    } else {
        return ensure_non_negative_zero((quotient / 10_u16) + 1_u16, sign);
    }
}

// Calculates the remainder of the division of a first i16 by a second i16.
// # Arguments
// * `a` - The i16 dividend.
// * `b` - The i16 divisor.
// # Returns
// * `i16` - The remainder of dividing `a` by `b`.
fn i16_rem(a: i16, b: i16) -> i16 {
    i16_check_sign_zero(a);
    // Check that the divisor is not zero.
    assert(b.mag != 0_u16, 'b can not be 0');

    return a - (b * (a / b));
}

/// Cf: IntegerTrait::div_rem docstring
fn i16_div_rem(a: i16, b: i16) -> (i16, i16) {
    let quotient = i16_div(a, b);
    let remainder = i16_rem(a, b);

    return (quotient, remainder);
}

// Compares two i16 integers for equality.
// # Arguments
// * `a` - The first i16 integer to compare.
// * `b` - The second i16 integer to compare.
// # Returns
// * `bool` - `true` if the two integers are equal, `false` otherwise.
fn i16_eq(a: i16, b: i16) -> bool {
    // Check if the two integers have the same sign and the same absolute value.
    if a.sign == b.sign && a.mag == b.mag {
        return true;
    }

    return false;
}

// Compares two i16 integers for inequality.
// # Arguments
// * `a` - The first i16 integer to compare.
// * `b` - The second i16 integer to compare.
// # Returns
// * `bool` - `true` if the two integers are not equal, `false` otherwise.
fn i16_ne(a: i16, b: i16) -> bool {
    // The result is the inverse of the equal function.
    return !i16_eq(a, b);
}

// Compares two i16 integers for greater than.
// # Arguments
// * `a` - The first i16 integer to compare.
// * `b` - The second i16 integer to compare.
// # Returns
// * `bool` - `true` if `a` is greater than `b`, `false` otherwise.
fn i16_gt(a: i16, b: i16) -> bool {
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

// Determines whether the first i16 is less than the second i16.
// # Arguments
// * `a` - The i16 to compare against the second i16.
// * `b` - The i16 to compare against the first i16.
// # Returns
// * `bool` - `true` if `a` is less than `b`, `false` otherwise.
fn i16_lt(a: i16, b: i16) -> bool {
    if (a.sign != b.sign) {
        return a.sign;
    } else {
        return a.mag != b.mag && (a.mag < b.mag) ^ a.sign;
    }
}

// Checks if the first i16 integer is less than or equal to the second.
// # Arguments
// * `a` - The first i16 integer to compare.
// * `b` - The second i16 integer to compare.
// # Returns
// * `bool` - `true` if `a` is less than or equal to `b`, `false` otherwise.
fn i16_le(a: i16, b: i16) -> bool {
    if (a == b || i16_lt(a, b) == true) {
        return true;
    } else {
        return false;
    }
}

// Checks if the first i16 integer is greater than or equal to the second.
// # Arguments
// * `a` - The first i16 integer to compare.
// * `b` - The second i16 integer to compare.
// # Returns
// * `bool` - `true` if `a` is greater than or equal to `b`, `false` otherwise.
fn i16_ge(a: i16, b: i16) -> bool {
    if (a == b || i16_gt(a, b) == true) {
        return true;
    } else {
        return false;
    }
}

// Negates the given i16 integer.
// # Arguments
// * `x` - The i16 integer to negate.
// # Returns
// * `i16` - The negation of `x`.
fn i16_neg(x: i16) -> i16 {
    // The negation of an integer is obtained by flipping its sign.
    return ensure_non_negative_zero(x.mag, !x.sign);
}

/// Cf: IntegerTrait::abs docstring
fn i16_abs(x: i16) -> i16 {
    return IntegerTrait::new(x.mag, false);
}

/// Cf: IntegerTrait::max docstring
fn i16_max(a: i16, b: i16) -> i16 {
    if (a > b) {
        return a;
    } else {
        return b;
    }
}

/// Cf: IntegerTrait::min docstring
fn i16_min(a: i16, b: i16) -> i16 {
    if (a < b) {
        return a;
    } else {
        return b;
    }
}

fn ensure_non_negative_zero(mag: u16, sign: bool) -> i16 {
    if mag == 0 {
        IntegerTrait::<i16>::new(mag, false)
    } else {
        IntegerTrait::<i16>::new(mag, sign)
    }
}
