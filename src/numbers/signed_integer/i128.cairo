use traits::Into;

use orion::numbers::signed_integer::integer_trait::IntegerTrait;

// ====================== INT 128 ======================

// i128 represents a 128-bit integer.
// The mag field holds the absolute value of the integer.
// The sign field is true for negative integers, and false for non-negative integers.
#[derive(Serde, Copy, Drop)]
struct i128 {
    mag: u128,
    sign: bool,
}

impl i128Impl of IntegerTrait<i128, u128> {
    fn new(mag: u128, sign: bool) -> i128 {
        i128_new(mag, sign)
    }

    fn div_rem(self: i128, other: i128) -> (i128, i128) {
        i128_div_rem(self, other)
    }

    fn abs(self: i128) -> i128 {
        i128_abs(self)
    }

    fn max(self: i128, other: i128) -> i128 {
        i128_max(self, other)
    }

    fn min(self: i128, other: i128) -> i128 {
        i128_min(self, other)
    }

    fn sign(self: i128) -> i128 {
        i128_sign(self)
    }
}

// Implements the Into trait for i128.
impl i32Into of Into<i128, felt252> {
    fn into(self: i128) -> felt252 {
        let mag_felt = self.mag.into();

        if (self.sign == true) {
            return mag_felt * -1;
        } else {
            return mag_felt;
        }
    }
}

// Implements the Add trait for i128.
impl i128Add of Add<i128> {
    fn add(lhs: i128, rhs: i128) -> i128 {
        i128_add(lhs, rhs)
    }
}

// Implements the AddEq trait for i128.
impl i128AddEq of AddEq<i128> {
    #[inline(always)]
    fn add_eq(ref self: i128, other: i128) {
        self = Add::add(self, other);
    }
}

// Implements the Sub trait for i128.
impl i128Sub of Sub<i128> {
    fn sub(lhs: i128, rhs: i128) -> i128 {
        i128_sub(lhs, rhs)
    }
}

// Implements the SubEq trait for i128.
impl i128SubEq of SubEq<i128> {
    #[inline(always)]
    fn sub_eq(ref self: i128, other: i128) {
        self = Sub::sub(self, other);
    }
}

// Implements the Mul trait for i128.
impl i128Mul of Mul<i128> {
    fn mul(lhs: i128, rhs: i128) -> i128 {
        i128_mul(lhs, rhs)
    }
}

// Implements the MulEq trait for i128.
impl i128MulEq of MulEq<i128> {
    #[inline(always)]
    fn mul_eq(ref self: i128, other: i128) {
        self = Mul::mul(self, other);
    }
}

// Implements the Div trait for i128.
impl i128Div of Div<i128> {
    fn div(lhs: i128, rhs: i128) -> i128 {
        i128_div(lhs, rhs)
    }
}

// Implements the DivEq trait for i128.
impl i128DivEq of DivEq<i128> {
    #[inline(always)]
    fn div_eq(ref self: i128, other: i128) {
        self = Div::div(self, other);
    }
}

// Implements the Rem trait for i128.
impl i128Rem of Rem<i128> {
    fn rem(lhs: i128, rhs: i128) -> i128 {
        i128_rem(lhs, rhs)
    }
}

// Implements the RemEq trait for i128.
impl i128RemEq of RemEq<i128> {
    #[inline(always)]
    fn rem_eq(ref self: i128, other: i128) {
        self = Rem::rem(self, other);
    }
}

// Implements the PartialEq trait for i128.
impl i128PartialEq of PartialEq<i128> {
    fn eq(lhs: @i128, rhs: @i128) -> bool {
        i128_eq(*lhs, *rhs)
    }

    fn ne(lhs: @i128, rhs: @i128) -> bool {
        i128_ne(*lhs, *rhs)
    }
}

// Implements the PartialOrd trait for i128.
impl i128PartialOrd of PartialOrd<i128> {
    fn le(lhs: i128, rhs: i128) -> bool {
        i128_le(lhs, rhs)
    }
    fn ge(lhs: i128, rhs: i128) -> bool {
        i128_ge(lhs, rhs)
    }

    fn lt(lhs: i128, rhs: i128) -> bool {
        i128_lt(lhs, rhs)
    }
    fn gt(lhs: i128, rhs: i128) -> bool {
        i128_gt(lhs, rhs)
    }
}

// Implements the Neg trait for i128.
impl i128Neg of Neg<i128> {
    fn neg(a: i128) -> i128 {
        i128_neg(a)
    }
}

// Checks if the given i128 integer is zero and has the correct sign.
// # Arguments
// * `x` - The i128 integer to check.
// # Panics
// Panics if `x` is zero and has a sign that is not false.
fn i128_check_sign_zero(x: i128) {
    if x.mag == 0_u128 {
        assert(x.sign == false, 'sign of 0 must be false');
    }
}

/// Cf: IntegerTrait::new docstring
fn i128_new(mag: u128, sign: bool) -> i128 {
    if sign == true {
        assert(mag <= 170141183460469231731687303715884105728_u128, 'int: out of range');
    } else {
        assert(mag <= 170141183460469231731687303715884105727_u128, 'int: out of range');
    }
    i128 { mag, sign }
}

// Adds two i128 integers.
// # Arguments
// * `a` - The first i128 to add.
// * `b` - The second i128 to add.
// # Returns
// * `i128` - The sum of `a` and `b`.
fn i128_add(a: i128, b: i128) -> i128 {
    i128_check_sign_zero(a);
    i128_check_sign_zero(b);

    // If both integers have the same sign, 
    // the sum of their absolute values can be returned.
    if a.sign == b.sign {
        let sum = a.mag + b.mag;
        if (sum == 0_u128) {
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

        if (difference == 0_u128) {
            return IntegerTrait::new(difference, false);
        }
        return ensure_non_negative_zero(difference, larger.sign);
    }
}

// Subtracts two i128 integers.
// # Arguments
// * `a` - The first i128 to subtract.
// * `b` - The second i128 to subtract.
// # Returns
// * `i128` - The difference of `a` and `b`.
fn i128_sub(a: i128, b: i128) -> i128 {
    i128_check_sign_zero(a);
    i128_check_sign_zero(b);

    if (b.mag == 0_u128) {
        return a;
    }

    // The subtraction of `a` to `b` is achieved by negating `b` sign and adding it to `a`.
    let neg_b = ensure_non_negative_zero(b.mag, !b.sign);
    return a + neg_b;
}

// Multiplies two i128 integers.
// 
// # Arguments
//
// * `a` - The first i128 to multiply.
// * `b` - The second i128 to multiply.
//
// # Returns
//
// * `i128` - The product of `a` and `b`.
fn i128_mul(a: i128, b: i128) -> i128 {
    i128_check_sign_zero(a);
    i128_check_sign_zero(b);

    // The sign of the product is the XOR of the signs of the operands.
    let sign = a.sign ^ b.sign;
    // The product is the product of the absolute values of the operands.
    let mag = a.mag * b.mag;

    if (mag == 0_u128) {
        return IntegerTrait::new(mag, false);
    }

    return ensure_non_negative_zero(mag, sign);
}

// Divides the first i128 by the second i128.
// # Arguments
// * `a` - The i128 dividend.
// * `b` - The i128 divisor.
// # Returns
// * `i128` - The quotient of `a` and `b`.
fn i128_div(a: i128, b: i128) -> i128 {
    i128_check_sign_zero(a);
    // Check that the divisor is not zero.
    assert(b.mag != 0_u128, 'b can not be 0');

    // The sign of the quotient is the XOR of the signs of the operands.
    let sign = a.sign ^ b.sign;

    if (sign == false) {
        // If the operands are positive, the quotient is simply their absolute value quotient.
        return ensure_non_negative_zero(a.mag / b.mag, sign);
    }

    // If the operands have different signs, rounding is necessary.
    // First, check if the quotient is an integer.
    if (a.mag % b.mag == 0_u128) {
        let quotient = a.mag / b.mag;
        if (quotient == 0_u128) {
            return IntegerTrait::new(quotient, false);
        }
        return ensure_non_negative_zero(quotient, sign);
    }

    // If the quotient is not an integer, multiply the dividend by 10 to move the decimal point over.
    let quotient = (a.mag * 10_u128) / b.mag;
    let last_digit = quotient % 10_u128;

    if (quotient == 0_u128) {
        return IntegerTrait::new(quotient, false);
    }

    // Check the last digit to determine rounding direction.
    if (last_digit <= 5_u128) {
        return ensure_non_negative_zero(quotient / 10_u128, sign);
    } else {
        return ensure_non_negative_zero((quotient / 10_u128) + 1_u128, sign);
    }
}

// Calculates the remainder of the division of a first i128 by a second i128.
// # Arguments
// * `a` - The i128 dividend.
// * `b` - The i128 divisor.
// # Returns
// * `i128` - The remainder of dividing `a` by `b`.
fn i128_rem(a: i128, b: i128) -> i128 {
    i128_check_sign_zero(a);
    // Check that the divisor is not zero.
    assert(b.mag != 0_u128, 'b can not be 0');

    return a - (b * (a / b));
}

/// Cf: IntegerTrait::div_rem docstring
fn i128_div_rem(a: i128, b: i128) -> (i128, i128) {
    let quotient = i128_div(a, b);
    let remainder = i128_rem(a, b);

    return (quotient, remainder);
}

// Compares two i128 integers for equality.
// # Arguments
// * `a` - The first i128 integer to compare.
// * `b` - The second i128 integer to compare.
// # Returns
// * `bool` - `true` if the two integers are equal, `false` otherwise.
fn i128_eq(a: i128, b: i128) -> bool {
    // Check if the two integers have the same sign and the same absolute value.
    if a.sign == b.sign && a.mag == b.mag {
        return true;
    }

    return false;
}

// Compares two i128 integers for inequality.
// # Arguments
// * `a` - The first i128 integer to compare.
// * `b` - The second i128 integer to compare.
// # Returns
// * `bool` - `true` if the two integers are not equal, `false` otherwise.
fn i128_ne(a: i128, b: i128) -> bool {
    // The result is the inverse of the equal function.
    return !i128_eq(a, b);
}

// Compares two i128 integers for greater than.
// # Arguments
// * `a` - The first i128 integer to compare.
// * `b` - The second i128 integer to compare.
// # Returns
// * `bool` - `true` if `a` is greater than `b`, `false` otherwise.
fn i128_gt(a: i128, b: i128) -> bool {
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

// Determines whether the first i128 is less than the second i128.
// # Arguments
// * `a` - The i128 to compare against the second i128.
// * `b` - The i128 to compare against the first i128.
// # Returns
// * `bool` - `true` if `a` is less than `b`, `false` otherwise.
fn i128_lt(a: i128, b: i128) -> bool {
    if (a.sign != b.sign) {
        return a.sign;
    } else {
        return a.mag != b.mag && (a.mag < b.mag) ^ a.sign;
    }
}

// Checks if the first i128 integer is less than or equal to the second.
// # Arguments
// * `a` - The first i128 integer to compare.
// * `b` - The second i128 integer to compare.
// # Returns
// * `bool` - `true` if `a` is less than or equal to `b`, `false` otherwise.
fn i128_le(a: i128, b: i128) -> bool {
    if (a == b || i128_lt(a, b) == true) {
        return true;
    } else {
        return false;
    }
}

// Checks if the first i128 integer is greater than or equal to the second.
// # Arguments
// * `a` - The first i128 integer to compare.
// * `b` - The second i128 integer to compare.
// # Returns
// * `bool` - `true` if `a` is greater than or equal to `b`, `false` otherwise.
fn i128_ge(a: i128, b: i128) -> bool {
    if (a == b || i128_gt(a, b) == true) {
        return true;
    } else {
        return false;
    }
}

// Negates the given i128 integer.
// # Arguments
// * `x` - The i128 integer to negate.
// # Returns
// * `i128` - The negation of `x`.
fn i128_neg(x: i128) -> i128 {
    // The negation of an integer is obtained by flipping its sign.
    return ensure_non_negative_zero(x.mag, !x.sign);
}

/// Cf: IntegerTrait::abs docstring
fn i128_abs(x: i128) -> i128 {
    return IntegerTrait::new(x.mag, false);
}

/// Cf: IntegerTrait::max docstring
fn i128_max(a: i128, b: i128) -> i128 {
    if (a > b) {
        return a;
    } else {
        return b;
    }
}

/// Cf: IntegerTrait::new docstring
fn i128_min(a: i128, b: i128) -> i128 {
    if (a < b) {
        return a;
    } else {
        return b;
    }
}

fn ensure_non_negative_zero(mag: u128, sign: bool) -> i128 {
    if mag == 0 {
        IntegerTrait::<i128>::new(mag, false)
    } else {
        IntegerTrait::<i128>::new(mag, sign)
    }
}

fn i128_sign(a: i128) -> i128 {
    if a.mag == 0 {
        IntegerTrait::<i128>::new(0, false)
    } else {
        IntegerTrait::<i128>::new(1, a.sign)
    }
}

fn i128_bitwise_and(a: i128, b: i128) -> i128 {
    IntegerTrait::<i128>::new(a.mag & b.mag, a.sign & b.sign)
}
