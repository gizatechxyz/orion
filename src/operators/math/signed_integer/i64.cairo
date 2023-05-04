use onnx_cairo::operators::math::signed_integer::integer_trait::IntegerTrait;
use onnx_cairo::utils::check_gas;

// ====================== INT 64 ======================

// i64 represents a 64-bit integer.
// The mag field holds the absolute value of the integer.
// The sign field is true for negative integers, and false for non-negative integers.
#[derive(Copy, Drop)]
struct i64 {
    mag: u64,
    sign: bool,
}

impl i64Impl of IntegerTrait::<i64, u64> {
    fn new(mag: u64, sign: bool) -> i64 {
        i64_new(mag, sign)
    }

    fn div_rem(self: i64, other: i64) -> (i64, i64) {
        i64_div_rem(self, other)
    }

    fn abs(self: i64) -> i64 {
        i64_abs(self)
    }

    fn max(self: i64, other: i64) -> i64 {
        i64_max(self, other)
    }

    fn min(self: i64, other: i64) -> i64 {
        i64_min(self, other)
    }
}

// Implements the Add trait for i64.
impl i64Add of Add::<i64> {
    fn add(a: i64, b: i64) -> i64 {
        i64_add(a, b)
    }
}

// Implements the AddEq trait for i64.
impl i64AddEq of AddEq::<i64> {
    #[inline(always)]
    fn add_eq(ref self: i64, other: i64) {
        self = Add::add(self, other);
    }
}

// Implements the Sub trait for i64.
impl i64Sub of Sub::<i64> {
    fn sub(a: i64, b: i64) -> i64 {
        i64_sub(a, b)
    }
}

// Implements the SubEq trait for i64.
impl i64SubEq of SubEq::<i64> {
    #[inline(always)]
    fn sub_eq(ref self: i64, other: i64) {
        self = Sub::sub(self, other);
    }
}

// Implements the Mul trait for i64.
impl i64Mul of Mul::<i64> {
    fn mul(a: i64, b: i64) -> i64 {
        i64_mul(a, b)
    }
}

// Implements the MulEq trait for i64.
impl i64MulEq of MulEq::<i64> {
    #[inline(always)]
    fn mul_eq(ref self: i64, other: i64) {
        self = Mul::mul(self, other);
    }
}

// Implements the Div trait for i64.
impl i64Div of Div::<i64> {
    fn div(a: i64, b: i64) -> i64 {
        i64_div(a, b)
    }
}

// Implements the DivEq trait for i64.
impl i64DivEq of DivEq::<i64> {
    #[inline(always)]
    fn div_eq(ref self: i64, other: i64) {
        self = Div::div(self, other);
    }
}

// Implements the Rem trait for i64.
impl i64Rem of Rem::<i64> {
    fn rem(a: i64, b: i64) -> i64 {
        i64_rem(a, b)
    }
}

// Implements the RemEq trait for i64.
impl i64RemEq of RemEq::<i64> {
    #[inline(always)]
    fn rem_eq(ref self: i64, other: i64) {
        self = Rem::rem(self, other);
    }
}

// Implements the PartialEq trait for i64.
impl i64PartialEq of PartialEq::<i64> {
    fn eq(a: i64, b: i64) -> bool {
        i64_eq(a, b)
    }

    fn ne(a: i64, b: i64) -> bool {
        i64_ne(a, b)
    }
}

// Implements the PartialOrd trait for i64.
impl i64PartialOrd of PartialOrd::<i64> {
    fn le(a: i64, b: i64) -> bool {
        i64_le(a, b)
    }
    fn ge(a: i64, b: i64) -> bool {
        i64_ge(a, b)
    }

    fn lt(a: i64, b: i64) -> bool {
        i64_lt(a, b)
    }
    fn gt(a: i64, b: i64) -> bool {
        i64_gt(a, b)
    }
}

// Implements the Neg trait for i64.
impl i64Neg of Neg::<i64> {
    fn neg(x: i64) -> i64 {
        i64_neg(x)
    }
}


// Checks if the given i64 integer is zero and has the correct sign.
// # Arguments
// * `x` - The i64 integer to check.
// # Panics
// Panics if `x` is zero and has a sign that is not false.
fn i64_check_sign_zero(x: i64) {
    if x.mag == 0_u64 {
        assert(x.sign == false, 'sign of 0 must be false');
    }
}

// Create a new int64.
// # Arguments
// * `mag` - The magnitude
// * `sign` - The sign of the integer
// # Panics
// Panics if `mag` is out of range.
fn i64_new(mag: u64, sign: bool) -> i64 {
    if sign == true {
        assert(mag <= 9223372036854775808_u64, 'int: out of range');
    } else {
        assert(mag <= 9223372036854775807_u64, 'int: out of range');
    }
    i64 { mag, sign }
}

// Adds two i64 integers.
// # Arguments
// * `a` - The first i64 to add.
// * `b` - The second i64 to add.
// # Returns
// * `i64` - The sum of `a` and `b`.
fn i64_add(a: i64, b: i64) -> i64 {
    i64_check_sign_zero(a);
    i64_check_sign_zero(b);

    // If both integers have the same sign, 
    // the sum of their absolute values can be returned.
    if a.sign == b.sign {
        let sum = a.mag + b.mag;
        if (sum == 0_u64) {
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

        if (difference == 0_u64) {
            return IntegerTrait::new(difference, false);
        }
        return IntegerTrait::new(difference, larger.sign);
    }
}

// Subtracts two i64 integers.
// # Arguments
// * `a` - The first i64 to subtract.
// * `b` - The second i64 to subtract.
// # Returns
// * `i64` - The difference of `a` and `b`.
fn i64_sub(a: i64, b: i64) -> i64 {
    i64_check_sign_zero(a);
    i64_check_sign_zero(b);

    if (b.mag == 0_u64) {
        return a;
    }

    // The subtraction of `a` to `b` is achieved by negating `b` sign and adding it to `a`.
    let neg_b = IntegerTrait::new(b.mag, !b.sign);
    return a + neg_b;
}

// Multiplies two i64 integers.
// 
// # Arguments
//
// * `a` - The first i64 to multiply.
// * `b` - The second i64 to multiply.
//
// # Returns
//
// * `i64` - The product of `a` and `b`.
fn i64_mul(a: i64, b: i64) -> i64 {
    i64_check_sign_zero(a);
    i64_check_sign_zero(b);

    // The sign of the product is the XOR of the signs of the operands.
    let sign = a.sign ^ b.sign;
    // The product is the product of the absolute values of the operands.
    let mag = a.mag * b.mag;

    if (mag == 0_u64) {
        return IntegerTrait::new(mag, false);
    }

    return IntegerTrait::new(mag, sign);
}

// Divides the first i64 by the second i64.
// # Arguments
// * `a` - The i64 dividend.
// * `b` - The i64 divisor.
// # Returns
// * `i64` - The quotient of `a` and `b`.
fn i64_div(a: i64, b: i64) -> i64 {
    i64_check_sign_zero(a);
    // Check that the divisor is not zero.
    assert(b.mag != 0_u64, 'b can not be 0');

    // The sign of the quotient is the XOR of the signs of the operands.
    let sign = a.sign ^ b.sign;

    if (sign == false) {
        // If the operands are positive, the quotient is simply their absolute value quotient.
        return IntegerTrait::new(a.mag / b.mag, sign);
    }

    // If the operands have different signs, rounding is necessary.
    // First, check if the quotient is an integer.
    if (a.mag % b.mag == 0_u64) {
        let quotient = a.mag / b.mag;
        if (quotient == 0_u64) {
            return IntegerTrait::new(quotient, false);
        }
        return IntegerTrait::new(quotient, sign);
    }

    // If the quotient is not an integer, multiply the dividend by 10 to move the decimal point over.
    let quotient = (a.mag * 10_u64) / b.mag;
    let last_digit = quotient % 10_u64;

    if (quotient == 0_u64) {
        return IntegerTrait::new(quotient, false);
    }

    // Check the last digit to determine rounding direction.
    if (last_digit <= 5_u64) {
        return IntegerTrait::new(quotient / 10_u64, sign);
    } else {
        return IntegerTrait::new((quotient / 10_u64) + 1_u64, sign);
    }
}

// Calculates the remainder of the division of a first i64 by a second i64.
// # Arguments
// * `a` - The i64 dividend.
// * `b` - The i64 divisor.
// # Returns
// * `i64` - The remainder of dividing `a` by `b`.
fn i64_rem(a: i64, b: i64) -> i64 {
    i64_check_sign_zero(a);
    // Check that the divisor is not zero.
    assert(b.mag != 0_u64, 'b can not be 0');

    return a - (b * (a / b));
}

// Calculates both the quotient and the remainder of the division of a first i64 by a second i64.
// # Arguments
// * `a` - The i64 dividend.
// * `b` - The i64 divisor.
// # Returns
// * `(i64, i64)` - A tuple containing the quotient and the remainder of dividing `a` by `b`.
fn i64_div_rem(a: i64, b: i64) -> (i64, i64) {
    check_gas();
    let quotient = i64_div(a, b);
    let remainder = i64_rem(a, b);

    return (quotient, remainder);
}

// Compares two i64 integers for equality.
// # Arguments
// * `a` - The first i64 integer to compare.
// * `b` - The second i64 integer to compare.
// # Returns
// * `bool` - `true` if the two integers are equal, `false` otherwise.
fn i64_eq(a: i64, b: i64) -> bool {
    // Check if the two integers have the same sign and the same absolute value.
    if a.sign == b.sign & a.mag == b.mag {
        return true;
    }

    return false;
}

// Compares two i64 integers for inequality.
// # Arguments
// * `a` - The first i64 integer to compare.
// * `b` - The second i64 integer to compare.
// # Returns
// * `bool` - `true` if the two integers are not equal, `false` otherwise.
fn i64_ne(a: i64, b: i64) -> bool {
    // The result is the inverse of the equal function.
    return !i64_eq(a, b);
}

// Compares two i64 integers for greater than.
// # Arguments
// * `a` - The first i64 integer to compare.
// * `b` - The second i64 integer to compare.
// # Returns
// * `bool` - `true` if `a` is greater than `b`, `false` otherwise.
fn i64_gt(a: i64, b: i64) -> bool {
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

// Determines whether the first i64 is less than the second i64.
// # Arguments
// * `a` - The i64 to compare against the second i64.
// * `b` - The i64 to compare against the first i64.
// # Returns
// * `bool` - `true` if `a` is less than `b`, `false` otherwise.
fn i64_lt(a: i64, b: i64) -> bool {
    // The result is the inverse of the greater than function.
    return !i64_gt(a, b);
}

// Checks if the first i64 integer is less than or equal to the second.
// # Arguments
// * `a` - The first i64 integer to compare.
// * `b` - The second i64 integer to compare.
// # Returns
// * `bool` - `true` if `a` is less than or equal to `b`, `false` otherwise.
fn i64_le(a: i64, b: i64) -> bool {
    if (a == b | i64_lt(a, b) == true) {
        return true;
    } else {
        return false;
    }
}

// Checks if the first i64 integer is greater than or equal to the second.
// # Arguments
// * `a` - The first i64 integer to compare.
// * `b` - The second i64 integer to compare.
// # Returns
// * `bool` - `true` if `a` is greater than or equal to `b`, `false` otherwise.
fn i64_ge(a: i64, b: i64) -> bool {
    if (a == b | i64_gt(a, b) == true) {
        return true;
    } else {
        return false;
    }
}

// Negates the given i64 integer.
// # Arguments
// * `x` - The i64 integer to negate.
// # Returns
// * `i64` - The negation of `x`.
fn i64_neg(x: i64) -> i64 {
    // The negation of an integer is obtained by flipping its sign.
    return IntegerTrait::new(x.mag, !x.sign);
}

// Computes the absolute value of the given i64 integer.
// # Arguments
// * `x` - The i64 integer to compute the absolute value of.
// # Returns
// * `i64` - The absolute value of `x`.
fn i64_abs(x: i64) -> i64 {
    return IntegerTrait::new(x.mag, false);
}

// Computes the maximum between two i64 integers.
// # Arguments
// * `a` - The first i64 integer to compare.
// * `b` - The second i64 integer to compare.
// # Returns
// * `i64` - The maximum between `a` and `b`.
fn i64_max(a: i64, b: i64) -> i64 {
    if (a > b) {
        return a;
    } else {
        return b;
    }
}

// Computes the minimum between two i64 integers.
// # Arguments
// * `a` - The first i64 integer to compare.
// * `b` - The second i64 integer to compare.
// # Returns
// * `i64` - The minimum between `a` and `b`.
fn i64_min(a: i64, b: i64) -> i64 {
    if (a < b) {
        return a;
    } else {
        return b;
    }
}
