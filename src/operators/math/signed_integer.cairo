// Signed integers : i8, i16, i32, i64, i128

trait IntegerTrait<T, U> {
    fn new(mag: U, sign: bool) -> T;
    fn div_rem(self: T, other: T) -> (T, T);
    fn abs(self: T) -> T;
    fn max(self: T, other: T) -> T;
    fn min(self: T, other: T) -> T;
}

// ====================== INT 8 ======================

// i8 represents a 8-bit integer.
// The mag field holds the absolute value of the integer.
// The sign field is true for negative integers, and false for non-negative integers.
#[derive(Copy, Drop)]
struct i8 {
    mag: u8,
    sign: bool,
}

impl i8Impl of IntegerTrait::<i8, u8> {
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

// Implements the Add trait for i8.
impl i8Add of Add::<i8> {
    fn add(a: i8, b: i8) -> i8 {
        i8_add(a, b)
    }
}

// Implements the AddEq trait for i8.
impl i8AddEq of AddEq::<i8> {
    #[inline(always)]
    fn add_eq(ref self: i8, other: i8) {
        self = Add::add(self, other);
    }
}

// Implements the Sub trait for i8.
impl i8Sub of Sub::<i8> {
    fn sub(a: i8, b: i8) -> i8 {
        i8_sub(a, b)
    }
}

// Implements the SubEq trait for i8.
impl i8SubEq of SubEq::<i8> {
    #[inline(always)]
    fn sub_eq(ref self: i8, other: i8) {
        self = Sub::sub(self, other);
    }
}

// Implements the Mul trait for i8.
impl i8Mul of Mul::<i8> {
    fn mul(a: i8, b: i8) -> i8 {
        i8_mul(a, b)
    }
}

// Implements the MulEq trait for i8.
impl i8MulEq of MulEq::<i8> {
    #[inline(always)]
    fn mul_eq(ref self: i8, other: i8) {
        self = Mul::mul(self, other);
    }
}

// Implements the Div trait for i8.
impl i8Div of Div::<i8> {
    fn div(a: i8, b: i8) -> i8 {
        i8_div(a, b)
    }
}

// Implements the DivEq trait for i8.
impl i8DivEq of DivEq::<i8> {
    #[inline(always)]
    fn div_eq(ref self: i8, other: i8) {
        self = Div::div(self, other);
    }
}

// Implements the Rem trait for i8.
impl i8Rem of Rem::<i8> {
    fn rem(a: i8, b: i8) -> i8 {
        i8_rem(a, b)
    }
}

// Implements the RemEq trait for i8.
impl i8RemEq of RemEq::<i8> {
    #[inline(always)]
    fn rem_eq(ref self: i8, other: i8) {
        self = Rem::rem(self, other);
    }
}

// Implements the PartialEq trait for i8.
impl i8PartialEq of PartialEq::<i8> {
    fn eq(a: i8, b: i8) -> bool {
        i8_eq(a, b)
    }

    fn ne(a: i8, b: i8) -> bool {
        i8_ne(a, b)
    }
}

// Implements the PartialOrd trait for i8.
impl i8PartialOrd of PartialOrd::<i8> {
    fn le(a: i8, b: i8) -> bool {
        i8_le(a, b)
    }
    fn ge(a: i8, b: i8) -> bool {
        i8_ge(a, b)
    }

    fn lt(a: i8, b: i8) -> bool {
        i8_lt(a, b)
    }
    fn gt(a: i8, b: i8) -> bool {
        i8_gt(a, b)
    }
}

// Implements the Neg trait for i8.
impl i8Neg of Neg::<i8> {
    fn neg(x: i8) -> i8 {
        i8_neg(x)
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

// Create a new int8.
// # Arguments
// * `mag` - The magnitude
// * `sign` - The sign of the integer
// # Panics
// Panics if `mag` is out of range.
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

// Calculates both the quotient and the remainder of the division of a first i8 by a second i8.
// # Arguments
// * `a` - The i8 dividend.
// * `b` - The i8 divisor.
// # Returns
// * `(i8, i8)` - A tuple containing the quotient and the remainder of dividing `a` by `b`.
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
    if a.sign == b.sign & a.mag == b.mag {
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
    // The result is the inverse of the greater than function.
    return !i8_gt(a, b);
}

// Checks if the first i8 integer is less than or equal to the second.
// # Arguments
// * `a` - The first i8 integer to compare.
// * `b` - The second i8 integer to compare.
// # Returns
// * `bool` - `true` if `a` is less than or equal to `b`, `false` otherwise.
fn i8_le(a: i8, b: i8) -> bool {
    if (a == b | i8_lt(a, b) == true) {
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
    if (a == b | i8_gt(a, b) == true) {
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

// Computes the absolute value of the given i8 integer.
// # Arguments
// * `x` - The i8 integer to compute the absolute value of.
// # Returns
// * `i8` - The absolute value of `x`.
fn i8_abs(x: i8) -> i8 {
    return IntegerTrait::new(x.mag, false);
}

// Computes the maximum between two i8 integers.
// # Arguments
// * `a` - The first i8 integer to compare.
// * `b` - The second i8 integer to compare.
// # Returns
// * `i8` - The maximum between `a` and `b`.
fn i8_max(a: i8, b: i8) -> i8 {
    if (a > b) {
        return a;
    } else {
        return b;
    }
}

// Computes the minimum between two i8 integers.
// # Arguments
// * `a` - The first i8 integer to compare.
// * `b` - The second i8 integer to compare.
// # Returns
// * `i8` - The minimum between `a` and `b`.
fn i8_min(a: i8, b: i8) -> i8 {
    if (a < b) {
        return a;
    } else {
        return b;
    }
}

// ====================== INT 16 ======================

// i16 represents a 16-bit integer.
// The mag field holds the absolute value of the integer.
// The sign field is true for negative integers, and false for non-negative integers.
#[derive(Copy, Drop)]
struct i16 {
    mag: u16,
    sign: bool,
}

impl i16Impl of IntegerTrait::<i16, u16> {
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

// Implements the Add trait for i16.
impl i16Add of Add::<i16> {
    fn add(a: i16, b: i16) -> i16 {
        i16_add(a, b)
    }
}

// Implements the AddEq trait for i16.
impl i16AddEq of AddEq::<i16> {
    #[inline(always)]
    fn add_eq(ref self: i16, other: i16) {
        self = Add::add(self, other);
    }
}

// Implements the Sub trait for i16.
impl i16Sub of Sub::<i16> {
    fn sub(a: i16, b: i16) -> i16 {
        i16_sub(a, b)
    }
}

// Implements the SubEq trait for i16.
impl i16SubEq of SubEq::<i16> {
    #[inline(always)]
    fn sub_eq(ref self: i16, other: i16) {
        self = Sub::sub(self, other);
    }
}

// Implements the Mul trait for i16.
impl i16Mul of Mul::<i16> {
    fn mul(a: i16, b: i16) -> i16 {
        i16_mul(a, b)
    }
}

// Implements the MulEq trait for i16.
impl i16MulEq of MulEq::<i16> {
    #[inline(always)]
    fn mul_eq(ref self: i16, other: i16) {
        self = Mul::mul(self, other);
    }
}

// Implements the Div trait for i16.
impl i16Div of Div::<i16> {
    fn div(a: i16, b: i16) -> i16 {
        i16_div(a, b)
    }
}

// Implements the DivEq trait for i16.
impl i16DivEq of DivEq::<i16> {
    #[inline(always)]
    fn div_eq(ref self: i16, other: i16) {
        self = Div::div(self, other);
    }
}

// Implements the Rem trait for i16.
impl i16Rem of Rem::<i16> {
    fn rem(a: i16, b: i16) -> i16 {
        i16_rem(a, b)
    }
}

// Implements the RemEq trait for i16.
impl i16RemEq of RemEq::<i16> {
    #[inline(always)]
    fn rem_eq(ref self: i16, other: i16) {
        self = Rem::rem(self, other);
    }
}

// Implements the PartialEq trait for i16.
impl i16PartialEq of PartialEq::<i16> {
    fn eq(a: i16, b: i16) -> bool {
        i16_eq(a, b)
    }

    fn ne(a: i16, b: i16) -> bool {
        i16_ne(a, b)
    }
}

// Implements the PartialOrd trait for i16.
impl i16PartialOrd of PartialOrd::<i16> {
    fn le(a: i16, b: i16) -> bool {
        i16_le(a, b)
    }
    fn ge(a: i16, b: i16) -> bool {
        i16_ge(a, b)
    }

    fn lt(a: i16, b: i16) -> bool {
        i16_lt(a, b)
    }
    fn gt(a: i16, b: i16) -> bool {
        i16_gt(a, b)
    }
}

// Implements the Neg trait for i16.
impl i16Neg of Neg::<i16> {
    fn neg(x: i16) -> i16 {
        i16_neg(x)
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

// Create a new int16.
// # Arguments
// * `mag` - The magnitude
// * `sign` - The sign of the integer
// # Panics
// Panics if `mag` is out of range.
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

        if (difference == 0_u16) {
            return IntegerTrait::new(difference, false);
        }
        return IntegerTrait::new(difference, larger.sign);
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
    let neg_b = IntegerTrait::new(b.mag, !b.sign);
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

    return IntegerTrait::new(mag, sign);
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
        return IntegerTrait::new(a.mag / b.mag, sign);
    }

    // If the operands have different signs, rounding is necessary.
    // First, check if the quotient is an integer.
    if (a.mag % b.mag == 0_u16) {
        let quotient = a.mag / b.mag;
        if (quotient == 0_u16) {
            return IntegerTrait::new(quotient, false);
        }
        return IntegerTrait::new(quotient, sign);
    }

    // If the quotient is not an integer, multiply the dividend by 10 to move the decimal point over.
    let quotient = (a.mag * 10_u16) / b.mag;
    let last_digit = quotient % 10_u16;

    if (quotient == 0_u16) {
        return IntegerTrait::new(quotient, false);
    }

    // Check the last digit to determine rounding direction.
    if (last_digit <= 5_u16) {
        return IntegerTrait::new(quotient / 10_u16, sign);
    } else {
        return IntegerTrait::new((quotient / 10_u16) + 1_u16, sign);
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

// Calculates both the quotient and the remainder of the division of a first i16 by a second i16.
// # Arguments
// * `a` - The i16 dividend.
// * `b` - The i16 divisor.
// # Returns
// * `(i16, i16)` - A tuple containing the quotient and the remainder of dividing `a` by `b`.
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
    if a.sign == b.sign & a.mag == b.mag {
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
    // The result is the inverse of the greater than function.
    return !i16_gt(a, b);
}

// Checks if the first i16 integer is less than or equal to the second.
// # Arguments
// * `a` - The first i16 integer to compare.
// * `b` - The second i16 integer to compare.
// # Returns
// * `bool` - `true` if `a` is less than or equal to `b`, `false` otherwise.
fn i16_le(a: i16, b: i16) -> bool {
    if (a == b | i16_lt(a, b) == true) {
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
    if (a == b | i16_gt(a, b) == true) {
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
    return IntegerTrait::new(x.mag, !x.sign);
}

// Computes the absolute value of the given i16 integer.
// # Arguments
// * `x` - The i16 integer to compute the absolute value of.
// # Returns
// * `i16` - The absolute value of `x`.
fn i16_abs(x: i16) -> i16 {
    return IntegerTrait::new(x.mag, false);
}

// Computes the maximum between two i16 integers.
// # Arguments
// * `a` - The first i16 integer to compare.
// * `b` - The second i16 integer to compare.
// # Returns
// * `i16` - The maximum between `a` and `b`.
fn i16_max(a: i16, b: i16) -> i16 {
    if (a > b) {
        return a;
    } else {
        return b;
    }
}

// Computes the minimum between two i16 integers.
// # Arguments
// * `a` - The first i16 integer to compare.
// * `b` - The second i16 integer to compare.
// # Returns
// * `i16` - The minimum between `a` and `b`.
fn i16_min(a: i16, b: i16) -> i16 {
    if (a < b) {
        return a;
    } else {
        return b;
    }
}

// ====================== INT 32 ======================

// i32 represents a 32-bit integer.
// The mag field holds the absolute value of the integer.
// The sign field is true for negative integers, and false for non-negative integers.
#[derive(Copy, Drop)]
struct i32 {
    mag: u32,
    sign: bool,
}

impl i32Impl of IntegerTrait::<i32, u32> {
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
}

// Implements the Add trait for i32.
impl i32Add of Add::<i32> {
    fn add(a: i32, b: i32) -> i32 {
        i32_add(a, b)
    }
}

// Implements the AddEq trait for i32.
impl i32AddEq of AddEq::<i32> {
    #[inline(always)]
    fn add_eq(ref self: i32, other: i32) {
        self = Add::add(self, other);
    }
}

// Implements the Sub trait for i32.
impl i32Sub of Sub::<i32> {
    fn sub(a: i32, b: i32) -> i32 {
        i32_sub(a, b)
    }
}

// Implements the SubEq trait for i32.
impl i32SubEq of SubEq::<i32> {
    #[inline(always)]
    fn sub_eq(ref self: i32, other: i32) {
        self = Sub::sub(self, other);
    }
}

// Implements the Mul trait for i32.
impl i32Mul of Mul::<i32> {
    fn mul(a: i32, b: i32) -> i32 {
        i32_mul(a, b)
    }
}

// Implements the MulEq trait for i32.
impl i32MulEq of MulEq::<i32> {
    #[inline(always)]
    fn mul_eq(ref self: i32, other: i32) {
        self = Mul::mul(self, other);
    }
}

// Implements the Div trait for i32.
impl i32Div of Div::<i32> {
    fn div(a: i32, b: i32) -> i32 {
        i32_div(a, b)
    }
}

// Implements the DivEq trait for i32.
impl i32DivEq of DivEq::<i32> {
    #[inline(always)]
    fn div_eq(ref self: i32, other: i32) {
        self = Div::div(self, other);
    }
}

// Implements the Rem trait for i32.
impl i32Rem of Rem::<i32> {
    fn rem(a: i32, b: i32) -> i32 {
        i32_rem(a, b)
    }
}

// Implements the RemEq trait for i32.
impl i32RemEq of RemEq::<i32> {
    #[inline(always)]
    fn rem_eq(ref self: i32, other: i32) {
        self = Rem::rem(self, other);
    }
}

// Implements the PartialEq trait for i32.
impl i32PartialEq of PartialEq::<i32> {
    fn eq(a: i32, b: i32) -> bool {
        i32_eq(a, b)
    }

    fn ne(a: i32, b: i32) -> bool {
        i32_ne(a, b)
    }
}

// Implements the PartialOrd trait for i32.
impl i32PartialOrd of PartialOrd::<i32> {
    fn le(a: i32, b: i32) -> bool {
        i32_le(a, b)
    }
    fn ge(a: i32, b: i32) -> bool {
        i32_ge(a, b)
    }

    fn lt(a: i32, b: i32) -> bool {
        i32_lt(a, b)
    }
    fn gt(a: i32, b: i32) -> bool {
        i32_gt(a, b)
    }
}

// Implements the Neg trait for i32.
impl i32Neg of Neg::<i32> {
    fn neg(x: i32) -> i32 {
        i32_neg(x)
    }
}


// Checks if the given i32 integer is zero and has the correct sign.
// # Arguments
// * `x` - The i32 integer to check.
// # Panics
// Panics if `x` is zero and has a sign that is not false.
fn i32_check_sign_zero(x: i32) {
    if x.mag == 0_u32 {
        assert(x.sign == false, 'sign of 0 must be false');
    }
}

// Create a new int32.
// # Arguments
// * `mag` - The magnitude
// * `sign` - The sign of the integer
// # Panics
// Panics if `mag` is out of range.
fn i32_new(mag: u32, sign: bool) -> i32 {
    if sign == true {
        assert(mag <= 2147483648_u32, 'int: out of range');
    } else {
        assert(mag <= 2147483647_u32, 'int: out of range');
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
        if (sum == 0_u32) {
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

        if (difference == 0_u32) {
            return IntegerTrait::new(difference, false);
        }
        return IntegerTrait::new(difference, larger.sign);
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

    if (b.mag == 0_u32) {
        return a;
    }

    // The subtraction of `a` to `b` is achieved by negating `b` sign and adding it to `a`.
    let neg_b = IntegerTrait::new(b.mag, !b.sign);
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

    if (mag == 0_u32) {
        return IntegerTrait::new(mag, false);
    }

    return IntegerTrait::new(mag, sign);
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
    assert(b.mag != 0_u32, 'b can not be 0');

    // The sign of the quotient is the XOR of the signs of the operands.
    let sign = a.sign ^ b.sign;

    if (sign == false) {
        // If the operands are positive, the quotient is simply their absolute value quotient.
        return IntegerTrait::new(a.mag / b.mag, sign);
    }

    // If the operands have different signs, rounding is necessary.
    // First, check if the quotient is an integer.
    if (a.mag % b.mag == 0_u32) {
        let quotient = a.mag / b.mag;
        if (quotient == 0_u32) {
            return IntegerTrait::new(quotient, false);
        }
        return IntegerTrait::new(quotient, sign);
    }

    // If the quotient is not an integer, multiply the dividend by 10 to move the decimal point over.
    let quotient = (a.mag * 10_u32) / b.mag;
    let last_digit = quotient % 10_u32;

    if (quotient == 0_u32) {
        return IntegerTrait::new(quotient, false);
    }

    // Check the last digit to determine rounding direction.
    if (last_digit <= 5_u32) {
        return IntegerTrait::new(quotient / 10_u32, sign);
    } else {
        return IntegerTrait::new((quotient / 10_u32) + 1_u32, sign);
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
    assert(b.mag != 0_u32, 'b can not be 0');

    return a - (b * (a / b));
}

// Calculates both the quotient and the remainder of the division of a first i32 by a second i32.
// # Arguments
// * `a` - The i32 dividend.
// * `b` - The i32 divisor.
// # Returns
// * `(i32, i32)` - A tuple containing the quotient and the remainder of dividing `a` by `b`.
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
    if a.sign == b.sign & a.mag == b.mag {
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
    // The result is the inverse of the greater than function.
    return !i32_gt(a, b);
}

// Checks if the first i32 integer is less than or equal to the second.
// # Arguments
// * `a` - The first i32 integer to compare.
// * `b` - The second i32 integer to compare.
// # Returns
// * `bool` - `true` if `a` is less than or equal to `b`, `false` otherwise.
fn i32_le(a: i32, b: i32) -> bool {
    if (a == b | i32_lt(a, b) == true) {
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
    if (a == b | i32_gt(a, b) == true) {
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
    return IntegerTrait::new(x.mag, !x.sign);
}

// Computes the absolute value of the given i32 integer.
// # Arguments
// * `x` - The i32 integer to compute the absolute value of.
// # Returns
// * `i32` - The absolute value of `x`.
fn i32_abs(x: i32) -> i32 {
    return IntegerTrait::new(x.mag, false);
}

// Computes the maximum between two i32 integers.
// # Arguments
// * `a` - The first i32 integer to compare.
// * `b` - The second i32 integer to compare.
// # Returns
// * `i32` - The maximum between `a` and `b`.
fn i32_max(a: i32, b: i32) -> i32 {
    if (a > b) {
        return a;
    } else {
        return b;
    }
}

// Computes the minimum between two i32 integers.
// # Arguments
// * `a` - The first i32 integer to compare.
// * `b` - The second i32 integer to compare.
// # Returns
// * `i32` - The minimum between `a` and `b`.
fn i32_min(a: i32, b: i32) -> i32 {
    if (a < b) {
        return a;
    } else {
        return b;
    }
}

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

// ====================== INT 128 ======================

// i128 represents a 128-bit integer.
// The mag field holds the absolute value of the integer.
// The sign field is true for negative integers, and false for non-negative integers.
#[derive(Copy, Drop)]
struct i128 {
    mag: u128,
    sign: bool,
}

impl i128Impl of IntegerTrait::<i128, u128> {
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
}

// Implements the Add trait for i128.
impl i128Add of Add::<i128> {
    fn add(a: i128, b: i128) -> i128 {
        i128_add(a, b)
    }
}

// Implements the AddEq trait for i128.
impl i128AddEq of AddEq::<i128> {
    #[inline(always)]
    fn add_eq(ref self: i128, other: i128) {
        self = Add::add(self, other);
    }
}

// Implements the Sub trait for i128.
impl i128Sub of Sub::<i128> {
    fn sub(a: i128, b: i128) -> i128 {
        i128_sub(a, b)
    }
}

// Implements the SubEq trait for i128.
impl i128SubEq of SubEq::<i128> {
    #[inline(always)]
    fn sub_eq(ref self: i128, other: i128) {
        self = Sub::sub(self, other);
    }
}

// Implements the Mul trait for i128.
impl i128Mul of Mul::<i128> {
    fn mul(a: i128, b: i128) -> i128 {
        i128_mul(a, b)
    }
}

// Implements the MulEq trait for i128.
impl i128MulEq of MulEq::<i128> {
    #[inline(always)]
    fn mul_eq(ref self: i128, other: i128) {
        self = Mul::mul(self, other);
    }
}

// Implements the Div trait for i128.
impl i128Div of Div::<i128> {
    fn div(a: i128, b: i128) -> i128 {
        i128_div(a, b)
    }
}

// Implements the DivEq trait for i128.
impl i128DivEq of DivEq::<i128> {
    #[inline(always)]
    fn div_eq(ref self: i128, other: i128) {
        self = Div::div(self, other);
    }
}

// Implements the Rem trait for i128.
impl i128Rem of Rem::<i128> {
    fn rem(a: i128, b: i128) -> i128 {
        i128_rem(a, b)
    }
}

// Implements the RemEq trait for i128.
impl i128RemEq of RemEq::<i128> {
    #[inline(always)]
    fn rem_eq(ref self: i128, other: i128) {
        self = Rem::rem(self, other);
    }
}

// Implements the PartialEq trait for i128.
impl i128PartialEq of PartialEq::<i128> {
    fn eq(a: i128, b: i128) -> bool {
        i128_eq(a, b)
    }

    fn ne(a: i128, b: i128) -> bool {
        i128_ne(a, b)
    }
}

// Implements the PartialOrd trait for i128.
impl i128PartialOrd of PartialOrd::<i128> {
    fn le(a: i128, b: i128) -> bool {
        i128_le(a, b)
    }
    fn ge(a: i128, b: i128) -> bool {
        i128_ge(a, b)
    }

    fn lt(a: i128, b: i128) -> bool {
        i128_lt(a, b)
    }
    fn gt(a: i128, b: i128) -> bool {
        i128_gt(a, b)
    }
}

// Implements the Neg trait for i128.
impl i128Neg of Neg::<i128> {
    fn neg(x: i128) -> i128 {
        i128_neg(x)
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

// Create a new int128.
// # Arguments
// * `mag` - The magnitude
// * `sign` - The sign of the integer
// # Panics
// Panics if `mag` is out of range.
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

        if (difference == 0_u128) {
            return IntegerTrait::new(difference, false);
        }
        return IntegerTrait::new(difference, larger.sign);
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
    let neg_b = IntegerTrait::new(b.mag, !b.sign);
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

    return IntegerTrait::new(mag, sign);
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
        return IntegerTrait::new(a.mag / b.mag, sign);
    }

    // If the operands have different signs, rounding is necessary.
    // First, check if the quotient is an integer.
    if (a.mag % b.mag == 0_u128) {
        let quotient = a.mag / b.mag;
        if (quotient == 0_u128) {
            return IntegerTrait::new(quotient, false);
        }
        return IntegerTrait::new(quotient, sign);
    }

    // If the quotient is not an integer, multiply the dividend by 10 to move the decimal point over.
    let quotient = (a.mag * 10_u128) / b.mag;
    let last_digit = quotient % 10_u128;

    if (quotient == 0_u128) {
        return IntegerTrait::new(quotient, false);
    }

    // Check the last digit to determine rounding direction.
    if (last_digit <= 5_u128) {
        return IntegerTrait::new(quotient / 10_u128, sign);
    } else {
        return IntegerTrait::new((quotient / 10_u128) + 1_u128, sign);
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

// Calculates both the quotient and the remainder of the division of a first i128 by a second i128.
// # Arguments
// * `a` - The i128 dividend.
// * `b` - The i128 divisor.
// # Returns
// * `(i128, i128)` - A tuple containing the quotient and the remainder of dividing `a` by `b`.
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
    if a.sign == b.sign & a.mag == b.mag {
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
    // The result is the inverse of the greater than function.
    return !i128_gt(a, b);
}

// Checks if the first i128 integer is less than or equal to the second.
// # Arguments
// * `a` - The first i128 integer to compare.
// * `b` - The second i128 integer to compare.
// # Returns
// * `bool` - `true` if `a` is less than or equal to `b`, `false` otherwise.
fn i128_le(a: i128, b: i128) -> bool {
    if (a == b | i128_lt(a, b) == true) {
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
    if (a == b | i128_gt(a, b) == true) {
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
    return IntegerTrait::new(x.mag, !x.sign);
}

// Computes the absolute value of the given i128 integer.
// # Arguments
// * `x` - The i128 integer to compute the absolute value of.
// # Returns
// * `i128` - The absolute value of `x`.
fn i128_abs(x: i128) -> i128 {
    return IntegerTrait::new(x.mag, false);
}

// Computes the maximum between two i128 integers.
// # Arguments
// * `a` - The first i128 integer to compare.
// * `b` - The second i128 integer to compare.
// # Returns
// * `i128` - The maximum between `a` and `b`.
fn i128_max(a: i128, b: i128) -> i128 {
    if (a > b) {
        return a;
    } else {
        return b;
    }
}

// Computes the minimum between two i128 integers.
// # Arguments
// * `a` - The first i128 integer to compare.
// * `b` - The second i128 integer to compare.
// # Returns
// * `i128` - The minimum between `a` and `b`.
fn i128_min(a: i128, b: i128) -> i128 {
    if (a < b) {
        return a;
    } else {
        return b;
    }
}