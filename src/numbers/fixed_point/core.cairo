/// A fixed point arithmetic library for handling signed fixed point numbers.
///
/// The library provides basic arithmetic operations, trigonometric functions,
/// logarithmic functions, and power functions for signed fixed point numbers.
/// Fixed point numbers are represented as a struct FixedType with a magnitude
/// and a sign. The magnitude represents the absolute value of the number, and
/// the sign indicates whether the number is positive or negative.
/// Fixed Point Q5.26 -> https://github.com/tensorflow/tensorflow/blob/305fec9fddc3bdb5bb574a134b955bf4b07fd795/tensorflow/lite/kernels/internal/reference/softmax.h#L66
/// Implemented from https://github.com/influenceth/cubit and adjusted to Q5.26

use traits::Into;

use onnx_cairo::numbers::fixed_point::types::{
    HALF_u128, MAX_u128, ONE_u128, ONE_u64, Fixed, FixedType,
};
use onnx_cairo::utils::check_gas;

//! PUBLIC

/// Returns the absolute value of a fixed point number.
///
/// # Arguments
///
/// * `a` - A fixed point number.
///
/// # Returns
///
/// * The absolute value of the input fixed point number.
fn abs(a: FixedType) -> FixedType {
    return Fixed::new(a.mag, false);
}

/// Adds two fixed point numbers and returns the result.
///
/// # Arguments
///
/// * `a` - The first fixed point number to add.
/// * `b` - The second fixed point number to add.
///
/// # Returns
///
/// * The sum of the input fixed point numbers.
fn add(a: FixedType, b: FixedType) -> FixedType {
    check_gas();
    return Fixed::from_felt(a.into() + b.into());
}

/// Returns the smallest integer greater than or equal to the input fixed point number.
///
/// # Arguments
///
/// * `a` - A fixed point number.
///
/// # Returns
///
/// * The smallest integer greater than or equal to the input fixed point number.
fn ceil(a: FixedType) -> FixedType {
    let (div_u128, rem_u128) = _split_unsigned(a);

    if (rem_u128 == 0_u128) {
        return a;
    } else if (a.sign == false) {
        return Fixed::new_unscaled(div_u128 + 1_u128, false);
    } else {
        return Fixed::from_unscaled_felt(div_u128.into() * -1);
    }
}

/// Divides the first fixed point number by the second fixed point number and returns the result.
///
/// # Arguments
///
/// * `a` - The dividend fixed point number.
/// * `b` - The divisor fixed point number.
///
/// # Returns
///
/// * The result of the division of the input fixed point numbers.
fn div(a: FixedType, b: FixedType) -> FixedType {
    check_gas();
    let res_sign = a.sign ^ b.sign;

    // Invert b to preserve precision as much as possible
    // TODO: replace if / when there is a felt div_rem supported
    let (a_high, a_low) = integer::u128_wide_mul(a.mag, ONE_u128);
    let b_inv = MAX_u128 / b.mag;
    let res_u128 = a_low / b.mag + a_high * b_inv;

    // Re-apply sign
    return FixedType { mag: res_u128, sign: res_sign };
}

/// Checks whether two fixed point numbers are equal.
///
/// # Arguments
///
/// * `a` - The first fixed point number to compare.
/// * `b` - The second fixed point number to compare.
///
/// # Returns
///
/// * A boolean value that indicates whether the input fixed point numbers are equal.
fn eq(a: FixedType, b: FixedType) -> bool {
    return a.mag == b.mag & a.sign == b.sign;
}

/// Calculates the natural exponent of a fixed point number: e^x
///
/// # Arguments
///
/// * `a` - A fixed point number.
///
/// # Returns
///
/// * The natural exponent of the input fixed point number.
fn exp(a: FixedType) -> FixedType {
    return exp2(Fixed::new(96817625_u128, false) * a); // log2(e) * 2^26 ≈ 96817625
}

/// Calculates the binary exponent of a fixed point number: 2^x
///
/// # Arguments
///
/// * `a` - A fixed point number.
///
/// # Returns
///
/// * The binary exponent of the input fixed point number.
fn exp2(a: FixedType) -> FixedType {
    if (a.mag == 0_u128) {
        return Fixed::new(ONE_u128, false);
    }

    let (int_part, frac_part) = _split_unsigned(a);
    let int_res = _pow_int(Fixed::new_unscaled(2_u128, false), int_part, false);

    let t8 = Fixed::new(152_u128, false);
    let t7 = Fixed::new(843_u128, false);
    let t6 = Fixed::new(10593_u128, false);
    let t5 = Fixed::new(89275_u128, false);
    let t4 = Fixed::new(645557_u128, false);
    let t3 = Fixed::new(3724792_u128, false);
    let t2 = Fixed::new(16121331_u128, false);
    let t1 = Fixed::new(46516320_u128, false);

    let frac_fixed = Fixed::new(frac_part, false);
    let r8 = t8 * frac_fixed;
    let r7 = (r8 + t7) * frac_fixed;
    let r6 = (r7 + t6) * frac_fixed;
    let r5 = (r6 + t5) * frac_fixed;
    let r4 = (r5 + t4) * frac_fixed;
    let r3 = (r4 + t3) * frac_fixed;
    let r2 = (r3 + t2) * frac_fixed;
    let r1 = (r2 + t1) * frac_fixed;
    let frac_res = r1 + Fixed::new(ONE_u128, false);
    let res_u = int_res * frac_res;

    if (a.sign == true) {
        return Fixed::new(ONE_u128, false) / res_u;
    } else {
        return res_u;
    }
}

/// Returns the largest integer less than or equal to the input fixed point number.
///
/// # Arguments
///
/// * `a` - A fixed point number.
///
/// # Returns
///
/// * The largest integer less than or equal to the input fixed point number.
fn floor(a: FixedType) -> FixedType {
    let (div_u128, rem_u128) = _split_unsigned(a);

    if (rem_u128 == 0_u128) {
        return a;
    } else if (a.sign == false) {
        return Fixed::new_unscaled(div_u128, false);
    } else {
        return Fixed::from_unscaled_felt(-1 * div_u128.into() - 1);
    }
}

/// Checks whether the first fixed point number is greater than or equal to the second fixed point number.
///
/// # Arguments
///
/// * `a` - The first fixed point number to compare.
/// * `b` - The second fixed point number to compare.
///
/// # Returns
///
/// * A boolean value that indicates whether the first fixed point number is greater than or equal to the second fixed point number.
fn ge(a: FixedType, b: FixedType) -> bool {
    if (a.sign != b.sign) {
        return !a.sign;
    } else {
        return (a.mag == b.mag) | ((a.mag > b.mag) ^ a.sign);
    }
}

/// Checks whether the first fixed point number is greater than the second fixed point number.
///
/// # Arguments
///
/// * `a` - The first fixed point number to compare.
/// * `b` - The second fixed point number to compare.
///
/// # Returns
///
/// * A boolean value that indicates whether the first fixed point number is greater than the second fixed point number.
fn gt(a: FixedType, b: FixedType) -> bool {
    if (a.sign != b.sign) {
        return !a.sign;
    } else {
        return (a.mag != b.mag) & ((a.mag > b.mag) ^ a.sign);
    }
}

/// Checks whether the first fixed point number is less than or equal to the second fixed point number.
///
/// # Arguments
///
/// * `a` - The first fixed point number to compare.
/// * `b` - The second fixed point number to compare.
///
/// # Returns
///
/// * A boolean value that indicates whether the first fixed point number is less than or equal to the second fixed point number.
fn le(a: FixedType, b: FixedType) -> bool {
    if (a.sign != b.sign) {
        return a.sign;
    } else {
        return (a.mag == b.mag) | ((a.mag < b.mag) ^ a.sign);
    }
}

/// Calculates the natural logarithm of a fixed point number: ln(x).
///
/// # Arguments
///
/// * `a` - A fixed point number greater than zero.
///
/// # Returns
///
/// * A FixedType value representing the natural logarithm of the input number.
fn ln(a: FixedType) -> FixedType {
    return Fixed::new(46516320_u128, false) * log2(a); // ln(2) = 0.693...
}

/// Calculates the binary logarithm of a fixed point number: log2(x).
///
/// # Arguments
///
/// * `a` - A fixed point number greater than zero.
///
/// # Returns
///
/// * A FixedType value representing the binary logarithm of the input number.
fn log2(a: FixedType) -> FixedType {
    check_gas();

    assert(a.sign == false, 'must be positive');

    if (a.mag == ONE_u128) {
        return Fixed::new(0_u128, false);
    } else if (a.mag < ONE_u128) {
        // Compute true inverse binary log if 0 < x < 1
        let div = Fixed::new_unscaled(1_u128, false) / a;
        return -log2(div);
    }

    let msb_u128 = _msb(a.mag / 2_u128);
    let divisor = _pow_int(Fixed::new_unscaled(2_u128, false), msb_u128, false);
    let norm = a / divisor;

    let t8 = Fixed::new(609947_u128, true);
    let t7 = Fixed::new(8311147_u128, false);
    let t6 = Fixed::new(50221432_u128, true);
    let t5 = Fixed::new(177085162_u128, false);
    let t4 = Fixed::new(403554714_u128, true);
    let t3 = Fixed::new(623171909_u128, false);
    let t2 = Fixed::new(671567545_u128, true);
    let t1 = Fixed::new(547259664_u128, false);
    let t0 = Fixed::new(229874243_u128, true);

    let r8 = t8 * norm;
    let r7 = (r8 + t7) * norm;
    let r6 = (r7 + t6) * norm;
    let r5 = (r6 + t5) * norm;
    let r4 = (r5 + t4) * norm;
    let r3 = (r4 + t3) * norm;
    let r2 = (r3 + t2) * norm;
    let r1 = (r2 + t1) * norm;
    return r1 + t0 + Fixed::new_unscaled(msb_u128, false);
}

/// Calculates the base 10 logarithm of a fixed point number: log10(x).
///
/// # Arguments
///
/// * `a` - A fixed point number greater than zero.
///
/// # Returns
///
/// * A FixedType value representing the base 10 logarithm of the input number.
fn log10(a: FixedType) -> FixedType {
    return Fixed::new(20201781_u128, false) * log2(a); // log10(2) = 0.301...
}

/// Checks whether the first fixed point number is less than the second fixed point number.
///
/// # Arguments
///
/// * `a` - The first fixed point number to compare.
/// * `b` - The second fixed point number to compare.
///
/// # Returns
///
/// * A boolean value that indicates whether the first fixed point number is less than the second fixed point number.
fn lt(a: FixedType, b: FixedType) -> bool {
    if (a.sign != b.sign) {
        return a.sign;
    } else {
        return (a.mag != b.mag) & ((a.mag < b.mag) ^ a.sign);
    }
}

/// Multiplies two fixed point numbers.
///
/// # Arguments
///
/// * `a` - The first fixed point number.
/// * `b` - The second fixed point number.
///
/// # Returns
///
/// * A FixedType value representing the product of the two input numbers.
fn mul(a: FixedType, b: FixedType) -> FixedType {
    check_gas();

    let res_sign = a.sign ^ b.sign;

    // Use u128 to multiply and shift back down
    // TODO: replace if / when there is a felt div_rem supported
    let (high, low) = integer::u128_wide_mul(a.mag, b.mag);
    let res_u128 = high + (low / ONE_u128);

    // Re-apply sign
    return FixedType { mag: res_u128, sign: res_sign };
}

/// Checks whether the first fixed point number is not equal to the second fixed point number.
///
/// # Arguments
///
/// * `a` - The first fixed point number to compare.
/// * `b` - The second fixed point number to compare.
///
/// # Returns
///
/// * A boolean value that indicates whether the first fixed point number is not equal to the second fixed point number.
fn ne(a: FixedType, b: FixedType) -> bool {
    return a.mag != b.mag | a.sign != b.sign;
}

/// Negates a fixed point number.
///
/// # Arguments
///
/// * `a` - The fixed point number to negate.
///
/// # Returns
///
/// * A FixedType value representing the negation of the input number.
fn neg(a: FixedType) -> FixedType {
    if (a.sign == false) {
        return Fixed::new(a.mag, true);
    } else {
        return Fixed::new(a.mag, false);
    }
}

/// Calculates the value of x^y for two fixed point numbers and checks for overflow before returning.
///
/// # Arguments
///
/// * `a` - The base fixed point number (x).
/// * `b` - The exponent fixed point number (y).
///
/// # Returns
///
/// * A fixed point number representing the result of x^y.
fn pow(a: FixedType, b: FixedType) -> FixedType {
    let (div_u128, rem_u128) = _split_unsigned(b);

    // use the more performant integer pow when y is an int
    if (rem_u128 == 0_u128) {
        return _pow_int(a, b.mag / ONE_u128, b.sign);
    }

    // x^y = exp(y*ln(x)) for x > 0 will error for x < 0
    return exp(b * ln(a));
}

/// Rounds a fixed point number to the nearest whole number.
///
/// # Arguments
///
/// * `a` - The fixed point number to round.
///
/// # Returns
///
/// * A fixed point number representing the rounded value.
fn round(a: FixedType) -> FixedType {
    let (div_u128, rem_u128) = _split_unsigned(a);

    if (HALF_u128 <= rem_u128) {
        return Fixed::new(ONE_u128 * (div_u128 + 1_u128), a.sign);
    } else {
        return Fixed::new(ONE_u128 * div_u128, a.sign);
    }
}

/// Calculates the square root of a positive fixed point number.
///
/// # Arguments
///
/// * `a` - The fixed point number to calculate the square root of. Must be positive.
///
/// # Returns
///
/// * A fixed point number representing the square root of the input value.
fn sqrt(a: FixedType) -> FixedType {
    assert(a.sign == false, 'must be positive');
    let root = integer::u128_sqrt(a.mag);
    let scale_root = integer::u128_sqrt(ONE_u128);
    let res_u64 = root * ONE_u64 / scale_root;
    return Fixed::new(res_u64.into(), false);
}

/// Subtracts one fixed point number from another.
///
/// # Arguments
///
/// * `a` - The minuend fixed point number.
/// * `b` - The subtrahend fixed point number.
///
/// # Returns
///
/// * A fixed point number representing the result of the subtraction.
fn sub(a: FixedType, b: FixedType) -> FixedType {
    check_gas();
    return Fixed::from_felt(a.into() - b.into());
}

/// Returns maximum value between two Fixed Points.
///
/// # Arguments
///
/// * `a` - The first fixed point number.
/// * `b` - The second fixed point number.
///
/// # Returns
///
/// * The max Fixed Point value .
fn max(a: FixedType, b: FixedType) -> FixedType {
    if (a >= b) {
        return a;
    } else {
        return b;
    }
}

/// Returns minimum value between two Fixed Points.
///
/// # Arguments
///
/// * `a` - The first fixed point number.
/// * `b` - The second fixed point number.
///
/// # Returns
///
/// * The min Fixed Point value.
fn min(a: FixedType, b: FixedType) -> FixedType {
    if (a <= b) {
        return a;
    } else {
        return b;
    }
}

/// INTERNAL

/// Calculates the most significant bit of a u128 number.
///
/// # Arguments
///
/// * `a` - The u128 number to find the most significant bit of.
///
/// # Returns
///
/// * A u128 value representing the most significant bit.
fn _msb(a: u128) -> u128 {
    check_gas();

    if (a <= ONE_u128) {
        return 0_u128;
    }

    return 1_u128 + _msb(a / 2_u128);
}

/// Calculates the value of x^y for a fixed point number `x` and a signed integer `y`, 
/// and checks for overflow before returning.
///
/// # Arguments
///
/// * `a` - The base fixed point number (x).
/// * `b` - The exponent as a u128 number (y).
/// * `sign` - A boolean value indicating the sign of the exponent.
///
/// # Returns
///
/// * A fixed point number representing the result of x^y.
fn _pow_int(a: FixedType, b: u128, sign: bool) -> FixedType {
    check_gas();

    if (sign == true) {
        return Fixed::new(ONE_u128, false) / _pow_int(a, b, false);
    }

    let (div, rem) = integer::u128_safe_divmod(b, integer::u128_as_non_zero(2_u128));

    if (b == 0_u128) {
        return Fixed::new(ONE_u128, false);
    } else if (rem == 0_u128) {
        return _pow_int(a * a, div, false);
    } else {
        return a * _pow_int(a * a, div, false);
    }
}

/// Ignores the sign and always returns false.
///
/// # Arguments
///
/// * `a` - The input fixed point number.
///
/// # Returns
///
/// * A tuple of two u128 numbers representing the division and remainder of the input number divided by `ONE_u128`.
fn _split_unsigned(a: FixedType) -> (u128, u128) {
    return integer::u128_safe_divmod(a.mag, integer::u128_as_non_zero(ONE_u128));
}
