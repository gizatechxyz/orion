use core::debug::PrintTrait;
use option::OptionTrait;
use result::{ResultTrait, ResultTraitImpl};
use traits::{Into, TryInto};
use integer::{u32_safe_divmod, u32_as_non_zero, u32_wide_mul};

use orion::numbers::fixed_point::implementations::fp8x23::core::{
    HALF, ONE, MAX, FP8x23, FP8x23Add, FP8x23Impl, FP8x23AddEq, FP8x23Sub, FP8x23Mul, FP8x23MulEq,
    FP8x23TryIntoU128, FP8x23PartialEq, FP8x23PartialOrd, FP8x23SubEq, FP8x23Neg, FP8x23Div,
    FP8x23IntoFelt252, FixedTrait
};
use orion::numbers::fixed_point::implementations::fp8x23::math::lut;

// PUBLIC

fn abs(a: FP8x23) -> FP8x23 {
    return FixedTrait::new(a.mag, false);
}

fn add(a: FP8x23, b: FP8x23) -> FP8x23 {
    if a.sign == b.sign {
        return FixedTrait::new(a.mag + b.mag, a.sign);
    }

    if a.mag == b.mag {
        return FixedTrait::ZERO();
    }

    if (a.mag > b.mag) {
        return FixedTrait::new(a.mag - b.mag, a.sign);
    } else {
        return FixedTrait::new(b.mag - a.mag, b.sign);
    }
}

fn ceil(a: FP8x23) -> FP8x23 {
    let (div, rem) = u32_safe_divmod(a.mag, u32_as_non_zero(ONE));

    if rem == 0 {
        return a;
    } else if !a.sign {
        return FixedTrait::new_unscaled(div + 1, false);
    } else if div == 0 {
        return FixedTrait::new_unscaled(0, false);
    } else {
        return FixedTrait::new_unscaled(div, true);
    }
}

fn div(a: FP8x23, b: FP8x23) -> FP8x23 {
    let a_u64 = integer::u32_wide_mul(a.mag, ONE);
    let res_u64 = a_u64 / b.mag.into();

    // Re-apply sign
    return FixedTrait::new(res_u64.try_into().unwrap(), a.sign ^ b.sign);
}

fn eq(a: @FP8x23, b: @FP8x23) -> bool {
    return (*a.mag == *b.mag) && (*a.sign == *b.sign);
}

// Calculates the natural exponent of x: e^x
fn exp(a: FP8x23) -> FP8x23 {
    return exp2(FixedTrait::new(12102203, false) * a); // log2(e) * 2^23 ≈ 12102203
}

// Calculates the binary exponent of x: 2^x
fn exp2(a: FP8x23) -> FP8x23 {
    if (a.mag == 0) {
        return FixedTrait::ONE();
    }

    let (int_part, frac_part) = integer::u32_safe_divmod(a.mag, u32_as_non_zero(ONE));
    let int_res = FixedTrait::new_unscaled(lut::exp2(int_part), false);
    let mut res_u = int_res;

    if frac_part != 0 {
        let frac = FixedTrait::new(frac_part, false);
        let r8 = FixedTrait::new(19, false) * frac;
        let r7 = (r8 + FixedTrait::new(105, false)) * frac;
        let r6 = (r7 + FixedTrait::new(1324, false)) * frac;
        let r5 = (r6 + FixedTrait::new(11159, false)) * frac;
        let r4 = (r5 + FixedTrait::new(80695, false)) * frac;
        let r3 = (r4 + FixedTrait::new(465599, false)) * frac;
        let r2 = (r3 + FixedTrait::new(2015166, false)) * frac;
        let r1 = (r2 + FixedTrait::new(5814540, false)) * frac;
        res_u = res_u * (r1 + FixedTrait::ONE());
    }

    if (a.sign == true) {
        return FixedTrait::ONE() / res_u;
    } else {
        return res_u;
    }
}

fn exp2_int(exp: u32) -> FP8x23 {
    return FixedTrait::new_unscaled(lut::exp2(exp), false);
}

fn floor(a: FP8x23) -> FP8x23 {
    let (div, rem) = integer::u32_safe_divmod(a.mag, u32_as_non_zero(ONE));

    if rem == 0 {
        return a;
    } else if !a.sign {
        return FixedTrait::new_unscaled(div, false);
    } else {
        return FixedTrait::new_unscaled(div + 1, true);
    }
}

fn ge(a: FP8x23, b: FP8x23) -> bool {
    if a.sign != b.sign {
        return !a.sign;
    } else {
        return (a.mag == b.mag) || ((a.mag > b.mag) ^ a.sign);
    }
}

fn gt(a: FP8x23, b: FP8x23) -> bool {
    if a.sign != b.sign {
        return !a.sign;
    } else {
        return (a.mag != b.mag) && ((a.mag > b.mag) ^ a.sign);
    }
}

fn le(a: FP8x23, b: FP8x23) -> bool {
    if a.sign != b.sign {
        return a.sign;
    } else {
        return (a.mag == b.mag) || ((a.mag < b.mag) ^ a.sign);
    }
}

// Calculates the natural logarithm of x: ln(x)
// self must be greater than zero
fn ln(a: FP8x23) -> FP8x23 {
    return FixedTrait::new(5814540, false) * log2(a); // ln(2) = 0.693...
}

// Calculates the binary logarithm of x: log2(x)
// self must be greather than zero
fn log2(a: FP8x23) -> FP8x23 {
    assert(a.sign == false, 'must be positive');

    if (a.mag == ONE) {
        return FixedTrait::ZERO();
    } else if (a.mag < ONE) {
        // Compute true inverse binary log if 0 < x < 1
        let div = FixedTrait::ONE() / a;
        return -log2(div);
    }

    let whole = a.mag / ONE;
    let (msb, div) = lut::msb(whole);

    if a.mag == div * ONE {
        return FixedTrait::new_unscaled(msb, false);
    } else {
        let norm = a / FixedTrait::new_unscaled(div, false);
        let r8 = FixedTrait::new(76243, true) * norm;
        let r7 = (r8 + FixedTrait::new(1038893, false)) * norm;
        let r6 = (r7 + FixedTrait::new(6277679, true)) * norm;
        let r5 = (r6 + FixedTrait::new(22135645, false)) * norm;
        let r4 = (r5 + FixedTrait::new(50444339, true)) * norm;
        let r3 = (r4 + FixedTrait::new(77896489, false)) * norm;
        let r2 = (r3 + FixedTrait::new(83945943, true)) * norm;
        let r1 = (r2 + FixedTrait::new(68407458, false)) * norm;
        return r1 + FixedTrait::new(28734280, true) + FixedTrait::new_unscaled(msb, false);
    }
}

// Calculates the base 10 log of x: log10(x)
// self must be greater than zero
fn log10(a: FP8x23) -> FP8x23 {
    return FixedTrait::new(2525223, false) * log2(a); // log10(2) = 0.301...
}

fn lt(a: FP8x23, b: FP8x23) -> bool {
    if a.sign != b.sign {
        return a.sign;
    } else {
        return (a.mag != b.mag) && ((a.mag < b.mag) ^ a.sign);
    }
}

fn mul(a: FP8x23, b: FP8x23) -> FP8x23 {
    let prod_u128 = integer::u32_wide_mul(a.mag, b.mag);

    // Re-apply sign
    return FixedTrait::new((prod_u128 / ONE.into()).try_into().unwrap(), a.sign ^ b.sign);
}

fn ne(a: @FP8x23, b: @FP8x23) -> bool {
    return (*a.mag != *b.mag) || (*a.sign != *b.sign);
}

fn neg(a: FP8x23) -> FP8x23 {
    if a.mag == 0 {
        return a;
    } else if !a.sign {
        return FixedTrait::new(a.mag, !a.sign);
    } else {
        return FixedTrait::new(a.mag, false);
    }
}

// Calclates the value of x^y and checks for overflow before returning
// self is a FP8x23 point value
// b is a FP8x23 point value
fn pow(a: FP8x23, b: FP8x23) -> FP8x23 {
    let (div, rem) = integer::u32_safe_divmod(b.mag, u32_as_non_zero(ONE));

    // use the more performant integer pow when y is an int
    if (rem == 0) {
        return pow_int(a, b.mag / ONE, b.sign);
    }

    // x^y = exp(y*ln(x)) for x > 0 will error for x < 0
    return exp(b * ln(a));
}

// Calclates the value of a^b and checks for overflow before returning
fn pow_int(a: FP8x23, b: u32, sign: bool) -> FP8x23 {
    let mut x = a;
    let mut n = b;

    if sign == true {
        x = FixedTrait::ONE() / x;
    }

    if n == 0 {
        return FixedTrait::ONE();
    }

    let mut y = FixedTrait::ONE();
    let two = integer::u32_as_non_zero(2);

    loop {
        if n <= 1 {
            break;
        }

        let (div, rem) = integer::u32_safe_divmod(n, two);

        if rem == 1 {
            y = x * y;
        }

        x = x * x;
        n = div;
    };

    return x * y;
}

fn rem(a: FP8x23, b: FP8x23) -> FP8x23 {
    return a - floor(a / b) * b;
}

fn round(a: FP8x23) -> FP8x23 {
    let (div, rem) = integer::u32_safe_divmod(a.mag, u32_as_non_zero(ONE));

    if (HALF <= rem) {
        return FixedTrait::new_unscaled(div + 1, a.sign);
    } else {
        return FixedTrait::new_unscaled(div, a.sign);
    }
}

// Calculates the square root of a FP8x23 point value
// x must be positive
fn sqrt(a: FP8x23) -> FP8x23 {
    assert(a.sign == false, 'must be positive');

    let root = integer::u64_sqrt(a.mag.into() * ONE.into());
    return FixedTrait::new(root.into(), false);
}

fn sub(a: FP8x23, b: FP8x23) -> FP8x23 {
    return add(a, -b);
}

fn sign(a: FP8x23) -> FP8x23 {

    if a.mag == 0 {
        FixedTrait::new(0, false)
    } 
    else {
        FixedTrait::new(ONE, a.sign)
    }
}

// Tests --------------------------------------------------------------------------------------------------------------

use orion::numbers::fixed_point::implementations::fp8x23::helpers::{
    assert_precise, assert_relative
};
use orion::numbers::fixed_point::implementations::fp8x23::math::trig::{PI, HALF_PI};

#[test]
fn test_into() {
    let a = FixedTrait::<FP8x23>::new_unscaled(5, false);
    assert(a.mag == 5 * ONE, 'invalid result');
}

#[test]
fn test_try_into_u128() {
    // Positive unscaled
    let a = FixedTrait::<FP8x23>::new_unscaled(5, false);
    assert(a.try_into().unwrap() == 5_u128, 'invalid result');

    // Positive scaled
    let b = FixedTrait::<FP8x23>::new(5 * ONE, false);
    assert(b.try_into().unwrap() == 5_u128, 'invalid result');

    // Zero
    let d = FixedTrait::<FP8x23>::new_unscaled(0, false);
    assert(d.try_into().unwrap() == 0_u128, 'invalid result');
}

#[test]
#[should_panic]
fn test_negative_try_into_u128() {
    let a = FixedTrait::<FP8x23>::new_unscaled(1, true);
    let a: u128 = a.try_into().unwrap();
}

#[test]
#[available_gas(1000000)]
fn test_acos() {
    let a = FixedTrait::<FP8x23>::ONE();
    assert(a.acos().into() == 0, 'invalid one');
}

#[test]
#[available_gas(1000000)]
fn test_asin() {
    let a = FixedTrait::ONE();
    assert_precise(a.asin(), HALF_PI.into(), 'invalid one', Option::None(())); // PI / 2
}

#[test]
#[available_gas(2000000)]
fn test_atan() {
    let a = FixedTrait::new(2 * ONE, false);
    assert_relative(a.atan(), 9287469, 'invalid two', Option::None(()));
}

#[test]
fn test_ceil() {
    let a = FixedTrait::new(24326963, false); // 2.9
    assert(ceil(a).mag == 3 * ONE, 'invalid pos decimal');
}

#[test]
fn test_floor() {
    let a = FixedTrait::new(24326963, false); // 2.9
    assert(floor(a).mag == 2 * ONE, 'invalid pos decimal');
}

#[test]
fn test_round() {
    let a = FixedTrait::new(24326963, false); // 2.9
    assert(round(a).mag == 3 * ONE, 'invalid pos decimal');
}

#[test]
#[should_panic]
fn test_sqrt_fail() {
    let a = FixedTrait::new_unscaled(25, true);
    sqrt(a);
}

#[test]
fn test_sqrt() {
    let mut a = FixedTrait::new_unscaled(0, false);
    assert(sqrt(a).mag == 0, 'invalid zero root');
    a = FixedTrait::new_unscaled(25, false);
    assert(sqrt(a).mag == 5 * ONE, 'invalid pos root');
}


#[test]
#[available_gas(100000)]
fn test_msb() {
    let a = FixedTrait::<FP8x23>::new_unscaled(100, false);
    let (msb, div) = lut::msb(a.mag / ONE);
    assert(msb == 6, 'invalid msb');
    assert(div == 64, 'invalid msb ceil');
}

#[test]
#[available_gas(600000)]
fn test_pow() {
    let a = FixedTrait::new_unscaled(3, false);
    let b = FixedTrait::new_unscaled(4, false);
    assert(pow(a, b).mag == 81 * ONE, 'invalid pos base power');
}

#[test]
#[available_gas(900000)]
fn test_pow_frac() {
    let a = FixedTrait::new_unscaled(3, false);
    let b = FixedTrait::new(4194304, false); // 0.5
    assert_relative(
        pow(a, b), 14529495, 'invalid pos base power', Option::None(())
    ); // 1.7320508075688772
}

#[test]
#[available_gas(1000000)]
fn test_exp() {
    let a = FixedTrait::new_unscaled(2, false);
    assert_relative(exp(a), 61983895, 'invalid exp of 2', Option::None(())); // 7.389056098793725
}

#[test]
#[available_gas(400000)]
fn test_exp2() {
    let a = FixedTrait::new_unscaled(5, false);
    assert(exp2(a).mag == 268435456, 'invalid exp2 of 2');
}

#[test]
#[available_gas(20000)]
fn test_exp2_int() {
    assert(exp2_int(5).into() == 268435456, 'invalid exp2 of 2');
}

#[test]
#[available_gas(1000000)]
fn test_ln() {
    let mut a = FixedTrait::new_unscaled(1, false);
    assert(ln(a).mag == 0, 'invalid ln of 1');

    a = FixedTrait::new(22802601, false);
    assert_relative(ln(a), ONE.into(), 'invalid ln of 2.7...', Option::None(()));
}

#[test]
#[available_gas(1000000)]
fn test_log2() {
    let mut a = FixedTrait::new_unscaled(32, false);
    assert(log2(a) == FixedTrait::new_unscaled(5, false), 'invalid log2 32');

    a = FixedTrait::new_unscaled(10, false);
    assert_relative(log2(a), 27866353, 'invalid log2 10', Option::None(())); // 3.321928094887362
}

#[test]
#[available_gas(1000000)]
fn test_log10() {
    let a = FixedTrait::new_unscaled(100, false);
    assert_relative(log10(a), 2 * ONE.into(), 'invalid log10', Option::None(()));
}

#[test]
fn test_eq() {
    let a = FixedTrait::new_unscaled(42, false);
    let b = FixedTrait::new_unscaled(42, false);
    let c = eq(@a, @b);
    assert(c == true, 'invalid result');
}

#[test]
fn test_ne() {
    let a = FixedTrait::new_unscaled(42, false);
    let b = FixedTrait::new_unscaled(42, false);
    let c = ne(@a, @b);
    assert(c == false, 'invalid result');
}

#[test]
fn test_add() {
    let a = FixedTrait::new_unscaled(1, false);
    let b = FixedTrait::new_unscaled(2, false);
    assert(add(a, b) == FixedTrait::new_unscaled(3, false), 'invalid result');
}

#[test]
fn test_add_eq() {
    let mut a = FixedTrait::new_unscaled(1, false);
    let b = FixedTrait::new_unscaled(2, false);
    a += b;
    assert(a == FixedTrait::<FP8x23>::new_unscaled(3, false), 'invalid result');
}

#[test]
fn test_sub() {
    let a = FixedTrait::new_unscaled(5, false);
    let b = FixedTrait::new_unscaled(2, false);
    let c = a - b;
    assert(c == FixedTrait::<FP8x23>::new_unscaled(3, false), 'false result invalid');
}

#[test]
fn test_sub_eq() {
    let mut a = FixedTrait::new_unscaled(5, false);
    let b = FixedTrait::new_unscaled(2, false);
    a -= b;
    assert(a == FixedTrait::<FP8x23>::new_unscaled(3, false), 'invalid result');
}

#[test]
#[available_gas(100000)]
fn test_mul_pos() {
    let a = FP8x23 { mag: 24326963, sign: false };
    let b = FP8x23 { mag: 24326963, sign: false };
    let c = a * b;
    assert(c.mag == 70548192, 'invalid result');
}

#[test]
fn test_mul_neg() {
    let a = FixedTrait::new_unscaled(5, false);
    let b = FixedTrait::new_unscaled(2, true);
    let c = a * b;
    assert(c == FixedTrait::<FP8x23>::new_unscaled(10, true), 'invalid result');
}

#[test]
fn test_mul_eq() {
    let mut a = FixedTrait::new_unscaled(5, false);
    let b = FixedTrait::new_unscaled(2, true);
    a *= b;
    assert(a == FixedTrait::<FP8x23>::new_unscaled(10, true), 'invalid result');
}

#[test]
fn test_div() {
    let a = FixedTrait::new_unscaled(10, false);
    let b = FixedTrait::<FP8x23>::new(24326963, false); // 2.9
    let c = a / b;
    assert(c.mag == 28926234, 'invalid pos decimal'); // 3.4482758620689653
}

#[test]
fn test_le() {
    let a = FixedTrait::new_unscaled(1, false);
    let b = FixedTrait::new_unscaled(0, false);
    let c = FixedTrait::<FP8x23>::new_unscaled(1, true);

    assert(a <= a, 'a <= a');
    assert(a <= b == false, 'a <= b');
    assert(a <= c == false, 'a <= c');

    assert(b <= a, 'b <= a');
    assert(b <= b, 'b <= b');
    assert(b <= c == false, 'b <= c');

    assert(c <= a, 'c <= a');
    assert(c <= b, 'c <= b');
    assert(c <= c, 'c <= c');
}

#[test]
fn test_lt() {
    let a = FixedTrait::new_unscaled(1, false);
    let b = FixedTrait::new_unscaled(0, false);
    let c = FixedTrait::<FP8x23>::new_unscaled(1, true);

    assert(a < a == false, 'a < a');
    assert(a < b == false, 'a < b');
    assert(a < c == false, 'a < c');

    assert(b < a, 'b < a');
    assert(b < b == false, 'b < b');
    assert(b < c == false, 'b < c');

    assert(c < a, 'c < a');
    assert(c < b, 'c < b');
    assert(c < c == false, 'c < c');
}

#[test]
fn test_ge() {
    let a = FixedTrait::new_unscaled(1, false);
    let b = FixedTrait::new_unscaled(0, false);
    let c = FixedTrait::<FP8x23>::new_unscaled(1, true);

    assert(a >= a, 'a >= a');
    assert(a >= b, 'a >= b');
    assert(a >= c, 'a >= c');

    assert(b >= a == false, 'b >= a');
    assert(b >= b, 'b >= b');
    assert(b >= c, 'b >= c');

    assert(c >= a == false, 'c >= a');
    assert(c >= b == false, 'c >= b');
    assert(c >= c, 'c >= c');
}

#[test]
fn test_gt() {
    let a = FixedTrait::new_unscaled(1, false);
    let b = FixedTrait::new_unscaled(0, false);
    let c = FixedTrait::<FP8x23>::new_unscaled(1, true);

    assert(a > a == false, 'a > a');
    assert(a > b, 'a > b');
    assert(a > c, 'a > c');

    assert(b > a == false, 'b > a');
    assert(b > b == false, 'b > b');
    assert(b > c, 'b > c');

    assert(c > a == false, 'c > a');
    assert(c > b == false, 'c > b');
    assert(c > c == false, 'c > c');
}

#[test]
#[available_gas(1000000)]
fn test_cos() {
    let a = FixedTrait::<FP8x23>::new(HALF_PI, false);
    assert(a.cos().into() == 0, 'invalid half pi');
}

#[test]
#[available_gas(1000000)]
fn test_sin() {
    let a = FixedTrait::new(HALF_PI, false);
    assert_precise(a.sin(), ONE.into(), 'invalid half pi', Option::None(()));
}

#[test]
#[available_gas(2000000)]
fn test_tan() {
    let a = FixedTrait::<FP8x23>::new(HALF_PI / 2, false);
    assert(a.tan().mag == 8388608, 'invalid quarter pi');
}

#[test]
#[available_gas(2000000)]
fn test_sign() {
    let a = FixedTrait::<FP8x23>::new(0, false);
    assert(a.sign().mag == 0 && !a.sign().sign, 'invalid sign (0, true)');

    let a = FixedTrait::<FP8x23>::new(HALF, true);
    assert(a.sign().mag == ONE && a.sign().sign, 'invalid sign (HALF, true)');

    let a = FixedTrait::<FP8x23>::new(HALF, false);
    assert(a.sign().mag == ONE && !a.sign().sign, 'invalid sign (HALF, false)');


    let a = FixedTrait::<FP8x23>::new(ONE, true);
    assert(a.sign().mag == ONE && a.sign().sign, 'invalid sign (ONE, true)');

    let a = FixedTrait::<FP8x23>::new(ONE, false);
    assert(a.sign().mag == ONE && !a.sign().sign, 'invalid sign (ONE, false)');
}

#[test]
#[should_panic]
#[available_gas(2000000)]
fn test_sign_fail() {
    let a = FixedTrait::<FP8x23>::new(HALF, true);
    assert(a.sign().mag != ONE && !a.sign().sign, 'invalid sign (HALF, true)');
}