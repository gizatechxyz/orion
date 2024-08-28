use core::num::traits::{WideMul, Sqrt};
use super::{F64, FixedTrait, F64Impl, ONE, HALF, NaN, lut};

// PUBLIC

pub(crate) fn abs(a: F64) -> F64 {
    if a.d == NaN {
        return F64 { d: NaN };
    }

    if a.d <= 0 {
        F64 { d: a.d * ONE }
    } else {
        a
    }
}

pub(crate) fn add(a: F64, b: F64) -> F64 {
    if a.d == NaN || b.d == NaN {
        return F64 { d: NaN };
    }
    F64 { d: a.d + b.d }
}

pub(crate) fn ceil(a: F64) -> F64 {
    if a.d == NaN {
        return F64 { d: NaN };
    }
    let div = Div::div(a.d, ONE);
    let rem = Rem::rem(a.d, ONE);

    if rem == 0 {
        FixedTrait::new_unscaled(div)
    } else {
        add(FixedTrait::new_unscaled(div), F64 { d: ONE })
    }
}

pub(crate) fn div(a: F64, b: F64) -> F64 {
    if a.d == NaN || b.d == NaN {
        return F64 { d: NaN };
    }
    let a_i128 = WideMul::wide_mul(a.d, ONE);
    let res_i128 = a_i128 / b.d.into();

    F64 { d: res_i128.try_into().unwrap() }
}

pub(crate) fn eq(a: @F64, b: @F64) -> bool {
    return (*a.d == *b.d);
}

// // Calculates the natural exponent of x: e^x
pub(crate) fn exp(a: F64) -> F64 {
    if a.d == NaN {
        return F64 { d: NaN };
    }
    return exp2(FixedTrait::new(6196328018) * a);
}


// Calculates the binary exponent of x: 2^x
pub(crate) fn exp2(a: F64) -> F64 {
    if a.d == NaN {
        return F64 { d: NaN };
    }
    if (a.d == 0) {
        return FixedTrait::ONE();
    }

    let a_abs = a.abs();

    let int_part = Div::div(a_abs.d, ONE);
    let frac_part = Rem::rem(a_abs.d, ONE);

    let int_res = FixedTrait::new_unscaled(lut::exp2(int_part));
    let mut res_u = int_res;

    if frac_part != 0 {
        let frac = FixedTrait::new(frac_part);
        let r8 = FixedTrait::new(9707) * frac;
        let r7 = (r8 + FixedTrait::new(53974)) * frac;
        let r6 = (r7 + FixedTrait::new(677974)) * frac;
        let r5 = (r6 + FixedTrait::new(5713580)) * frac;
        let r4 = (r5 + FixedTrait::new(41315679)) * frac;
        let r3 = (r4 + FixedTrait::new(238386709)) * frac;
        let r2 = (r3 + FixedTrait::new(1031765214)) * frac;
        let r1 = (r2 + FixedTrait::new(2977044459)) * frac;
        res_u = res_u * (r1 + FixedTrait::ONE());
    }

    if (a.d < 0) {
        return FixedTrait::ONE() / res_u;
    } else {
        return res_u;
    }
}

pub(crate) fn exp2_int(exp: i64) -> F64 {
    return FixedTrait::new_unscaled(lut::exp2(exp));
}

pub(crate) fn floor(a: F64) -> F64 {
    if a.d == NaN {
        return F64 { d: NaN };
    }

    let div = Div::div(a.d, ONE);
    let rem = Rem::rem(a.d, ONE);

    if rem == 0 {
        a
    } else if a.d >= 0 {
        FixedTrait::new_unscaled(div)
    } else {
        FixedTrait::new_unscaled(div - 1)
    }
}


pub(crate) fn ge(a: F64, b: F64) -> bool {
    a.d >= b.d
}

pub(crate) fn gt(a: F64, b: F64) -> bool {
    a.d > b.d
}

pub(crate) fn le(a: F64, b: F64) -> bool {
    a.d <= b.d
}

// Calculates the natural logarithm of x: ln(x)
// self must be greater than zero
pub(crate) fn ln(a: F64) -> F64 {
    if a.d == NaN {
        return F64 { d: NaN };
    }

    return FixedTrait::new(2977044472) * log2(a); // ln(2) = 0.693...
}

// Calculates the binary logarithm of x: log2(x)
// self must be greather than zero
pub(crate) fn log2(a: F64) -> F64 {
    if a.d == NaN {
        return F64 { d: NaN };
    }

    if a.d < 0 {
        return F64 { d: NaN };
    }

    if (a.d == ONE) {
        return FixedTrait::ZERO();
    } else if (a.d < ONE) {
        // Compute true inverse binary log if 0 < x < 1
        let div = FixedTrait::ONE() / a;
        return -log2(div);
    }

    let whole = a.d / ONE;
    let (msb, div) = lut::msb(whole);

    if a.d == div * ONE {
        return FixedTrait::new_unscaled(msb);
    } else {
        let norm = a / FixedTrait::new_unscaled(div);
        let r8 = FixedTrait::new(-39036580) * norm;
        let r7 = (r8 + FixedTrait::new(531913440)) * norm;
        let r6 = (r7 + FixedTrait::new(-3214171660)) * norm;
        let r5 = (r6 + FixedTrait::new(11333450393)) * norm;
        let r4 = (r5 + FixedTrait::new(-25827501665)) * norm;
        let r3 = (r4 + FixedTrait::new(39883002199)) * norm;
        let r2 = (r3 + FixedTrait::new(-42980322874)) * norm;
        let r1 = (r2 + FixedTrait::new(35024618493)) * norm;
        return r1 + FixedTrait::new(-14711951564) + FixedTrait::new_unscaled(msb);
    }
}

// Calculates the base 10 log of x: log10(x)
// self must be greater than zero
pub(crate) fn log10(a: F64) -> F64 {
    if a.d == NaN {
        return F64 { d: NaN };
    }

    return FixedTrait::new(1292913986) * log2(a); // log10(2) = 0.301...
}

pub(crate) fn lt(a: F64, b: F64) -> bool {
    a.d < b.d
}

pub(crate) fn mul(a: F64, b: F64) -> F64 {
    if a.d == NaN {
        return F64 { d: NaN };
    }

    let prod_i128 = WideMul::wide_mul(a.d, b.d);

    // Re-apply sign
    return FixedTrait::new((prod_i128 / ONE.into()).try_into().unwrap());
}

pub(crate) fn ne(a: @F64, b: @F64) -> bool {
    *a.d != *b.d
}

pub(crate) fn neg(a: F64) -> F64 {
    if a.d == NaN {
        return F64 { d: NaN };
    }

    a * F64 { d: -ONE }
}

// Calclates the value of x^y and checks for overflow before returning
// self is a Fixed point value
// b is a Fixed point value
pub(crate) fn pow(a: F64, b: F64) -> F64 {
    if a.d == NaN {
        return F64 { d: NaN };
    }

    let rem = Rem::rem(b.d, ONE);

    // use the more performant integer pow when y is an int
    if (rem == 0) {
        return pow_int(a, b.d / ONE, b.d < 0);
    }

    // x^y = exp(y*ln(x)) for x > 0 will error for x < 0
    return exp(b * ln(a));
}

// Calclates the value of a^b and checks for overflow before returning
pub(crate) fn pow_int(a: F64, b: i64, sign: bool) -> F64 {
    if a.d == NaN {
        return F64 { d: NaN };
    }

    let mut x = a;
    let mut n = b;

    if sign == true {
        x = FixedTrait::ONE() / x;
    }

    if n == 0 {
        return FixedTrait::ONE();
    }

    let mut y = FixedTrait::ONE();
    let two = 2;

    loop {
        if n <= 1 {
            break;
        }

        let div = Div::div(n, two);
        let rem = Rem::rem(n, two);

        if rem == 1 {
            y = x * y;
        }

        x = x * x;
        n = div;
    };

    return x * y;
}

pub(crate) fn rem(a: F64, b: F64) -> F64 {
    if a.d == NaN || b.d == NaN || b.d == 0 {
        return F64 { d: NaN };
    }

    return F64 { d: a.d - (a.d / b.d) * b.d };
}

pub(crate) fn round(a: F64) -> F64 {
    if a.d == NaN {
        return F64 { d: NaN };
    }

    let div = Div::div(a.d, ONE);
    let rem = Rem::rem(a.d, ONE);

    if (HALF <= rem) {
        return FixedTrait::new_unscaled(div + 1);
    } else {
        return FixedTrait::new_unscaled(div);
    }
}

// Calculates the square root of a FP16x16 point value
// x must be positive
pub(crate) fn sqrt(a: F64) -> F64 {
    if a.d == NaN {
        return F64 { d: NaN };
    }

    if a.d < 0 {
        return F64 { d: NaN };
    }

    let a: u128 = a.d.try_into().unwrap();
    let one: u128 = ONE.try_into().unwrap();

    let root: u64 = Sqrt::sqrt(a * one);

    FixedTrait::new(root.try_into().unwrap())
}

pub(crate) fn sub(a: F64, b: F64) -> F64 {
    if a.d == NaN {
        return F64 { d: NaN };
    }

    return add(a, -b);
}


// Tests
// --------------------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use orion_numbers::f64::helpers::{assert_precise, assert_relative};

    use super::{
        FixedTrait, ONE, NaN, round, floor, sqrt, ceil, lut, exp, exp2, exp2_int, pow, log10, log2,
        ln, eq, ne, add, F64, F64Impl
    };

    #[test]
    fn test_into() {
        let a: F64 = FixedTrait::new_unscaled(5);
        assert(a.d == 5 * ONE, 'invalid result');
    }


    #[test]
    fn test_ceil() {
        let a = FixedTrait::new(12455405158); // 2.9
        assert(ceil(a).d == 3 * ONE, 'invalid pos decimal');
    }

    #[test]
    fn test_floor() {
        let a = FixedTrait::new(12455405158); // 2.9
        assert(floor(a).d == 2 * ONE, 'invalid pos decimal');
    }

    #[test]
    fn test_round() {
        let a = FixedTrait::new(12455405158); // 2.9
        assert(round(a).d == 3 * ONE, 'invalid pos decimal');
    }

    #[test]
    fn test_sqrt_nan() {
        let a = FixedTrait::new_unscaled(-25);
        assert(sqrt(a).d == NaN, 'should return NaN');
    }

    #[test]
    fn test_sqrt() {
        let mut a = FixedTrait::new_unscaled(0);
        assert(sqrt(a).d == 0, 'invalid zero root');
        a = FixedTrait::new_unscaled(25);
        assert(sqrt(a).d == 5 * ONE, 'invalid pos root');
    }

    #[test]
    #[available_gas(100000)]
    fn test_msb() {
        let a: F64 = FixedTrait::new_unscaled(1000000);
        let (msb, div) = lut::msb(a.d / ONE);
        assert(msb == 19, 'invalid msb');
        assert(div == 524288, 'invalid msb ceil');
    }

    #[test]
    #[available_gas(600000)] // 430k
    fn test_pow() {
        let a = FixedTrait::new_unscaled(3);
        let b = FixedTrait::new_unscaled(4);
        assert(pow(a, b).d == 81 * ONE, 'invalid pos base power');
    }

    #[test]
    #[available_gas(900000)] // 350k
    fn test_pow_frac() {
        let a = FixedTrait::new_unscaled(3);
        let b = FixedTrait::new(2147483648); // 0.5
        assert_relative(
            pow(a, b), 7439101574, 'invalid pos base power', Option::None(())
        ); // 1.7320508075688772
    }

    #[test]
    #[available_gas(1000000)] // 167k
    fn test_exp() {
        let a = FixedTrait::new_unscaled(2);
        assert_relative(
            exp(a), 31735754293, 'invalid exp of 2', Option::None(())
        ); // 7.389056098793725
    }

    #[test]
    #[available_gas(400000)]
    fn test_exp2() {
        let a = FixedTrait::new_unscaled(24);
        assert(exp2(a).d == 72057594037927936, 'invalid exp2 of 2');
    }

    #[test]
    #[available_gas(20000)]
    fn test_exp2_int() {
        assert(exp2_int(24).into() == 72057594037927936, 'invalid exp2 of 2');
    }

    #[test]
    #[available_gas(1000000)]
    fn test_ln() {
        let mut a = FixedTrait::new_unscaled(1);
        assert(ln(a).d == 0, 'invalid ln of 1');

        a = FixedTrait::new(11674931554);
        assert_relative(ln(a), ONE.into(), 'invalid ln of 2.7...', Option::None(()));
    }

    #[test]
    #[available_gas(1000000)]
    fn test_log2() {
        let mut a = FixedTrait::new_unscaled(32);
        assert(log2(a) == FixedTrait::new_unscaled(5), 'invalid log2 32');

        a = FixedTrait::new_unscaled(10);
        assert_relative(
            log2(a), 14267572527, 'invalid log2 10', Option::None(())
        ); // 3.321928094887362
    }

    #[test]
    #[available_gas(1000000)]
    fn test_log10() {
        let a = FixedTrait::new_unscaled(100);
        assert_relative(log10(a), 2 * ONE.into(), 'invalid log10', Option::None(()));
    }

    #[test]
    fn test_eq() {
        let a = FixedTrait::new_unscaled(42);
        let b = FixedTrait::new_unscaled(42);
        let c = eq(@a, @b);
        assert(c == true, 'invalid result');
    }

    #[test]
    fn test_ne() {
        let a = FixedTrait::new_unscaled(42);
        let b = FixedTrait::new_unscaled(42);
        let c = ne(@a, @b);
        assert(c == false, 'invalid result');
    }


    #[test]
    fn test_add() {
        let a = FixedTrait::new_unscaled(1);
        let b = FixedTrait::new_unscaled(2);
        assert(add(a, b) == FixedTrait::new_unscaled(3), 'invalid result');
    }

    #[test]
    fn test_add_eq() {
        let mut a: F64 = FixedTrait::new_unscaled(1);
        let b = FixedTrait::new_unscaled(2);
        a += b;
        assert(a == FixedTrait::new_unscaled(3), 'invalid result');
    }

    #[test]
    fn test_sub() {
        let a: F64 = FixedTrait::new_unscaled(5);
        let b = FixedTrait::new_unscaled(2);
        let c = a - b;
        assert(c == FixedTrait::new_unscaled(3), 'false result invalid');
    }

    #[test]
    fn test_sub_eq() {
        let mut a: F64 = FixedTrait::new_unscaled(5);
        let b = FixedTrait::new_unscaled(2);
        a -= b;
        assert(a == FixedTrait::new_unscaled(3), 'invalid result');
    }

    #[test]
    #[available_gas(100000)] // 13k
    fn test_mul_pos() {
        let a = F64 { d: 12455405158 };
        let b = F64 { d: 12455405158 };
        let c = a * b;
        assert(c.d == 36120674957, 'invalid result');
    }

    #[test]
    fn test_mul_neg() {
        let a: F64 = FixedTrait::new_unscaled(5);
        let b = FixedTrait::new_unscaled(-2);
        let c = a * b;
        assert(c == FixedTrait::new_unscaled(-10), 'invalid result');
    }

    #[test]
    fn test_mul_eq() {
        let mut a: F64 = FixedTrait::new_unscaled(5);
        let b = FixedTrait::new_unscaled(-2);
        a *= b;
        assert(a == FixedTrait::new_unscaled(-10), 'invalid result');
    }

    #[test]
    fn test_div() {
        let a: F64 = FixedTrait::new_unscaled(10);
        let b = FixedTrait::new(12455405158); // 2.9
        let c = a / b;
        assert(c.d == 14810232055, 'invalid pos decimal'); // 3.4482758620689653
    }

    #[test]
    fn test_le() {
        let a: F64 = FixedTrait::new_unscaled(1);
        let b = FixedTrait::new_unscaled(0);
        let c = FixedTrait::new_unscaled(-1);

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
        let a: F64 = FixedTrait::new_unscaled(1);
        let b = FixedTrait::new_unscaled(0);
        let c = FixedTrait::new_unscaled(-1);

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
        let a: F64 = FixedTrait::new_unscaled(1);
        let b = FixedTrait::new_unscaled(0);
        let c = FixedTrait::new_unscaled(-1);

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
        let a: F64 = FixedTrait::new_unscaled(1);
        let b = FixedTrait::new_unscaled(0);
        let c = FixedTrait::new_unscaled(-1);

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
}
