use core::integer;
use orion_numbers::f16x16::{core::{F16x16Impl, f16x16, ONE, HALF}, lut};

use orion_numbers::core_trait::{I32Rem, I32Div, I64Div}; //I32TryIntoNonZero, I32DivRem


pub fn abs(a: f16x16) -> f16x16 {
    if a >= 0 {
        a
    } else {
        a * -1_i32
    }
}

pub fn add(a: f16x16, b: f16x16) -> f16x16 {
    a + b
}

pub fn sub(a: f16x16, b: f16x16) -> f16x16 {
    a - b
}

pub fn ceil(a: f16x16) -> f16x16 {
    //let (div, rem) = DivRem::div_rem(a, ONE.try_into().unwrap());
    let div = Div::div(a, ONE);
    let rem = Rem::rem(a, ONE);

    if rem == 0 {
        F16x16Impl::new_unscaled(div)
    } else {
        F16x16Impl::new_unscaled(div) + ONE
    }
}

pub fn div(a: f16x16, b: f16x16) -> f16x16 {
    let a_i64 = integer::i32_wide_mul(a, ONE);
    let res_i64 = a_i64 / b.into();

    // Re-apply sign
    F16x16Impl::new(res_i64.try_into().unwrap())
}

// Calculates the natural exponent of x: e^x
pub fn exp(a: f16x16) -> f16x16 {
    exp2(F16x16Impl::mul(F16x16Impl::new(94548), a)) // log2(e) * 2^23 ≈ 12102203
}

// Calculates the binary exponent of x: 2^x
pub fn exp2(a: f16x16) -> f16x16 {
    if (a == 0) {
        return F16x16Impl::ONE();
    }

    //let (int_part, frac_part) = DivRem::div_rem(a.abs(), ONE.try_into().unwrap());
    let int_part = Div::div(a.abs(), ONE);
    let frac_part = Rem::rem(a.abs(), ONE);

    let int_res = F16x16Impl::new_unscaled(lut::exp2(int_part));
    let mut res_u = int_res;

    if frac_part != 0 {
        let frac = F16x16Impl::new(frac_part);
        let r7 = F16x16Impl::mul(F16x16Impl::new(1), frac);
        let r6 = F16x16Impl::mul((r7 + F16x16Impl::new(10)), frac);
        let r5 = F16x16Impl::mul((r6 + F16x16Impl::new(87)), frac);
        let r4 = F16x16Impl::mul((r5 + F16x16Impl::new(630)), frac);
        let r3 = F16x16Impl::mul((r4 + F16x16Impl::new(3638)), frac);
        let r2 = F16x16Impl::mul((r3 + F16x16Impl::new(15743)), frac);
        let r1 = F16x16Impl::mul((r2 + F16x16Impl::new(45426)), frac);
        res_u = F16x16Impl::mul(res_u, (r1 + F16x16Impl::ONE()));
    }

    if a < 0 {
        F16x16Impl::div(F16x16Impl::ONE(), res_u)
    } else {
        res_u
    }
}

fn exp2_int(exp: i32) -> f16x16 {
    F16x16Impl::new_unscaled(lut::exp2(exp))
}

pub fn floor(a: f16x16) -> f16x16 {
    //let (div, rem) = DivRem::div_rem(a, ONE.try_into().unwrap());
    let div = Div::div(a, ONE);
    let rem = Rem::rem(a, ONE);

    if rem == 0 {
        a
    } else if a >= 0 {
        F16x16Impl::new_unscaled(div)
    } else {
        F16x16Impl::new_unscaled(div - 1)
    }
}

// Calculates the natural logarithm of x: ln(x)
// self must be greater than zero
pub fn ln(a: f16x16) -> f16x16 {
    F16x16Impl::mul(F16x16Impl::new(45426), log2(a)) // ln(2) = 0.693...
}

// Calculates the binary logarithm of x: log2(x)
// self must be greather than zero
pub fn log2(a: f16x16) -> f16x16 {
    assert(a >= 0, 'must be positive');

    if (a == ONE) {
        return F16x16Impl::ZERO();
    } else if (a < ONE) {
        // Compute true inverse binary log if 0 < x < 1
        let div = F16x16Impl::div(F16x16Impl::ONE(), a);
        return -log2(div);
    }

    let whole = a / ONE;
    let (msb, div) = lut::msb(whole);

    if a == div * ONE {
        F16x16Impl::new_unscaled(msb)
    } else {
        let norm = F16x16Impl::div(a, F16x16Impl::new_unscaled(div));
        let r8 = F16x16Impl::mul(F16x16Impl::new(-596), norm);
        let r7 = F16x16Impl::mul((r8 + F16x16Impl::new(8116)), norm);
        let r6 = F16x16Impl::mul((r7 + F16x16Impl::new(-49044)), norm);
        let r5 = F16x16Impl::mul((r6 + F16x16Impl::new(172935)), norm);
        let r4 = F16x16Impl::mul((r5 + F16x16Impl::new(-394096)), norm);
        let r3 = F16x16Impl::mul((r4 + F16x16Impl::new(608566)), norm);
        let r2 = F16x16Impl::mul((r3 + F16x16Impl::new(-655828)), norm);
        let r1 = F16x16Impl::mul((r2 + F16x16Impl::new(534433)), norm);

        r1 + F16x16Impl::new(-224487) + F16x16Impl::new_unscaled(msb)
    }
}

// Calculates the base 10 log of x: log10(x)
// self must be greater than zero
pub fn log10(a: f16x16) -> f16x16 {
    F16x16Impl::mul(F16x16Impl::new(19728), log2(a)) // log10(2) = 0.301...
}

pub fn mul(a: f16x16, b: f16x16) -> f16x16 {
    let prod_i64 = integer::i32_wide_mul(a, b);

    // Re-apply sign
    F16x16Impl::new((prod_i64 / ONE.into()).try_into().unwrap())
}

// Calclates the value of x^y and checks for overflow before returning
// self is a FP16x16 point value
// b is a FP16x16 point value
pub fn pow(a: f16x16, b: f16x16) -> f16x16 {
    let rem = Rem::rem(b, ONE);

    // use the more performant integer pow when y is an int
    if (rem == 0) {
        return pow_int(a, b / ONE);
    }

    // x^y = exp(y*ln(x)) for x > 0 will error for x < 0
    exp(F16x16Impl::mul(b, ln(a)))
}

// Calclates the value of a^b and checks for overflow before returning
fn pow_int(a: f16x16, b: i32) -> f16x16 {
    let mut x = a;
    let mut n = b.abs();

    if b < 0 {
        x = F16x16Impl::div(ONE, x);
    }

    if n == 0 {
        return ONE;
    }

    let mut y = ONE;
    let two: i32 = 2;

    while n > 1 {
        //let (div, rem) = DivRem::div_rem(n, two.try_into().unwrap());
        let div = Div::div(n, two);
        let rem = Rem::rem(n, two);

        if rem == 1 {
            y = F16x16Impl::mul(x, y);
        }

        x = F16x16Impl::mul(x, x);
        n = div;
    };

    F16x16Impl::mul(x, y)
}

pub fn round(a: f16x16) -> f16x16 {
    //let (div, rem) = DivRem::div_rem(a, ONE.try_into().unwrap());
    let div = Div::div(a, ONE);
    let rem = Rem::rem(a, ONE);

    if (HALF <= rem) {
        F16x16Impl::new_unscaled(div + 1)
    } else {
        F16x16Impl::new_unscaled(div)
    }
}

pub fn sign(a: f16x16) -> f16x16 {
    if a == 0 {
        F16x16Impl::new(0)
    } else if a > 0 {
        ONE
    } else {
        -ONE
    }
}

// Calculates the square root of a FP16x16 point value
// x must be positive
pub fn sqrt(a: f16x16) -> f16x16 {
    assert(a >= 0, 'must be positive');
    //let a: usize = a.try_into().unwrap();

    let root = integer::u64_sqrt(a.try_into().unwrap() * ONE.try_into().unwrap());

    F16x16Impl::new(root.try_into().unwrap())
}


// Tests
//
// 
// --------------------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use orion_numbers::f16x16::helpers::{assert_precise, assert_relative};

    use super::{
        F16x16Impl, ONE, HALF, f16x16, integer, lut, ceil, add, sqrt, floor, exp, exp2, exp2_int,
        ln, log2, log10, pow, round, sign
    };

    use orion_numbers::core_trait::{I32Rem, I32Div};

    #[test]
    fn test_into() {
        let a = F16x16Impl::new_unscaled(5);
        assert(a == 5 * ONE, 'invalid result');
    }

    #[test]
    fn test_ceil() {
        let a = F16x16Impl::new(190054); // 2.9
        assert(ceil(a) == 3 * ONE, 'invalid pos decimal');
    }

    #[test]
    #[available_gas(10000000)]
    fn test_exp() {
        let a = F16x16Impl::new_unscaled(2);
        assert_relative(exp(a), 484249, 'invalid exp of 2', Option::None(())); // 7.389056098793725
    }

    #[test]
    #[available_gas(400000)]
    fn test_exp2() {
        let a = F16x16Impl::new_unscaled(5);
        assert(exp2(a) == 2097152, 'invalid exp2 of 2');
    }

    #[test]
    #[available_gas(20000)]
    fn test_exp2_int() {
        assert(exp2_int(5).into() == 2097152, 'invalid exp2 of 2');
    }

    #[test]
    fn test_floor() {
        let a = F16x16Impl::new(190054); // 2.9
        assert(floor(a) == 2 * ONE, 'invalid pos decimal');
    }

    #[test]
    #[available_gas(1000000)]
    fn test_ln() {
        let mut a = F16x16Impl::new_unscaled(1);
        assert(ln(a) == 0, 'invalid ln of 1');

        a = F16x16Impl::new(178145);
        assert_relative(ln(a), ONE.into(), 'invalid ln of 2.7...', Option::None(()));
    }

    #[test]
    #[available_gas(1000000)]
    fn test_log2() {
        let mut a = F16x16Impl::new_unscaled(32);
        assert(log2(a) == F16x16Impl::new_unscaled(5), 'invalid log2 32');

        a = F16x16Impl::new_unscaled(10);
        assert_relative(log2(a), 217706, 'invalid log2 10', Option::None(())); // 3.321928094887362
    }

    #[test]
    #[available_gas(1000000)]
    fn test_log10() {
        let a = F16x16Impl::new_unscaled(100);
        assert_relative(log10(a), 2 * ONE.into(), 'invalid log10', Option::None(()));
    }


    #[test]
    #[available_gas(600000)]
    fn test_pow() {
        let a = F16x16Impl::new_unscaled(3);
        let b = F16x16Impl::new_unscaled(4);
        assert(pow(a, b) == 81 * ONE, 'invalid pos base power');
    }

    #[test]
    #[available_gas(900000)]
    fn test_pow_frac() {
        let a = F16x16Impl::new_unscaled(3);
        let b = F16x16Impl::new(32768); // 0.5
        assert_relative(
            pow(a, b), 113512, 'invalid pos base power', Option::None(())
        ); // 1.7320508075688772
    }

    #[test]
    fn test_round() {
        let a = F16x16Impl::new(190054); // 2.9
        assert(round(a) == 3 * ONE, 'invalid pos decimal');
    }


    #[test]
    fn test_sqrt() {
        let mut a = F16x16Impl::new_unscaled(0);

        assert(sqrt(a) == 0, 'invalid zero root');
        a = F16x16Impl::new_unscaled(25);
        assert(sqrt(a) == 5 * ONE, 'invalid pos root');
    }

    #[test]
    #[should_panic]
    fn test_sqrt_fail() {
        let a = F16x16Impl::new_unscaled(-25);
        sqrt(a);
    }

    #[test]
    #[available_gas(2000000)]
    fn test_sign() {
        let a = F16x16Impl::new(0);
        assert(a.sign() == 0, 'invalid sign (0)');

        let a = F16x16Impl::new(-HALF);
        assert(a.sign() == -ONE, 'invalid sign (-HALF)');

        let a = F16x16Impl::new(HALF);
        assert(a.sign() == ONE, 'invalid sign (HALF)');

        let a = F16x16Impl::new(-ONE);
        assert(a.sign() == -ONE, 'invalid sign (-ONE)');

        let a = F16x16Impl::new(ONE);
        assert(a.sign() == ONE, 'invalid sign (ONE)');
    }

    #[test]
    #[available_gas(100000)]
    fn test_msb() {
        let a = F16x16Impl::new_unscaled(100);
        let (msb, div) = lut::msb(a / ONE);
        assert(msb == 6, 'invalid msb');
        assert(div == 64, 'invalid msb ceil');
    }

    #[test]
    fn test_eq() {
        let a = F16x16Impl::new_unscaled(42);
        let b = F16x16Impl::new_unscaled(42);
        let c = a == b;
        assert(c, 'invalid result');
    }

    #[test]
    fn test_ne() {
        let a = F16x16Impl::new_unscaled(42);
        let b = F16x16Impl::new_unscaled(42);
        let c = a != b;
        assert(!c, 'invalid result');
    }

    #[test]
    fn test_add() {
        let a = F16x16Impl::new_unscaled(1);
        let b = F16x16Impl::new_unscaled(2);
        assert(add(a, b) == F16x16Impl::new_unscaled(3), 'invalid result');
    }

    #[test]
    fn test_add_eq() {
        let mut a = F16x16Impl::new_unscaled(1);
        let b = F16x16Impl::new_unscaled(2);
        a += b;
        assert(a == F16x16Impl::new_unscaled(3), 'invalid result');
    }

    #[test]
    fn test_sub() {
        let a = F16x16Impl::new_unscaled(5);
        let b = F16x16Impl::new_unscaled(2);
        let c = a - b;
        assert(c == F16x16Impl::new_unscaled(3), 'false result invalid');
    }

    #[test]
    fn test_sub_eq() {
        let mut a = F16x16Impl::new_unscaled(5);
        let b = F16x16Impl::new_unscaled(2);
        a -= b;
        assert(a == F16x16Impl::new_unscaled(3), 'invalid result');
    }

    #[test]
    #[available_gas(100000)]
    fn test_mul_pos() {
        let a = 190054;
        let b = 190054;
        let c = F16x16Impl::mul(a, b);
        assert(c == 551155, 'invalid result');
    }

    #[test]
    fn test_mul_neg() {
        let a = F16x16Impl::new_unscaled(5);
        let b = F16x16Impl::new_unscaled(-2);
        let c = F16x16Impl::mul(a, b);
        assert(c == F16x16Impl::new_unscaled(-10), 'invalid result');
    }


    #[test]
    fn test_div() {
        let a = F16x16Impl::new_unscaled(10);
        let b = F16x16Impl::new(190054); // 2.9
        let c = F16x16Impl::div(a, b);
        assert(c == 225986, 'invalid pos decimal'); // 3.4482758620689653
    }

    #[test]
    fn test_le() {
        let a = F16x16Impl::new_unscaled(1);
        let b = F16x16Impl::new_unscaled(0);
        let c = F16x16Impl::new_unscaled(-1);

        assert(a <= a, 'a <= a');
        assert(!(a <= b), 'a <= b');
        assert(!(a <= c), 'a <= c');

        assert(b <= a, 'b <= a');
        assert(b <= b, 'b <= b');
        assert(!(b <= c), 'b <= c');

        assert(c <= a, 'c <= a');
        assert(c <= b, 'c <= b');
        assert(c <= c, 'c <= c');
    }

    #[test]
    fn test_lt() {
        let a = F16x16Impl::new_unscaled(1);
        let b = F16x16Impl::new_unscaled(0);
        let c = F16x16Impl::new_unscaled(-1);

        assert(!(a < a), 'a < a');
        assert(!(a < b), 'a < b');
        assert(!(a < c), 'a < c');

        assert(b < a, 'b < a');
        assert(!(b < b), 'b < b');
        assert(!(b < c), 'b < c');

        assert(c < a, 'c < a');
        assert(c < b, 'c < b');
        assert(!(c < c), 'c < c');
    }

    #[test]
    fn test_ge() {
        let a = F16x16Impl::new_unscaled(1);
        let b = F16x16Impl::new_unscaled(0);
        let c = F16x16Impl::new_unscaled(-1);

        assert(a >= a, 'a >= a');
        assert(a >= b, 'a >= b');
        assert(a >= c, 'a >= c');

        assert(!(b >= a), 'b >= a');
        assert(b >= b, 'b >= b');
        assert(b >= c, 'b >= c');

        assert(!(c >= a), 'c >= a');
        assert(!(c >= b), 'c >= b');
        assert(c >= c, 'c >= c');
    }

    #[test]
    fn test_gt() {
        let a = F16x16Impl::new_unscaled(1);
        let b = F16x16Impl::new_unscaled(0);
        let c = F16x16Impl::new_unscaled(-1);

        assert(!(a > a), 'a > a');
        assert(a > b, 'a > b');
        assert(a > c, 'a > c');

        assert(!(b > a), 'b > a');
        assert(!(b > b), 'b > b');
        assert(b > c, 'b > c');

        assert(!(c > a), 'c > a');
        assert(!(c > b), 'c > b');
        assert(!(c > c), 'c > c');
    }
}
