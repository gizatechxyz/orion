use option::OptionTrait;
use traits::Into;

use onnx_cairo::numbers::fixed_point::types::{ONE, _felt_abs, _felt_sign, Fixed, };

use onnx_cairo::numbers::fixed_point::core;

#[test]
fn test_into() {
    let a = Fixed::from_unscaled_felt(5);
    assert(a.into() == 5 * ONE, 'invalid result');
}

#[test]
#[should_panic]
fn test_overflow_large() {
    let too_large = 0x100000000000000000000000000000000;
    Fixed::from_felt(too_large);
}

#[test]
#[should_panic]
fn test_overflow_small() {
    let too_small = -0x100000000000000000000000000000000;
    Fixed::from_felt(too_small);
}

#[test]
fn test_sign() {
    let min = -1809251394333065606848661391547535052811553607665798349986546028067936010240;
    let max = 1809251394333065606848661391547535052811553607665798349986546028067936010240;
    assert(_felt_sign(min) == true, 'invalid result');
    assert(_felt_sign(-1) == true, 'invalid result');
    assert(_felt_sign(0) == false, 'invalid result');
    assert(_felt_sign(1) == false, 'invalid result');
    assert(_felt_sign(max) == false, 'invalid result');
}

#[test]
fn test_abs() {
    assert(_felt_abs(5) == 5, 'abs of pos should be pos');
    assert(_felt_abs(-5) == 5, 'abs of neg should be pos');
    assert(_felt_abs(0) == 0, 'abs of 0 should be 0');
}

#[test]
fn test_ceil() {
    let a = Fixed::from_felt(194615506); // 2.9
    assert(a.ceil().into() == 3 * ONE, 'invalid pos decimal');
}

#[test]
fn test_floor() {
    let a = Fixed::from_felt(194615506); // 2.9
    assert(a.floor().into() == 2 * ONE, 'invalid pos decimal');
}

#[test]
fn test_round() {
    let a = Fixed::from_felt(194615506); // 2.9
    assert(a.round().into() == 3 * ONE, 'invalid pos decimal');
}

#[test]
#[should_panic]
fn test_sqrt_fail() {
    let a = Fixed::from_unscaled_felt(-25);
    a.sqrt();
}

#[test]
fn test_sqrt() {
    let a = Fixed::from_unscaled_felt(0);
    assert(a.sqrt().into() == 0, 'invalid zero root');
}

#[test]
#[available_gas(10000000)]
fn test_pow() {
    let a = Fixed::from_unscaled_felt(3);
    let b = Fixed::from_unscaled_felt(4);
    assert(a.pow(b).into() == 81 * ONE, 'invalid pos base power');
}

#[test]
#[available_gas(10000000)]
fn test_exp() {
    let a = Fixed::from_unscaled_felt(2);
    assert(a.exp().into() == 495871144, 'invalid exp of 2'); // 7.389056317241236
}

#[test]
#[available_gas(10000000)]
fn test_exp2() {
    let a = Fixed::from_unscaled_felt(2);
    assert(a.exp2().into() == 268435456, 'invalid exp2 of 2'); // 3.99999957248 = 4
}

#[test]
#[available_gas(10000000)]
fn test_ln() {
    let a = Fixed::from_unscaled_felt(1);
    assert(a.ln().into() == 0, 'invalid ln of 1');
}

#[test]
#[available_gas(10000000)]
fn test_log2() {
    let a = Fixed::from_unscaled_felt(32);
    assert(a.log2().into() == 335544129, 'invalid log2'); // 4.99999995767848
}

#[test]
#[available_gas(10000000)]
fn test_log10() {
    let a = Fixed::from_unscaled_felt(100);
    assert(a.log10().into() == 134217717, 'invalid log10'); // 1.9999999873985543
}

#[test]
fn test_eq() {
    let a = Fixed::from_unscaled_felt(42);
    let b = Fixed::from_unscaled_felt(42);
    let c = a == b;
    assert(c == true, 'invalid result');
}

#[test]
fn test_ne_() {
    let a = Fixed::from_unscaled_felt(42);
    let b = Fixed::from_unscaled_felt(42);
    let c = a != b;
    assert(c == false, 'invalid result');

    let a = Fixed::from_unscaled_felt(42);
    let b = Fixed::from_unscaled_felt(-42);
    let c = a != b;
    assert(c == true, 'invalid result');
}

#[test]
#[available_gas(2000000)]
fn test_add() {
    let a = Fixed::from_unscaled_felt(1);
    let b = Fixed::from_unscaled_felt(2);
    assert(a + b == Fixed::from_unscaled_felt(3), 'invalid result');
}

#[test]
#[available_gas(2000000)]
fn test_add_eq() {
    let mut a = Fixed::from_unscaled_felt(1);
    let b = Fixed::from_unscaled_felt(2);
    a += b;
    assert(a.into() == 3 * ONE, 'invalid result');
}

#[test]
#[available_gas(2000000)]
fn test_sub() {
    let a = Fixed::from_unscaled_felt(5);
    let b = Fixed::from_unscaled_felt(2);
    let c = a - b;
    assert(c.into() == 3 * ONE, 'false result invalid');
}

#[test]
#[available_gas(2000000)]
fn test_sub_eq() {
    let mut a = Fixed::from_unscaled_felt(5);
    let b = Fixed::from_unscaled_felt(2);
    a -= b;
    assert(a.into() == 3 * ONE, 'invalid result');
}

#[test]
#[available_gas(2000000)]
fn test_mul_pos() {
    let a = Fixed::from_unscaled_felt(5);
    let b = Fixed::from_unscaled_felt(2);
    let c = a * b;
    assert(c.into() == 10 * ONE, 'invalid result');

    let a = Fixed::from_unscaled_felt(9);
    let b = Fixed::from_unscaled_felt(9);
    let c = a * b;
    assert(c.into() == 81 * ONE, 'invalid result');

    let a = Fixed::from_felt(83886080); // 1.25
    let b = Fixed::from_felt(154350387); // 2.3
    let c = a * b;
    assert(c.into() == 192937983, 'invalid result'); // 2.875

    let a = Fixed::from_unscaled_felt(0);
    let b = Fixed::from_felt(154350387); // 2.3
    let c = a * b;
    assert(c.into() == 0, 'invalid result');
}

#[test]
#[available_gas(2000000)]
fn test_mul_neg() {
    let a = Fixed::from_unscaled_felt(5);
    let b = Fixed::from_unscaled_felt(-2);
    let c = a * b;
    assert(c.into() == -10 * ONE, 'true result invalid');
}

#[test]
#[available_gas(2000000)]
fn test_mul_eq() {
    let mut a = Fixed::from_unscaled_felt(5);
    let b = Fixed::from_unscaled_felt(-2);
    a *= b;
    assert(a.into() == -10 * ONE, 'invalid result');
}

#[test]
#[available_gas(2000000)]
fn test_div_() {
    let a = Fixed::from_unscaled_felt(10);
    let b = Fixed::from_felt(194615706); // 2.9
    let c = a / b;
    assert(c.into() == 231409875, 'invalid pos decimal'); // 3.4482758620689653

    let a = Fixed::from_unscaled_felt(10);
    let b = Fixed::from_unscaled_felt(5);
    let c = a / b;
    assert(c.into() == 134217728, 'invalid pos integer'); // 2

    let a = Fixed::from_unscaled_felt(-2);
    let b = Fixed::from_unscaled_felt(5);
    let c = a / b;
    assert(c.into() == -26843545, 'invalid neg decimal'); // 0.4

    let a = Fixed::from_unscaled_felt(-1000);
    let b = Fixed::from_unscaled_felt(12500);
    let c = a / b;
    assert(c.into() == -5368709, 'invalid neg decimal'); // 0.08

    let a = Fixed::from_unscaled_felt(-10);
    let b = Fixed::from_unscaled_felt(123456789);
    let c = a / b;
    assert(c.into() == -5, 'invalid neg decimal'); // 8.100000073706917e-8

    let a = Fixed::from_unscaled_felt(123456789);
    let b = Fixed::from_unscaled_felt(-10);
    let c = a / b;
    assert(c.into() == -828504486287769, 'invalid neg decimal'); // -12345678.9
}

#[test]
fn test_le() {
    let a = Fixed::from_unscaled_felt(1);
    let b = Fixed::from_unscaled_felt(0);
    let c = Fixed::from_unscaled_felt(-1);

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
    let a = Fixed::from_unscaled_felt(1);
    let b = Fixed::from_unscaled_felt(0);
    let c = Fixed::from_unscaled_felt(-1);

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
    let a = Fixed::from_unscaled_felt(1);
    let b = Fixed::from_unscaled_felt(0);
    let c = Fixed::from_unscaled_felt(-1);

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
    let a = Fixed::from_unscaled_felt(1);
    let b = Fixed::from_unscaled_felt(0);
    let c = Fixed::from_unscaled_felt(-1);

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
