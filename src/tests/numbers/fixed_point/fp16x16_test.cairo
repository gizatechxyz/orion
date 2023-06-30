use option::OptionTrait;
use traits::Into;

use orion::numbers::fixed_point::implementations::impl_16x16::{
    ONE, _felt_abs, _felt_sign, FP16x16Impl, FP16x16Into, FP16x16Add, FP16x16AddEq, FP16x16Sub,
    FP16x16SubEq, FP16x16Mul, FP16x16MulEq, FP16x16Div, FP16x16DivEq, FP16x16PartialOrd,
    FP16x16PartialEq
};
use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
use orion::numbers::fixed_point::core;

#[test]
fn test_into() {
    let a = FixedTrait::from_unscaled_felt(5);
    assert(a.into() == 5 * ONE.into(), 'invalid result');
}

#[test]
#[should_panic]
fn test_overflow_large() {
    let too_large = 0x100000000000000000000000000000000;
    FixedTrait::from_felt(too_large);
}

#[test]
#[should_panic]
fn test_overflow_small() {
    let too_small = -0x100000000000000000000000000000000;
    FixedTrait::from_felt(too_small);
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
    let a = FixedTrait::from_felt(190054); // 2.9
    assert(a.ceil().into() == 3 * ONE.into(), 'invalid pos decimal');
}

#[test]
fn test_floor() {
    let a = FixedTrait::from_felt(190054); // 2.9
    assert(a.floor().into() == 2 * ONE.into(), 'invalid pos decimal');
}

#[test]
fn test_round() {
    let a = FixedTrait::from_felt(190054); // 2.9
    assert(a.round().into() == 3 * ONE.into(), 'invalid pos decimal');
}

#[test]
#[should_panic]
fn test_sqrt_fail() {
    let a = FixedTrait::from_unscaled_felt(-25);
    a.sqrt();
}

#[test]
fn test_sqrt() {
    let a = FixedTrait::from_unscaled_felt(0);
    assert(a.sqrt().into() == 0, 'invalid zero root');
}

#[test]
#[available_gas(10000000)]
fn test_pow() {
    let a = FixedTrait::from_unscaled_felt(3);
    let b = FixedTrait::from_unscaled_felt(4);
    assert(a.pow(b).into() == 81 * ONE.into(), 'invalid pos base power');
}

#[test]
#[available_gas(10000000)]
fn test_exp() {
    let a = FixedTrait::from_unscaled_felt(2);
    assert(a.exp().into() == 484232, 'invalid exp of 2');
}

#[test]
#[available_gas(10000000)]
fn test_exp2() {
    let a = FixedTrait::from_unscaled_felt(2);
    assert(a.exp2().into() == 262144, 'invalid exp2 of 2');
}

#[test]
#[available_gas(10000000)]
fn test_ln() {
    let a = FixedTrait::from_unscaled_felt(1);
    assert(a.ln().into() == 0, 'invalid ln of 1');
}

#[test]
#[available_gas(10000000)]
fn test_log2() {
    let a = FixedTrait::from_unscaled_felt(31);
    assert(a.log2().into() == 324653, 'invalid log2');
}

#[test]
#[available_gas(10000000)]
fn test_log10() {
    let a = FixedTrait::from_unscaled_felt(30);
    assert(a.log10().into() == 96802, 'invalid log10');
}

#[test]
fn test_eq() {
    let a = FixedTrait::from_unscaled_felt(25);
    let b = FixedTrait::from_unscaled_felt(25);
    let c = a == b;
    assert(c == true, 'invalid result');
}

#[test]
fn test_ne_() {
    let a = FixedTrait::from_unscaled_felt(25);
    let b = FixedTrait::from_unscaled_felt(25);
    let c = a != b;
    assert(c == false, 'invalid result');

    let a = FixedTrait::from_unscaled_felt(25);
    let b = FixedTrait::from_unscaled_felt(-25);
    let c = a != b;
    assert(c == true, 'invalid result');
}

#[test]
#[available_gas(2000000)]
fn test_add() {
    let a = FixedTrait::from_unscaled_felt(1);
    let b = FixedTrait::from_unscaled_felt(2);
    assert(a + b == FixedTrait::from_unscaled_felt(3), 'invalid result');
}

#[test]
#[available_gas(2000000)]
fn test_add_eq() {
    let mut a = FixedTrait::from_unscaled_felt(1);
    let b = FixedTrait::from_unscaled_felt(2);
    a += b;
    assert(a.into() == 3 * ONE.into(), 'invalid result');
}

#[test]
#[available_gas(2000000)]
fn test_sub() {
    let a = FixedTrait::from_unscaled_felt(5);
    let b = FixedTrait::from_unscaled_felt(2);
    let c = a - b;
    assert(c.into() == 3 * ONE.into(), 'false result invalid');
}

#[test]
#[available_gas(2000000)]
fn test_sub_eq() {
    let mut a = FixedTrait::from_unscaled_felt(5);
    let b = FixedTrait::from_unscaled_felt(2);
    a -= b;
    assert(a.into() == 3 * ONE.into(), 'invalid result');
}

#[test]
#[available_gas(2000000)]
fn test_mul_pos() {
    let a = FixedTrait::from_unscaled_felt(5);
    let b = FixedTrait::from_unscaled_felt(2);
    let c = a * b;
    assert(c.into() == 10 * ONE.into(), 'invalid result');

    let a = FixedTrait::from_unscaled_felt(9);
    let b = FixedTrait::from_unscaled_felt(9);
    let c = a * b;
    assert(c.into() == 81 * ONE.into(), 'invalid result');

    let a = FixedTrait::from_felt(81920); // 1.25
    let b = FixedTrait::from_felt(150733); // 2.3
    let c = a * b;
    assert(c.into() == 188416, 'invalid result'); // 2.875

    let a = FixedTrait::from_unscaled_felt(0);
    let b = FixedTrait::from_felt(150733); // 2.3
    let c = a * b;
    assert(c.into() == 0, 'invalid result');
}

#[test]
#[available_gas(2000000)]
fn test_mul_neg() {
    let a = FixedTrait::from_unscaled_felt(5);
    let b = FixedTrait::from_unscaled_felt(-2);
    let c = a * b;
    assert(c.into() == -10 * ONE.into(), 'true result invalid');
}

#[test]
#[available_gas(2000000)]
fn test_mul_eq() {
    let mut a = FixedTrait::from_unscaled_felt(5);
    let b = FixedTrait::from_unscaled_felt(-2);
    a *= b;
    assert(a.into() == -10 * ONE.into(), 'invalid result');
}

#[test]
#[available_gas(2000000)]
fn test_div_() {
    let a = FixedTrait::from_unscaled_felt(10);
    let b = FixedTrait::from_felt(190054); // 2.9
    let c = a / b;
    assert(c.into() == 225986, 'invalid pos decimal'); // 3.4482758620689653

    let a = FixedTrait::from_unscaled_felt(10);
    let b = FixedTrait::from_unscaled_felt(5);
    let c = a / b;
    assert(c.into() == 131072, 'invalid pos integer'); // 2

    let a = FixedTrait::from_unscaled_felt(-2);
    let b = FixedTrait::from_unscaled_felt(5);
    let c = a / b;
    assert(c.into() == -26214, 'invalid neg decimal'); // -0.4

    let a = FixedTrait::from_unscaled_felt(-1);
    let b = FixedTrait::from_unscaled_felt(12);
    let c = a / b;
    assert(c.into() == -5461, 'invalid neg decimal'); // -0.08333333333333333

    let a = FixedTrait::from_unscaled_felt(12);
    let b = FixedTrait::from_unscaled_felt(-10);
    let c = a / b;
    assert(c.into() == -78643, 'invalid neg decimal'); // -1.2
}

#[test]
fn test_le() {
    let a = FixedTrait::from_unscaled_felt(1);
    let b = FixedTrait::from_unscaled_felt(0);
    let c = FixedTrait::from_unscaled_felt(-1);

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
    let a = FixedTrait::from_unscaled_felt(1);
    let b = FixedTrait::from_unscaled_felt(0);
    let c = FixedTrait::from_unscaled_felt(-1);

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
    let a = FixedTrait::from_unscaled_felt(1);
    let b = FixedTrait::from_unscaled_felt(0);
    let c = FixedTrait::from_unscaled_felt(-1);

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
    let a = FixedTrait::from_unscaled_felt(1);
    let b = FixedTrait::from_unscaled_felt(0);
    let c = FixedTrait::from_unscaled_felt(-1);

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
#[available_gas(10000000)]
fn test_sinh() {
    let a = FixedTrait::from_unscaled_felt(1);
    assert(a.sinh().into() == 77016, 'invalid sinh of 1');
}

#[test]
#[available_gas(10000000)]
fn test_tanh() {
    let a = FixedTrait::from_unscaled_felt(1);
    assert(a.tanh().into() == 49911, 'invalid tanh of 1');
}

#[test]
#[available_gas(10000000)]
fn test_cosh() {
    let a = FixedTrait::from_unscaled_felt(1);
    assert(a.cosh().into() == 101125, 'invalid cosh of 1');
}

#[test]
#[available_gas(10000000)]
fn test_acosh() {
    let a = FixedTrait::from_unscaled_felt(1);
    assert(a.acosh().into() == 0, 'invalid cosh of 1');
}

#[test]
#[available_gas(10000000)]
fn test_acosh_2() {
    let a = FixedTrait::from_unscaled_felt(2);
    assert(a.acosh().into() == 86255, 'invalid cosh of 2');
}

#[test]
#[available_gas(10000000)]
#[should_panic]
fn test_acosh_zero() {
    //should panic with a value less than 1
    let a = FixedTrait::from_unscaled_felt(0);
    a.acosh();
}