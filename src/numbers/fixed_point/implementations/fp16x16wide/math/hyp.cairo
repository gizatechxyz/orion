use core::debug::PrintTrait;
use orion::numbers::fixed_point::implementations::fp16x16wide::core::{
    HALF, ONE, TWO, FP16x16W, FP16x16WImpl, FP16x16WAdd, FP16x16WAddEq, FP16x16WSub, FP16x16WMul,
    FP16x16WMulEq, FP16x16WTryIntoU128, FP16x16WPartialEq, FP16x16WPartialOrd, FP16x16WSubEq, FP16x16WNeg,
    FP16x16WDiv, FP16x16WIntoFelt252, FixedTrait
};

// Calculates hyperbolic cosine of a (fixed point)
fn cosh(a: FP16x16W) -> FP16x16W {
    let ea = a.exp();
    return (ea + (FixedTrait::ONE() / ea)) / FixedTrait::new(TWO, false);
}

// Calculates hyperbolic sine of a (fixed point)
fn sinh(a: FP16x16W) -> FP16x16W {
    let ea = a.exp();
    return (ea - (FixedTrait::ONE() / ea)) / FixedTrait::new(TWO, false);
}

// Calculates hyperbolic tangent of a (fixed point)
fn tanh(a: FP16x16W) -> FP16x16W {
    let ea = a.exp();
    let ea_i = FixedTrait::ONE() / ea;
    return (ea - ea_i) / (ea + ea_i);
}

// Calculates inverse hyperbolic cosine of a (fixed point)
fn acosh(a: FP16x16W) -> FP16x16W {
    let root = (a * a - FixedTrait::ONE()).sqrt();
    return (a + root).ln();
}

// Calculates inverse hyperbolic sine of a (fixed point)
fn asinh(a: FP16x16W) -> FP16x16W {
    let root = (a * a + FixedTrait::ONE()).sqrt();
    return (a + root).ln();
}

// Calculates inverse hyperbolic tangent of a (fixed point)
fn atanh(a: FP16x16W) -> FP16x16W {
    let one = FixedTrait::ONE();
    let ln_arg = (one + a) / (one - a);
    return ln_arg.ln() / FixedTrait::new(TWO, false);
}

// Tests --------------------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {


use option::OptionTrait;
use traits::Into;

use orion::numbers::fixed_point::implementations::fp16x16wide::helpers::assert_precise;

    use super::{FixedTrait, TWO, cosh, ONE, sinh, tanh, acosh, asinh, atanh, HALF};


#[test]
#[available_gas(10000000)]
fn test_cosh() {
    let a = FixedTrait::new(TWO, false);
    assert_precise(cosh(a), 246550, 'invalid two', Option::None(())); // 3.5954653836066

    let a = FixedTrait::ONE();
    assert_precise(cosh(a), 101127, 'invalid one', Option::None(())); // 1.42428174592510

    let a = FixedTrait::ZERO();
    assert_precise(cosh(a), ONE.into(), 'invalid zero', Option::None(()));

    let a = FixedTrait::ONE();
    assert_precise(cosh(a), 101127, 'invalid neg one', Option::None(())); // 1.42428174592510

    let a = FixedTrait::new(TWO, true);
    assert_precise(cosh(a), 246568, 'invalid neg two', Option::None(())); // 3.5954653836066
}

#[test]
#[available_gas(10000000)]
fn test_sinh() {
    let a = FixedTrait::new(TWO, false);
    assert_precise(sinh(a), 237681, 'invalid two', Option::None(())); // 3.48973469357602

    let a = FixedTrait::ONE();
    assert_precise(sinh(a), 77018, 'invalid one', Option::None(())); // 1.13687593250230

    let a = FixedTrait::ZERO();
    assert(sinh(a).into() == 0, 'invalid zero');

    let a = FixedTrait::new(ONE, true);
    assert_precise(sinh(a), -77018, 'invalid neg one', Option::None(())); // -1.13687593250230

    let a = FixedTrait::new(TWO, true);
    assert_precise(sinh(a), -237699, 'invalid neg two', Option::None(())); // -3.48973469357602
}

#[test]
#[available_gas(10000000)]
fn test_tanh() {
    let a = FixedTrait::new(TWO, false);
    assert_precise(tanh(a), 63179, 'invalid two', Option::None(())); // 0.75314654693321

    let a = FixedTrait::ONE();
    assert_precise(tanh(a), 49912, 'invalid one', Option::None(())); // 0.59499543433175

    let a = FixedTrait::ZERO();
    assert(tanh(a).into() == 0, 'invalid zero');

    let a = FixedTrait::new(ONE, true);
    assert_precise(tanh(a), -49912, 'invalid neg one', Option::None(())); // -0.59499543433175

    let a = FixedTrait::new(TWO, true);
    assert_precise(tanh(a), -63179, 'invalid neg two', Option::None(())); // 0.75314654693321
}

#[test]
#[available_gas(10000000)]
fn test_acosh() {
    let a = FixedTrait::new(246559, false); // 3.5954653836066
    assert_precise(acosh(a), 131072, 'invalid two', Option::None(()));

    let a = FixedTrait::new(101127, false); // 1.42428174592510
    assert_precise(acosh(a), ONE.into(), 'invalid one', Option::None(()));

    let a = FixedTrait::ONE(); // 1
    assert(acosh(a).into() == 0, 'invalid zero');
}

#[test]
#[available_gas(10000000)]
fn test_asinh() {
    let a = FixedTrait::new(237690, false); // 3.48973469357602
    assert_precise(asinh(a), 131072, 'invalid two', Option::None(()));

    let a = FixedTrait::new(77018, false); // 1.13687593250230
    assert_precise(asinh(a), ONE.into(), 'invalid one', Option::None(()));

    let a = FixedTrait::ZERO();
    assert(asinh(a).into() == 0, 'invalid zero');

    let a = FixedTrait::new(77018, true); // -1.13687593250230
    assert_precise(asinh(a), -ONE.into(), 'invalid neg one', Option::None(()));

    let a = FixedTrait::new(237690, true); // -3.48973469357602
    assert_precise(asinh(a), -131017, 'invalid neg two', Option::None(()));
}

#[test]
#[available_gas(10000000)]
fn test_atanh() {
    let a = FixedTrait::new(58982, false); // 0.9
    assert_precise(atanh(a), 96483, 'invalid 0.9', Option::None(())); // 1.36892147623689

    let a = FixedTrait::new(HALF, false); // 0.5
    assert_precise(atanh(a), 35999, 'invalid half', Option::None(())); // 0.42914542526098

    let a = FixedTrait::ZERO();
    assert(atanh(a).into() == 0, 'invalid zero');

    let a = FixedTrait::new(HALF, true); // 0.5
    assert_precise(atanh(a), -35999, 'invalid neg half', Option::None(())); // 0.42914542526098

    let a = FixedTrait::new(58982, true); // 0.9
    assert_precise(atanh(a), -96483, 'invalid -0.9', Option::None(())); // 1.36892147623689
}

}