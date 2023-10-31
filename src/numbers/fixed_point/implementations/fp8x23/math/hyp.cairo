use core::debug::PrintTrait;
use orion::numbers::fixed_point::implementations::fp8x23::core::{
    HALF, ONE, TWO, FP8x23, FP8x23Impl, FP8x23Add, FP8x23AddEq, FP8x23Sub, FP8x23Mul, FP8x23MulEq,
    FP8x23TryIntoU128, FP8x23PartialEq, FP8x23PartialOrd, FP8x23SubEq, FP8x23Neg, FP8x23Div,
    FP8x23IntoFelt252, FixedTrait
};

// Calculates hyperbolic cosine of a (fixed point)
fn cosh(a: FP8x23) -> FP8x23 {
    let ea = a.exp();
    return (ea + (FixedTrait::ONE() / ea)) / FixedTrait::new(TWO, false);
}

// Calculates hyperbolic sine of a (fixed point)
fn sinh(a: FP8x23) -> FP8x23 {
    let ea = a.exp();
    return (ea - (FixedTrait::ONE() / ea)) / FixedTrait::new(TWO, false);
}

// Calculates hyperbolic tangent of a (fixed point)
fn tanh(a: FP8x23) -> FP8x23 {
    let ea = a.exp();
    let ea_i = FixedTrait::ONE() / ea;
    return (ea - ea_i) / (ea + ea_i);
}

// Calculates inverse hyperbolic cosine of a (fixed point)
fn acosh(a: FP8x23) -> FP8x23 {
    let root = (a * a - FixedTrait::ONE()).sqrt();
    return (a + root).ln();
}

// Calculates inverse hyperbolic sine of a (fixed point)
fn asinh(a: FP8x23) -> FP8x23 {
    let root = (a * a + FixedTrait::ONE()).sqrt();
    return (a + root).ln();
}

// Calculates inverse hyperbolic tangent of a (fixed point)
fn atanh(a: FP8x23) -> FP8x23 {
    let one = FixedTrait::ONE();
    let ln_arg = (one + a) / (one - a);
    return ln_arg.ln() / FixedTrait::new(TWO, false);
}

// Tests --------------------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use option::OptionTrait;
    use traits::Into;

    use orion::numbers::fixed_point::implementations::fp8x23::helpers::assert_precise;

    use super::{FixedTrait, TWO, cosh, ONE, sinh, tanh, acosh, asinh, atanh, HALF};

    #[test]
    #[available_gas(10000000)]
    fn test_cosh() {
        let a = FixedTrait::new(TWO, false);
        assert_precise(cosh(a), 31559585, 'invalid two', Option::None(())); // 3.762195691016423

        let a = FixedTrait::ONE();
        assert_precise(cosh(a), 12944299, 'invalid one', Option::None(())); // 1.5430806347841253

        let a = FixedTrait::ZERO();
        assert_precise(cosh(a), ONE.into(), 'invalid zero', Option::None(()));

        let a = FixedTrait::ONE();
        assert_precise(
            cosh(a), 12944299, 'invalid neg one', Option::None(())
        ); // 1.5430806347841253

        let a = FixedTrait::new(TWO, true);
        assert_precise(cosh(a), 31559602, 'invalid neg two', Option::None(())); // 3.762195691016423
    }

    #[test]
    #[available_gas(10000000)]
    fn test_sinh() {
        let a = FixedTrait::new(TWO, false);
        assert_precise(sinh(a), 30424310, 'invalid two', Option::None(())); // 3.6268604077773023

        let a = FixedTrait::ONE();
        assert_precise(sinh(a), 9858302, 'invalid one', Option::None(())); // 1.1752011936029418

        let a = FixedTrait::ZERO();
        assert(sinh(a).into() == 0, 'invalid zero');

        let a = FixedTrait::new(ONE, true);
        assert_precise(
            sinh(a), -9858302, 'invalid neg one', Option::None(())
        ); // -1.1752011936029418

        let a = FixedTrait::new(TWO, true);
        assert_precise(
            sinh(a), -30424328, 'invalid neg two', Option::None(())
        ); // -3.6268604077773023
    }

    #[test]
    #[available_gas(10000000)]
    fn test_tanh() {
        let a = FixedTrait::new(TWO, false);
        assert_precise(tanh(a), 8086849, 'invalid two', Option::None(())); // 0.9640275800745076

        let a = FixedTrait::ONE();
        assert_precise(tanh(a), 6388715, 'invalid one', Option::None(())); // 0.7615941559446443

        let a = FixedTrait::ZERO();
        assert(tanh(a).into() == 0, 'invalid zero');

        let a = FixedTrait::new(ONE, true);
        assert_precise(
            tanh(a), -6388715, 'invalid neg one', Option::None(())
        ); // -0.7615941559446443

        let a = FixedTrait::new(TWO, true);
        assert_precise(
            tanh(a), -8086849, 'invalid neg two', Option::None(())
        ); // 0.9640275800745076
    }

    #[test]
    #[available_gas(10000000)]
    fn test_acosh() {
        let a = FixedTrait::new(31559585, false); // 3.762195691016423
        assert_precise(acosh(a), 16777257, 'invalid two', Option::None(()));

        let a = FixedTrait::new(12944299, false); // 1.5430806347841253
        assert_precise(acosh(a), ONE.into(), 'invalid one', Option::None(()));

        let a = FixedTrait::ONE(); // 1
        assert(acosh(a).into() == 0, 'invalid zero');
    }

    #[test]
    #[available_gas(10000000)]
    fn test_asinh() {
        let a = FixedTrait::new(30424310, false); // 3.6268604077773023
        assert_precise(asinh(a), 16777257, 'invalid two', Option::None(()));

        let a = FixedTrait::new(9858302, false); // 1.1752011936029418
        assert_precise(asinh(a), ONE.into(), 'invalid one', Option::None(()));

        let a = FixedTrait::ZERO();
        assert(asinh(a).into() == 0, 'invalid zero');

        let a = FixedTrait::new(9858302, true); // -1.1752011936029418
        assert_precise(asinh(a), -ONE.into(), 'invalid neg one', Option::None(()));

        let a = FixedTrait::new(30424310, true); // -3.6268604077773023
        assert_precise(asinh(a), -16777238, 'invalid neg two', Option::None(()));
    }

    #[test]
    #[available_gas(10000000)]
    fn test_atanh() {
        let a = FixedTrait::new(7549747, false); // 0.9
        assert_precise(atanh(a), 12349872, 'invalid 0.9', Option::None(())); // 1.4722194895832204

        let a = FixedTrait::new(HALF, false); // 0.5
        assert_precise(atanh(a), 4607914, 'invalid half', Option::None(())); // 0.5493061443340548

        let a = FixedTrait::ZERO();
        assert(atanh(a).into() == 0, 'invalid zero');

        let a = FixedTrait::new(HALF, true); // 0.5
        assert_precise(
            atanh(a), -4607914, 'invalid neg half', Option::None(())
        ); // 0.5493061443340548

        let a = FixedTrait::new(7549747, true); // 0.9
        assert_precise(atanh(a), -12349872, 'invalid -0.9', Option::None(())); // 1.4722194895832204
    }
}
