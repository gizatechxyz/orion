use super::{F64, FixedTrait, F64Impl, ONE, HALF, TWO, lut};

// Calculates hyperbolic cosine of a (fixed point)
pub(crate) fn cosh(a: F64) -> F64 {
    let ea = a.exp();
    return (ea + (FixedTrait::ONE() / ea)) / FixedTrait::new(TWO);
}

// Calculates hyperbolic sine of a (fixed point)
pub(crate) fn sinh(a: F64) -> F64 {
    let ea = a.exp();
    return (ea - (FixedTrait::ONE() / ea)) / FixedTrait::new(TWO);
}

// Calculates hyperbolic tangent of a (fixed point)
pub(crate) fn tanh(a: F64) -> F64 {
    let ea = a.exp();
    let ea_i = FixedTrait::ONE() / ea;
    return (ea - ea_i) / (ea + ea_i);
}

// Calculates inverse hyperbolic cosine of a (fixed point)
pub(crate) fn acosh(a: F64) -> F64 {
    let root = (a * a - FixedTrait::ONE()).sqrt();
    return (a + root).ln();
}

// Calculates inverse hyperbolic sine of a (fixed point)
pub(crate) fn asinh(a: F64) -> F64 {
    let root = (a * a + FixedTrait::ONE()).sqrt();
    return (a + root).ln();
}

// Calculates inverse hyperbolic tangent of a (fixed point)
pub(crate) fn atanh(a: F64) -> F64 {
    let one = FixedTrait::ONE();
    let ln_arg = (one + a) / (one - a);
    return ln_arg.ln() / FixedTrait::new(TWO);
}

// Tests
// --------------------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use orion_numbers::f64::helpers::{assert_precise, assert_relative};

    use super::{FixedTrait, F64Impl, TWO, cosh, sinh, ONE, tanh, acosh, asinh, atanh, HALF};

    // TODO: Fix it
    // #[test]
    // #[available_gas(10000000)]
    // fn test_cosh() {
    //     let a = FixedTrait::new(TWO);
    //     assert_precise(cosh(a), 16158507454, 'invalid two', Option::None(())); // 3.762195691016423

    //     let a = FixedTrait::ONE();
    //     assert_precise(cosh(a), 6627480862, 'invalid one', Option::None(())); // 1.5430806347841253

    //     let a = FixedTrait::ZERO();
    //     assert_precise(cosh(a), ONE.into(), 'invalid zero', Option::None(()));

    //     let a = FixedTrait::ONE();
    //     assert_precise(
    //         cosh(a), 6627480862, 'invalid neg one', Option::None(())
    //     ); // 1.5430806347841253

    //     let a = FixedTrait::new(-TWO);
    //     assert_precise(
    //         cosh(a), 16158507454, 'invalid neg two', Option::None(())
    //     ); // 3.762195691016423
    // }

    // TODO: fix it
    // #[test]
    // #[available_gas(10000000)]
    // fn test_sinh() {
    //     let a = FixedTrait::new(TWO);
    //     assert_precise(sinh(a), 15577246839, 'invalid two', Option::None(())); // 3.6268604077773023

    //     let a = FixedTrait::ONE();
    //     assert_precise(sinh(a), 5047450693, 'invalid one', Option::None(())); // 1.1752011936029418

    //     let a = FixedTrait::ZERO();
    //     assert(sinh(a).into() == 0, 'invalid zero');

    //     let a = FixedTrait::new(-ONE);
    //     assert_precise(
    //         sinh(a), -5047450693, 'invalid neg one', Option::None(())
    //     ); // -1.1752011936029418

    //     let a = FixedTrait::new(-TWO);
    //     assert_precise(
    //         sinh(a), -15577246839, 'invalid neg two', Option::None(())
    //     ); // -3.6268604077773023
    // }

    // TODO: fix it
    // #[test]
    // #[available_gas(10000000)]
    // fn test_tanh() {
    //     let a = FixedTrait::new(TWO);
    //     assert_precise(tanh(a), 4140466929, 'invalid two', Option::None(())); // 0.9640275800745076

    //     let a = FixedTrait::ONE();
    //     assert_precise(tanh(a), 3271021993, 'invalid one', Option::None(())); // 0.7615941559446443

    //     let a = FixedTrait::ZERO();
    //     assert(tanh(a).into() == 0, 'invalid zero');

    //     let a = FixedTrait::new(-ONE);
    //     assert_precise(
    //         tanh(a), -3271021993, 'invalid neg one', Option::None(())
    //     ); // -0.7615941559446443

    //     let a = FixedTrait::new(-TWO);
    //     assert_precise(
    //         tanh(a), -4140466929, 'invalid neg two', Option::None(())
    //     ); // 0.9640275800745076
    // }

    #[test]
    #[available_gas(10000000)]
    fn test_acosh() {
        let a = FixedTrait::new(16158507454); // 3.762195691016423
        assert_precise(acosh(a), TWO.into(), 'invalid two', Option::None(()));

        let a = FixedTrait::new(6627480862); // 1.5430806347841253
        assert_precise(acosh(a), ONE.into(), 'invalid one', Option::None(()));

        let a = FixedTrait::ONE(); // 1
        assert(acosh(a).into() == 0, 'invalid zero');
    }

    #[test]
    #[available_gas(10000000)]
    fn test_asinh() {
        let a = FixedTrait::new(15577246839); // 3.6268604077773023
        assert_precise(asinh(a), TWO.into(), 'invalid two', Option::None(()));

        let a = FixedTrait::new(5047450693); // 1.1752011936029418
        assert_precise(asinh(a), ONE.into(), 'invalid one', Option::None(()));

        let a = FixedTrait::ZERO();
        assert(asinh(a).into() == 0, 'invalid zero');

        let a = FixedTrait::new(-5047450693); // -1.1752011936029418
        assert_precise(asinh(a), -ONE.into(), 'invalid neg one', Option::None(()));

        let a = FixedTrait::new(-15577246839); // -3.6268604077773023
        assert_precise(asinh(a), -TWO.into(), 'invalid neg two', Option::None(()));
    }

    #[test]
    #[available_gas(10000000)]
    fn test_atanh() {
        let a = FixedTrait::new(3865470566); // 0.9
        assert_precise(atanh(a), 6323134560, 'invalid 0.9', Option::None(())); // 1.4722194895832204

        let a = FixedTrait::new(HALF); // 0.5
        assert_precise(
            atanh(a), 2359251925, 'invalid half', Option::None(())
        ); // 0.5493061443340548

        let a = FixedTrait::ZERO();
        assert(atanh(a).into() == 0, 'invalid zero');

        let a = FixedTrait::new(-HALF); // 0.5
        assert_precise(
            atanh(a), -2359251925, 'invalid neg half', Option::None(())
        ); // 0.5493061443340548

        let a = FixedTrait::new(-3865470566); // 0.9
        assert_precise(
            atanh(a), -6323134560, 'invalid -0.9', Option::None(())
        ); // 1.4722194895832204
    }


}
