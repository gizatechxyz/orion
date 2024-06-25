use core::option::OptionTrait;
use core::traits::TryInto;
use core::integer;
use orion_numbers::f16x16::core::{FixedTrait, f16x16, ONE, HALF, TWO};
use orion_numbers::f16x16::lut;

use orion_numbers::f16x16::core_trait::{I32Div, I32Rem};

// CONSTANTS
const TWO_PI: i32 = 411775;
const PI: i32 = 205887;
const HALF_PI: i32 = 102944;

// PUBLIC

// Calculates arccos(a) for -1 <= a <= 1 (fixed point)
// arccos(a) = arcsin(sqrt(1 - a^2)) - arctan identity has discontinuity at zero
pub fn acos_fast(a: f16x16) -> f16x16 {
    let asin_arg = (FixedTrait::ONE() - FixedTrait::mul(a, a)).sqrt(); // will fail if a > 1
    let asin_res = asin_fast(asin_arg);

    if a < 0 {
        FixedTrait::new(PI) - asin_res
    } else {
        asin_res
    }
}


// Calculates arcsin(a) for -1 <= a <= 1 (fixed point)
// arcsin(a) = arctan(a / sqrt(1 - a^2))
pub fn asin_fast(a: f16x16) -> f16x16 {
    if (a == ONE) {
        return FixedTrait::new(HALF_PI);
    }

    if (a == -ONE) {
        return FixedTrait::new(-HALF_PI);
    }

    let div = (FixedTrait::ONE() - FixedTrait::mul(a, a)).sqrt(); // will fail if a > 1

    atan_fast(FixedTrait::div(a, div))
}

// Calculates arctan(a) (fixed point)
// See https://stackoverflow.com/a/50894477 for range adjustments
pub fn atan_fast(a: f16x16) -> f16x16 {
    let mut at = a.abs();
    let mut shift = false;
    let mut invert = false;

    // Invert value when a > 1
    if (at > ONE) {
        at = FixedTrait::div(FixedTrait::ONE(), at);
        invert = true;
    }

    // Account for lack of precision in polynomaial when a > 0.7
    if (at > 45875) {
        let sqrt3_3 = FixedTrait::new(37837); // sqrt(3) / 3
        at = FixedTrait::div(at - sqrt3_3, FixedTrait::ONE() + FixedTrait::mul(at, sqrt3_3));
        shift = true;
    }

    let (start, low, high) = lut::atan(at);
    let partial_step = FixedTrait::div(FixedTrait::new(at - start).abs(), FixedTrait::new(459));
    let mut res = FixedTrait::mul(partial_step, FixedTrait::new(high - low).abs())
        + FixedTrait::new(low);

    // Adjust for sign change, inversion, and shift
    if (shift) {
        res = res + FixedTrait::new(34315); // pi / 6
    }

    if (invert) {
        res = res - FixedTrait::new(HALF_PI);
    }

    FixedTrait::mul(FixedTrait::new(res), a.sign())
}


// Calculates cos(a) with a in radians (fixed point)
pub fn cos_fast(a: f16x16) -> f16x16 {
    sin_fast(FixedTrait::new(HALF_PI) - a)
}

pub fn sin_fast(a: f16x16) -> f16x16 {
    let a1 = a.abs() % TWO_PI;
    //let (whole_rem, mut partial_rem) = DivRem::div_rem(a1, PI.try_into().unwrap());
    let whole_rem = Div::div(a1, PI);
    let mut partial_rem = Rem::rem(a1, PI);

    let partial_sign = whole_rem == 1;

    if partial_rem >= HALF_PI {
        partial_rem = PI - partial_rem;
    }

    let (start, low, high) = lut::sin(partial_rem);
    let partial_step = FixedTrait::div(
        FixedTrait::new(partial_rem - start).abs(), FixedTrait::new(402)
    );
    let res = FixedTrait::mul(partial_step, (FixedTrait::new(high) - FixedTrait::new(low)))
        + FixedTrait::new(low);

    if (a < 0) ^ partial_sign && res != 0 {
        FixedTrait::new(-res)
    } else {
        FixedTrait::new(res)
    }
}

// Calculates tan(a) with a in radians (fixed point)
pub fn tan_fast(a: f16x16) -> f16x16 {
    let sinx = sin_fast(a);
    let cosx = cos_fast(a);
    assert(cosx != 0, 'tan undefined');

    FixedTrait::div(sinx, cosx)
}

// Calculates inverse hyperbolic cosine of a (fixed point)
pub fn acosh(a: f16x16) -> f16x16 {
    let root = (FixedTrait::mul(a, a) - FixedTrait::ONE()).sqrt();

    (a + root).ln()
}

// Calculates inverse hyperbolic sine of a (fixed point)
pub fn asinh(a: f16x16) -> f16x16 {
    let root = (FixedTrait::mul(a, a) + FixedTrait::ONE()).sqrt();

    (a + root).ln()
}

// Calculates inverse hyperbolic tangent of a (fixed point)
pub fn atanh(a: f16x16) -> f16x16 {
    let one = FixedTrait::ONE();
    let ln_arg = FixedTrait::div((one + a), (one - a));

    FixedTrait::div(ln_arg.ln(), FixedTrait::new(TWO))
}

// Calculates hyperbolic cosine of a (fixed point)
pub fn cosh(a: f16x16) -> f16x16 {
    let ea = a.exp();

    FixedTrait::div((ea + FixedTrait::div(FixedTrait::ONE(), ea)), FixedTrait::new(TWO))
}

// Calculates hyperbolic sine of a (fixed point)
pub fn sinh(a: f16x16) -> f16x16 {
    let ea = a.exp();

    FixedTrait::div((ea - FixedTrait::div(FixedTrait::ONE(), ea)), FixedTrait::new(TWO))
}

// Calculates hyperbolic tangent of a (fixed point)
pub fn tanh(a: f16x16) -> f16x16 {
    let ea = a.exp();
    let ea_i = FixedTrait::div(FixedTrait::ONE(), ea);

    FixedTrait::div((ea - ea_i), (ea + ea_i))
}


// Tests
//
// 
// --------------------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use orion_numbers::f16x16::helpers::{assert_precise, assert_relative};

    use super::{
        FixedTrait, PI, HALF_PI, ONE, HALF, TWO, acos_fast, atan_fast, asin_fast, cos_fast,
        sin_fast, tan_fast, acosh, asinh, atanh, cosh, sinh, tanh
    };

    use orion_numbers::f16x16::core_trait::I32Div;

    #[test]
    #[available_gas(8000000)]
    fn test_acos_fast() {
        let error = Option::Some(84); // 1e-5

        let a = FixedTrait::ONE();
        assert(acos_fast(a).into() == 0, 'invalid one');

        let a = FixedTrait::new(ONE / 2);
        assert_relative(acos_fast(a), 68629, 'invalid half', error); // 1.3687308642680

        let a = FixedTrait::ZERO();
        assert_relative(acos_fast(a), HALF_PI.into(), 'invalid zero', Option::None(())); // PI / 2

        let a = FixedTrait::new(-ONE / 2);
        assert_relative(acos_fast(a), 137258, 'invalid neg half', error); // 2.737461741902

        let a = FixedTrait::new(-ONE);
        assert_relative(acos_fast(a), PI.into(), 'invalid neg one', Option::None(())); // PI
    }

    #[test]
    #[should_panic]
    #[available_gas(8000000)]
    fn test_acos_fail() {
        let a = FixedTrait::new(2 * ONE);
        acos_fast(a);
    }

    #[test]
    #[available_gas(8000000)]
    fn test_atan_fast() {
        let error = Option::Some(84); // 1e-5

        let a = FixedTrait::new(2 * ONE);
        assert_relative(atan_fast(a), 72558, 'invalid two', error);

        let a = FixedTrait::ONE();
        assert_relative(atan_fast(a), 51472, 'invalid one', error);

        let a = FixedTrait::new(ONE / 2);
        assert_relative(atan_fast(a), 30386, 'invalid half', error);

        let a = FixedTrait::ZERO();
        assert(atan_fast(a).into() == 0, 'invalid zero');

        let a = FixedTrait::new(-ONE / 2);
        assert_relative(atan_fast(a), -30386, 'invalid neg half', error);

        let a = FixedTrait::new(-ONE);
        assert_relative(atan_fast(a), -51472, 'invalid neg one', error);

        let a = FixedTrait::new(-2 * ONE);
        assert_relative(atan_fast(a), -72558, 'invalid neg two', error);
    }

    #[test]
    #[available_gas(8000000)]
    fn test_asin() {
        let error = Option::Some(84); // 1e-5

        let a = FixedTrait::ONE();
        assert_relative(asin_fast(a), HALF_PI.into(), 'invalid one', Option::None(())); // PI / 2

        let a = FixedTrait::new(ONE / 2);
        assert_relative(asin_fast(a), 34315, 'invalid half', error);

        let a = FixedTrait::ZERO();
        assert_precise(asin_fast(a), 0, 'invalid zero', Option::None(()));

        let a = FixedTrait::new(-ONE / 2);
        assert_relative(asin_fast(a), -34315, 'invalid neg half', error);

        let a = FixedTrait::new(-ONE);
        assert_relative(
            asin_fast(a), -HALF_PI.into(), 'invalid neg one', Option::None(())
        ); // -PI / 2
    }

    #[test]
    #[should_panic]
    #[available_gas(8000000)]
    fn test_asin_fail() {
        let a = FixedTrait::new(2 * ONE);
        asin_fast(a);
    }

    #[test]
    #[available_gas(8000000)]
    fn test_cos_fast() {
        let error = Option::Some(84); // 1e-5

        let a = FixedTrait::new(HALF_PI);
        assert(cos_fast(a).into() == 0, 'invalid half pi');

        let a = FixedTrait::new(HALF_PI / 2);
        assert_precise(cos_fast(a), 46341, 'invalid quarter pi', error); // 0.55242717280199

        let a = FixedTrait::new(PI);
        assert_precise(cos_fast(a), -1 * ONE.into(), 'invalid pi', error);

        let a = FixedTrait::new(HALF_PI);
        assert_precise(cos_fast(a), 0, 'invalid neg half pi', Option::None(()));

        let a = FixedTrait::new_unscaled(17);
        assert_precise(cos_fast(a), -18033, 'invalid 17', error); // -0.21497123284870
    }

    #[test]
    #[available_gas(8000000)]
    fn test_sin_fast() {
        let error = Option::Some(84); // 1e-5

        let a = FixedTrait::new(HALF_PI);
        assert_precise(sin_fast(a), ONE.into(), 'invalid half pi', error);

        let a = FixedTrait::new(HALF_PI / 2);
        assert_precise(sin_fast(a), 46341, 'invalid quarter pi', error); // 0.55242717280199

        let a = FixedTrait::new(PI);
        assert(sin_fast(a).into() == 0, 'invalid pi');

        let a = FixedTrait::new(-HALF_PI);
        assert_precise(sin_fast(a), -ONE.into(), 'invalid neg half pi', error); // 0.78124999999529

        let a = FixedTrait::new_unscaled(17);
        assert_precise(sin_fast(a), -63006, 'invalid 17', error); // -0.75109179053073

        let a = FixedTrait::new_unscaled(-17);
        assert_precise(sin_fast(a), 63006, 'invalid -17', error); // 0.75109179053073
    }

    #[test]
    #[available_gas(8000000)]
    fn test_tan_fast() {
        let a = FixedTrait::new(HALF_PI / 2);
        assert_precise(tan_fast(a), ONE.into(), 'invalid quarter pi', Option::None(()));

        let a = FixedTrait::new(PI);
        assert_precise(tan_fast(a), 0, 'invalid pi', Option::None(()));

        let a = FixedTrait::new_unscaled(17);
        assert_precise(tan_fast(a), 228990, 'invalid 17', Option::None(())); // 3.3858731852805

        let a = FixedTrait::new_unscaled(-17);

        assert_precise(tan_fast(a), -228952, 'invalid -17', Option::None(())); // -3.3858731852805
    }

    #[test]
    #[available_gas(10000000)]
    fn test_acosh() {
        let a = FixedTrait::new(246559); // 3.5954653836066
        assert_precise(acosh(a), 131072, 'invalid two', Option::None(()));

        let a = FixedTrait::new(101127); // 1.42428174592510
        assert_precise(acosh(a), ONE.into(), 'invalid one', Option::None(()));

        let a = FixedTrait::ONE(); // 1
        assert(acosh(a).into() == 0, 'invalid zero');
    }

    #[test]
    #[available_gas(10000000)]
    fn test_asinh() {
        let a = FixedTrait::new(237690); // 3.48973469357602
        assert_precise(asinh(a), 131072, 'invalid two', Option::None(()));

        let a = FixedTrait::new(77018); // 1.13687593250230
        assert_precise(asinh(a), ONE.into(), 'invalid one', Option::None(()));

        let a = FixedTrait::ZERO();
        assert(asinh(a).into() == 0, 'invalid zero');

        let a = FixedTrait::new(-77018); // -1.13687593250230
        assert_precise(asinh(a), -ONE.into(), 'invalid neg one', Option::None(()));

        let a = FixedTrait::new(-237690); // -3.48973469357602
        assert_precise(asinh(a), -131017, 'invalid neg two', Option::None(()));
    }

    #[test]
    #[available_gas(10000000)]
    fn test_atanh() {
        let a = FixedTrait::new(58982); // 0.9
        assert_precise(atanh(a), 96483, 'invalid 0.9', Option::None(())); // 1.36892147623689

        let a = FixedTrait::new(HALF); // 0.5
        assert_precise(atanh(a), 35999, 'invalid half', Option::None(())); // 0.42914542526098

        let a = FixedTrait::ZERO();
        assert(atanh(a).into() == 0, 'invalid zero');

        let a = FixedTrait::new(-HALF); // 0.5
        assert_precise(atanh(a), -35999, 'invalid neg half', Option::None(())); // 0.42914542526098

        let a = FixedTrait::new(-58982); // 0.9
        assert_precise(atanh(a), -96483, 'invalid -0.9', Option::None(())); // 1.36892147623689
    }

    #[test]
    #[available_gas(10000000)]
    fn test_cosh() {
        let a = FixedTrait::new(TWO);
        assert_precise(cosh(a), 246550, 'invalid two', Option::None(())); // 3.5954653836066

        let a = FixedTrait::ONE();
        assert_precise(cosh(a), 101127, 'invalid one', Option::None(())); // 1.42428174592510

        let a = FixedTrait::ZERO();
        assert_precise(cosh(a), ONE.into(), 'invalid zero', Option::None(()));

        let a = -FixedTrait::ONE();
        assert_precise(cosh(a), 101127, 'invalid neg one', Option::None(())); // 1.42428174592510

        let a = FixedTrait::new(-TWO);
        assert_precise(cosh(a), 246568, 'invalid neg two', Option::None(())); // 3.5954653836066
    }

    #[test]
    #[available_gas(10000000)]
    fn test_sinh() {
        let a = FixedTrait::new(TWO);
        assert_precise(sinh(a), 237681, 'invalid two', Option::None(())); // 3.48973469357602

        let a = FixedTrait::ONE();
        assert_precise(sinh(a), 77018, 'invalid one', Option::None(())); // 1.13687593250230

        let a = FixedTrait::ZERO();
        assert(sinh(a).into() == 0, 'invalid zero');

        let a = FixedTrait::new(-ONE);
        assert_precise(sinh(a), -77018, 'invalid neg one', Option::None(())); // -1.13687593250230

        let a = FixedTrait::new(-TWO);
        assert_precise(sinh(a), -237699, 'invalid neg two', Option::None(())); // -3.48973469357602
    }

    #[test]
    #[available_gas(10000000)]
    fn test_tanh() {
        let a = FixedTrait::new(TWO);
        assert_precise(tanh(a), 63179, 'invalid two', Option::None(())); // 0.75314654693321

        let a = FixedTrait::ONE();
        assert_precise(tanh(a), 49912, 'invalid one', Option::None(())); // 0.59499543433175

        let a = FixedTrait::ZERO();
        assert(tanh(a).into() == 0, 'invalid zero');

        let a = FixedTrait::new(-ONE);
        assert_precise(tanh(a), -49912, 'invalid neg one', Option::None(())); // -0.59499543433175

        let a = FixedTrait::new(-TWO);
        assert_precise(tanh(a), -63179, 'invalid neg two', Option::None(())); // 0.75314654693321
    }
}
