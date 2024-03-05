use core::integer;

use orion::numbers::fixed_point::implementations::fp8x23::math::lut;
use orion::numbers::fixed_point::implementations::fp8x23::core::{
    HALF, ONE, TWO, FP8x23, FP8x23Impl, FP8x23Add, FP8x23Sub, FP8x23Mul, FP8x23Div,
    FP8x23IntoFelt252, FixedTrait
};

// CONSTANTS
const TWO_PI: u32 = 52707178;
const PI: u32 = 26353589;
const HALF_PI: u32 = 13176795;

// PUBLIC

// Calculates arccos(a) for -1 <= a <= 1 (fixed point)
// arccos(a) = arcsin(sqrt(1 - a^2)) - arctan identity has discontinuity at zero
fn acos(a: FP8x23) -> FP8x23 {
    let asin_arg = (FixedTrait::ONE() - a * a).sqrt(); // will fail if a > 1
    let asin_res = asin(asin_arg);

    if (a.sign) {
        FixedTrait::new(PI, false) - asin_res
    } else {
        asin_res
    }
}

fn acos_fast(a: FP8x23) -> FP8x23 {
    let asin_arg = (FixedTrait::ONE() - a * a).sqrt(); // will fail if a > 1
    let asin_res = asin_fast(asin_arg);

    if (a.sign) {
        FixedTrait::new(PI, false) - asin_res
    } else {
        asin_res
    }
}

// Calculates arcsin(a) for -1 <= a <= 1 (fixed point)
// arcsin(a) = arctan(a / sqrt(1 - a^2))
fn asin(a: FP8x23) -> FP8x23 {
    if (a.mag == ONE) {
        return FixedTrait::new(HALF_PI, a.sign);
    }

    let div = (FixedTrait::ONE() - a * a).sqrt(); // will fail if a > 1

    atan(a / div)
}

fn asin_fast(a: FP8x23) -> FP8x23 {
    if (a.mag == ONE) {
        return FixedTrait::new(HALF_PI, a.sign);
    }

    let div = (FixedTrait::ONE() - a * a).sqrt(); // will fail if a > 1

    atan_fast(a / div)
}

// Calculates arctan(a) (fixed point)
// See https://stackoverflow.com/a/50894477 for range adjustments
fn atan(a: FP8x23) -> FP8x23 {
    let mut at = a.abs();
    let mut shift = false;
    let mut invert = false;

    // Invert value when a > 1
    if (at.mag > ONE) {
        at = FixedTrait::ONE() / at;
        invert = true;
    }

    // Account for lack of precision in polynomaial when a > 0.7
    if (at.mag > 5872026) {
        let sqrt3_3 = FixedTrait::new(4843165, false); // sqrt(3) / 3
        at = (at - sqrt3_3) / (FixedTrait::ONE() + at * sqrt3_3);
        shift = true;
    }

    let r10 = FixedTrait::new(15363, true) * at;
    let r9 = (r10 + FixedTrait::new(392482, true)) * at;
    let r8 = (r9 + FixedTrait::new(1629064, false)) * at;
    let r7 = (r8 + FixedTrait::new(2197820, true)) * at;
    let r6 = (r7 + FixedTrait::new(366693, false)) * at;
    let r5 = (r6 + FixedTrait::new(1594324, false)) * at;
    let r4 = (r5 + FixedTrait::new(11519, false)) * at;
    let r3 = (r4 + FixedTrait::new(2797104, true)) * at;
    let r2 = (r3 + FixedTrait::new(34, false)) * at;
    let mut res = (r2 + FixedTrait::new(8388608, false)) * at;

    // Adjust for sign change, inversion, and shift
    if (shift) {
        res = res + FixedTrait::new(4392265, false); // pi / 6
    }

    if (invert) {
        res = res - FixedTrait::new(HALF_PI, false);
    }

    FixedTrait::new(res.mag, a.sign)
}

fn atan_fast(a: FP8x23) -> FP8x23 {
    let mut at = a.abs();
    let mut shift = false;
    let mut invert = false;

    // Invert value when a > 1
    if (at.mag > ONE) {
        at = FixedTrait::ONE() / at;
        invert = true;
    }

    // Account for lack of precision in polynomaial when a > 0.7
    if (at.mag > 5872026) {
        let sqrt3_3 = FixedTrait::new(4843165, false); // sqrt(3) / 3
        at = (at - sqrt3_3) / (FixedTrait::ONE() + at * sqrt3_3);
        shift = true;
    }

    let (start, low, high) = lut::atan(at.mag);
    let partial_step = FixedTrait::new(at.mag - start, false) / FixedTrait::new(58720, false);
    let mut res = partial_step * FixedTrait::new(high - low, false) + FixedTrait::new(low, false);

    // Adjust for sign change, inversion, and shift
    if (shift) {
        res = res + FixedTrait::new(4392265, false); // pi / 6
    }

    if (invert) {
        res = res - FixedTrait::<FP8x23>::new(HALF_PI, false);
    }

    FixedTrait::new(res.mag, a.sign)
}

// Calculates cos(a) with a in radians (fixed point)
fn cos(a: FP8x23) -> FP8x23 {
    sin(FixedTrait::new(HALF_PI, false) - a)
}

fn cos_fast(a: FP8x23) -> FP8x23 {
    sin_fast(FixedTrait::new(HALF_PI, false) - a)
}

fn sin(a: FP8x23) -> FP8x23 {
    let a1 = a.mag % TWO_PI;
    let (whole_rem, partial_rem) = integer::u32_safe_divmod(a1, integer::u32_as_non_zero(PI));
    let a2 = FixedTrait::new(partial_rem, false);
    let partial_sign = whole_rem == 1;

    let loop_res = a2 * _sin_loop(a2, 7, FixedTrait::ONE());

    FixedTrait::new(loop_res.mag, a.sign ^ partial_sign && loop_res.mag != 0)
}

fn sin_fast(a: FP8x23) -> FP8x23 {
    let a1 = a.mag % TWO_PI;
    let (whole_rem, mut partial_rem) = integer::u32_safe_divmod(a1, integer::u32_as_non_zero(PI));
    let partial_sign = whole_rem == 1;

    if partial_rem >= HALF_PI {
        partial_rem = PI - partial_rem;
    }

    let (start, low, high) = lut::sin(partial_rem);
    let partial_step = FixedTrait::new(partial_rem - start, false) / FixedTrait::new(51472, false);
    let res = partial_step * (FixedTrait::new(high, false) - FixedTrait::new(low, false))
        + FixedTrait::<FP8x23>::new(low, false);

    FixedTrait::new(res.mag, a.sign ^ partial_sign && res.mag != 0)
}

// Calculates tan(a) with a in radians (fixed point)
fn tan(a: FP8x23) -> FP8x23 {
    let sinx = sin(a);
    let cosx = cos(a);
    assert(cosx.mag != 0, 'tan undefined');

    sinx / cosx
}

fn tan_fast(a: FP8x23) -> FP8x23 {
    let sinx = sin_fast(a);
    let cosx = cos_fast(a);
    assert(cosx.mag != 0, 'tan undefined');

    sinx / cosx
}

// Helper function to calculate Taylor series for sin
fn _sin_loop(a: FP8x23, i: u32, acc: FP8x23) -> FP8x23 {
    let div = (2 * i + 2) * (2 * i + 3);
    let term = a * a * acc / FixedTrait::new_unscaled(div, false);
    let new_acc = FixedTrait::ONE() - term;

    if (i == 0) {
        return new_acc;
    }

    _sin_loop(a, i - 1, new_acc)
}

// Tests --------------------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use orion::numbers::fixed_point::implementations::fp8x23::helpers::{
        assert_precise, assert_relative
    };

    use super::{
        FixedTrait, acos, HALF_PI, ONE, acos_fast, PI, atan_fast, atan, asin, cos, cos_fast, sin,
        sin_fast, tan
    };

    #[test]
    #[available_gas(3000000)]
    fn test_acos() {
        let error = Option::Some(84); // 1e-5

        let a = FixedTrait::ONE();
        assert(acos(a).into() == 0, 'invalid one');

        let a = FixedTrait::new(ONE / 2, false);
        assert_relative(acos(a), 8784530, 'invalid half', error); // 1.0471975506263043

        let a = FixedTrait::ZERO();
        assert_relative(acos(a), HALF_PI.into(), 'invalid zero', Option::None(())); // PI / 2

        let a = FixedTrait::new(ONE / 2, true);
        assert_relative(acos(a), 17569060, 'invalid neg half', error); // 2.094395102963489

        let a = FixedTrait::new(ONE, true);
        assert_relative(acos(a), PI.into(), 'invalid neg one', Option::None(())); // PI
    }

    #[test]
    #[available_gas(3000000)]
    fn test_acos_fast() {
        let error = Option::Some(84); // 1e-5

        let a = FixedTrait::ONE();
        assert(acos_fast(a).into() == 0, 'invalid one');

        let a = FixedTrait::new(ONE / 2, false);
        assert_relative(acos_fast(a), 8784530, 'invalid half', error); // 1.0471975506263043

        let a = FixedTrait::ZERO();
        assert_relative(acos_fast(a), HALF_PI.into(), 'invalid zero', Option::None(())); // PI / 2

        let a = FixedTrait::new(ONE / 2, true);
        assert_relative(acos_fast(a), 17569060, 'invalid neg half', error); // 2.094395102963489

        let a = FixedTrait::new(ONE, true);
        assert_relative(acos_fast(a), PI.into(), 'invalid neg one', Option::None(())); // PI
    }

    #[test]
    #[should_panic]
    #[available_gas(1000000)]
    fn test_acos_fail() {
        let a = FixedTrait::new(2 * ONE, true);
        acos(a);
    }

    #[test]
    #[available_gas(1400000)]
    fn test_atan_fast() {
        let error = Option::Some(84); // 1e-5

        let a = FixedTrait::new(2 * ONE, false);
        assert_relative(atan_fast(a), 9287437, 'invalid two', error);

        let a = FixedTrait::ONE();
        assert_relative(atan_fast(a), 6588397, 'invalid one', error);

        let a = FixedTrait::new(ONE / 2, false);
        assert_relative(atan_fast(a), 3889358, 'invalid half', error);

        let a = FixedTrait::ZERO();
        assert(atan_fast(a).into() == 0, 'invalid zero');

        let a = FixedTrait::new(ONE / 2, true);
        assert_relative(atan_fast(a), -3889358, 'invalid neg half', error);

        let a = FixedTrait::new(ONE, true);
        assert_relative(atan_fast(a), -6588397, 'invalid neg one', error);

        let a = FixedTrait::new(2 * ONE, true);
        assert_relative(atan_fast(a), -9287437, 'invalid neg two', error);
    }

    #[test]
    #[available_gas(2600000)]
    fn test_atan() {
        let a = FixedTrait::new(2 * ONE, false);
        assert_relative(atan(a), 9287437, 'invalid two', Option::None(()));

        let a = FixedTrait::ONE();
        assert_relative(atan(a), 6588397, 'invalid one', Option::None(()));

        let a = FixedTrait::new(ONE / 2, false);
        assert_relative(atan(a), 3889358, 'invalid half', Option::None(()));

        let a = FixedTrait::ZERO();
        assert(atan(a).into() == 0, 'invalid zero');

        let a = FixedTrait::new(ONE / 2, true);
        assert_relative(atan(a), -3889358, 'invalid neg half', Option::None(()));

        let a = FixedTrait::new(ONE, true);
        assert_relative(atan(a), -6588397, 'invalid neg one', Option::None(()));

        let a = FixedTrait::new(2 * ONE, true);
        assert_relative(atan(a), -9287437, 'invalid neg two', Option::None(()));
    }

    #[test]
    #[available_gas(3000000)]
    fn test_asin() {
        let error = Option::Some(84); // 1e-5

        let a = FixedTrait::ONE();
        assert_relative(asin(a), HALF_PI.into(), 'invalid one', Option::None(())); // PI / 2

        let a = FixedTrait::new(ONE / 2, false);
        assert_relative(asin(a), 4392265, 'invalid half', error);

        let a = FixedTrait::ZERO();
        assert_precise(asin(a), 0, 'invalid zero', Option::None(()));

        let a = FixedTrait::new(ONE / 2, true);
        assert_relative(asin(a), -4392265, 'invalid neg half', error);

        let a = FixedTrait::new(ONE, true);
        assert_relative(asin(a), -HALF_PI.into(), 'invalid neg one', Option::None(())); // -PI / 2
    }

    #[test]
    #[should_panic]
    #[available_gas(1000000)]
    fn test_asin_fail() {
        let a = FixedTrait::new(2 * ONE, false);
        asin(a);
    }

    #[test]
    #[available_gas(6000000)]
    fn test_cos() {
        let a = FixedTrait::new(HALF_PI, false);
        assert(cos(a).into() == 0, 'invalid half pi');

        let a = FixedTrait::new(HALF_PI / 2, false);
        assert_relative(
            cos(a), 5931642, 'invalid quarter pi', Option::None(())
        ); // 0.7071067811865475

        let a = FixedTrait::new(PI, false);
        assert_relative(cos(a), -1 * ONE.into(), 'invalid pi', Option::None(()));

        let a = FixedTrait::new(HALF_PI, true);
        assert_precise(cos(a), 0, 'invalid neg half pi', Option::None(()));

        let a = FixedTrait::new_unscaled(17, false);
        assert_relative(cos(a), -2308239, 'invalid 17', Option::None(())); // -0.2751631780463348

        let a = FixedTrait::new_unscaled(17, true);
        assert_relative(cos(a), -2308236, 'invalid -17', Option::None(())); // -0.2751631780463348
    }

    #[test]
    #[available_gas(6000000)]
    fn test_cos_fast() {
        let error = Option::Some(84); // 1e-5

        let a = FixedTrait::new(HALF_PI, false);
        assert(cos_fast(a).into() == 0, 'invalid half pi');

        let a = FixedTrait::new(HALF_PI / 2, false);
        assert_precise(cos_fast(a), 5931642, 'invalid quarter pi', error); // 0.7071067811865475

        let a = FixedTrait::new(PI, false);
        assert_precise(cos_fast(a), -1 * ONE.into(), 'invalid pi', error);

        let a = FixedTrait::new(HALF_PI, true);
        assert_precise(cos(a), 0, 'invalid neg half pi', Option::None(()));

        let a = FixedTrait::new_unscaled(17, false);
        assert_precise(cos_fast(a), -2308239, 'invalid 17', error); // -0.2751631780463348
    }

    #[test]
    #[available_gas(6000000)]
    fn test_sin() {
        let a = FixedTrait::new(HALF_PI, false);
        assert_precise(sin(a), ONE.into(), 'invalid half pi', Option::None(()));

        let a = FixedTrait::new(HALF_PI / 2, false);
        assert_precise(
            sin(a), 5931642, 'invalid quarter pi', Option::None(())
        ); // 0.7071067811865475

        let a = FixedTrait::new(PI, false);
        assert(sin(a).into() == 0, 'invalid pi');

        let a = FixedTrait::new(HALF_PI, true);
        assert_precise(
            sin(a), -ONE.into(), 'invalid neg half pi', Option::None(())
        ); // 0.9999999999939766

        let a = FixedTrait::new_unscaled(17, false);
        assert_precise(sin(a), -8064787, 'invalid 17', Option::None(())); // -0.9613974918793389

        let a = FixedTrait::new_unscaled(17, true);
        assert_precise(sin(a), 8064787, 'invalid -17', Option::None(())); // 0.9613974918793389
    }

    #[test]
    #[available_gas(1000000)]
    fn test_sin_fast() {
        let error = Option::Some(84); // 1e-5

        let a = FixedTrait::new(HALF_PI, false);
        assert_precise(sin_fast(a), ONE.into(), 'invalid half pi', error);

        let a = FixedTrait::new(HALF_PI / 2, false);
        assert_precise(sin_fast(a), 5931642, 'invalid quarter pi', error); // 0.7071067811865475

        let a = FixedTrait::new(PI, false);
        assert(sin_fast(a).into() == 0, 'invalid pi');

        let a = FixedTrait::new(HALF_PI, true);
        assert_precise(
            sin_fast(a), -ONE.into(), 'invalid neg half pi', error
        ); // 0.9999999999939766

        let a = FixedTrait::new_unscaled(17, false);
        assert_precise(sin_fast(a), -8064787, 'invalid 17', error); // -0.9613974918793389

        let a = FixedTrait::new_unscaled(17, true);
        assert_precise(sin_fast(a), 8064787, 'invalid -17', error); // 0.9613974918793389
    }

    #[test]
    #[available_gas(8000000)]
    fn test_tan() {
        let a = FixedTrait::new(HALF_PI / 2, false);
        assert_precise(tan(a), ONE.into(), 'invalid quarter pi', Option::None(()));

        let a = FixedTrait::new(PI, false);
        assert_precise(tan(a), 0, 'invalid pi', Option::None(()));

        let a = FixedTrait::new_unscaled(17, false);
        assert_precise(tan(a), 29309069, 'invalid 17', Option::None(())); // 3.493917677159002

        let a = FixedTrait::new_unscaled(17, true);
        assert_precise(tan(a), -29309106, 'invalid -17', Option::None(())); // -3.493917677159002
    }
}
