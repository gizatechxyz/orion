use debug::PrintTrait;
use integer::{u32_safe_divmod, u32_as_non_zero};
use option::OptionTrait;

use orion::numbers::fixed_point::implementations::fp16x16::math::lut;
use orion::numbers::fixed_point::implementations::fp16x16::core::{
    HALF, ONE, TWO, FixedType, FP16x16Impl, FP16x16Add, FP16x16Sub, FP16x16Mul, FP16x16Div,
    FP16x16IntoFelt252, FixedTrait
};

// CONSTANTS

const TWO_PI: u32 = 411775;
const PI: u32 = 205887;
const HALF_PI: u32 = 102944;

// PUBLIC

// Calculates arccos(a) for -1 <= a <= 1 (fixed point)
// arccos(a) = arcsin(sqrt(1 - a^2)) - arctan identity has discontinuity at zero
fn acos(a: FixedType) -> FixedType {
    let asin_arg = (FixedTrait::ONE() - a * a).sqrt(); // will fail if a > 1
    let asin_res = asin(asin_arg);

    if (a.sign) {
        return FixedTrait::new(PI, false) - asin_res;
    } else {
        return asin_res;
    }
}

fn acos_fast(a: FixedType) -> FixedType {
    let asin_arg = (FixedTrait::ONE() - a * a).sqrt(); // will fail if a > 1
    let asin_res = asin_fast(asin_arg);

    if (a.sign) {
        return FixedTrait::new(PI, false) - asin_res;
    } else {
        return asin_res;
    }
}

// Calculates arcsin(a) for -1 <= a <= 1 (fixed point)
// arcsin(a) = arctan(a / sqrt(1 - a^2))
fn asin(a: FixedType) -> FixedType {
    if (a.mag == ONE) {
        return FixedTrait::new(HALF_PI, a.sign);
    }

    let div = (FixedTrait::ONE() - a * a).sqrt(); // will fail if a > 1
    return atan(a / div);
}

fn asin_fast(a: FixedType) -> FixedType {
    if (a.mag == ONE) {
        return FixedTrait::new(HALF_PI, a.sign);
    }

    let div = (FixedTrait::ONE() - a * a).sqrt(); // will fail if a > 1
    return atan_fast(a / div);
}

// Calculates arctan(a) (fixed point)
// See https://stackoverflow.com/a/50894477 for range adjustments
fn atan(a: FixedType) -> FixedType {
    let mut at = a.abs();
    let mut shift = false;
    let mut invert = false;

    // Invert value when a > 1
    if (at.mag > ONE) {
        at = FixedTrait::ONE() / at;
        invert = true;
    }

    // Account for lack of precision in polynomaial when a > 0.7
    if (at.mag > 45875) {
        let sqrt3_3 = FixedTrait::new(37837, false); // sqrt(3) / 3
        at = (at - sqrt3_3) / (FixedTrait::ONE() + at * sqrt3_3);
        shift = true;
    }

    let r10 = FixedTrait::new(120, true) * at;
    let r9 = (r10 + FixedTrait::new(3066, true)) * at;
    let r8 = (r9 + FixedTrait::new(12727, false)) * at;
    let r7 = (r8 + FixedTrait::new(17170, true)) * at;
    let r6 = (r7 + FixedTrait::new(2865, false)) * at;
    let r5 = (r6 + FixedTrait::new(12456, false)) * at;
    let r4 = (r5 + FixedTrait::new(90, false)) * at;
    let r3 = (r4 + FixedTrait::new(21852, true)) * at;
    let r2 = r3 * at;
    let mut res = (r2 + FixedTrait::new(65536, false)) * at;

    // Adjust for sign change, inversion, and shift
    if (shift) {
        res = res + FixedTrait::new(34315, false); // pi / 6
    }

    if (invert) {
        res = res - FixedTrait::new(HALF_PI, false);
    }

    return FixedTrait::new(res.mag, a.sign);
}


fn atan_fast(a: FixedType) -> FixedType {
    let mut at = a.abs();
    let mut shift = false;
    let mut invert = false;

    // Invert value when a > 1
    if (at.mag > ONE) {
        at = FixedTrait::ONE() / at;
        invert = true;
    }

    // Account for lack of precision in polynomaial when a > 0.7
    if (at.mag > 45875) {
        let sqrt3_3 = FixedTrait::new(37837, false); // sqrt(3) / 3
        at = (at - sqrt3_3) / (FixedTrait::ONE() + at * sqrt3_3);
        shift = true;
    }

    let (start, low, high) = lut::atan(at.mag);
    let partial_step = FixedTrait::new(at.mag - start, false) / FixedTrait::new(459, false);
    let mut res = partial_step * FixedTrait::new(high - low, false) + FixedTrait::new(low, false);

    // Adjust for sign change, inversion, and shift
    if (shift) {
        res = res + FixedTrait::new(34315, false); // pi / 6
    }

    if (invert) {
        res = res - FixedTrait::new(HALF_PI, false);
    }

    return FixedTrait::new(res.mag, a.sign);
}

// Calculates cos(a) with a in radians (fixed point)
fn cos(a: FixedType) -> FixedType {
    return sin(FixedTrait::new(HALF_PI, false) - a);
}

fn cos_fast(a: FixedType) -> FixedType {
    return sin_fast(FixedTrait::new(HALF_PI, false) - a);
}

fn sin(a: FixedType) -> FixedType {
    let a1 = a.mag % TWO_PI;
    let (whole_rem, partial_rem) = u32_safe_divmod(a1, u32_as_non_zero(PI));
    let a2 = FixedTrait::new(partial_rem, false);
    let partial_sign = whole_rem == 1;

    let loop_res = a2 * _sin_loop(a2, 7, FixedTrait::ONE());
    return FixedTrait::new(loop_res.mag, a.sign ^ partial_sign && loop_res.mag != 0);
}

fn sin_fast(a: FixedType) -> FixedType {
    let a1 = a.mag % TWO_PI;
    let (whole_rem, mut partial_rem) = u32_safe_divmod(a1, u32_as_non_zero(PI));
    let partial_sign = whole_rem == 1;

    if partial_rem >= HALF_PI {
        partial_rem = PI - partial_rem;
    }

    let (start, low, high) = lut::sin(partial_rem);
    let partial_step = FixedTrait::new(partial_rem - start, false) / FixedTrait::new(402, false);
    let res = partial_step * (FixedTrait::new(high, false) - FixedTrait::new(low, false))
        + FixedTrait::new(low, false);

    return FixedTrait::new(res.mag, a.sign ^ partial_sign && res.mag != 0);
}

// Calculates tan(a) with a in radians (fixed point)
fn tan(a: FixedType) -> FixedType {
    let sinx = sin(a);
    let cosx = cos(a);
    assert(cosx.mag != 0, 'tan undefined');
    return sinx / cosx;
}

fn tan_fast(a: FixedType) -> FixedType {
    let sinx = sin_fast(a);
    let cosx = cos_fast(a);
    assert(cosx.mag != 0, 'tan undefined');
    return sinx / cosx;
}

// Helper function to calculate Taylor series for sin
fn _sin_loop(a: FixedType, i: u32, acc: FixedType) -> FixedType {
    let div = (2 * i + 2) * (2 * i + 3);
    let term = a * a * acc / FixedTrait::new_unscaled(div, false);
    let new_acc = FixedTrait::ONE() - term;

    if (i == 0) {
        return new_acc;
    }

    return _sin_loop(a, i - 1, new_acc);
}

// Tests --------------------------------------------------------------------------------------------------------------

use traits::Into;

use orion::numbers::fixed_point::implementations::fp16x16::helpers::{
    assert_precise, assert_relative
};
use orion::numbers::fixed_point::implementations::fp16x16::core::{FP16x16PartialEq, FP16x16Print};

#[test]
#[available_gas(8000000)]
fn test_acos() {
    let error = Option::Some(84); // 1e-5

    let a = FixedTrait::ONE();
    assert(acos(a).into() == 0, 'invalid one');

    let a = FixedTrait::new(ONE / 2, false);
    assert_relative(acos(a), 68629, 'invalid half', error); // 1.3687308642680

    let a = FixedTrait::ZERO();
    assert_relative(acos(a), HALF_PI.into(), 'invalid zero', Option::None(())); // PI / 2

    let a = FixedTrait::new(ONE / 2, true);
    assert_relative(acos(a), 137258, 'invalid neg half', error); // 2.737461741902

    let a = FixedTrait::new(ONE, true);
    assert_relative(acos(a), PI.into(), 'invalid neg one', Option::None(())); // PI
}

#[test]
#[available_gas(8000000)]
fn test_acos_fast() {
    let error = Option::Some(84); // 1e-5

    let a = FixedTrait::ONE();
    assert(acos_fast(a).into() == 0, 'invalid one');

    let a = FixedTrait::new(ONE / 2, false);
    assert_relative(acos_fast(a), 68629, 'invalid half', error); // 1.3687308642680

    let a = FixedTrait::ZERO();
    assert_relative(acos_fast(a), HALF_PI.into(), 'invalid zero', Option::None(())); // PI / 2

    let a = FixedTrait::new(ONE / 2, true);
    assert_relative(acos_fast(a), 137258, 'invalid neg half', error); // 2.737461741902

    let a = FixedTrait::new(ONE, true);
    assert_relative(acos_fast(a), PI.into(), 'invalid neg one', Option::None(())); // PI
}

#[test]
#[should_panic]
#[available_gas(8000000)]
fn test_acos_fail() {
    let a = FixedTrait::new(2 * ONE, true);
    acos(a);
}

#[test]
#[available_gas(8000000)]
fn test_atan_fast() {
    let error = Option::Some(84); // 1e-5

    let a = FixedTrait::new(2 * ONE, false);
    assert_relative(atan_fast(a), 72558, 'invalid two', error);

    let a = FixedTrait::ONE();
    assert_relative(atan_fast(a), 51472, 'invalid one', error);

    let a = FixedTrait::new(ONE / 2, false);
    assert_relative(atan_fast(a), 30386, 'invalid half', error);

    let a = FixedTrait::ZERO();
    assert(atan_fast(a).into() == 0, 'invalid zero');

    let a = FixedTrait::new(ONE / 2, true);
    assert_relative(atan_fast(a), -30386, 'invalid neg half', error);

    let a = FixedTrait::new(ONE, true);
    assert_relative(atan_fast(a), -51472, 'invalid neg one', error);

    let a = FixedTrait::new(2 * ONE, true);
    assert_relative(atan_fast(a), -72558, 'invalid neg two', error);
}

#[test]
#[available_gas(8000000)]
fn test_atan() {
    let a = FixedTrait::new(2 * ONE, false);
    assert_relative(atan(a), 72558, 'invalid two', Option::None(()));

    let a = FixedTrait::ONE();
    assert_relative(atan(a), 51472, 'invalid one', Option::None(()));

    let a = FixedTrait::new(ONE / 2, false);
    assert_relative(atan(a), 30386, 'invalid half', Option::None(()));

    let a = FixedTrait::ZERO();
    assert(atan(a).into() == 0, 'invalid zero');

    let a = FixedTrait::new(ONE / 2, true);
    assert_relative(atan(a), -30386, 'invalid neg half', Option::None(()));

    let a = FixedTrait::new(ONE, true);
    assert_relative(atan(a), -51472, 'invalid neg one', Option::None(()));

    let a = FixedTrait::new(2 * ONE, true);
    assert_relative(atan(a), -72558, 'invalid neg two', Option::None(()));
}

#[test]
#[available_gas(8000000)]
fn test_asin() {
    let error = Option::Some(84); // 1e-5

    let a = FixedTrait::ONE();
    assert_relative(asin(a), HALF_PI.into(), 'invalid one', Option::None(())); // PI / 2

    let a = FixedTrait::new(ONE / 2, false);
    assert_relative(asin(a), 34315, 'invalid half', error);

    let a = FixedTrait::ZERO();
    assert_precise(asin(a), 0, 'invalid zero', Option::None(()));

    let a = FixedTrait::new(ONE / 2, true);
    assert_relative(asin(a), -34315, 'invalid neg half', error);

    let a = FixedTrait::new(ONE, true);
    assert_relative(asin(a), -HALF_PI.into(), 'invalid neg one', Option::None(())); // -PI / 2
}

#[test]
#[should_panic]
#[available_gas(8000000)]
fn test_asin_fail() {
    let a = FixedTrait::new(2 * ONE, false);
    asin(a);
}

#[test]
#[available_gas(8000000)]
fn test_cos() {
    let a = FixedTrait::new(HALF_PI, false);
    assert(cos(a).into() == 0, 'invalid half pi');

    let a = FixedTrait::new(HALF_PI / 2, false);
    assert_relative(cos(a), 46341, 'invalid quarter pi', Option::None(())); // 0.55242717280199

    let a = FixedTrait::new(PI, false);
    assert_relative(cos(a), -1 * ONE.into(), 'invalid pi', Option::None(()));

    let a = FixedTrait::new(HALF_PI, true);
    assert_precise(cos(a), 0, 'invalid neg half pi', Option::None(()));

    let a = FixedTrait::new_unscaled(17, false);
    assert_relative(cos(a), -18033, 'invalid 17', Option::None(())); // -0.21497123284870

    let a = FixedTrait::new_unscaled(17, true);
    assert_relative(cos(a), -18033, 'invalid -17', Option::None(())); // -0.21497123284870
}

#[test]
#[available_gas(8000000)]
fn test_cos_fast() {
    let error = Option::Some(84); // 1e-5

    let a = FixedTrait::new(HALF_PI, false);
    assert(cos_fast(a).into() == 0, 'invalid half pi');

    let a = FixedTrait::new(HALF_PI / 2, false);
    assert_precise(cos_fast(a), 46341, 'invalid quarter pi', error); // 0.55242717280199

    let a = FixedTrait::new(PI, false);
    assert_precise(cos_fast(a), -1 * ONE.into(), 'invalid pi', error);

    let a = FixedTrait::new(HALF_PI, true);
    assert_precise(cos(a), 0, 'invalid neg half pi', Option::None(()));

    let a = FixedTrait::new_unscaled(17, false);
    assert_precise(cos_fast(a), -18033, 'invalid 17', error); // -0.21497123284870
}

#[test]
#[available_gas(8000000)]
fn test_sin() {
    let a = FixedTrait::new(HALF_PI, false);
    assert_precise(sin(a), ONE.into(), 'invalid half pi', Option::None(()));

    let a = FixedTrait::new(HALF_PI / 2, false);
    assert_precise(sin(a), 46341, 'invalid quarter pi', Option::None(())); // 0.55242717280199

    let a = FixedTrait::new(PI, false);
    assert(sin(a).into() == 0, 'invalid pi');

    let a = FixedTrait::new(HALF_PI, true);
    assert_precise(
        sin(a), -ONE.into(), 'invalid neg half pi', Option::None(())
    ); // 0.78124999999529

    let a = FixedTrait::new_unscaled(17, false);
    assert_precise(sin(a), -63006, 'invalid 17', Option::None(())); // -0.75109179053073

    let a = FixedTrait::new_unscaled(17, true);
    assert_precise(sin(a), 63006, 'invalid -17', Option::None(())); // 0.75109179053073
}

#[test]
#[available_gas(8000000)]
fn test_sin_fast() {
    let error = Option::Some(84); // 1e-5

    let a = FixedTrait::new(HALF_PI, false);
    assert_precise(sin_fast(a), ONE.into(), 'invalid half pi', error);

    let a = FixedTrait::new(HALF_PI / 2, false);
    assert_precise(sin_fast(a), 46341, 'invalid quarter pi', error); // 0.55242717280199

    let a = FixedTrait::new(PI, false);
    assert(sin_fast(a).into() == 0, 'invalid pi');

    let a = FixedTrait::new(HALF_PI, true);
    assert_precise(sin_fast(a), -ONE.into(), 'invalid neg half pi', error); // 0.78124999999529

    let a = FixedTrait::new_unscaled(17, false);
    assert_precise(sin_fast(a), -63006, 'invalid 17', error); // -0.75109179053073

    let a = FixedTrait::new_unscaled(17, true);
    assert_precise(sin_fast(a), 63006, 'invalid -17', error); // 0.75109179053073
}

#[test]
#[available_gas(8000000)]
fn test_tan() {
    let a = FixedTrait::new(HALF_PI / 2, false);
    assert_precise(tan(a), ONE.into(), 'invalid quarter pi', Option::None(()));

    let a = FixedTrait::new(PI, false);
    assert_precise(tan(a), 0, 'invalid pi', Option::None(()));

    let a = FixedTrait::new_unscaled(17, false);
    assert_precise(tan(a), 228990, 'invalid 17', Option::None(())); // 3.3858731852805

    let a = FixedTrait::new_unscaled(17, true);
    assert_precise(tan(a), -228952, 'invalid -17', Option::None(())); // -3.3858731852805
}
