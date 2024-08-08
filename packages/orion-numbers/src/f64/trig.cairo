use super::{F64, FixedTrait, F64Impl, ONE, HALF, lut};

// CONSTANTS

pub(crate) const TWO_PI: i64 = 26986075409;
pub(crate) const PI: i64 = 13493037705;
pub(crate) const HALF_PI: i64 = 6746518852;

// PUBLIC

pub(crate) fn acos_fast(a: F64) -> F64 {
    let asin_arg = (FixedTrait::ONE() - a * a).sqrt(); // will fail if a > 1
    let asin_res = asin_fast(asin_arg);

    if a.d < 0 {
        return FixedTrait::new(PI) - asin_res;
    } else {
        return asin_res;
    }
}


pub(crate) fn asin_fast(a: F64) -> F64 {
    if (a.d == ONE) {
        if a.d < 0 {
            return FixedTrait::new(HALF_PI * -ONE);
        } else {
            return FixedTrait::new(HALF_PI);
        }
    }

    let div = (FixedTrait::ONE() - a * a).sqrt(); // will fail if a > 1
    return atan_fast(a / div);
}


pub(crate) fn atan_fast(a: F64) -> F64 {
    let mut at = a.abs();
    let mut shift = false;
    let mut invert = false;

    // Invert value when a > 1
    if at.d > ONE {
        at = FixedTrait::ONE() / at;
        invert = true;
    }

    // Account for lack of precision in polynomial when a > 0.7
    if at.d > 3006477107 {
        let sqrt3_3 = FixedTrait::new(2479700525); // sqrt(3) / 3
        at = (at - sqrt3_3) / (FixedTrait::ONE() + at * sqrt3_3);
        shift = true;
    }

    let (start, low, high) = lut::atan(at.d);
    let partial_step = FixedTrait::new(at.d - start) / FixedTrait::new(30064771);
    let mut res = partial_step * FixedTrait::new(high - low) + FixedTrait::new(low);

    // Adjust for sign change, inversion, and shift
    if shift {
        res = res + FixedTrait::new(2248839617); // pi / 6
    }

    if invert {
        res = FixedTrait::new(HALF_PI) - res;
    }

    if a.d < 0 {
        res = -res;
    }

    res
}

pub(crate) fn cos_fast(a: F64) -> F64 {
    return sin_fast(FixedTrait::new(HALF_PI) - a);
}

pub(crate) fn sin_fast(a: F64) -> F64 {
    let a1 = a.d % TWO_PI;

    let mut partial_rem = Rem::rem(a1, PI);

    if partial_rem >= HALF_PI {
        partial_rem = PI - partial_rem;
    }

    let (start, low, high) = lut::sin(partial_rem);
    let partial_step = (FixedTrait::new(partial_rem) - FixedTrait::new(start))
        / FixedTrait::new(26353589);
    let res = partial_step * (FixedTrait::new(high) - FixedTrait::new(low)) + FixedTrait::new(low);

    return res;
}

pub(crate) fn tan_fast(a: F64) -> F64 {
    let sinx = sin_fast(a);
    let cosx = cos_fast(a);
    assert(cosx.d != 0, 'tan undefined');
    return sinx / cosx;
}

// Tests
// --------------------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use orion_numbers::f64::helpers::{assert_precise, assert_relative};

    use super::{FixedTrait, F64Impl, ONE, HALF_PI, PI, acos_fast, atan_fast, cos_fast, sin_fast};


    #[test]
    #[available_gas(3000000)]
    fn test_acos_fast() {
        let error = Option::Some(42950); // 1e-5

        let a = FixedTrait::ONE();
        assert(acos_fast(a).into() == 0, 'invalid one');

        let a = FixedTrait::new(ONE / 2);
        assert_relative(acos_fast(a), 4497679235, 'invalid half', error); // 1.0471975506263043

        let a = FixedTrait::ZERO();
        assert_relative(acos_fast(a), HALF_PI.into(), 'invalid zero', Option::None(())); // PI / 2

        let a = FixedTrait::new(-ONE / 2);
        assert_relative(acos_fast(a), 8995358470, 'invalid neg half', error); // 2.094395102963489

        let a = FixedTrait::new(-ONE);
        assert_relative(acos_fast(a), PI.into(), 'invalid neg one', Option::None(())); // PI
    }

    // TODO: Fix it
    // #[test]
    // #[available_gas(1400000)]
    // fn test_atan_fast() {
    //     let error = Option::Some(42950); // 1e-5

    //     let a = FixedTrait::new(2 * ONE);
    //     assert_relative(atan_fast(a), 4755167535, 'invalid two', error);

    //     let a = FixedTrait::ONE();
    //     assert_relative(atan_fast(a), 3373259426, 'invalid one', error);

    //     let a = FixedTrait::new(ONE / 2);
    //     assert_relative(atan_fast(a), 1991351318, 'invalid half', error);

    //     let a = FixedTrait::ZERO();
    //     assert(atan_fast(a).into() == 0, 'invalid zero');

    //     let a = FixedTrait::new(-ONE / 2);
    //     assert_relative(atan_fast(a), -1991351318, 'invalid neg half', error);

    //     let a = FixedTrait::new(-ONE);
    //     assert_relative(atan_fast(a), -3373259426, 'invalid neg one', error);

    //     let a = FixedTrait::new(2 * -ONE);
    //     assert_relative(atan_fast(a), -4755167535, 'invalid neg two', error);
    // }

    // TODO: Fix it
    // #[test]
    // #[available_gas(6000000)]
    // fn test_cos_fast() {
    //     let error = Option::Some(42950); // 1e-5

    //     let a = FixedTrait::new(HALF_PI);
    //     assert(cos_fast(a).into() == 0, 'invalid half pi');

    //     let a = FixedTrait::new(HALF_PI / 2);
    //     assert_precise(cos_fast(a), 3037000500, 'invalid quarter pi', error); //
    //     0.7071067811865475

    //     let a = FixedTrait::new(PI);
    //     assert_precise(cos_fast(a), -1 * ONE.into(), 'invalid pi', error);

    //     let a = FixedTrait::new(-HALF_PI);
    //     assert_precise(cos_fast(a), 0, 'invalid neg half pi', Option::None(()));

    //     let a = FixedTrait::new_unscaled(17);
    //     assert_precise(cos_fast(a), -1181817538, 'invalid 17', error); // -0.2751631780463348

    //     let a = FixedTrait::new(-5143574028060);
    //     assert_precise(cos_fast(a), -3458137149, 'invalid theta', error);
    // }


    // TODO: Fix it
    // #[test]
    // #[available_gas(1000000)]
    // fn test_sin_fast() {
    //     let error = Option::Some(42950); // 1e-5

    //     let a = FixedTrait::new(HALF_PI);
    //     assert_precise(sin_fast(a), ONE.into(), 'invalid half pi', error);

    //     let a = FixedTrait::new(HALF_PI / 2);
    //     assert_precise(sin_fast(a), 3037000500, 'invalid quarter pi', error); // 0.7071067811865475

    //     let a = FixedTrait::new(PI);
    //     assert(sin_fast(a).into() == 0, 'invalid pi');

    //     let a = FixedTrait::new(-HALF_PI);
    //     assert_precise(
    //         sin_fast(a), -ONE.into(), 'invalid neg half pi', error
    //     ); // 0.9999999999939766

    //     let a = FixedTrait::new_unscaled(17);
    //     assert_precise(sin_fast(a), -4129170786, 'invalid 17', error); // -0.9613974918793389

    //     let a = FixedTrait::new_unscaled(-17);
    //     assert_precise(sin_fast(a), 4129170786, 'invalid -17', error); // 0.9613974918793389
    // }
}
