use super::{F64, NaN, F64Impl, lut, helpers::abs_and_sign, HALF, ONE};

const TWO_PI: i64 = 26986075409;
const PI: i64 = 13493037705;
const HALF_PI: i64 = 6746518852;


pub(crate) fn sin_fast(a: F64) -> F64 {
    if a.d == NaN {
        return F64 { d: NaN };
    }

    let (a_abs, _) = abs_and_sign(a.d);

    let a1 = a_abs.try_into().unwrap() % TWO_PI;
    let whole_rem = a1 / PI;
    let mut partial_rem = a1 % PI;
    let partial_sign = whole_rem == 1;

    if partial_rem >= HALF_PI {
        partial_rem = PI - partial_rem;
    }

    let (start, low, high) = lut::sin(partial_rem);
    let partial_step = (F64 { d: partial_rem } - F64 { d: start }) / F64 { d: 26353589 };
    let res = partial_step * (F64 { d: high } - F64 { d: low }) + F64 { d: low };

    let sign = if a.d < 0 {
        !partial_sign
    } else {
        partial_sign
    };
    F64 { d: if sign {
        -res.d
    } else {
        res.d
    } }
}


// Tests
// --------------------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use orion_numbers::f64::helpers::{assert_precise, assert_relative};
    use super::{F64, F64Impl, PI, HALF_PI, sin_fast};
    use orion_numbers::f64::ONE;

    #[test]
    fn test_sin_fast() {
        let error = Option::Some(42950); // 1e-5

        let a = F64Impl::new(HALF_PI);
        assert_precise(sin_fast(a), ONE.into(), 'invalid half pi', error);

        let a = F64Impl::new(HALF_PI / 2);
        assert_precise(sin_fast(a), 3037000500, 'invalid quarter pi', error); // 0.7071067811865475

        let a = F64Impl::new(PI);
        assert(sin_fast(a).into() == 0, 'invalid pi');

        let a = F64Impl::new(-HALF_PI);
        assert_precise(
            sin_fast(a), -ONE.into(), 'invalid neg half pi', error
        ); // 0.9999999999939766

        let a = F64Impl::new_unscaled(17);
        assert_precise(sin_fast(a), -4129170786, 'invalid 17', error); // -0.9613974918793389

        let a = F64Impl::new_unscaled(-17);
        assert_precise(sin_fast(a), 4129170786, 'invalid -17', error); // 0.9613974918793389
    }
}
