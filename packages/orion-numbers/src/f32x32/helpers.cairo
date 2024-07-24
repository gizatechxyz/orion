use orion_numbers::f32x32::core::{F32x32Impl, f32x32, ONE, HALF};

const DEFAULT_PRECISION: i64 = 429497; // 1e-4

// To use `DEFAULT_PRECISION`, final arg is: `Option::None(())`.
// To use `custom_precision` of 430_i64: `Option::Some(430_i64)`.
pub fn assert_precise(
    result: f32x32, expected: felt252, msg: felt252, custom_precision: Option<i64>
) {
    let precision = match custom_precision {
        Option::Some(val) => val,
        Option::None => DEFAULT_PRECISION,
    };

    let diff = (result - F32x32Impl::from_felt(expected));

    if (diff > precision) {
        //println!("{}", result);
        assert(diff <= precision, msg);
    }
}

pub fn assert_relative(
    result: f32x32, expected: felt252, msg: felt252, custom_precision: Option<i64>
) {
    let precision = match custom_precision {
        Option::Some(val) => val,
        Option::None => DEFAULT_PRECISION,
    };

    let diff = result - F32x32Impl::from_felt(expected);
    let rel_diff = diff / result;

    if (rel_diff > precision) {
        //println!("{}", result);
        assert(rel_diff <= precision, msg);
    }
}

