use orion_numbers::f16x16::core::{F16x16Impl, f16x16, ONE, HALF};

const DEFAULT_PRECISION: i32 = 7; // 1e-4

// To use `DEFAULT_PRECISION`, final arg is: `Option::None(())`.
// To use `custom_precision` of 430_i32: `Option::Some(430_i32)`.
pub fn assert_precise(
    result: f16x16, expected: felt252, msg: felt252, custom_precision: Option<i32>
) {
    let precision = match custom_precision {
        Option::Some(val) => val,
        Option::None => DEFAULT_PRECISION,
    };

    let diff = (result - F16x16Impl::from_felt(expected));

    if (diff > precision) {
        //println!("{}", result);
        assert(diff <= precision, msg);
    }
}

pub fn assert_relative(
    result: f16x16, expected: felt252, msg: felt252, custom_precision: Option<i32>
) {
    let precision = match custom_precision {
        Option::Some(val) => val,
        Option::None => DEFAULT_PRECISION,
    };

    let diff = result - F16x16Impl::from_felt(expected);
    let rel_diff = diff / result;

    if (rel_diff > precision) {
        //println!("{}", result);
        assert(rel_diff <= precision, msg);
    }
}

