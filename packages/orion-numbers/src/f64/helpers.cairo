use super::{F64Impl, F64, FixedTrait};

const DEFAULT_PRECISION: i64 = 430; // 1e-7

// To use `DEFAULT_PRECISION`, final arg is: `Option::None(())`.
// To use `custom_precision` of 430_i64: `Option::Some(430_i64)`.
pub(crate) fn assert_precise(
    result: F64, expected: felt252, msg: felt252, custom_precision: Option<i64>
) {
    let precision = match custom_precision {
        Option::Some(val) => val,
        Option::None(_) => DEFAULT_PRECISION,
    };

    let diff = (result - FixedTrait::from_felt(expected)).d;

    if (diff > precision) {
        assert(diff <= precision, msg);
    }
}

pub(crate) fn assert_relative(
    result: F64, expected: felt252, msg: felt252, custom_precision: Option<i64>
) {
    let precision = match custom_precision {
        Option::Some(val) => val,
        Option::None(_) => DEFAULT_PRECISION,
    };

    let diff = result - FixedTrait::from_felt(expected);
    let rel_diff = (diff / result).d;

    if (rel_diff > precision) {
        assert(rel_diff <= precision, msg);
    }
}
