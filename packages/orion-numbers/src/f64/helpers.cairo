use super::{F64Impl, F64, FixedTrait};

const DEFAULT_PRECISION: i64 = 430; // 1e-7

// To use `DEFAULT_PRECISION`, final arg is: `Option::None(())`.
// To use `custom_precision` of 430_i64: `Option::Some(430_i64)`.
pub fn assert_precise(result: F64, expected: felt252, msg: felt252, custom_precision: Option<i64>) {
    let precision = match custom_precision {
        Option::Some(val) => val,
        Option::None(_) => DEFAULT_PRECISION,
    };

    let diff = (result - FixedTrait::from_felt(expected)).d;

    if (diff > precision) {
        assert(diff <= precision, msg);
    }
}

pub fn assert_precise_span(
    results: Span<F64>, expected: Span<felt252>, msg: felt252, custom_precision: Option<i64>
) {
    assert(results.len() == expected.len(), 'Arrays must have same length');

    let mut i: usize = 0;
    loop {
        if i == results.len() {
            break;
        }

        let result = *results.at(i);
        let expected_val = *expected.at(i);

        assert_precise(result, expected_val, msg, custom_precision);

        i += 1;
    }
}

pub fn assert_relative(
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

pub fn assert_relative_span(
    results: Span<F64>, expected: Span<felt252>, msg: felt252, custom_precision: Option<i64>
) {
    assert(results.len() == expected.len(), 'Arrays must have same length');

    let mut i: usize = 0;
    loop {
        if i == results.len() {
            break;
        }

        let result = *results.at(i);
        let expected_val = *expected.at(i);

        assert_relative(result, expected_val, msg, custom_precision);

        i += 1;
    }
}
