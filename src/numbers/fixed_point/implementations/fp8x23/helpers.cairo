use core::debug::PrintTrait;

use orion::numbers::fixed_point::implementations::fp8x23::core::{
    HALF, ONE, TWO, FP8x23, FP8x23Sub, FP8x23Div, FixedTrait, FP8x23Print
};

const DEFAULT_PRECISION: u32 = 8; // 1e-6

// To use `DEFAULT_PRECISION`, final arg is: `Option::None(())`.
// To use `custom_precision` of 430_u32: `Option::Some(430_u32)`.
fn assert_precise(result: FP8x23, expected: felt252, msg: felt252, custom_precision: Option<u32>) {
    let precision = match custom_precision {
        Option::Some(val) => val,
        Option::None => DEFAULT_PRECISION,
    };

    let diff = (result - FixedTrait::from_felt(expected)).mag;

    if (diff > precision) {
        result.print();
        assert(diff <= precision, msg);
    }
}

fn assert_relative(result: FP8x23, expected: felt252, msg: felt252, custom_precision: Option<u32>) {
    let precision = match custom_precision {
        Option::Some(val) => val,
        Option::None => DEFAULT_PRECISION,
    };

    let diff = result - FixedTrait::from_felt(expected);
    let rel_diff = (diff / result).mag;

    if (rel_diff > precision) {
        result.print();
        assert(rel_diff <= precision, msg);
    }
}
