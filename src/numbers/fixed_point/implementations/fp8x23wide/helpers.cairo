use debug::PrintTrait;
use traits::Into;

use orion::numbers::fixed_point::implementations::fp8x23wide::core::{
    HALF, ONE, TWO, FP8x23W, FP8x23WSub, FP8x23WDiv, FixedTrait, FP8x23WPrint
};

const DEFAULT_PRECISION: u64 = 8; // 1e-6

// To use `DEFAULT_PRECISION`, final arg is: `Option::None(())`.
// To use `custom_precision` of 430_u64: `Option::Some(430_u64)`.
fn assert_precise(result: FP8x23W, expected: felt252, msg: felt252, custom_precision: Option<u64>) {
    let precision = match custom_precision {
        Option::Some(val) => val,
        Option::None(_) => DEFAULT_PRECISION,
    };

    let diff = (result - FixedTrait::from_felt(expected)).mag;

    if (diff > precision) {
        result.print();
        assert(diff <= precision, msg);
    }
}

fn assert_relative(result: FP8x23W, expected: felt252, msg: felt252, custom_precision: Option<u64>) {
    let precision = match custom_precision {
        Option::Some(val) => val,
        Option::None(_) => DEFAULT_PRECISION,
    };

    let diff = result - FixedTrait::from_felt(expected);
    let rel_diff = (diff / result).mag;

    if (rel_diff > precision) {
        result.print();
        assert(rel_diff <= precision, msg);
    }
}
