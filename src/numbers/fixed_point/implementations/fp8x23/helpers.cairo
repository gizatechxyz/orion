use debug::PrintTrait;
use traits::Into;

use orion::numbers::fixed_point::implementations::fp8x23::core::{
    HALF, ONE, TWO, FixedType, FP8x23Impl, FP8x23Sub, FP8x23Div, FixedTrait, FP8x23Print
};

const DEFAULT_PRECISION: u32 = 8; // 1e-6

// To use `DEFAULT_PRECISION`, final arg is: `Option::None(())`.
// To use `custom_precision` of 430_u32: `Option::Some(430_u32)`.
fn assert_precise(
    result: FixedType, expected: felt252, msg: felt252, custom_precision: Option<u32>
) {
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

fn assert_relative(
    result: FixedType, expected: felt252, msg: felt252, custom_precision: Option<u32>
) {
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
