use debug::PrintTrait;
use traits::Into;

use orion::numbers::fixed_point::implementations::fp16x16wide::core::{
    HALF, ONE, TWO, FP16x16W, FP16x16WImpl, FP16x16WSub, FP16x16WDiv, FixedTrait, FP16x16WPrint
};

const DEFAULT_PRECISION: u64 = 7; // 1e-4

// To use `DEFAULT_PRECISION`, final arg is: `Option::None(())`.
// To use `custom_precision` of 430_u32: `Option::Some(430_u32)`.
fn assert_precise(result: FP16x16W, expected: felt252, msg: felt252, custom_precision: Option<u64>) {
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
    result: FP16x16W, expected: felt252, msg: felt252, custom_precision: Option<u64>
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
