use orion::numbers::{FP32x32, FixedTrait};
use orion::numbers::FP32x32Impl;

fn xor(a: FP32x32, b: FP32x32) -> bool {
    if (a == FixedTrait::new(0, false) || b == FixedTrait::new(0, false)) && (a != b) {
        return true;
    } else {
        return false;
    }
}

fn or(a: FP32x32, b: FP32x32) -> bool {
    let zero = FixedTrait::new(0, false);
    if a == zero && b == zero {
        return false;
    } else {
        return true;
    }
}
