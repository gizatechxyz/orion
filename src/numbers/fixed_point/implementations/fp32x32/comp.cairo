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

fn and(a: FP32x32, b: FP32x32) -> bool {
    let zero = FixedTrait::new(0, false);
    if a == zero || b == zero {
        return false;
    } else {
        return true;
    }
}

fn where(a: FP32x32, b: FP32x32, c: FP32x32) -> FP32x32 {
    if a == FixedTrait::new(0, false) {
        return c;
    } else {
        return b;
    }
}

fn bitwise_and(a: FP32x32, b: FP32x32) -> FP32x32 {
    return FixedTrait::new(a.mag & b.mag, a.sign & b.sign);
}
