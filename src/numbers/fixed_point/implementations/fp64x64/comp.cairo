use orion::numbers::{FP64x64, FixedTrait};
use orion::numbers::FP64x64Impl;

fn xor(a: FP64x64, b: FP64x64) -> bool {
    if (a == FixedTrait::new(0, false) || b == FixedTrait::new(0, false)) && (a != b) {
        return true;
    } else {
        return false;
    }
}

fn or(a: FP64x64, b: FP64x64) -> bool {
    let zero = FixedTrait::new(0, false);
    if a == zero && b == zero {
        return false;
    } else {
        return true;
    }
}

fn and(a: FP64x64, b: FP64x64) -> bool {
    let zero = FixedTrait::new(0, false);
    if a == zero || b == zero {
        return false;
    } else {
        return true;
    }
}

fn where(a: FP64x64, b: FP64x64, c: FP64x64) -> FP64x64 {
    if a == FixedTrait::new(0, false) {
        return c;
    } else {
        return b;
    }
}

fn bitwise_and(a: FP64x64, b: FP64x64) -> FP64x64 {
    return FixedTrait::new(a.mag & b.mag, a.sign & b.sign);
}
