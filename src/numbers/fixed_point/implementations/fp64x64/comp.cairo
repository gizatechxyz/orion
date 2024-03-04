use orion::numbers::{FP64x64, FixedTrait};
use orion::numbers::FP64x64Impl;

fn xor(a: FP64x64, b: FP64x64) -> bool {
    if (a == FixedTrait::new(0, false) || b == FixedTrait::new(0, false)) && (a != b) {
        true
    } else {
        false
    }
}

fn or(a: FP64x64, b: FP64x64) -> bool {
    let zero = FixedTrait::new(0, false);
    if a == zero && b == zero {
        false
    } else {
        true
    }
}

fn and(a: FP64x64, b: FP64x64) -> bool {
    let zero = FixedTrait::new(0, false);
    if a == zero || b == zero {
        false
    } else {
        true
    }
}

fn where(a: FP64x64, b: FP64x64, c: FP64x64) -> FP64x64 {
    if a == FixedTrait::new(0, false) {
        c
    } else {
        b
    }
}

fn bitwise_and(a: FP64x64, b: FP64x64) -> FP64x64 {
    FixedTrait::new(a.mag & b.mag, a.sign & b.sign)
}

fn bitwise_xor(a: FP64x64, b: FP64x64) -> FP64x64 {
    FixedTrait::new(a.mag ^ b.mag, a.sign ^ b.sign)
}

fn bitwise_or(a: FP64x64, b: FP64x64) -> FP64x64 {
    FixedTrait::new(a.mag | b.mag, a.sign | b.sign)
}
