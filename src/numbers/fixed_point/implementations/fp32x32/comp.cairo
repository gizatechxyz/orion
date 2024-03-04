use orion::numbers::{FP32x32, FP32x32Impl, FixedTrait};

fn xor(a: FP32x32, b: FP32x32) -> bool {
    if (a == FixedTrait::new(0, false) || b == FixedTrait::new(0, false)) && (a != b) {
        true
    } else {
        false
    }
}

fn or(a: FP32x32, b: FP32x32) -> bool {
    let zero = FixedTrait::new(0, false);
    if a == zero && b == zero {
        false
    } else {
        true
    }
}

fn and(a: FP32x32, b: FP32x32) -> bool {
    let zero = FixedTrait::new(0, false);
    if a == zero || b == zero {
        false
    } else {
        true
    }
}

fn where(a: FP32x32, b: FP32x32, c: FP32x32) -> FP32x32 {
    if a == FixedTrait::new(0, false) {
        c
    } else {
        b
    }
}

fn bitwise_and(a: FP32x32, b: FP32x32) -> FP32x32 {
    FixedTrait::new(a.mag & b.mag, a.sign & b.sign)
}

fn bitwise_xor(a: FP32x32, b: FP32x32) -> FP32x32 {
    FixedTrait::new(a.mag ^ b.mag, a.sign ^ b.sign)
}

fn bitwise_or(a: FP32x32, b: FP32x32) -> FP32x32 {
    FixedTrait::new(a.mag | b.mag, a.sign | b.sign)
}
