pub fn sign_i128(a: i128) -> i128 {
    if a == 0 {
        0
    } else if a > 0 {
        1
    } else {
        -1
    }
}

pub fn sign_i32(a: i32) -> i32 {
    if a == 0 {
        0
    } else if a > 0 {
        1
    } else {
        -1
    }
}

pub fn sign_i64(a: i64) -> i64 {
    if a == 0 {
        0
    } else if a > 0 {
        1
    } else {
        -1
    }
}
