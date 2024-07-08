// Basic Arithmetic Trait on integer 32, 64 and 128, should be included in Cairo core soon.

pub impl I32Div of Div<i32> {
    fn div(lhs: i32, rhs: i32) -> i32 {
        assert(rhs != 0, 'divisor cannot be 0');

        let mut lhs_positive = lhs;
        let mut rhs_positive = rhs;

        if lhs < 0 {
            lhs_positive = lhs * -1;
        }
        if rhs < 0 {
            rhs_positive = rhs * -1;
        }

        let lhs_u32: u32 = lhs_positive.try_into().unwrap();
        let rhs_u32: u32 = rhs_positive.try_into().unwrap();

        let mut result = lhs_u32 / rhs_u32;
        let felt_result: felt252 = result.into();
        let signed_int_result: i32 = felt_result.try_into().unwrap();

        // avoids mul overflow for f16x16
        if sign_i32(lhs) * rhs < 0 {
            signed_int_result * -1
        } else {
            signed_int_result
        }
    }
}

pub impl I32Rem of Rem<i32> {
    fn rem(lhs: i32, rhs: i32) -> i32 {
        let div = Div::div(lhs, rhs);
        lhs - rhs * div
    }
}


pub impl I64Div of Div<i64> {
    fn div(lhs: i64, rhs: i64) -> i64 {
        assert(rhs != 0, 'divisor cannot be 0');

        let mut lhs_positive = lhs;
        let mut rhs_positive = rhs;

        if lhs < 0 {
            lhs_positive = lhs * -1;
        }
        if rhs < 0 {
            rhs_positive = rhs * -1;
        }

        let lhs_u64: u64 = lhs_positive.try_into().unwrap();
        let rhs_u64: u64 = rhs_positive.try_into().unwrap();

        let mut result = lhs_u64 / rhs_u64;
        let felt_result: felt252 = result.into();
        let signed_int_result: i64 = felt_result.try_into().unwrap();

        // avoids mul overflow for f16x16
        if sign_i64(lhs) * rhs < 0 {
            signed_int_result * -1
        } else {
            signed_int_result
        }
    }
}

pub impl I64Rem of Rem<i64> {
    fn rem(lhs: i64, rhs: i64) -> i64 {
        let div = Div::div(lhs, rhs);
        lhs - rhs * div
    }
}

pub impl I128Div of Div<i128> {
    fn div(lhs: i128, rhs: i128) -> i128 {
        assert(rhs != 0, 'divisor cannot be 0');

        let mut lhs_positive = lhs;
        let mut rhs_positive = rhs;

        if lhs < 0 {
            lhs_positive = lhs * -1;
        }
        if rhs < 0 {
            rhs_positive = rhs * -1;
        }

        let lhs_u128: u128 = lhs_positive.try_into().unwrap();
        let rhs_u128: u128 = rhs_positive.try_into().unwrap();

        let mut result = lhs_u128 / rhs_u128;
        let felt_result: felt252 = result.into();
        let signed_int_result: i128 = felt_result.try_into().unwrap();

        // avoids mul overflow for f16x16
        if sign_i128(lhs) * rhs < 0 {
            signed_int_result * -1
        } else {
            signed_int_result
        }
    }
}

pub impl I128Rem of Rem<i128> {
    fn rem(lhs: i128, rhs: i128) -> i128 {
        let div = Div::div(lhs, rhs);
        lhs - rhs * div
    }
}


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
