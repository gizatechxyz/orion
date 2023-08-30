mod fixed_point;
mod signed_integer;

// Common methods from Fixed Point and Signed Integers.
trait NumberTrait<T, MAG> {
    fn zero() -> T;
    fn is_zero(self: T) -> bool;

    fn one() -> T;
    fn is_one(self: T) -> bool;

    fn abs(self: T) -> T;

    fn min_value() -> T;
    fn max_value() -> T;

    fn min(self: T, other: T) -> T;
    fn max(self: T, other: T) -> T;

    fn mag(self: T) -> MAG;
    fn is_neg(self: T) -> bool;
}

use orion::numbers::fixed_point::implementations::fp8x23::core::{FP8x23Impl, FP8x23};
use orion::numbers::fixed_point::implementations::fp8x23::math::core as core_fp8x23;
use orion::numbers::fixed_point::implementations::fp8x23::math::comp as comp_fp8x23;

impl FP8x23Number of NumberTrait<FP8x23, u32> {
    fn zero() -> FP8x23 {
        FP8x23Impl::ZERO()
    }
    fn is_zero(self: FP8x23) -> bool {
        core_fp8x23::eq(@self, @FP8x23Impl::ZERO())
    }

    fn one() -> FP8x23 {
        FP8x23Impl::ONE()
    }
    fn is_one(self: FP8x23) -> bool {
        core_fp8x23::eq(@self, @FP8x23Impl::ONE())
    }

    fn abs(self: FP8x23) -> FP8x23 {
        core_fp8x23::abs(self)
    }

    fn min_value() -> FP8x23 {
        FP8x23 { mag: core_fp8x23::MAX, sign: true }
    }

    fn max_value() -> FP8x23 {
        FP8x23 { mag: core_fp8x23::MAX, sign: false }
    }

    fn min(self: FP8x23, other: FP8x23) -> FP8x23 {
        comp_fp8x23::min(self, other)
    }

    fn max(self: FP8x23, other: FP8x23) -> FP8x23 {
        comp_fp8x23::max(self, other)
    }

    fn mag(self: FP8x23) -> u32 {
        self.mag
    }

    fn is_neg(self: FP8x23) -> bool {
        self.sign
    }
}

use orion::numbers::fixed_point::implementations::fp16x16::core::{FP16x16Impl, FP16x16};
use orion::numbers::fixed_point::implementations::fp16x16::math::core as core_fp16x16;
use orion::numbers::fixed_point::implementations::fp16x16::math::comp as comp_fp16x16;

impl FP16x16Number of NumberTrait<FP16x16, u32> {
    fn zero() -> FP16x16 {
        FP16x16Impl::ZERO()
    }
    fn is_zero(self: FP16x16) -> bool {
        core_fp16x16::eq(@self, @FP16x16Impl::ZERO())
    }

    fn one() -> FP16x16 {
        FP16x16Impl::ONE()
    }
    fn is_one(self: FP16x16) -> bool {
        core_fp16x16::eq(@self, @FP16x16Impl::ONE())
    }

    fn abs(self: FP16x16) -> FP16x16 {
        core_fp16x16::abs(self)
    }

    fn min_value() -> FP16x16 {
        FP16x16 { mag: core_fp16x16::MAX, sign: true }
    }

    fn max_value() -> FP16x16 {
        FP16x16 { mag: core_fp16x16::MAX, sign: false }
    }

    fn min(self: FP16x16, other: FP16x16) -> FP16x16 {
        comp_fp16x16::min(self, other)
    }

    fn max(self: FP16x16, other: FP16x16) -> FP16x16 {
        comp_fp16x16::max(self, other)
    }

    fn mag(self: FP16x16) -> u32 {
        self.mag
    }

    fn is_neg(self: FP16x16) -> bool {
        self.sign
    }
}

use orion::numbers::signed_integer::i8 as i8_core;
use orion::numbers::signed_integer::i8::i8;

impl I8Number of NumberTrait<i8, u8> {
    fn zero() -> i8 {
        i8 { mag: 0, sign: false }
    }
    fn is_zero(self: i8) -> bool {
        i8_core::i8_eq(self, i8 { mag: 0, sign: false })
    }

    fn one() -> i8 {
        i8 { mag: 1, sign: false }
    }
    fn is_one(self: i8) -> bool {
        i8_core::i8_eq(self, i8 { mag: 1, sign: false })
    }

    fn abs(self: i8) -> i8 {
        i8_core::i8_abs(self)
    }

    fn min_value() -> i8 {
        i8 { mag: 128, sign: true }
    }

    fn max_value() -> i8 {
        i8 { mag: 127, sign: false }
    }

    fn min(self: i8, other: i8) -> i8 {
        i8_core::i8_min(self, other)
    }

    fn max(self: i8, other: i8) -> i8 {
        i8_core::i8_max(self, other)
    }

    fn mag(self: i8) -> u8 {
        self.mag
    }

    fn is_neg(self: i8) -> bool {
        self.sign
    }
}

use orion::numbers::signed_integer::i16 as i16_core;
use orion::numbers::signed_integer::i16::i16;

impl i16Number of NumberTrait<i16, u16> {
    fn zero() -> i16 {
        i16 { mag: 0, sign: false }
    }
    fn is_zero(self: i16) -> bool {
        i16_core::i16_eq(self, i16 { mag: 0, sign: false })
    }

    fn one() -> i16 {
        i16 { mag: 1, sign: false }
    }
    fn is_one(self: i16) -> bool {
        i16_core::i16_eq(self, i16 { mag: 1, sign: false })
    }

    fn abs(self: i16) -> i16 {
        i16_core::i16_abs(self)
    }

    fn min_value() -> i16 {
        i16 { mag: 32768, sign: true }
    }

    fn max_value() -> i16 {
        i16 { mag: 32767, sign: false }
    }

    fn min(self: i16, other: i16) -> i16 {
        i16_core::i16_min(self, other)
    }

    fn max(self: i16, other: i16) -> i16 {
        i16_core::i16_max(self, other)
    }

    fn mag(self: i16) -> u16 {
        self.mag
    }

    fn is_neg(self: i16) -> bool {
        self.sign
    }
}

use orion::numbers::signed_integer::i32 as i32_core;
use orion::numbers::signed_integer::i32::i32;

impl i32Number of NumberTrait<i32, u32> {
    fn zero() -> i32 {
        i32 { mag: 0, sign: false }
    }
    fn is_zero(self: i32) -> bool {
        i32_core::i32_eq(self, i32 { mag: 0, sign: false })
    }

    fn one() -> i32 {
        i32 { mag: 1, sign: false }
    }
    fn is_one(self: i32) -> bool {
        i32_core::i32_eq(self, i32 { mag: 1, sign: false })
    }

    fn abs(self: i32) -> i32 {
        i32_core::i32_abs(self)
    }

    fn min_value() -> i32 {
        i32 { mag: 2147483648, sign: true }
    }

    fn max_value() -> i32 {
        i32 { mag: 2147483647, sign: false }
    }

    fn min(self: i32, other: i32) -> i32 {
        i32_core::i32_min(self, other)
    }

    fn max(self: i32, other: i32) -> i32 {
        i32_core::i32_max(self, other)
    }

    fn mag(self: i32) -> u32 {
        self.mag
    }

    fn is_neg(self: i32) -> bool {
        self.sign
    }
}

use orion::numbers::signed_integer::i64 as i64_core;
use orion::numbers::signed_integer::i64::i64;

impl i64Number of NumberTrait<i64, u64> {
    fn zero() -> i64 {
        i64 { mag: 0, sign: false }
    }
    fn is_zero(self: i64) -> bool {
        i64_core::i64_eq(self, i64 { mag: 0, sign: false })
    }

    fn one() -> i64 {
        i64 { mag: 1, sign: false }
    }
    fn is_one(self: i64) -> bool {
        i64_core::i64_eq(self, i64 { mag: 1, sign: false })
    }

    fn abs(self: i64) -> i64 {
        i64_core::i64_abs(self)
    }

    fn min_value() -> i64 {
        i64 { mag: 9223372036854775808, sign: true }
    }

    fn max_value() -> i64 {
        i64 { mag: 9223372036854775807, sign: false }
    }

    fn min(self: i64, other: i64) -> i64 {
        i64_core::i64_min(self, other)
    }

    fn max(self: i64, other: i64) -> i64 {
        i64_core::i64_max(self, other)
    }

    fn mag(self: i64) -> u64 {
        self.mag
    }

    fn is_neg(self: i64) -> bool {
        self.sign
    }
}

use orion::numbers::signed_integer::i128 as i128_core;
use orion::numbers::signed_integer::i128::i128;

impl i128Number of NumberTrait<i128, u128> {
    fn zero() -> i128 {
        i128 { mag: 0, sign: false }
    }
    fn is_zero(self: i128) -> bool {
        i128_core::i128_eq(self, i128 { mag: 0, sign: false })
    }

    fn one() -> i128 {
        i128 { mag: 1, sign: false }
    }
    fn is_one(self: i128) -> bool {
        i128_core::i128_eq(self, i128 { mag: 1, sign: false })
    }

    fn abs(self: i128) -> i128 {
        i128_core::i128_abs(self)
    }

    fn min_value() -> i128 {
        i128 { mag: 170141183460469231731687303715884105728, sign: true }
    }

    fn max_value() -> i128 {
        i128 { mag: 170141183460469231731687303715884105727, sign: false }
    }

    fn min(self: i128, other: i128) -> i128 {
        i128_core::i128_min(self, other)
    }

    fn max(self: i128, other: i128) -> i128 {
        i128_core::i128_max(self, other)
    }

    fn mag(self: i128) -> u128 {
        self.mag
    }

    fn is_neg(self: i128) -> bool {
        self.sign
    }
}

impl u32Number of NumberTrait<u32, u32> {
    fn zero() -> u32 {
        0
    }
    fn is_zero(self: u32) -> bool {
        self == 0
    }

    fn one() -> u32 {
        1
    }
    fn is_one(self: u32) -> bool {
        self == 1
    }

    fn abs(self: u32) -> u32 {
        self
    }

    fn min_value() -> u32 {
        0
    }

    fn max_value() -> u32 {
        4294967295
    }

    fn min(self: u32, other: u32) -> u32 {
        if self < other {
            return self;
        } else {
            other
        }
    }

    fn max(self: u32, other: u32) -> u32 {
        if self > other {
            return self;
        } else {
            other
        }
    }

    fn mag(self: u32) -> u32 {
        self
    }

    fn is_neg(self: u32) -> bool {
        false
    }
}
