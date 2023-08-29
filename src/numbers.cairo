mod fixed_point;
mod signed_integer;

// Common methods from Fixed Point and Signed Integers.
trait NumberTrait<T> {
    fn zero() -> T;
    fn is_zero(self: T) -> bool;

    fn one() -> T;
    fn is_one(self: T) -> bool;

    fn abs(self: T) -> T;
}

use orion::numbers::fixed_point::implementations::fp8x23::core::{FP8x23Impl, FP8x23};
use orion::numbers::fixed_point::implementations::fp8x23::math::core as core_fp8x23;

impl FP8x23Number of NumberTrait<FP8x23> {
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
}

use orion::numbers::fixed_point::implementations::fp16x16::core::{FP16x16Impl, FP16x16};
use orion::numbers::fixed_point::implementations::fp16x16::math::core as core_fp16x16;

impl FP16x16Number of NumberTrait<FP16x16> {
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
}

use orion::numbers::signed_integer::i8 as i8_core;
use orion::numbers::signed_integer::i8::i8;

impl I8Number of NumberTrait<i8> {
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
}

use orion::numbers::signed_integer::i16 as i16_core;
use orion::numbers::signed_integer::i16::i16;

impl i16Number of NumberTrait<i16> {
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
}

use orion::numbers::signed_integer::i32 as i32_core;
use orion::numbers::signed_integer::i32::i32;

impl i32Number of NumberTrait<i32> {
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
}

use orion::numbers::signed_integer::i64 as i64_core;
use orion::numbers::signed_integer::i64::i64;

impl i64Number of NumberTrait<i64> {
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
}

use orion::numbers::signed_integer::i128 as i128_core;
use orion::numbers::signed_integer::i128::i128;

impl i128Number of NumberTrait<i128> {
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
}

impl u32Number of NumberTrait<u32> {
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
}
