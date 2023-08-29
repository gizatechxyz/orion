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

use orion::numbers::fixed_point::core::FixedType;
use orion::numbers::fixed_point::implementations::fp8x23::core::FP8x23Impl;
use orion::numbers::fixed_point::implementations::fp8x23::math::core as core_fp8x23;

impl FP8x23Number of NumberTrait<FixedType> {
    fn zero() -> FixedType {
        FP8x23Impl::ZERO()
    }
    fn is_zero(self: FixedType) -> bool {
        core_fp8x23::eq(@self, @FP8x23Impl::ZERO())
    }

    fn one() -> FixedType {
        FP8x23Impl::ONE()
    }
    fn is_one(self: FixedType) -> bool {
        core_fp8x23::eq(@self, @FP8x23Impl::ONE())
    }

    fn abs(self: FixedType) -> FixedType {
        core_fp8x23::abs(self)
    }
}

use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16Impl;
use orion::numbers::fixed_point::implementations::fp16x16::math::core as core_fp16x16;

impl FP16x16Number of NumberTrait<FixedType> {
    fn zero() -> FixedType {
        FP16x16Impl::ZERO()
    }
    fn is_zero(self: FixedType) -> bool {
        core_fp16x16::eq(@self, @FP16x16Impl::ZERO())
    }

    fn one() -> FixedType {
        FP16x16Impl::ONE()
    }
    fn is_one(self: FixedType) -> bool {
        core_fp16x16::eq(@self, @FP16x16Impl::ONE())
    }

    fn abs(self: FixedType) -> FixedType {
        core_fp16x16::abs(self)
    }
}

use orion::numbers::signed_integer::i8 as i8_core;
use orion::numbers::signed_integer::i8::i8;

impl I8Number of NumberTrait<i8> {
    fn zero() -> i8 {
        i8 {mag: 0, sign: false}
    }
    fn is_zero(self: i8) -> bool {
        i8_core::i8_eq(self, i8 {mag: 0, sign: false})
    }

    fn one() -> i8 {
        i8 {mag: 1, sign: false}
    }
    fn is_one(self: i8) -> bool {
        i8_core::i8_eq(self, i8 {mag: 1, sign: false})
    }

    fn abs(self: i8) -> i8 {
        i8_core::i8_abs(self)
    }
}

use orion::numbers::signed_integer::i16 as i16_core;
use orion::numbers::signed_integer::i16::i16;

impl i16Number of NumberTrait<i16> {
    fn zero() -> i16 {
        i16 {mag: 0, sign: false}
    }
    fn is_zero(self: i16) -> bool {
        i16_core::i16_eq(self, i16 {mag: 0, sign: false})
    }

    fn one() -> i16 {
        i16 {mag: 1, sign: false}
    }
    fn is_one(self: i16) -> bool {
        i16_core::i16_eq(self, i16 {mag: 1, sign: false})
    }

    fn abs(self: i16) -> i16 {
        i16_core::i16_abs(self)
    }
}

use orion::numbers::signed_integer::i32 as i32_core;
use orion::numbers::signed_integer::i32::i32;

impl i32Number of NumberTrait<i32> {
    fn zero() -> i32 {
        i32 {mag: 0, sign: false}
    }
    fn is_zero(self: i32) -> bool {
        i32_core::i32_eq(self, i32 {mag: 0, sign: false})
    }

    fn one() -> i32 {
        i32 {mag: 1, sign: false}
    }
    fn is_one(self: i32) -> bool {
        i32_core::i32_eq(self, i32 {mag: 1, sign: false})
    }

    fn abs(self: i32) -> i32 {
        i32_core::i32_abs(self)
    }
}

use orion::numbers::signed_integer::i64 as i64_core;
use orion::numbers::signed_integer::i64::i64;

impl i64Number of NumberTrait<i64> {
    fn zero() -> i64 {
        i64 {mag: 0, sign: false}
    }
    fn is_zero(self: i64) -> bool {
        i64_core::i64_eq(self, i64 {mag: 0, sign: false})
    }

    fn one() -> i64 {
        i64 {mag: 1, sign: false}
    }
    fn is_one(self: i64) -> bool {
        i64_core::i64_eq(self, i64 {mag: 1, sign: false})
    }

    fn abs(self: i64) -> i64 {
        i64_core::i64_abs(self)
    }
}

use orion::numbers::signed_integer::i128 as i128_core;
use orion::numbers::signed_integer::i128::i128;

impl i128Number of NumberTrait<i128> {
    fn zero() -> i128 {
        i128 {mag: 0, sign: false}
    }
    fn is_zero(self: i128) -> bool {
        i128_core::i128_eq(self, i128 {mag: 0, sign: false})
    }

    fn one() -> i128 {
        i128 {mag: 1, sign: false}
    }
    fn is_one(self: i128) -> bool {
        i128_core::i128_eq(self, i128 {mag: 1, sign: false})
    }

    fn abs(self: i128) -> i128 {
        i128_core::i128_abs(self)
    }
}