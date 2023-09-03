mod fixed_point;
mod signed_integer;

use orion::numbers::signed_integer::integer_trait::IntegerTrait;
use orion::numbers::fixed_point::core::FixedTrait;

// Common methods from Fixed Point and Signed Integers.
trait NumberTrait<T, MAG> {
    fn new(mag: MAG, sign: bool) -> T;
    fn new_unscaled(mag: MAG, sign: bool) -> T;
    fn from_felt(val: felt252) -> T;
    fn abs(self: T) -> T;
    fn ceil(self: T) -> T;
    fn exp(self: T) -> T;
    fn exp2(self: T) -> T;
    fn floor(self: T) -> T;
    fn ln(self: T) -> T;
    fn log2(self: T) -> T;
    fn log10(self: T) -> T;
    fn pow(self: T, b: T) -> T;
    fn round(self: T) -> T;
    fn sqrt(self: T) -> T;
    fn acos(self: T) -> T;
    fn asin(self: T) -> T;
    fn atan(self: T) -> T;
    fn cos(self: T) -> T;
    fn sin(self: T) -> T;
    fn tan(self: T) -> T;
    fn acosh(self: T) -> T;
    fn asinh(self: T) -> T;
    fn atanh(self: T) -> T;
    fn cosh(self: T) -> T;
    fn sinh(self: T) -> T;
    fn tanh(self: T) -> T;
    fn zero() -> T;
    fn is_zero(self: T) -> bool;
    fn one() -> T;
    fn is_one(self: T) -> bool;
    fn neg_one() -> T;
    fn min_value() -> T;
    fn max_value() -> T;
    fn min(self: T, other: T) -> T;
    fn max(self: T, other: T) -> T;
    fn mag(self: T) -> MAG;
    fn is_neg(self: T) -> bool;
    fn xor(lhs: T, rhs: T) -> bool;
    fn or(lhs: T, rhs: T) -> bool;
}

use orion::numbers::fixed_point::implementations::fp8x23::core::{FP8x23Impl, FP8x23};
use orion::numbers::fixed_point::implementations::fp8x23::math::core as core_fp8x23;
use orion::numbers::fixed_point::implementations::fp8x23::math::comp as comp_fp8x23;

impl FP8x23Number of NumberTrait<FP8x23, u32> {
    fn new(mag: u32, sign: bool) -> FP8x23 {
        FP8x23Impl::new(mag, sign)
    }

    fn new_unscaled(mag: u32, sign: bool) -> FP8x23 {
        FP8x23Impl::new_unscaled(mag, sign)
    }

    fn from_felt(val: felt252) -> FP8x23 {
        FP8x23Impl::from_felt(val)
    }

    fn ceil(self: FP8x23) -> FP8x23 {
        FP8x23Impl::ceil(self)
    }

    fn exp(self: FP8x23) -> FP8x23 {
        FP8x23Impl::exp(self)
    }

    fn exp2(self: FP8x23) -> FP8x23 {
        FP8x23Impl::exp2(self)
    }

    fn floor(self: FP8x23) -> FP8x23 {
        FP8x23Impl::floor(self)
    }

    fn ln(self: FP8x23) -> FP8x23 {
        FP8x23Impl::ln(self)
    }

    fn log2(self: FP8x23) -> FP8x23 {
        FP8x23Impl::log2(self)
    }

    fn log10(self: FP8x23) -> FP8x23 {
        FP8x23Impl::log10(self)
    }

    fn pow(self: FP8x23, b: FP8x23) -> FP8x23 {
        FP8x23Impl::pow(self, b)
    }

    fn round(self: FP8x23) -> FP8x23 {
        FP8x23Impl::round(self)
    }

    fn sqrt(self: FP8x23) -> FP8x23 {
        FP8x23Impl::sqrt(self)
    }

    fn acos(self: FP8x23) -> FP8x23 {
        FP8x23Impl::acos(self)
    }

    fn asin(self: FP8x23) -> FP8x23 {
        FP8x23Impl::asin(self)
    }

    fn atan(self: FP8x23) -> FP8x23 {
        FP8x23Impl::atan(self)
    }

    fn cos(self: FP8x23) -> FP8x23 {
        FP8x23Impl::cos(self)
    }

    fn sin(self: FP8x23) -> FP8x23 {
        FP8x23Impl::sin(self)
    }

    fn tan(self: FP8x23) -> FP8x23 {
        FP8x23Impl::tan(self)
    }

    fn acosh(self: FP8x23) -> FP8x23 {
        FP8x23Impl::acosh(self)
    }

    fn asinh(self: FP8x23) -> FP8x23 {
        FP8x23Impl::asinh(self)
    }

    fn atanh(self: FP8x23) -> FP8x23 {
        FP8x23Impl::atanh(self)
    }

    fn cosh(self: FP8x23) -> FP8x23 {
        FP8x23Impl::cosh(self)
    }

    fn sinh(self: FP8x23) -> FP8x23 {
        FP8x23Impl::sinh(self)
    }

    fn tanh(self: FP8x23) -> FP8x23 {
        FP8x23Impl::tanh(self)
    }

    fn zero() -> FP8x23 {
        FP8x23Impl::ZERO()
    }
    fn is_zero(self: FP8x23) -> bool {
        core_fp8x23::eq(@self, @FP8x23Impl::ZERO())
    }

    fn one() -> FP8x23 {
        FP8x23Impl::ONE()
    }

    fn neg_one() -> FP8x23 {
        FP8x23 { mag: core_fp8x23::ONE, sign: true }
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

    fn xor(lhs: FP8x23, rhs: FP8x23) -> bool {
        comp_fp8x23::xor(lhs, rhs)
    }

    fn or(lhs: FP8x23, rhs: FP8x23) -> bool {
        comp_fp8x23::or(lhs, rhs)
    }
}

use orion::numbers::fixed_point::implementations::fp16x16::core::{FP16x16Impl, FP16x16};
use orion::numbers::fixed_point::implementations::fp16x16::math::core as core_fp16x16;
use orion::numbers::fixed_point::implementations::fp16x16::math::comp as comp_fp16x16;

impl FP16x16Number of NumberTrait<FP16x16, u32> {
    fn new(mag: u32, sign: bool) -> FP16x16 {
        FP16x16Impl::new(mag, sign)
    }

    fn new_unscaled(mag: u32, sign: bool) -> FP16x16 {
        FP16x16Impl::new_unscaled(mag, sign)
    }

    fn from_felt(val: felt252) -> FP16x16 {
        FP16x16Impl::from_felt(val)
    }

    fn ceil(self: FP16x16) -> FP16x16 {
        FP16x16Impl::ceil(self)
    }

    fn exp(self: FP16x16) -> FP16x16 {
        FP16x16Impl::exp(self)
    }

    fn exp2(self: FP16x16) -> FP16x16 {
        FP16x16Impl::exp2(self)
    }

    fn floor(self: FP16x16) -> FP16x16 {
        FP16x16Impl::floor(self)
    }

    fn ln(self: FP16x16) -> FP16x16 {
        FP16x16Impl::ln(self)
    }

    fn log2(self: FP16x16) -> FP16x16 {
        FP16x16Impl::log2(self)
    }

    fn log10(self: FP16x16) -> FP16x16 {
        FP16x16Impl::log10(self)
    }

    fn pow(self: FP16x16, b: FP16x16) -> FP16x16 {
        FP16x16Impl::pow(self, b)
    }

    fn round(self: FP16x16) -> FP16x16 {
        FP16x16Impl::round(self)
    }

    fn sqrt(self: FP16x16) -> FP16x16 {
        FP16x16Impl::sqrt(self)
    }

    fn acos(self: FP16x16) -> FP16x16 {
        FP16x16Impl::acos(self)
    }

    fn asin(self: FP16x16) -> FP16x16 {
        FP16x16Impl::asin(self)
    }

    fn atan(self: FP16x16) -> FP16x16 {
        FP16x16Impl::atan(self)
    }

    fn cos(self: FP16x16) -> FP16x16 {
        FP16x16Impl::cos(self)
    }

    fn sin(self: FP16x16) -> FP16x16 {
        FP16x16Impl::sin(self)
    }

    fn tan(self: FP16x16) -> FP16x16 {
        FP16x16Impl::tan(self)
    }

    fn acosh(self: FP16x16) -> FP16x16 {
        FP16x16Impl::acosh(self)
    }

    fn asinh(self: FP16x16) -> FP16x16 {
        FP16x16Impl::asinh(self)
    }

    fn atanh(self: FP16x16) -> FP16x16 {
        FP16x16Impl::atanh(self)
    }

    fn cosh(self: FP16x16) -> FP16x16 {
        FP16x16Impl::cosh(self)
    }

    fn sinh(self: FP16x16) -> FP16x16 {
        FP16x16Impl::sinh(self)
    }

    fn tanh(self: FP16x16) -> FP16x16 {
        FP16x16Impl::tanh(self)
    }

    fn zero() -> FP16x16 {
        FP16x16Impl::ZERO()
    }
    fn is_zero(self: FP16x16) -> bool {
        core_fp16x16::eq(@self, @FP16x16Impl::ZERO())
    }

    fn one() -> FP16x16 {
        FP16x16Impl::ONE()
    }

    fn neg_one() -> FP16x16 {
        FP16x16 { mag: core_fp16x16::ONE, sign: true }
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

    fn xor(lhs: FP16x16, rhs: FP16x16) -> bool {
        comp_fp16x16::xor(lhs, rhs)
    }

    fn or(lhs: FP16x16, rhs: FP16x16) -> bool {
        comp_fp16x16::or(lhs, rhs)
    }
}

use orion::numbers::signed_integer::i8 as i8_core;
use orion::numbers::signed_integer::i8::i8;

impl I8Number of NumberTrait<i8, u8> {
    fn new(mag: u8, sign: bool) -> i8 {
        i8 { mag: mag, sign: false }
    }

    fn new_unscaled(mag: u8, sign: bool) -> i8 {
        i8 { mag: mag, sign: false }
    }

    fn from_felt(val: felt252) -> i8 {
        panic(array!['not supported!'])
    }

    fn ceil(self: i8) -> i8 {
        panic(array!['not supported!'])
    }

    fn exp(self: i8) -> i8 {
        panic(array!['not supported!'])
    }

    fn exp2(self: i8) -> i8 {
        panic(array!['not supported!'])
    }

    fn floor(self: i8) -> i8 {
        panic(array!['not supported!'])
    }

    fn ln(self: i8) -> i8 {
        panic(array!['not supported!'])
    }

    fn log2(self: i8) -> i8 {
        panic(array!['not supported!'])
    }

    fn log10(self: i8) -> i8 {
        panic(array!['not supported!'])
    }

    fn pow(self: i8, b: i8) -> i8 {
        panic(array!['not supported!'])
    }

    fn round(self: i8) -> i8 {
        panic(array!['not supported!'])
    }

    fn sqrt(self: i8) -> i8 {
        panic(array!['not supported!'])
    }

    fn acos(self: i8) -> i8 {
        panic(array!['not supported!'])
    }

    fn asin(self: i8) -> i8 {
        panic(array!['not supported!'])
    }

    fn atan(self: i8) -> i8 {
        panic(array!['not supported!'])
    }

    fn cos(self: i8) -> i8 {
        panic(array!['not supported!'])
    }

    fn sin(self: i8) -> i8 {
        panic(array!['not supported!'])
    }

    fn tan(self: i8) -> i8 {
        panic(array!['not supported!'])
    }

    fn acosh(self: i8) -> i8 {
        panic(array!['not supported!'])
    }

    fn asinh(self: i8) -> i8 {
        panic(array!['not supported!'])
    }

    fn atanh(self: i8) -> i8 {
        panic(array!['not supported!'])
    }

    fn cosh(self: i8) -> i8 {
        panic(array!['not supported!'])
    }

    fn sinh(self: i8) -> i8 {
        panic(array!['not supported!'])
    }

    fn tanh(self: i8) -> i8 {
        panic(array!['not supported!'])
    }

    fn zero() -> i8 {
        i8 { mag: 0, sign: false }
    }
    fn is_zero(self: i8) -> bool {
        i8_core::i8_eq(self, i8 { mag: 0, sign: false })
    }

    fn one() -> i8 {
        i8 { mag: 1, sign: false }
    }

    fn neg_one() -> i8 {
        i8 { mag: 1, sign: true }
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

    fn xor(lhs: i8, rhs: i8) -> bool {
        if (lhs.mag == 0 || rhs.mag == 0) && lhs.mag != rhs.mag {
            return true;
        } else {
            return false;
        }
    }

    fn or(lhs: i8, rhs: i8) -> bool {
        if (lhs.mag == 0 && rhs.mag == 0) {
            return false;
        } else {
            return true;
        }
    }
}

use orion::numbers::signed_integer::i16 as i16_core;
use orion::numbers::signed_integer::i16::i16;

impl i16Number of NumberTrait<i16, u16> {
    fn new(mag: u16, sign: bool) -> i16 {
        i16 { mag: mag, sign: false }
    }

    fn new_unscaled(mag: u16, sign: bool) -> i16 {
        i16 { mag: mag, sign: false }
    }

    fn from_felt(val: felt252) -> i16 {
        panic(array!['not supported!'])
    }

    fn ceil(self: i16) -> i16 {
        panic(array!['not supported!'])
    }

    fn exp(self: i16) -> i16 {
        panic(array!['not supported!'])
    }

    fn exp2(self: i16) -> i16 {
        panic(array!['not supported!'])
    }

    fn floor(self: i16) -> i16 {
        panic(array!['not supported!'])
    }

    fn ln(self: i16) -> i16 {
        panic(array!['not supported!'])
    }

    fn log2(self: i16) -> i16 {
        panic(array!['not supported!'])
    }

    fn log10(self: i16) -> i16 {
        panic(array!['not supported!'])
    }

    fn pow(self: i16, b: i16) -> i16 {
        panic(array!['not supported!'])
    }

    fn round(self: i16) -> i16 {
        panic(array!['not supported!'])
    }

    fn sqrt(self: i16) -> i16 {
        panic(array!['not supported!'])
    }

    fn acos(self: i16) -> i16 {
        panic(array!['not supported!'])
    }

    fn asin(self: i16) -> i16 {
        panic(array!['not supported!'])
    }

    fn atan(self: i16) -> i16 {
        panic(array!['not supported!'])
    }

    fn cos(self: i16) -> i16 {
        panic(array!['not supported!'])
    }

    fn sin(self: i16) -> i16 {
        panic(array!['not supported!'])
    }

    fn tan(self: i16) -> i16 {
        panic(array!['not supported!'])
    }

    fn acosh(self: i16) -> i16 {
        panic(array!['not supported!'])
    }

    fn asinh(self: i16) -> i16 {
        panic(array!['not supported!'])
    }

    fn atanh(self: i16) -> i16 {
        panic(array!['not supported!'])
    }

    fn cosh(self: i16) -> i16 {
        panic(array!['not supported!'])
    }

    fn sinh(self: i16) -> i16 {
        panic(array!['not supported!'])
    }

    fn tanh(self: i16) -> i16 {
        panic(array!['not supported!'])
    }

    fn zero() -> i16 {
        i16 { mag: 0, sign: false }
    }
    fn is_zero(self: i16) -> bool {
        i16_core::i16_eq(self, i16 { mag: 0, sign: false })
    }

    fn one() -> i16 {
        i16 { mag: 1, sign: false }
    }

    fn neg_one() -> i16 {
        i16 { mag: 1, sign: true }
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

    fn xor(lhs: i16, rhs: i16) -> bool {
        if (lhs.mag == 0 || rhs.mag == 0) && lhs.mag != rhs.mag {
            return true;
        } else {
            return false;
        }
    }

    fn or(lhs: i16, rhs: i16) -> bool {
        if (lhs.mag == 0 && rhs.mag == 0) {
            return false;
        } else {
            return true;
        }
    }
}

use orion::numbers::signed_integer::i32 as i32_core;
use orion::numbers::signed_integer::i32::i32;

impl i32Number of NumberTrait<i32, u32> {
    fn new(mag: u32, sign: bool) -> i32 {
        i32 { mag: mag, sign: false }
    }

    fn new_unscaled(mag: u32, sign: bool) -> i32 {
        i32 { mag: mag, sign: false }
    }

    fn from_felt(val: felt252) -> i32 {
        panic(array!['not supported!'])
    }

    fn ceil(self: i32) -> i32 {
        panic(array!['not supported!'])
    }

    fn exp(self: i32) -> i32 {
        panic(array!['not supported!'])
    }

    fn exp2(self: i32) -> i32 {
        panic(array!['not supported!'])
    }

    fn floor(self: i32) -> i32 {
        panic(array!['not supported!'])
    }

    fn ln(self: i32) -> i32 {
        panic(array!['not supported!'])
    }

    fn log2(self: i32) -> i32 {
        panic(array!['not supported!'])
    }

    fn log10(self: i32) -> i32 {
        panic(array!['not supported!'])
    }

    fn pow(self: i32, b: i32) -> i32 {
        panic(array!['not supported!'])
    }

    fn round(self: i32) -> i32 {
        panic(array!['not supported!'])
    }

    fn sqrt(self: i32) -> i32 {
        panic(array!['not supported!'])
    }

    fn acos(self: i32) -> i32 {
        panic(array!['not supported!'])
    }

    fn asin(self: i32) -> i32 {
        panic(array!['not supported!'])
    }

    fn atan(self: i32) -> i32 {
        panic(array!['not supported!'])
    }

    fn cos(self: i32) -> i32 {
        panic(array!['not supported!'])
    }

    fn sin(self: i32) -> i32 {
        panic(array!['not supported!'])
    }

    fn tan(self: i32) -> i32 {
        panic(array!['not supported!'])
    }

    fn acosh(self: i32) -> i32 {
        panic(array!['not supported!'])
    }

    fn asinh(self: i32) -> i32 {
        panic(array!['not supported!'])
    }

    fn atanh(self: i32) -> i32 {
        panic(array!['not supported!'])
    }

    fn cosh(self: i32) -> i32 {
        panic(array!['not supported!'])
    }

    fn sinh(self: i32) -> i32 {
        panic(array!['not supported!'])
    }

    fn tanh(self: i32) -> i32 {
        panic(array!['not supported!'])
    }

    fn zero() -> i32 {
        i32 { mag: 0, sign: false }
    }
    fn is_zero(self: i32) -> bool {
        i32_core::i32_eq(self, i32 { mag: 0, sign: false })
    }

    fn one() -> i32 {
        i32 { mag: 1, sign: false }
    }

    fn neg_one() -> i32 {
        i32 { mag: 1, sign: true }
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

    fn xor(lhs: i32, rhs: i32) -> bool {
        if (lhs.mag == 0 || rhs.mag == 0) && lhs.mag != rhs.mag {
            return true;
        } else {
            return false;
        }
    }

    fn or(lhs: i32, rhs: i32) -> bool {
        if (lhs.mag == 0 && rhs.mag == 0) {
            return false;
        } else {
            return true;
        }
    }
}

use orion::numbers::signed_integer::i64 as i64_core;
use orion::numbers::signed_integer::i64::i64;

impl i64Number of NumberTrait<i64, u64> {
    fn new(mag: u64, sign: bool) -> i64 {
        i64 { mag: mag, sign: false }
    }

    fn new_unscaled(mag: u64, sign: bool) -> i64 {
        i64 { mag: mag, sign: false }
    }

    fn from_felt(val: felt252) -> i64 {
        panic(array!['not supported!'])
    }

    fn ceil(self: i64) -> i64 {
        panic(array!['not supported!'])
    }

    fn exp(self: i64) -> i64 {
        panic(array!['not supported!'])
    }

    fn exp2(self: i64) -> i64 {
        panic(array!['not supported!'])
    }

    fn floor(self: i64) -> i64 {
        panic(array!['not supported!'])
    }

    fn ln(self: i64) -> i64 {
        panic(array!['not supported!'])
    }

    fn log2(self: i64) -> i64 {
        panic(array!['not supported!'])
    }

    fn log10(self: i64) -> i64 {
        panic(array!['not supported!'])
    }

    fn pow(self: i64, b: i64) -> i64 {
        panic(array!['not supported!'])
    }

    fn round(self: i64) -> i64 {
        panic(array!['not supported!'])
    }

    fn sqrt(self: i64) -> i64 {
        panic(array!['not supported!'])
    }

    fn acos(self: i64) -> i64 {
        panic(array!['not supported!'])
    }

    fn asin(self: i64) -> i64 {
        panic(array!['not supported!'])
    }

    fn atan(self: i64) -> i64 {
        panic(array!['not supported!'])
    }

    fn cos(self: i64) -> i64 {
        panic(array!['not supported!'])
    }

    fn sin(self: i64) -> i64 {
        panic(array!['not supported!'])
    }

    fn tan(self: i64) -> i64 {
        panic(array!['not supported!'])
    }

    fn acosh(self: i64) -> i64 {
        panic(array!['not supported!'])
    }

    fn asinh(self: i64) -> i64 {
        panic(array!['not supported!'])
    }

    fn atanh(self: i64) -> i64 {
        panic(array!['not supported!'])
    }

    fn cosh(self: i64) -> i64 {
        panic(array!['not supported!'])
    }

    fn sinh(self: i64) -> i64 {
        panic(array!['not supported!'])
    }

    fn tanh(self: i64) -> i64 {
        panic(array!['not supported!'])
    }

    fn zero() -> i64 {
        i64 { mag: 0, sign: false }
    }
    fn is_zero(self: i64) -> bool {
        i64_core::i64_eq(self, i64 { mag: 0, sign: false })
    }

    fn one() -> i64 {
        i64 { mag: 1, sign: false }
    }

    fn neg_one() -> i64 {
        i64 { mag: 1, sign: true }
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

    fn xor(lhs: i64, rhs: i64) -> bool {
        if (lhs.mag == 0 || rhs.mag == 0) && lhs.mag != rhs.mag {
            return true;
        } else {
            return false;
        }
    }

    fn or(lhs: i64, rhs: i64) -> bool {
        if (lhs.mag == 0 && rhs.mag == 0) {
            return false;
        } else {
            return true;
        }
    }
}

use orion::numbers::signed_integer::i128 as i128_core;
use orion::numbers::signed_integer::i128::i128;

impl i128Number of NumberTrait<i128, u128> {
    fn new(mag: u128, sign: bool) -> i128 {
        i128 { mag: mag, sign: false }
    }

    fn new_unscaled(mag: u128, sign: bool) -> i128 {
        i128 { mag: mag, sign: false }
    }

    fn from_felt(val: felt252) -> i128 {
        panic(array!['not supported!'])
    }

    fn ceil(self: i128) -> i128 {
        panic(array!['not supported!'])
    }

    fn exp(self: i128) -> i128 {
        panic(array!['not supported!'])
    }

    fn exp2(self: i128) -> i128 {
        panic(array!['not supported!'])
    }

    fn floor(self: i128) -> i128 {
        panic(array!['not supported!'])
    }

    fn ln(self: i128) -> i128 {
        panic(array!['not supported!'])
    }

    fn log2(self: i128) -> i128 {
        panic(array!['not supported!'])
    }

    fn log10(self: i128) -> i128 {
        panic(array!['not supported!'])
    }

    fn pow(self: i128, b: i128) -> i128 {
        panic(array!['not supported!'])
    }

    fn round(self: i128) -> i128 {
        panic(array!['not supported!'])
    }

    fn sqrt(self: i128) -> i128 {
        panic(array!['not supported!'])
    }

    fn acos(self: i128) -> i128 {
        panic(array!['not supported!'])
    }

    fn asin(self: i128) -> i128 {
        panic(array!['not supported!'])
    }

    fn atan(self: i128) -> i128 {
        panic(array!['not supported!'])
    }

    fn cos(self: i128) -> i128 {
        panic(array!['not supported!'])
    }

    fn sin(self: i128) -> i128 {
        panic(array!['not supported!'])
    }

    fn tan(self: i128) -> i128 {
        panic(array!['not supported!'])
    }

    fn acosh(self: i128) -> i128 {
        panic(array!['not supported!'])
    }

    fn asinh(self: i128) -> i128 {
        panic(array!['not supported!'])
    }

    fn atanh(self: i128) -> i128 {
        panic(array!['not supported!'])
    }

    fn cosh(self: i128) -> i128 {
        panic(array!['not supported!'])
    }

    fn sinh(self: i128) -> i128 {
        panic(array!['not supported!'])
    }

    fn tanh(self: i128) -> i128 {
        panic(array!['not supported!'])
    }


    fn zero() -> i128 {
        i128 { mag: 0, sign: false }
    }
    fn is_zero(self: i128) -> bool {
        i128_core::i128_eq(self, i128 { mag: 0, sign: false })
    }

    fn one() -> i128 {
        i128 { mag: 1, sign: false }
    }

    fn neg_one() -> i128 {
        i128 { mag: 1, sign: true }
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

    fn xor(lhs: i128, rhs: i128) -> bool {
        if (lhs.mag == 0 || rhs.mag == 0) && lhs.mag != rhs.mag {
            return true;
        } else {
            return false;
        }
    }

    fn or(lhs: i128, rhs: i128) -> bool {
        if (lhs.mag == 0 && rhs.mag == 0) {
            return false;
        } else {
            return true;
        }
    }
}

impl u32Number of NumberTrait<u32, u32> {
    fn new(mag: u32, sign: bool) -> u32 {
        mag
    }

    fn new_unscaled(mag: u32, sign: bool) -> u32 {
        mag
    }

    fn from_felt(val: felt252) -> u32 {
        panic(array!['not supported!'])
    }

    fn ceil(self: u32) -> u32 {
        panic(array!['not supported!'])
    }

    fn exp(self: u32) -> u32 {
        panic(array!['not supported!'])
    }

    fn exp2(self: u32) -> u32 {
        panic(array!['not supported!'])
    }

    fn floor(self: u32) -> u32 {
        panic(array!['not supported!'])
    }

    fn ln(self: u32) -> u32 {
        panic(array!['not supported!'])
    }

    fn log2(self: u32) -> u32 {
        panic(array!['not supported!'])
    }

    fn log10(self: u32) -> u32 {
        panic(array!['not supported!'])
    }

    fn pow(self: u32, b: u32) -> u32 {
        panic(array!['not supported!'])
    }

    fn round(self: u32) -> u32 {
        panic(array!['not supported!'])
    }

    fn sqrt(self: u32) -> u32 {
        panic(array!['not supported!'])
    }

    fn acos(self: u32) -> u32 {
        panic(array!['not supported!'])
    }

    fn asin(self: u32) -> u32 {
        panic(array!['not supported!'])
    }

    fn atan(self: u32) -> u32 {
        panic(array!['not supported!'])
    }

    fn cos(self: u32) -> u32 {
        panic(array!['not supported!'])
    }

    fn sin(self: u32) -> u32 {
        panic(array!['not supported!'])
    }

    fn tan(self: u32) -> u32 {
        panic(array!['not supported!'])
    }

    fn acosh(self: u32) -> u32 {
        panic(array!['not supported!'])
    }

    fn asinh(self: u32) -> u32 {
        panic(array!['not supported!'])
    }

    fn atanh(self: u32) -> u32 {
        panic(array!['not supported!'])
    }

    fn cosh(self: u32) -> u32 {
        panic(array!['not supported!'])
    }

    fn sinh(self: u32) -> u32 {
        panic(array!['not supported!'])
    }

    fn tanh(self: u32) -> u32 {
        panic(array!['not supported!'])
    }


    fn zero() -> u32 {
        0
    }
    fn is_zero(self: u32) -> bool {
        self == 0
    }

    fn one() -> u32 {
        1
    }

    fn neg_one() -> u32 {
        panic(array!['not supported'])
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

    fn xor(lhs: u32, rhs: u32) -> bool {
        if (lhs == 0 || rhs == 0) && lhs != rhs {
            return true;
        } else {
            return false;
        }
    }

    fn or(lhs: u32, rhs: u32) -> bool {
        if (lhs == 0 && rhs == 0) {
            return false;
        } else {
            return true;
        }
    }
}
