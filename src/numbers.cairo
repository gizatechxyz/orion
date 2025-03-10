mod fixed_point;
mod complex_number;

use orion::numbers::fixed_point::core::FixedTrait;
use orion::numbers::fixed_point::implementations::fp8x23::core::{ONE as ONE_fp8x23};
use orion::numbers::fixed_point::implementations::fp16x16::core::{ONE as ONE_fp16x16};
use orion::numbers::fixed_point::implementations::fp64x64::core::{ONE as ONE_fp64x64};
use orion::numbers::fixed_point::implementations::fp32x32::core::{ONE as ONE_fp32x32};

// Common methods from Fixed Point and Signed Integers.
trait NumberTrait<T, MAG> {
    fn new(mag: MAG, sign: bool) -> T;
    fn new_unscaled(mag: MAG, sign: bool) -> T;
    fn from_felt(val: felt252) -> T;
    fn abs(self: T) -> T;
    fn neg(self: T) -> T;
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
    fn half() -> T;
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
    fn sign(self: T) -> T;
    fn and(lhs: T, rhs: T) -> bool;
    fn where(self: T, x: T, y: T) -> T;
    fn NaN() -> T;
    fn is_nan(self: T) -> bool;
    fn INF() -> T;
    fn is_inf(self: T) -> bool;
    fn is_pos_inf(self: T) -> bool;
    fn is_neg_inf(self: T) -> bool;
    fn bitwise_and(lhs: T, rhs: T) -> T;
    fn bitwise_xor(lhs: T, rhs: T) -> T;
    fn bitwise_or(lhs: T, rhs: T) -> T;
    fn add(lhs: T, rhs: T) -> T;
    fn sub(lhs: T, rhs: T) -> T;
}

use orion::numbers::fixed_point::implementations::fp8x23::core::{
    FP8x23Impl, FP8x23, FP8x23Add, FP8x23Sub
};
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

    fn half() -> FP8x23 {
        FP8x23Impl::HALF()
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

    fn neg(self: FP8x23) -> FP8x23 {
        core_fp8x23::neg(self)
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

    fn sign(self: FP8x23) -> FP8x23 {
        core_fp8x23::sign(self)
    }

    fn and(lhs: FP8x23, rhs: FP8x23) -> bool {
        comp_fp8x23::and(lhs, rhs)
    }

    fn where(self: FP8x23, x: FP8x23, y: FP8x23) -> FP8x23 {
        comp_fp8x23::where(self, x, y)
    }

    fn NaN() -> FP8x23 {
        FP8x23Impl::NaN()
    }

    fn is_nan(self: FP8x23) -> bool {
        FP8x23Impl::is_nan(self)
    }

    fn INF() -> FP8x23 {
        FP8x23Impl::INF()
    }

    fn is_inf(self: FP8x23) -> bool {
        FP8x23Impl::is_inf(self)
    }

    fn is_pos_inf(self: FP8x23) -> bool {
        FP8x23Impl::is_pos_inf(self)
    }

    fn is_neg_inf(self: FP8x23) -> bool {
        FP8x23Impl::is_neg_inf(self)
    }

    fn bitwise_and(lhs: FP8x23, rhs: FP8x23) -> FP8x23 {
        comp_fp8x23::bitwise_and(lhs, rhs)
    }

    fn bitwise_xor(lhs: FP8x23, rhs: FP8x23) -> FP8x23 {
        comp_fp8x23::bitwise_xor(lhs, rhs)
    }

    fn bitwise_or(lhs: FP8x23, rhs: FP8x23) -> FP8x23 {
        comp_fp8x23::bitwise_or(lhs, rhs)
    }

    fn add(lhs: FP8x23, rhs: FP8x23) -> FP8x23 {
        FP8x23Add::add(lhs, rhs)
    }

    fn sub(lhs: FP8x23, rhs: FP8x23) -> FP8x23 {
        FP8x23Sub::sub(lhs, rhs)
    }
}

use orion::numbers::fixed_point::implementations::fp8x23wide::core::{
    FP8x23WImpl, FP8x23W, FP8x23WAdd, FP8x23WSub
};
use orion::numbers::fixed_point::implementations::fp8x23wide::math::core as core_fp8x23wide;
use orion::numbers::fixed_point::implementations::fp8x23wide::math::comp as comp_fp8x23wide;

impl FP8x23WNumber of NumberTrait<FP8x23W, u64> {
    fn new(mag: u64, sign: bool) -> FP8x23W {
        FP8x23WImpl::new(mag, sign)
    }

    fn new_unscaled(mag: u64, sign: bool) -> FP8x23W {
        FP8x23WImpl::new_unscaled(mag, sign)
    }

    fn from_felt(val: felt252) -> FP8x23W {
        FP8x23WImpl::from_felt(val)
    }

    fn ceil(self: FP8x23W) -> FP8x23W {
        FP8x23WImpl::ceil(self)
    }

    fn exp(self: FP8x23W) -> FP8x23W {
        FP8x23WImpl::exp(self)
    }

    fn exp2(self: FP8x23W) -> FP8x23W {
        FP8x23WImpl::exp2(self)
    }

    fn floor(self: FP8x23W) -> FP8x23W {
        FP8x23WImpl::floor(self)
    }

    fn ln(self: FP8x23W) -> FP8x23W {
        FP8x23WImpl::ln(self)
    }

    fn log2(self: FP8x23W) -> FP8x23W {
        FP8x23WImpl::log2(self)
    }

    fn log10(self: FP8x23W) -> FP8x23W {
        FP8x23WImpl::log10(self)
    }

    fn pow(self: FP8x23W, b: FP8x23W) -> FP8x23W {
        FP8x23WImpl::pow(self, b)
    }

    fn round(self: FP8x23W) -> FP8x23W {
        FP8x23WImpl::round(self)
    }

    fn sqrt(self: FP8x23W) -> FP8x23W {
        FP8x23WImpl::sqrt(self)
    }

    fn acos(self: FP8x23W) -> FP8x23W {
        FP8x23WImpl::acos(self)
    }

    fn asin(self: FP8x23W) -> FP8x23W {
        FP8x23WImpl::asin(self)
    }

    fn atan(self: FP8x23W) -> FP8x23W {
        FP8x23WImpl::atan(self)
    }

    fn cos(self: FP8x23W) -> FP8x23W {
        FP8x23WImpl::cos(self)
    }

    fn sin(self: FP8x23W) -> FP8x23W {
        FP8x23WImpl::sin(self)
    }

    fn tan(self: FP8x23W) -> FP8x23W {
        FP8x23WImpl::tan(self)
    }

    fn acosh(self: FP8x23W) -> FP8x23W {
        FP8x23WImpl::acosh(self)
    }

    fn asinh(self: FP8x23W) -> FP8x23W {
        FP8x23WImpl::asinh(self)
    }

    fn atanh(self: FP8x23W) -> FP8x23W {
        FP8x23WImpl::atanh(self)
    }

    fn cosh(self: FP8x23W) -> FP8x23W {
        FP8x23WImpl::cosh(self)
    }

    fn sinh(self: FP8x23W) -> FP8x23W {
        FP8x23WImpl::sinh(self)
    }

    fn tanh(self: FP8x23W) -> FP8x23W {
        FP8x23WImpl::tanh(self)
    }

    fn zero() -> FP8x23W {
        FP8x23WImpl::ZERO()
    }
    fn is_zero(self: FP8x23W) -> bool {
        core_fp8x23wide::eq(@self, @FP8x23WImpl::ZERO())
    }

    fn half() -> FP8x23W {
        FP8x23WImpl::HALF()
    }

    fn one() -> FP8x23W {
        FP8x23WImpl::ONE()
    }

    fn neg_one() -> FP8x23W {
        FP8x23W { mag: core_fp8x23wide::ONE, sign: true }
    }

    fn is_one(self: FP8x23W) -> bool {
        core_fp8x23wide::eq(@self, @FP8x23WImpl::ONE())
    }

    fn abs(self: FP8x23W) -> FP8x23W {
        core_fp8x23wide::abs(self)
    }

    fn neg(self: FP8x23W) -> FP8x23W {
        core_fp8x23wide::neg(self)
    }

    fn min_value() -> FP8x23W {
        FP8x23W { mag: core_fp8x23wide::MAX, sign: true }
    }

    fn max_value() -> FP8x23W {
        FP8x23W { mag: core_fp8x23wide::MAX, sign: false }
    }

    fn min(self: FP8x23W, other: FP8x23W) -> FP8x23W {
        comp_fp8x23wide::min(self, other)
    }

    fn max(self: FP8x23W, other: FP8x23W) -> FP8x23W {
        comp_fp8x23wide::max(self, other)
    }

    fn mag(self: FP8x23W) -> u64 {
        self.mag
    }

    fn is_neg(self: FP8x23W) -> bool {
        self.sign
    }

    fn xor(lhs: FP8x23W, rhs: FP8x23W) -> bool {
        comp_fp8x23wide::xor(lhs, rhs)
    }

    fn or(lhs: FP8x23W, rhs: FP8x23W) -> bool {
        comp_fp8x23wide::or(lhs, rhs)
    }

    fn sign(self: FP8x23W) -> FP8x23W {
        core_fp8x23wide::sign(self)
    }

    fn and(lhs: FP8x23W, rhs: FP8x23W) -> bool {
        comp_fp8x23wide::and(lhs, rhs)
    }

    fn where(self: FP8x23W, x: FP8x23W, y: FP8x23W) -> FP8x23W {
        comp_fp8x23wide::where(self, x, y)
    }

    fn NaN() -> FP8x23W {
        FP8x23WImpl::NaN()
    }

    fn is_nan(self: FP8x23W) -> bool {
        FP8x23WImpl::is_nan(self)
    }

    fn INF() -> FP8x23W {
        FP8x23WImpl::INF()
    }

    fn is_inf(self: FP8x23W) -> bool {
        FP8x23WImpl::is_inf(self)
    }

    fn is_pos_inf(self: FP8x23W) -> bool {
        FP8x23WImpl::is_pos_inf(self)
    }

    fn is_neg_inf(self: FP8x23W) -> bool {
        FP8x23WImpl::is_neg_inf(self)
    }

    fn bitwise_and(lhs: FP8x23W, rhs: FP8x23W) -> FP8x23W {
        comp_fp8x23wide::bitwise_and(lhs, rhs)
    }

    fn bitwise_xor(lhs: FP8x23W, rhs: FP8x23W) -> FP8x23W {
        comp_fp8x23wide::bitwise_xor(lhs, rhs)
    }

    fn bitwise_or(lhs: FP8x23W, rhs: FP8x23W) -> FP8x23W {
        comp_fp8x23wide::bitwise_or(lhs, rhs)
    }

    fn add(lhs: FP8x23W, rhs: FP8x23W) -> FP8x23W {
        FP8x23WAdd::add(lhs, rhs)
    }

    fn sub(lhs: FP8x23W, rhs: FP8x23W) -> FP8x23W {
        FP8x23WSub::sub(lhs, rhs)
    }
}

use orion::numbers::fixed_point::implementations::fp16x16::core::{
    FP16x16Impl, FP16x16, FP16x16Add, FP16x16Sub
};
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

    fn half() -> FP16x16 {
        FP16x16Impl::HALF()
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

    fn neg(self: FP16x16) -> FP16x16 {
        core_fp16x16::neg(self)
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

    fn sign(self: FP16x16) -> FP16x16 {
        core_fp16x16::sign(self)
    }

    fn and(lhs: FP16x16, rhs: FP16x16) -> bool {
        comp_fp16x16::and(lhs, rhs)
    }

    fn where(self: FP16x16, x: FP16x16, y: FP16x16) -> FP16x16 {
        comp_fp16x16::where(self, x, y)
    }

    fn NaN() -> FP16x16 {
        FP16x16Impl::NaN()
    }

    fn is_nan(self: FP16x16) -> bool {
        FP16x16Impl::is_nan(self)
    }

    fn INF() -> FP16x16 {
        FP16x16Impl::INF()
    }

    fn is_inf(self: FP16x16) -> bool {
        FP16x16Impl::is_inf(self)
    }

    fn is_pos_inf(self: FP16x16) -> bool {
        FP16x16Impl::is_pos_inf(self)
    }

    fn is_neg_inf(self: FP16x16) -> bool {
        FP16x16Impl::is_neg_inf(self)
    }

    fn bitwise_and(lhs: FP16x16, rhs: FP16x16) -> FP16x16 {
        comp_fp16x16::bitwise_and(lhs, rhs)
    }

    fn bitwise_xor(lhs: FP16x16, rhs: FP16x16) -> FP16x16 {
        comp_fp16x16::bitwise_xor(lhs, rhs)
    }

    fn bitwise_or(lhs: FP16x16, rhs: FP16x16) -> FP16x16 {
        comp_fp16x16::bitwise_or(lhs, rhs)
    }

    fn add(lhs: FP16x16, rhs: FP16x16) -> FP16x16 {
        FP16x16Add::add(lhs, rhs)
    }

    fn sub(lhs: FP16x16, rhs: FP16x16) -> FP16x16 {
        FP16x16Sub::sub(lhs, rhs)
    }
}

use orion::numbers::fixed_point::implementations::fp16x16wide::core::{
    FP16x16WImpl, FP16x16W, FP16x16WAdd, FP16x16WSub
};
use orion::numbers::fixed_point::implementations::fp16x16wide::math::core as core_fp16x16wide;
use orion::numbers::fixed_point::implementations::fp16x16wide::math::comp as comp_fp16x16wide;

impl FP16x16WNumber of NumberTrait<FP16x16W, u64> {
    fn new(mag: u64, sign: bool) -> FP16x16W {
        FP16x16WImpl::new(mag, sign)
    }

    fn new_unscaled(mag: u64, sign: bool) -> FP16x16W {
        FP16x16WImpl::new_unscaled(mag, sign)
    }

    fn from_felt(val: felt252) -> FP16x16W {
        FP16x16WImpl::from_felt(val)
    }

    fn ceil(self: FP16x16W) -> FP16x16W {
        FP16x16WImpl::ceil(self)
    }

    fn exp(self: FP16x16W) -> FP16x16W {
        FP16x16WImpl::exp(self)
    }

    fn exp2(self: FP16x16W) -> FP16x16W {
        FP16x16WImpl::exp2(self)
    }

    fn floor(self: FP16x16W) -> FP16x16W {
        FP16x16WImpl::floor(self)
    }

    fn ln(self: FP16x16W) -> FP16x16W {
        FP16x16WImpl::ln(self)
    }

    fn log2(self: FP16x16W) -> FP16x16W {
        FP16x16WImpl::log2(self)
    }

    fn log10(self: FP16x16W) -> FP16x16W {
        FP16x16WImpl::log10(self)
    }

    fn pow(self: FP16x16W, b: FP16x16W) -> FP16x16W {
        FP16x16WImpl::pow(self, b)
    }

    fn round(self: FP16x16W) -> FP16x16W {
        FP16x16WImpl::round(self)
    }

    fn sqrt(self: FP16x16W) -> FP16x16W {
        FP16x16WImpl::sqrt(self)
    }

    fn acos(self: FP16x16W) -> FP16x16W {
        FP16x16WImpl::acos(self)
    }

    fn asin(self: FP16x16W) -> FP16x16W {
        FP16x16WImpl::asin(self)
    }

    fn atan(self: FP16x16W) -> FP16x16W {
        FP16x16WImpl::atan(self)
    }

    fn cos(self: FP16x16W) -> FP16x16W {
        FP16x16WImpl::cos(self)
    }

    fn sin(self: FP16x16W) -> FP16x16W {
        FP16x16WImpl::sin(self)
    }

    fn tan(self: FP16x16W) -> FP16x16W {
        FP16x16WImpl::tan(self)
    }

    fn acosh(self: FP16x16W) -> FP16x16W {
        FP16x16WImpl::acosh(self)
    }

    fn asinh(self: FP16x16W) -> FP16x16W {
        FP16x16WImpl::asinh(self)
    }

    fn atanh(self: FP16x16W) -> FP16x16W {
        FP16x16WImpl::atanh(self)
    }

    fn cosh(self: FP16x16W) -> FP16x16W {
        FP16x16WImpl::cosh(self)
    }

    fn sinh(self: FP16x16W) -> FP16x16W {
        FP16x16WImpl::sinh(self)
    }

    fn tanh(self: FP16x16W) -> FP16x16W {
        FP16x16WImpl::tanh(self)
    }

    fn zero() -> FP16x16W {
        FP16x16WImpl::ZERO()
    }
    fn is_zero(self: FP16x16W) -> bool {
        core_fp16x16wide::eq(@self, @FP16x16WImpl::ZERO())
    }

    fn half() -> FP16x16W {
        FP16x16WImpl::HALF()
    }

    fn one() -> FP16x16W {
        FP16x16WImpl::ONE()
    }

    fn neg_one() -> FP16x16W {
        FP16x16W { mag: core_fp16x16wide::ONE, sign: true }
    }

    fn is_one(self: FP16x16W) -> bool {
        core_fp16x16wide::eq(@self, @FP16x16WImpl::ONE())
    }

    fn abs(self: FP16x16W) -> FP16x16W {
        core_fp16x16wide::abs(self)
    }

    fn neg(self: FP16x16W) -> FP16x16W {
        core_fp16x16wide::neg(self)
    }

    fn min_value() -> FP16x16W {
        FP16x16W { mag: core_fp16x16wide::MAX, sign: true }
    }

    fn max_value() -> FP16x16W {
        FP16x16W { mag: core_fp16x16wide::MAX, sign: false }
    }

    fn min(self: FP16x16W, other: FP16x16W) -> FP16x16W {
        comp_fp16x16wide::min(self, other)
    }

    fn max(self: FP16x16W, other: FP16x16W) -> FP16x16W {
        comp_fp16x16wide::max(self, other)
    }

    fn mag(self: FP16x16W) -> u64 {
        self.mag
    }

    fn is_neg(self: FP16x16W) -> bool {
        self.sign
    }

    fn xor(lhs: FP16x16W, rhs: FP16x16W) -> bool {
        comp_fp16x16wide::xor(lhs, rhs)
    }

    fn or(lhs: FP16x16W, rhs: FP16x16W) -> bool {
        comp_fp16x16wide::or(lhs, rhs)
    }

    fn sign(self: FP16x16W) -> FP16x16W {
        core_fp16x16wide::sign(self)
    }

    fn and(lhs: FP16x16W, rhs: FP16x16W) -> bool {
        comp_fp16x16wide::and(lhs, rhs)
    }

    fn where(self: FP16x16W, x: FP16x16W, y: FP16x16W) -> FP16x16W {
        comp_fp16x16wide::where(self, x, y)
    }

    fn NaN() -> FP16x16W {
        FP16x16WImpl::NaN()
    }

    fn is_nan(self: FP16x16W) -> bool {
        FP16x16WImpl::is_nan(self)
    }

    fn INF() -> FP16x16W {
        FP16x16WImpl::INF()
    }

    fn is_inf(self: FP16x16W) -> bool {
        FP16x16WImpl::is_inf(self)
    }

    fn is_pos_inf(self: FP16x16W) -> bool {
        FP16x16WImpl::is_pos_inf(self)
    }

    fn is_neg_inf(self: FP16x16W) -> bool {
        FP16x16WImpl::is_neg_inf(self)
    }

    fn bitwise_and(lhs: FP16x16W, rhs: FP16x16W) -> FP16x16W {
        comp_fp16x16wide::bitwise_and(lhs, rhs)
    }

    fn bitwise_xor(lhs: FP16x16W, rhs: FP16x16W) -> FP16x16W {
        comp_fp16x16wide::bitwise_xor(lhs, rhs)
    }

    fn bitwise_or(lhs: FP16x16W, rhs: FP16x16W) -> FP16x16W {
        comp_fp16x16wide::bitwise_or(lhs, rhs)
    }

    fn add(lhs: FP16x16W, rhs: FP16x16W) -> FP16x16W {
        FP16x16WAdd::add(lhs, rhs)
    }

    fn sub(lhs: FP16x16W, rhs: FP16x16W) -> FP16x16W {
        FP16x16WSub::sub(lhs, rhs)
    }
}

use orion::numbers::fixed_point::implementations::fp64x64::core::{
    FP64x64Impl, FP64x64, FP64x64Add, FP64x64Sub
};
use orion::numbers::fixed_point::implementations::fp64x64::{core as core_fp64x64};
use orion::numbers::fixed_point::implementations::fp64x64::comp as comp_fp64x64;
use cubit::f128 as fp64x64;

impl FP64x64Number of NumberTrait<FP64x64, u128> {
    fn new(mag: u128, sign: bool) -> FP64x64 {
        FP64x64Impl::new(mag, sign)
    }

    fn new_unscaled(mag: u128, sign: bool) -> FP64x64 {
        FP64x64Impl::new_unscaled(mag, sign)
    }

    fn from_felt(val: felt252) -> FP64x64 {
        FP64x64Impl::from_felt(val)
    }

    fn ceil(self: FP64x64) -> FP64x64 {
        FP64x64Impl::ceil(self)
    }

    fn exp(self: FP64x64) -> FP64x64 {
        FP64x64Impl::exp(self)
    }

    fn exp2(self: FP64x64) -> FP64x64 {
        FP64x64Impl::exp2(self)
    }

    fn floor(self: FP64x64) -> FP64x64 {
        FP64x64Impl::floor(self)
    }

    fn ln(self: FP64x64) -> FP64x64 {
        FP64x64Impl::ln(self)
    }

    fn log2(self: FP64x64) -> FP64x64 {
        FP64x64Impl::log2(self)
    }

    fn log10(self: FP64x64) -> FP64x64 {
        FP64x64Impl::log10(self)
    }

    fn pow(self: FP64x64, b: FP64x64) -> FP64x64 {
        FP64x64Impl::pow(self, b)
    }

    fn round(self: FP64x64) -> FP64x64 {
        FP64x64Impl::round(self)
    }

    fn sqrt(self: FP64x64) -> FP64x64 {
        FP64x64Impl::sqrt(self)
    }

    fn acos(self: FP64x64) -> FP64x64 {
        FP64x64Impl::acos(self)
    }

    fn asin(self: FP64x64) -> FP64x64 {
        FP64x64Impl::asin(self)
    }

    fn atan(self: FP64x64) -> FP64x64 {
        FP64x64Impl::atan(self)
    }

    fn cos(self: FP64x64) -> FP64x64 {
        FP64x64Impl::cos(self)
    }

    fn sin(self: FP64x64) -> FP64x64 {
        FP64x64Impl::sin(self)
    }

    fn tan(self: FP64x64) -> FP64x64 {
        FP64x64Impl::tan(self)
    }

    fn acosh(self: FP64x64) -> FP64x64 {
        FP64x64Impl::acosh(self)
    }

    fn asinh(self: FP64x64) -> FP64x64 {
        FP64x64Impl::asinh(self)
    }

    fn atanh(self: FP64x64) -> FP64x64 {
        FP64x64Impl::atanh(self)
    }

    fn cosh(self: FP64x64) -> FP64x64 {
        FP64x64Impl::cosh(self)
    }

    fn sinh(self: FP64x64) -> FP64x64 {
        FP64x64Impl::sinh(self)
    }

    fn tanh(self: FP64x64) -> FP64x64 {
        FP64x64Impl::tanh(self)
    }

    fn zero() -> FP64x64 {
        FP64x64Impl::ZERO()
    }
    fn is_zero(self: FP64x64) -> bool {
        fp64x64::ops::eq(@self, @FP64x64Impl::ZERO())
    }

    fn half() -> FP64x64 {
        FP64x64Impl::HALF()
    }

    fn one() -> FP64x64 {
        FP64x64Impl::ONE()
    }

    fn neg_one() -> FP64x64 {
        FP64x64 { mag: core_fp64x64::ONE, sign: true }
    }

    fn is_one(self: FP64x64) -> bool {
        core_fp64x64::eq(@self, @FP64x64Impl::ONE())
    }

    fn abs(self: FP64x64) -> FP64x64 {
        fp64x64::ops::abs(self)
    }

    fn neg(self: FP64x64) -> FP64x64 {
        fp64x64::ops::neg(self)
    }

    fn min_value() -> FP64x64 {
        FP64x64 { mag: core_fp64x64::MAX, sign: true }
    }

    fn max_value() -> FP64x64 {
        FP64x64 { mag: core_fp64x64::MAX, sign: false }
    }

    fn min(self: FP64x64, other: FP64x64) -> FP64x64 {
        fp64x64::comp::min(self, other)
    }

    fn max(self: FP64x64, other: FP64x64) -> FP64x64 {
        fp64x64::comp::max(self, other)
    }

    fn mag(self: FP64x64) -> u128 {
        self.mag
    }

    fn is_neg(self: FP64x64) -> bool {
        self.sign
    }

    fn xor(lhs: FP64x64, rhs: FP64x64) -> bool {
        comp_fp64x64::xor(lhs, rhs)
    }

    fn or(lhs: FP64x64, rhs: FP64x64) -> bool {
        comp_fp64x64::or(lhs, rhs)
    }

    fn sign(self: FP64x64) -> FP64x64 {
        FP64x64Impl::sign(self)
    }

    fn and(lhs: FP64x64, rhs: FP64x64) -> bool {
        comp_fp64x64::and(lhs, rhs)
    }

    fn where(self: FP64x64, x: FP64x64, y: FP64x64) -> FP64x64 {
        comp_fp64x64::where(self, x, y)
    }

    fn NaN() -> FP64x64 {
        FP64x64Impl::NaN()
    }

    fn is_nan(self: FP64x64) -> bool {
        FP64x64Impl::is_nan(self)
    }

    fn INF() -> FP64x64 {
        FP64x64Impl::INF()
    }

    fn is_inf(self: FP64x64) -> bool {
        FP64x64Impl::is_inf(self)
    }

    fn is_pos_inf(self: FP64x64) -> bool {
        FP64x64Impl::is_pos_inf(self)
    }

    fn is_neg_inf(self: FP64x64) -> bool {
        FP64x64Impl::is_neg_inf(self)
    }

    fn bitwise_and(lhs: FP64x64, rhs: FP64x64) -> FP64x64 {
        comp_fp64x64::bitwise_and(lhs, rhs)
    }

    fn bitwise_xor(lhs: FP64x64, rhs: FP64x64) -> FP64x64 {
        comp_fp64x64::bitwise_xor(lhs, rhs)
    }

    fn bitwise_or(lhs: FP64x64, rhs: FP64x64) -> FP64x64 {
        comp_fp64x64::bitwise_or(lhs, rhs)
    }

    fn add(lhs: FP64x64, rhs: FP64x64) -> FP64x64 {
        FP64x64Add::add(lhs, rhs)
    }

    fn sub(lhs: FP64x64, rhs: FP64x64) -> FP64x64 {
        FP64x64Sub::sub(lhs, rhs)
    }
}

use orion::numbers::fixed_point::implementations::fp32x32::core::{
    FP32x32Impl, FP32x32, FP32x32Add, FP32x32Sub
};
use orion::numbers::fixed_point::implementations::fp32x32::core as core_fp32x32;
use orion::numbers::fixed_point::implementations::fp32x32::comp as comp_fp32x32;
use cubit::f64 as fp32x32;

impl FP32x32Number of NumberTrait<FP32x32, u64> {
    fn new(mag: u64, sign: bool) -> FP32x32 {
        FP32x32Impl::new(mag, sign)
    }

    fn new_unscaled(mag: u64, sign: bool) -> FP32x32 {
        FP32x32Impl::new_unscaled(mag, sign)
    }

    fn from_felt(val: felt252) -> FP32x32 {
        FP32x32Impl::from_felt(val)
    }

    fn ceil(self: FP32x32) -> FP32x32 {
        FP32x32Impl::ceil(self)
    }

    fn exp(self: FP32x32) -> FP32x32 {
        FP32x32Impl::exp(self)
    }

    fn exp2(self: FP32x32) -> FP32x32 {
        FP32x32Impl::exp2(self)
    }

    fn floor(self: FP32x32) -> FP32x32 {
        FP32x32Impl::floor(self)
    }

    fn ln(self: FP32x32) -> FP32x32 {
        FP32x32Impl::ln(self)
    }

    fn log2(self: FP32x32) -> FP32x32 {
        FP32x32Impl::log2(self)
    }

    fn log10(self: FP32x32) -> FP32x32 {
        FP32x32Impl::log10(self)
    }

    fn pow(self: FP32x32, b: FP32x32) -> FP32x32 {
        FP32x32Impl::pow(self, b)
    }

    fn round(self: FP32x32) -> FP32x32 {
        FP32x32Impl::round(self)
    }

    fn sqrt(self: FP32x32) -> FP32x32 {
        FP32x32Impl::sqrt(self)
    }

    fn acos(self: FP32x32) -> FP32x32 {
        FP32x32Impl::acos(self)
    }

    fn asin(self: FP32x32) -> FP32x32 {
        FP32x32Impl::asin(self)
    }

    fn atan(self: FP32x32) -> FP32x32 {
        FP32x32Impl::atan(self)
    }

    fn cos(self: FP32x32) -> FP32x32 {
        FP32x32Impl::cos(self)
    }

    fn sin(self: FP32x32) -> FP32x32 {
        FP32x32Impl::sin(self)
    }

    fn tan(self: FP32x32) -> FP32x32 {
        FP32x32Impl::tan(self)
    }

    fn acosh(self: FP32x32) -> FP32x32 {
        FP32x32Impl::acosh(self)
    }

    fn asinh(self: FP32x32) -> FP32x32 {
        FP32x32Impl::asinh(self)
    }

    fn atanh(self: FP32x32) -> FP32x32 {
        FP32x32Impl::atanh(self)
    }

    fn cosh(self: FP32x32) -> FP32x32 {
        FP32x32Impl::cosh(self)
    }

    fn sinh(self: FP32x32) -> FP32x32 {
        FP32x32Impl::sinh(self)
    }

    fn tanh(self: FP32x32) -> FP32x32 {
        FP32x32Impl::tanh(self)
    }

    fn zero() -> FP32x32 {
        FP32x32Impl::ZERO()
    }
    fn is_zero(self: FP32x32) -> bool {
        fp32x32::ops::eq(@self, @FP32x32Impl::ZERO())
    }

    fn half() -> FP32x32 {
        FP32x32Impl::HALF()
    }

    fn one() -> FP32x32 {
        FP32x32Impl::ONE()
    }

    fn neg_one() -> FP32x32 {
        FP32x32 { mag: core_fp32x32::ONE, sign: true }
    }

    fn is_one(self: FP32x32) -> bool {
        core_fp32x32::eq(@self, @FP32x32Impl::ONE())
    }

    fn abs(self: FP32x32) -> FP32x32 {
        fp32x32::ops::abs(self)
    }

    fn neg(self: FP32x32) -> FP32x32 {
        fp32x32::ops::neg(self)
    }

    fn min_value() -> FP32x32 {
        FP32x32 { mag: core_fp32x32::MAX, sign: true }
    }

    fn max_value() -> FP32x32 {
        FP32x32 { mag: core_fp32x32::MAX, sign: false }
    }

    fn min(self: FP32x32, other: FP32x32) -> FP32x32 {
        fp32x32::comp::min(self, other)
    }

    fn max(self: FP32x32, other: FP32x32) -> FP32x32 {
        fp32x32::comp::max(self, other)
    }

    fn mag(self: FP32x32) -> u64 {
        self.mag
    }

    fn is_neg(self: FP32x32) -> bool {
        self.sign
    }

    fn xor(lhs: FP32x32, rhs: FP32x32) -> bool {
        comp_fp32x32::xor(lhs, rhs)
    }

    fn or(lhs: FP32x32, rhs: FP32x32) -> bool {
        comp_fp32x32::or(lhs, rhs)
    }

    fn sign(self: FP32x32) -> FP32x32 {
        FP32x32Impl::sign(self)
    }

    fn and(lhs: FP32x32, rhs: FP32x32) -> bool {
        comp_fp32x32::and(lhs, rhs)
    }

    fn where(self: FP32x32, x: FP32x32, y: FP32x32) -> FP32x32 {
        comp_fp32x32::where(self, x, y)
    }

    fn NaN() -> FP32x32 {
        FP32x32Impl::NaN()
    }

    fn is_nan(self: FP32x32) -> bool {
        FP32x32Impl::is_nan(self)
    }

    fn INF() -> FP32x32 {
        FP32x32Impl::INF()
    }

    fn is_inf(self: FP32x32) -> bool {
        FP32x32Impl::is_inf(self)
    }

    fn is_pos_inf(self: FP32x32) -> bool {
        FP32x32Impl::is_pos_inf(self)
    }

    fn is_neg_inf(self: FP32x32) -> bool {
        FP32x32Impl::is_neg_inf(self)
    }

    fn bitwise_and(lhs: FP32x32, rhs: FP32x32) -> FP32x32 {
        comp_fp32x32::bitwise_and(lhs, rhs)
    }

    fn bitwise_xor(lhs: FP32x32, rhs: FP32x32) -> FP32x32 {
        comp_fp32x32::bitwise_xor(lhs, rhs)
    }

    fn bitwise_or(lhs: FP32x32, rhs: FP32x32) -> FP32x32 {
        comp_fp32x32::bitwise_or(lhs, rhs)
    }

    fn add(lhs: FP32x32, rhs: FP32x32) -> FP32x32 {
        FP32x32Add::add(lhs, rhs)
    }

    fn sub(lhs: FP32x32, rhs: FP32x32) -> FP32x32 {
        FP32x32Sub::sub(lhs, rhs)
    }
}

impl I8Number of NumberTrait<i8, i8> {
    fn new(mag: i8, sign: bool) -> i8 {
        if sign {
            return -mag;
        }

        mag
    }

    fn new_unscaled(mag: i8, sign: bool) -> i8 {
        mag
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
        0
    }
    fn is_zero(self: i8) -> bool {
        self == 0
    }

    fn half() -> i8 {
        panic(array!['not supported!'])
    }

    fn one() -> i8 {
        1
    }

    fn neg_one() -> i8 {
        -1
    }

    fn is_one(self: i8) -> bool {
        self == 1
    }

    fn abs(self: i8) -> i8 {
        if self >= 0 {
            self
        } else {
            self * -1_i8
        }
    }

    fn neg(self: i8) -> i8 {
        self * -1_i8
    }

    fn min_value() -> i8 {
        -127
    }

    fn max_value() -> i8 {
        127
    }

    fn min(self: i8, other: i8) -> i8 {
        if self < other {
            self
        } else {
            other
        }
    }

    fn max(self: i8, other: i8) -> i8 {
        if self > other {
            self
        } else {
            other
        }
    }

    fn mag(self: i8) -> i8 {
        self
    }

    fn is_neg(self: i8) -> bool {
        self < 0
    }

    fn xor(lhs: i8, rhs: i8) -> bool {
        if (lhs == 0 || rhs == 0) && lhs != rhs {
            true
        } else {
            false
        }
    }

    fn or(lhs: i8, rhs: i8) -> bool {
        if lhs == 0 && rhs == 0 {
            false
        } else {
            true
        }
    }

    fn sign(self: i8) -> i8 {
        if self == 0 {
            0_i8
        } else if self > 0 {
            1_i8
        } else {
            -1_i8
        }
    }

    fn and(lhs: i8, rhs: i8) -> bool {
        if lhs == 0 || rhs == 0 {
            false
        } else {
            true
        }
    }

    fn where(self: i8, x: i8, y: i8) -> i8 {
        if self == 0 {
            y
        } else {
            x
        }
    }

    fn NaN() -> i8 {
        panic(array!['not supported!'])
    }

    fn is_nan(self: i8) -> bool {
        panic(array!['not supported!'])
    }

    fn INF() -> i8 {
        127
    }

    fn is_inf(self: i8) -> bool {
        self == 127 || self == -127
    }

    fn is_pos_inf(self: i8) -> bool {
        self == 127
    }

    fn is_neg_inf(self: i8) -> bool {
        self == -127
    }

    fn bitwise_and(lhs: i8, rhs: i8) -> i8 {
        panic(array!['not supported!'])
    }

    fn bitwise_xor(lhs: i8, rhs: i8) -> i8 {
        panic(array!['not supported!'])
    }

    fn bitwise_or(lhs: i8, rhs: i8) -> i8 {
        panic(array!['not supported!'])
    }

    fn add(lhs: i8, rhs: i8) -> i8 {
        lhs + rhs
    }

    fn sub(lhs: i8, rhs: i8) -> i8 {
        lhs - rhs
    }
}

impl I8Div of Div<i8> {
    fn div(lhs: i8, rhs: i8) -> i8 {
        assert(rhs != 0, 'divisor cannot be 0');

        let mut lhs_positive = lhs;
        let mut rhs_positive = rhs;

        // making sure everything is positive
        if lhs < 0 {
            lhs_positive = lhs * -1;
        }
        if rhs < 0 {
            rhs_positive = rhs * -1;
        }

        //felt252 plays role of a bridge for type casting
        let lhs_felt: felt252 = lhs_positive.into();
        let rhs_felt: felt252 = rhs_positive.into();
        let lhs_u128: u128 = lhs_felt.try_into().unwrap();
        let rhs_u128: u128 = rhs_felt.try_into().unwrap();
        let mut result = lhs_u128 / rhs_u128;
        let felt_result: felt252 = result.into();
        let signed_int_result: i8 = felt_result.try_into().unwrap();

        if lhs * rhs < 0 {
            signed_int_result * -1
        } else {
            signed_int_result
        }
    }
}

impl I8DivEq of DivEq<i8> {
    #[inline(always)]
    fn div_eq(ref self: i8, other: i8) {
        self = Div::div(self, other);
    }
}

impl I8IntoFP8x23 of Into<i8, FP8x23> {
    fn into(self: i8) -> FP8x23 {
        let number_sign: bool = self < 0;
        let mut self_positive: i8 = self;

        if number_sign {
            self_positive = self_positive * -1_i8
        }

        let number_felt: felt252 = self_positive.into();
        let number_u32: u32 = number_felt.try_into().unwrap();

        FP8x23 { mag: number_u32 * ONE_fp8x23, sign: number_sign }
    }
}

impl I8IntoFP16x16 of Into<i8, FP16x16> {
    fn into(self: i8) -> FP16x16 {
        let number_sign: bool = self < 0;
        let mut self_positive: i8 = self;

        if number_sign {
            self_positive = self_positive * -1_i8
        }

        let number_felt: felt252 = self_positive.into();
        let number_u32: u32 = number_felt.try_into().unwrap();

        FP16x16 { mag: number_u32 * ONE_fp16x16, sign: number_sign }
    }
}

impl I16Number of NumberTrait<i16, i16> {
    fn new(mag: i16, sign: bool) -> i16 {
        if sign {
            return -mag;
        }

        mag
    }

    fn new_unscaled(mag: i16, sign: bool) -> i16 {
        mag
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
        0
    }
    fn is_zero(self: i16) -> bool {
        self == 0
    }

    fn half() -> i16 {
        panic(array!['not supported!'])
    }

    fn one() -> i16 {
        1
    }

    fn neg_one() -> i16 {
        -1
    }

    fn is_one(self: i16) -> bool {
        self == 1
    }

    fn abs(self: i16) -> i16 {
        if self >= 0 {
            self
        } else {
            self * -1_i16
        }
    }

    fn neg(self: i16) -> i16 {
        self * -1_i16
    }

    fn min_value() -> i16 {
        -32767
    }

    fn max_value() -> i16 {
        32767
    }

    fn min(self: i16, other: i16) -> i16 {
        if self < other {
            self
        } else {
            other
        }
    }

    fn max(self: i16, other: i16) -> i16 {
        if self > other {
            self
        } else {
            other
        }
    }

    fn mag(self: i16) -> i16 {
        self
    }

    fn is_neg(self: i16) -> bool {
        self < 0
    }

    fn xor(lhs: i16, rhs: i16) -> bool {
        if (lhs == 0 || rhs == 0) && lhs != rhs {
            true
        } else {
            false
        }
    }

    fn or(lhs: i16, rhs: i16) -> bool {
        if lhs == 0 && rhs == 0 {
            false
        } else {
            true
        }
    }

    fn sign(self: i16) -> i16 {
        if self == 0 {
            0_i16
        } else if self > 0 {
            1_i16
        } else {
            -1_i16
        }
    }

    fn and(lhs: i16, rhs: i16) -> bool {
        if lhs == 0 || rhs == 0 {
            false
        } else {
            true
        }
    }

    fn where(self: i16, x: i16, y: i16) -> i16 {
        if self == 0 {
            y
        } else {
            x
        }
    }

    fn NaN() -> i16 {
        panic(array!['not supported!'])
    }

    fn is_nan(self: i16) -> bool {
        panic(array!['not supported!'])
    }

    fn INF() -> i16 {
        32767
    }

    fn is_inf(self: i16) -> bool {
        self == 32767 || self == -32767
    }

    fn is_pos_inf(self: i16) -> bool {
        self == 32767
    }

    fn is_neg_inf(self: i16) -> bool {
        self == -32767
    }

    fn bitwise_and(lhs: i16, rhs: i16) -> i16 {
        panic(array!['not supported!'])
    }

    fn bitwise_xor(lhs: i16, rhs: i16) -> i16 {
        panic(array!['not supported!'])
    }

    fn bitwise_or(lhs: i16, rhs: i16) -> i16 {
        panic(array!['not supported!'])
    }

    fn add(lhs: i16, rhs: i16) -> i16 {
        lhs + rhs
    }

    fn sub(lhs: i16, rhs: i16) -> i16 {
        lhs - rhs
    }
}

impl I16Div of Div<i16> {
    fn div(lhs: i16, rhs: i16) -> i16 {
        assert(rhs != 0, 'divisor cannot be 0');

        let mut lhs_positive = lhs;
        let mut rhs_positive = rhs;

        // making sure everything is positive
        if lhs < 0 {
            lhs_positive = lhs * -1;
        }
        if rhs < 0 {
            rhs_positive = rhs * -1;
        }

        //felt252 plays role of a bridge for type casting
        let lhs_felt: felt252 = lhs_positive.into();
        let rhs_felt: felt252 = rhs_positive.into();
        let lhs_u128: u128 = lhs_felt.try_into().unwrap();
        let rhs_u128: u128 = rhs_felt.try_into().unwrap();
        let mut result = lhs_u128 / rhs_u128;
        let felt_result: felt252 = result.into();
        let signed_int_result: i16 = felt_result.try_into().unwrap();

        if lhs * rhs < 0 {
            signed_int_result * -1
        } else {
            signed_int_result
        }
    }
}

impl I16DivEq of DivEq<i16> {
    #[inline(always)]
    fn div_eq(ref self: i16, other: i16) {
        self = Div::div(self, other);
    }
}

impl I32Number of NumberTrait<i32, i32> {
    fn new(mag: i32, sign: bool) -> i32 {
        if sign {
            return -mag;
        }

        mag
    }

    fn new_unscaled(mag: i32, sign: bool) -> i32 {
        mag
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
        0
    }
    fn is_zero(self: i32) -> bool {
        self == 0
    }

    fn half() -> i32 {
        panic(array!['not supported!'])
    }

    fn one() -> i32 {
        1
    }

    fn neg_one() -> i32 {
        -1
    }

    fn is_one(self: i32) -> bool {
        self == 1
    }

    fn abs(self: i32) -> i32 {
        if self >= 0 {
            self
        } else {
            self * -1_i32
        }
    }

    fn neg(self: i32) -> i32 {
        self * -1_i32
    }

    fn min_value() -> i32 {
        -2147483647
    }

    fn max_value() -> i32 {
        2147483647
    }

    fn min(self: i32, other: i32) -> i32 {
        if self < other {
            self
        } else {
            other
        }
    }

    fn max(self: i32, other: i32) -> i32 {
        if self > other {
            self
        } else {
            other
        }
    }

    fn mag(self: i32) -> i32 {
        self
    }

    fn is_neg(self: i32) -> bool {
        self < 0
    }

    fn xor(lhs: i32, rhs: i32) -> bool {
        if (lhs == 0 || rhs == 0) && lhs != rhs {
            true
        } else {
            false
        }
    }

    fn or(lhs: i32, rhs: i32) -> bool {
        if lhs == 0 && rhs == 0 {
            false
        } else {
            true
        }
    }

    fn sign(self: i32) -> i32 {
        if self == 0 {
            0_i32
        } else if self > 0 {
            1_i32
        } else {
            -1_i32
        }
    }

    fn and(lhs: i32, rhs: i32) -> bool {
        if lhs == 0 || rhs == 0 {
            false
        } else {
            true
        }
    }

    fn where(self: i32, x: i32, y: i32) -> i32 {
        if self == 0 {
            y
        } else {
            x
        }
    }

    fn NaN() -> i32 {
        panic(array!['not supported!'])
    }

    fn is_nan(self: i32) -> bool {
        panic(array!['not supported!'])
    }

    fn INF() -> i32 {
        2147483647
    }

    fn is_inf(self: i32) -> bool {
        self == 2147483647 || self == -2147483647
    }

    fn is_pos_inf(self: i32) -> bool {
        self == 2147483647
    }

    fn is_neg_inf(self: i32) -> bool {
        self == -2147483647
    }

    fn bitwise_and(lhs: i32, rhs: i32) -> i32 {
        panic(array!['not supported!'])
    }

    fn bitwise_xor(lhs: i32, rhs: i32) -> i32 {
        panic(array!['not supported!'])
    }

    fn bitwise_or(lhs: i32, rhs: i32) -> i32 {
        panic(array!['not supported!'])
    }

    fn add(lhs: i32, rhs: i32) -> i32 {
        lhs + rhs
    }

    fn sub(lhs: i32, rhs: i32) -> i32 {
        lhs - rhs
    }
}

impl I32Div of Div<i32> {
    fn div(lhs: i32, rhs: i32) -> i32 {
        assert(rhs != 0, 'divisor cannot be 0');

        let mut lhs_positive = lhs;
        let mut rhs_positive = rhs;

        // making sure everything is positive
        if lhs < 0 {
            lhs_positive = lhs * -1;
        }
        if rhs < 0 {
            rhs_positive = rhs * -1;
        }

        //felt252 plays role of a bridge for type casting
        let lhs_felt: felt252 = lhs_positive.into();
        let rhs_felt: felt252 = rhs_positive.into();
        let lhs_u128: u128 = lhs_felt.try_into().unwrap();
        let rhs_u128: u128 = rhs_felt.try_into().unwrap();
        let mut result = lhs_u128 / rhs_u128;
        let felt_result: felt252 = result.into();
        let signed_int_result: i32 = felt_result.try_into().unwrap();

        if lhs * rhs < 0 {
            signed_int_result * -1
        } else {
            signed_int_result
        }
    }
}

impl I32DivEq of DivEq<i32> {
    #[inline(always)]
    fn div_eq(ref self: i32, other: i32) {
        self = Div::div(self, other);
    }
}

impl I32IntoU32 of Into<i32, u32> {
    fn into(self: i32) -> u32 {
        let number_sign: bool = self < 0;
        let mut self_positive: i32 = self;

        if number_sign {
            self_positive = self_positive * -1_i32
        }

        let number_felt: felt252 = self_positive.into();
        let number_u32: u32 = number_felt.try_into().unwrap();

        number_u32
    }
}

impl I64Number of NumberTrait<i64, i64> {
    fn new(mag: i64, sign: bool) -> i64 {
        if sign {
            return -mag;
        }

        mag
    }

    fn new_unscaled(mag: i64, sign: bool) -> i64 {
        mag
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
        0
    }
    fn is_zero(self: i64) -> bool {
        self == 0
    }

    fn half() -> i64 {
        panic(array!['not supported!'])
    }

    fn one() -> i64 {
        1
    }

    fn neg_one() -> i64 {
        -1
    }

    fn is_one(self: i64) -> bool {
        self == 1
    }

    fn abs(self: i64) -> i64 {
        if self >= 0 {
            self
        } else {
            self * -1_i64
        }
    }

    fn neg(self: i64) -> i64 {
        self * -1_i64
    }

    fn min_value() -> i64 {
        -9223372036854775807
    }

    fn max_value() -> i64 {
        9223372036854775807
    }

    fn min(self: i64, other: i64) -> i64 {
        if self < other {
            self
        } else {
            other
        }
    }

    fn max(self: i64, other: i64) -> i64 {
        if self > other {
            self
        } else {
            other
        }
    }

    fn mag(self: i64) -> i64 {
        self
    }

    fn is_neg(self: i64) -> bool {
        self < 0
    }

    fn xor(lhs: i64, rhs: i64) -> bool {
        if (lhs == 0 || rhs == 0) && lhs != rhs {
            true
        } else {
            false
        }
    }

    fn or(lhs: i64, rhs: i64) -> bool {
        if lhs == 0 && rhs == 0 {
            false
        } else {
            true
        }
    }

    fn sign(self: i64) -> i64 {
        if self == 0 {
            0_i64
        } else if self > 0 {
            1_i64
        } else {
            -1_i64
        }
    }

    fn and(lhs: i64, rhs: i64) -> bool {
        if lhs == 0 || rhs == 0 {
            false
        } else {
            true
        }
    }

    fn where(self: i64, x: i64, y: i64) -> i64 {
        if self == 0 {
            y
        } else {
            x
        }
    }

    fn NaN() -> i64 {
        panic(array!['not supported!'])
    }

    fn is_nan(self: i64) -> bool {
        panic(array!['not supported!'])
    }

    fn INF() -> i64 {
        9223372036854775807
    }

    fn is_inf(self: i64) -> bool {
        self == 9223372036854775807 || self == -9223372036854775807
    }

    fn is_pos_inf(self: i64) -> bool {
        self == 9223372036854775807
    }

    fn is_neg_inf(self: i64) -> bool {
        self == -9223372036854775807
    }

    fn bitwise_and(lhs: i64, rhs: i64) -> i64 {
        panic(array!['not supported!'])
    }

    fn bitwise_xor(lhs: i64, rhs: i64) -> i64 {
        panic(array!['not supported!'])
    }

    fn bitwise_or(lhs: i64, rhs: i64) -> i64 {
        panic(array!['not supported!'])
    }

    fn add(lhs: i64, rhs: i64) -> i64 {
        lhs + rhs
    }

    fn sub(lhs: i64, rhs: i64) -> i64 {
        lhs - rhs
    }
}

impl I64Div of Div<i64> {
    fn div(lhs: i64, rhs: i64) -> i64 {
        assert(rhs != 0, 'divisor cannot be 0');

        let mut lhs_positive = lhs;
        let mut rhs_positive = rhs;

        // making sure everything is positive
        if lhs < 0 {
            lhs_positive = lhs * -1;
        }
        if rhs < 0 {
            rhs_positive = rhs * -1;
        }

        //felt252 plays role of a bridge for type casting
        let lhs_felt: felt252 = lhs_positive.into();
        let rhs_felt: felt252 = rhs_positive.into();
        let lhs_u128: u128 = lhs_felt.try_into().unwrap();
        let rhs_u128: u128 = rhs_felt.try_into().unwrap();
        let mut result = lhs_u128 / rhs_u128;
        let felt_result: felt252 = result.into();
        let signed_int_result: i64 = felt_result.try_into().unwrap();

        if lhs * rhs < 0 {
            signed_int_result * -1
        } else {
            signed_int_result
        }
    }
}

impl I64DivEq of DivEq<i64> {
    #[inline(always)]
    fn div_eq(ref self: i64, other: i64) {
        self = Div::div(self, other);
    }
}

impl I128Number of NumberTrait<i128, i128> {
    fn new(mag: i128, sign: bool) -> i128 {
        if sign {
            return -mag;
        }

        mag
    }

    fn new_unscaled(mag: i128, sign: bool) -> i128 {
        mag
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
        0
    }
    fn is_zero(self: i128) -> bool {
        self == 0
    }

    fn half() -> i128 {
        panic(array!['not supported!'])
    }

    fn one() -> i128 {
        1
    }

    fn neg_one() -> i128 {
        -1
    }

    fn is_one(self: i128) -> bool {
        self == 1
    }

    fn abs(self: i128) -> i128 {
        if self >= 0 {
            self
        } else {
            self * -1_i128
        }
    }

    fn neg(self: i128) -> i128 {
        self * -1_i128
    }

    fn min_value() -> i128 {
        -170141183460469231731687303715884105727
    }

    fn max_value() -> i128 {
        170141183460469231731687303715884105727
    }

    fn min(self: i128, other: i128) -> i128 {
        if self < other {
            self
        } else {
            other
        }
    }

    fn max(self: i128, other: i128) -> i128 {
        if self > other {
            self
        } else {
            other
        }
    }

    fn mag(self: i128) -> i128 {
        self
    }

    fn is_neg(self: i128) -> bool {
        self < 0
    }

    fn xor(lhs: i128, rhs: i128) -> bool {
        if (lhs == 0 || rhs == 0) && lhs != rhs {
            true
        } else {
            false
        }
    }

    fn or(lhs: i128, rhs: i128) -> bool {
        if lhs == 0 && rhs == 0 {
            false
        } else {
            true
        }
    }

    fn sign(self: i128) -> i128 {
        if self == 0 {
            0_i128
        } else if self > 0 {
            1_i128
        } else {
            -1_i128
        }
    }

    fn and(lhs: i128, rhs: i128) -> bool {
        if lhs == 0 || rhs == 0 {
            false
        } else {
            true
        }
    }

    fn where(self: i128, x: i128, y: i128) -> i128 {
        if self == 0 {
            y
        } else {
            x
        }
    }

    fn NaN() -> i128 {
        panic(array!['not supported!'])
    }

    fn is_nan(self: i128) -> bool {
        panic(array!['not supported!'])
    }

    fn INF() -> i128 {
        170141183460469231731687303715884105727
    }

    fn is_inf(self: i128) -> bool {
        self == 170141183460469231731687303715884105727
            || self == -170141183460469231731687303715884105727
    }

    fn is_pos_inf(self: i128) -> bool {
        self == 170141183460469231731687303715884105727
    }

    fn is_neg_inf(self: i128) -> bool {
        self == -170141183460469231731687303715884105727
    }

    fn bitwise_and(lhs: i128, rhs: i128) -> i128 {
        panic(array!['not supported!'])
    }

    fn bitwise_xor(lhs: i128, rhs: i128) -> i128 {
        panic(array!['not supported!'])
    }

    fn bitwise_or(lhs: i128, rhs: i128) -> i128 {
        panic(array!['not supported!'])
    }

    fn add(lhs: i128, rhs: i128) -> i128 {
        lhs + rhs
    }

    fn sub(lhs: i128, rhs: i128) -> i128 {
        lhs - rhs
    }
}

impl I128Div of Div<i128> {
    fn div(lhs: i128, rhs: i128) -> i128 {
        assert(rhs != 0, 'divisor cannot be 0');

        let mut lhs_positive = lhs;
        let mut rhs_positive = rhs;

        // making sure everything is positive
        if lhs < 0 {
            lhs_positive = lhs * -1;
        }
        if rhs < 0 {
            rhs_positive = rhs * -1;
        }

        //felt252 plays role of a bridge for type casting
        let lhs_felt: felt252 = lhs_positive.into();
        let rhs_felt: felt252 = rhs_positive.into();
        let lhs_u128: u128 = lhs_felt.try_into().unwrap();
        let rhs_u128: u128 = rhs_felt.try_into().unwrap();
        let mut result = lhs_u128 / rhs_u128;
        let felt_result: felt252 = result.into();
        let signed_int_result: i128 = felt_result.try_into().unwrap();

        // assigning the sign and returning
        if lhs * rhs < 0 {
            signed_int_result * -1
        } else {
            signed_int_result
        }
    }
}

impl I128DivEq of DivEq<i128> {
    #[inline(always)]
    fn div_eq(ref self: i128, other: i128) {
        self = Div::div(self, other);
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

    fn half() -> u32 {
        panic(array!['not supported!'])
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

    fn neg(self: u32) -> u32 {
        panic(array!['not supported'])
    }

    fn min_value() -> u32 {
        0
    }

    fn max_value() -> u32 {
        4294967295
    }

    fn min(self: u32, other: u32) -> u32 {
        if self < other {
            self
        } else {
            other
        }
    }

    fn max(self: u32, other: u32) -> u32 {
        if self > other {
            self
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
            true
        } else {
            false
        }
    }

    fn or(lhs: u32, rhs: u32) -> bool {
        if lhs == 0 && rhs == 0 {
            false
        } else {
            true
        }
    }

    fn sign(self: u32) -> u32 {
        panic(array!['not supported!'])
    }

    fn and(lhs: u32, rhs: u32) -> bool {
        if lhs == 0 || rhs == 0 {
            false
        } else {
            true
        }
    }

    fn where(self: u32, x: u32, y: u32) -> u32 {
        if self == 0 {
            y
        } else {
            x
        }
    }

    fn NaN() -> u32 {
        4242424242
    }

    fn is_nan(self: u32) -> bool {
        self == 4242424242
    }

    fn INF() -> u32 {
        4294967295
    }

    fn is_inf(self: u32) -> bool {
        self == 4294967295
    }

    fn is_pos_inf(self: u32) -> bool {
        self == 4294967295
    }

    fn is_neg_inf(self: u32) -> bool {
        panic(array!['not supported!'])
    }

    fn bitwise_and(lhs: u32, rhs: u32) -> u32 {
        lhs & rhs
    }

    fn bitwise_xor(lhs: u32, rhs: u32) -> u32 {
        lhs ^ rhs
    }

    fn bitwise_or(lhs: u32, rhs: u32) -> u32 {
        lhs | rhs
    }

    fn add(lhs: u32, rhs: u32) -> u32 {
        lhs + rhs
    }

    fn sub(lhs: u32, rhs: u32) -> u32 {
        lhs - rhs
    }
}


use orion::numbers::complex_number::complex_trait::ComplexTrait;
use orion::numbers::complex_number::complex64::{
    Complex64Impl, complex64, Complex64Add, Complex64Sub
};


impl Complex64Number of NumberTrait<complex64, FP64x64> {
    fn new(mag: FP64x64, sign: bool) -> complex64 {
        panic(array!['not supported!'])
    }

    fn new_unscaled(mag: FP64x64, sign: bool) -> complex64 {
        panic(array!['not supported!'])
    }

    fn from_felt(val: felt252) -> complex64 {
        panic(array!['not supported!'])
    }

    fn ceil(self: complex64) -> complex64 {
        panic(array!['not supported!'])
    }

    fn exp(self: complex64) -> complex64 {
        Complex64Impl::exp(self)
    }

    fn exp2(self: complex64) -> complex64 {
        Complex64Impl::exp2(self)
    }

    fn floor(self: complex64) -> complex64 {
        panic(array!['not supported!'])
    }

    fn ln(self: complex64) -> complex64 {
        Complex64Impl::ln(self)
    }

    fn log2(self: complex64) -> complex64 {
        Complex64Impl::log2(self)
    }

    fn log10(self: complex64) -> complex64 {
        Complex64Impl::log10(self)
    }

    fn pow(self: complex64, b: complex64) -> complex64 {
        Complex64Impl::pow(self, b)
    }

    fn round(self: complex64) -> complex64 {
        panic(array!['not supported!'])
    }

    fn sqrt(self: complex64) -> complex64 {
        Complex64Impl::sqrt(self)
    }

    fn acos(self: complex64) -> complex64 {
        Complex64Impl::acos(self)
    }

    fn asin(self: complex64) -> complex64 {
        Complex64Impl::asin(self)
    }

    fn atan(self: complex64) -> complex64 {
        Complex64Impl::atan(self)
    }

    fn cos(self: complex64) -> complex64 {
        Complex64Impl::cos(self)
    }

    fn sin(self: complex64) -> complex64 {
        Complex64Impl::sin(self)
    }

    fn tan(self: complex64) -> complex64 {
        Complex64Impl::tan(self)
    }

    fn acosh(self: complex64) -> complex64 {
        Complex64Impl::acosh(self)
    }

    fn asinh(self: complex64) -> complex64 {
        Complex64Impl::asinh(self)
    }

    fn atanh(self: complex64) -> complex64 {
        Complex64Impl::atanh(self)
    }

    fn cosh(self: complex64) -> complex64 {
        Complex64Impl::cosh(self)
    }

    fn sinh(self: complex64) -> complex64 {
        Complex64Impl::sinh(self)
    }

    fn tanh(self: complex64) -> complex64 {
        Complex64Impl::tanh(self)
    }

    fn zero() -> complex64 {
        Complex64Impl::zero()
    }

    fn is_zero(self: complex64) -> bool {
        if self == Complex64Impl::zero() {
            return true;
        }

        false
    }

    fn half() -> complex64 {
        panic(array!['not supported!'])
    }

    fn one() -> complex64 {
        Complex64Impl::one()
    }

    fn neg_one() -> complex64 {
        Complex64Impl::new(FP64x64 { mag: core_fp64x64::ONE, sign: true }, FP64x64Impl::ZERO())
    }

    fn is_one(self: complex64) -> bool {
        if self == Complex64Impl::one() {
            return true;
        }

        false
    }

    fn abs(self: complex64) -> complex64 {
        Complex64Impl::new(Complex64Impl::mag(self), FP64x64Impl::ZERO())
    }

    fn neg(self: complex64) -> complex64 {
        panic(array!['not supported!'])
    }

    fn min_value() -> complex64 {
        panic(array!['not supported!'])
    }

    fn max_value() -> complex64 {
        panic(array!['not supported!'])
    }

    fn min(self: complex64, other: complex64) -> complex64 {
        panic(array!['not supported!'])
    }

    fn max(self: complex64, other: complex64) -> complex64 {
        panic(array!['not supported!'])
    }

    fn mag(self: complex64) -> FP64x64 {
        Complex64Impl::mag(self)
    }

    fn is_neg(self: complex64) -> bool {
        panic(array!['not supported!'])
    }

    fn xor(lhs: complex64, rhs: complex64) -> bool {
        panic(array!['not supported!'])
    }

    fn or(lhs: complex64, rhs: complex64) -> bool {
        panic(array!['not supported!'])
    }

    fn sign(self: complex64) -> complex64 {
        panic(array!['not supported!'])
    }

    fn and(lhs: complex64, rhs: complex64) -> bool {
        panic(array!['not supported!'])
    }

    fn where(self: complex64, x: complex64, y: complex64) -> complex64 {
        panic(array!['not supported!'])
    }

    fn NaN() -> complex64 {
        panic(array!['not supported!'])
    }

    fn is_nan(self: complex64) -> bool {
        panic(array!['not supported!'])
    }

    fn INF() -> complex64 {
        panic(array!['not supported!'])
    }

    fn is_inf(self: complex64) -> bool {
        panic(array!['not supported!'])
    }

    fn is_pos_inf(self: complex64) -> bool {
        panic(array!['not supported!'])
    }

    fn is_neg_inf(self: complex64) -> bool {
        panic(array!['not supported!'])
    }

    fn bitwise_and(lhs: complex64, rhs: complex64) -> complex64 {
        panic(array!['not supported!'])
    }

    fn bitwise_xor(lhs: complex64, rhs: complex64) -> complex64 {
        panic(array!['not supported!'])
    }

    fn bitwise_or(lhs: complex64, rhs: complex64) -> complex64 {
        panic(array!['not supported!'])
    }

    fn add(lhs: complex64, rhs: complex64) -> complex64 {
        Complex64Add::add(lhs, rhs)
    }

    fn sub(lhs: complex64, rhs: complex64) -> complex64 {
        Complex64Sub::sub(lhs, rhs)
    }
}

impl U32IntoI32 of Into<u32, i32> {
    fn into(self: u32) -> i32 {
        let number_felt: felt252 = self.into();
        let number_i32: i32 = number_felt.try_into().unwrap();

        number_i32
    }
}
