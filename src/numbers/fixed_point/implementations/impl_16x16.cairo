use option::OptionTrait;
use debug::PrintTrait;
use traits::Into;

use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
use orion::numbers::fixed_point::math::math_16x16;

const PRIME: felt252 = 3618502788666131213697322783095070105623107215331596699973092056135872020480;
const HALF_PRIME: felt252 =
    1809251394333065606848661391547535052811553607665798349986546028067936010240;
const ONE: u128 = 65536; // 2 ** 16
const ONE_u64: u64 = 65536; // 2 ** 16
const HALF: u128 = 32768; // 2 ** 15
const MAX: u128 = 4294967295; // 2 ** 32 - 1
const MIN_MAG: u128 = 4294967296; // 2 ** 32

/// IMPLS

impl FP16x16Impl of FixedTrait {
    fn new(mag: u128, sign: bool) -> FixedType {
        // TODO: check range
        return FixedType { mag: mag, sign: sign };
    }

    fn new_unscaled(mag: u128, sign: bool) -> FixedType {
        return FixedTrait::new(mag * ONE, sign);
    }

    fn from_felt(val: felt252) -> FixedType {
        let mag = integer::u128_try_from_felt252(_felt_abs(val)).unwrap();
        return FixedTrait::new(mag, _felt_sign(val));
    }

    fn from_unscaled_felt(val: felt252) -> FixedType {
        return FixedTrait::from_felt(val * ONE.into());
    }

    fn abs(self: FixedType) -> FixedType {
        return math_16x16::abs(self);
    }


    fn ceil(self: FixedType) -> FixedType {
        return math_16x16::ceil(self);
    }


    fn floor(self: FixedType) -> FixedType {
        return math_16x16::floor(self);
    }

    fn exp(self: FixedType) -> FixedType {
        return math_16x16::exp(self);
    }

    fn exp2(self: FixedType) -> FixedType {
        return math_16x16::exp2(self);
    }

    fn ln(self: FixedType) -> FixedType {
        return math_16x16::ln(self);
    }

    fn log2(self: FixedType) -> FixedType {
        return math_16x16::log2(self);
    }

    fn log10(self: FixedType) -> FixedType {
        return math_16x16::log10(self);
    }

    fn pow(self: FixedType, b: FixedType) -> FixedType {
        return math_16x16::pow(self, b);
    }

    fn round(self: FixedType) -> FixedType {
        return math_16x16::round(self);
    }


    fn sqrt(self: FixedType) -> FixedType {
        return math_16x16::sqrt(self);
    }

    fn sinh(self: FixedType) -> FixedType {
        return math_16x16::sinh(self);
    }

    fn tanh(self: FixedType) -> FixedType {
        return math_16x16::tanh(self);
    }

    fn cosh(self: FixedType) -> FixedType {
        return math_16x16::cosh(self);
    }

    fn asinh(self: FixedType) -> FixedType {
        return math_16x16::asinh(self);
    }
}

impl FP16x16Print of PrintTrait<FixedType> {
    fn print(self: FixedType) {
        self.sign.print();
        self.mag.print();
    }
}

impl FP16x16Into of Into<FixedType, felt252> {
    fn into(self: FixedType) -> felt252 {
        let mag_felt = self.mag.into();

        if (self.sign == true) {
            return mag_felt * -1;
        } else {
            return mag_felt;
        }
    }
}

impl FP16x16PartialEq of PartialEq<FixedType> {
    #[inline(always)]
    fn eq(lhs: @FixedType, rhs: @FixedType) -> bool {
        return math_16x16::eq(*lhs, *rhs);
    }

    #[inline(always)]
    fn ne(lhs: @FixedType, rhs: @FixedType) -> bool {
        return math_16x16::ne(*lhs, *rhs);
    }
}

impl FP16x16Add of Add<FixedType> {
    fn add(lhs: FixedType, rhs: FixedType) -> FixedType {
        return math_16x16::add(lhs, rhs);
    }
}

impl FP16x16AddEq of AddEq<FixedType> {
    #[inline(always)]
    fn add_eq(ref self: FixedType, other: FixedType) {
        self = Add::add(self, other);
    }
}

impl FP16x16Sub of Sub<FixedType> {
    fn sub(lhs: FixedType, rhs: FixedType) -> FixedType {
        return math_16x16::sub(lhs, rhs);
    }
}

impl FP16x16SubEq of SubEq<FixedType> {
    #[inline(always)]
    fn sub_eq(ref self: FixedType, other: FixedType) {
        self = Sub::sub(self, other);
    }
}

impl FP16x16Mul of Mul<FixedType> {
    fn mul(lhs: FixedType, rhs: FixedType) -> FixedType {
        return math_16x16::mul(lhs, rhs);
    }
}

impl FP16x16MulEq of MulEq<FixedType> {
    #[inline(always)]
    fn mul_eq(ref self: FixedType, other: FixedType) {
        self = Mul::mul(self, other);
    }
}

impl FP16x16Div of Div<FixedType> {
    fn div(lhs: FixedType, rhs: FixedType) -> FixedType {
        return math_16x16::div(lhs, rhs);
    }
}

impl FP16x16DivEq of DivEq<FixedType> {
    #[inline(always)]
    fn div_eq(ref self: FixedType, other: FixedType) {
        self = Div::div(self, other);
    }
}

impl FP16x16PartialOrd of PartialOrd<FixedType> {
    #[inline(always)]
    fn ge(lhs: FixedType, rhs: FixedType) -> bool {
        return math_16x16::ge(lhs, rhs);
    }

    #[inline(always)]
    fn gt(lhs: FixedType, rhs: FixedType) -> bool {
        return math_16x16::gt(lhs, rhs);
    }

    #[inline(always)]
    fn le(lhs: FixedType, rhs: FixedType) -> bool {
        return math_16x16::le(lhs, rhs);
    }

    #[inline(always)]
    fn lt(lhs: FixedType, rhs: FixedType) -> bool {
        return math_16x16::lt(lhs, rhs);
    }
}

impl FP16x16Neg of Neg<FixedType> {
    #[inline(always)]
    fn neg(a: FixedType) -> FixedType {
        return math_16x16::neg(a);
    }
}

/// INTERNAL

fn _felt_sign(a: felt252) -> bool {
    return integer::u256_from_felt252(a) > integer::u256_from_felt252(HALF_PRIME);
}

fn _felt_abs(a: felt252) -> felt252 {
    let a_sign = _felt_sign(a);

    if (a_sign == true) {
        return a * -1;
    } else {
        return a;
    }
}

