use option::OptionTrait;
use debug::PrintTrait;
use traits::{Into, TryInto};

use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
use orion::numbers::fixed_point::math::math_8x23;
use orion::numbers::signed_integer::{i32::i32, i8::i8};

const PRIME: felt252 = 3618502788666131213697322783095070105623107215331596699973092056135872020480;
const HALF_PRIME: felt252 =
    1809251394333065606848661391547535052811553607665798349986546028067936010240;
const ONE: u128 = 8388608; // 2 ** 23
const ONE_u64: u64 = 8388608; // 2 ** 23
const HALF: u128 = 4194304; // 2 ** 22
const MAX: u128 = 2147483647; // 2 ** 31 - 1
const MIN_MAG: u128 = 2147483648; // 2 ** 31 
const PI: u128 = 26353589_u128;
const HALF_PI: u128 = 13176794_u128;

/// IMPLS

impl FP8x23Impl of FixedTrait {
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
        return math_8x23::abs(self);
    }


    fn ceil(self: FixedType) -> FixedType {
        return math_8x23::ceil(self);
    }


    fn floor(self: FixedType) -> FixedType {
        return math_8x23::floor(self);
    }

    fn exp(self: FixedType) -> FixedType {
        return math_8x23::exp(self);
    }

    fn exp2(self: FixedType) -> FixedType {
        return math_8x23::exp2(self);
    }

    fn ln(self: FixedType) -> FixedType {
        return math_8x23::ln(self);
    }

    fn log2(self: FixedType) -> FixedType {
        return math_8x23::log2(self);
    }

    fn log10(self: FixedType) -> FixedType {
        return math_8x23::log10(self);
    }

    fn pow(self: FixedType, b: FixedType) -> FixedType {
        return math_8x23::pow(self, b);
    }

    fn round(self: FixedType) -> FixedType {
        return math_8x23::round(self);
    }


    fn sqrt(self: FixedType) -> FixedType {
        return math_8x23::sqrt(self);
    }

    fn sin(self: FixedType) -> FixedType {
        return math_8x23::sin(self);
    }

    fn cos(self: FixedType) -> FixedType {
        return math_8x23::cos(self);
    }

    fn asin(self: FixedType) -> FixedType {
        return math_8x23::asin(self);
    }

    fn sinh(self: FixedType) -> FixedType {
        return math_8x23::sinh(self);
    }

    fn tanh(self: FixedType) -> FixedType {
        return math_8x23::tanh(self);
    }

    fn cosh(self: FixedType) -> FixedType {
        return math_8x23::cosh(self);
    }

    fn acosh(self: FixedType) -> FixedType {
        return math_8x23::acosh(self);
    }

    fn asinh(self: FixedType) -> FixedType {
        return math_8x23::asinh(self);
    }

    fn atan(self: FixedType) -> FixedType {
        return math_8x23::atan(self);
    }

    fn acos(self: FixedType) -> FixedType {
        return math_8x23::acos(self);
    }
}

impl FP8x23Print of PrintTrait<FixedType> {
    fn print(self: FixedType) {
        self.sign.print();
        self.mag.print();
    }
}

impl FP8x23Into of Into<FixedType, felt252> {
    fn into(self: FixedType) -> felt252 {
        let mag_felt = self.mag.into();

        if (self.sign == true) {
            return mag_felt * -1;
        } else {
            return mag_felt;
        }
    }
}

impl FP8x23PartialEq of PartialEq<FixedType> {
    #[inline(always)]
    fn eq(lhs: @FixedType, rhs: @FixedType) -> bool {
        return math_8x23::equal(*lhs, *rhs);
    }

    #[inline(always)]
    fn ne(lhs: @FixedType, rhs: @FixedType) -> bool {
        return math_8x23::ne(*lhs, *rhs);
    }
}

impl FP8x23Add of Add<FixedType> {
    fn add(lhs: FixedType, rhs: FixedType) -> FixedType {
        return math_8x23::add(lhs, rhs);
    }
}

impl FP8x23AddEq of AddEq<FixedType> {
    #[inline(always)]
    fn add_eq(ref self: FixedType, other: FixedType) {
        self = Add::add(self, other);
    }
}

impl FP8x23Sub of Sub<FixedType> {
    fn sub(lhs: FixedType, rhs: FixedType) -> FixedType {
        return math_8x23::sub(lhs, rhs);
    }
}

impl FP8x23SubEq of SubEq<FixedType> {
    #[inline(always)]
    fn sub_eq(ref self: FixedType, other: FixedType) {
        self = Sub::sub(self, other);
    }
}

impl FP8x23Mul of Mul<FixedType> {
    fn mul(lhs: FixedType, rhs: FixedType) -> FixedType {
        return math_8x23::mul(lhs, rhs);
    }
}

impl FP8x23MulEq of MulEq<FixedType> {
    #[inline(always)]
    fn mul_eq(ref self: FixedType, other: FixedType) {
        self = Mul::mul(self, other);
    }
}

impl FP8x23Div of Div<FixedType> {
    fn div(lhs: FixedType, rhs: FixedType) -> FixedType {
        return math_8x23::div(lhs, rhs);
    }
}

impl FP8x23DivEq of DivEq<FixedType> {
    #[inline(always)]
    fn div_eq(ref self: FixedType, other: FixedType) {
        self = Div::div(self, other);
    }
}

impl FP8x23PartialOrd of PartialOrd<FixedType> {
    #[inline(always)]
    fn ge(lhs: FixedType, rhs: FixedType) -> bool {
        return math_8x23::ge(lhs, rhs);
    }

    #[inline(always)]
    fn gt(lhs: FixedType, rhs: FixedType) -> bool {
        return math_8x23::gt(lhs, rhs);
    }

    #[inline(always)]
    fn le(lhs: FixedType, rhs: FixedType) -> bool {
        return math_8x23::le(lhs, rhs);
    }

    #[inline(always)]
    fn lt(lhs: FixedType, rhs: FixedType) -> bool {
        return math_8x23::lt(lhs, rhs);
    }
}

impl FP8x23Neg of Neg<FixedType> {
    #[inline(always)]
    fn neg(a: FixedType) -> FixedType {
        return math_8x23::neg(a);
    }
}

impl FP8x23TryIntoI32 of TryInto<FixedType, i32> {
    fn try_into(self: FixedType) -> Option<i32> {
        _i32_try_from_fp(self)
    }
}

impl FP8x23TryIntoI8 of TryInto<FixedType, i8> {
    fn try_into(self: FixedType) -> Option<i8> {
        _i8_try_from_fp(self)
    }
}

impl FP8x23TryIntoU32 of TryInto<FixedType, u32> {
    fn try_into(self: FixedType) -> Option<u32> {
        _u32_try_from_fp(self)
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
        return a * 1;
    }
}

fn _i32_try_from_fp(x: FixedType) -> Option<i32> {
    let unscaled_mag: Option<u32> = (x.mag / ONE).try_into();

    match unscaled_mag {
        Option::Some(val) => Option::Some(i32 { mag: unscaled_mag.unwrap(), sign: x.sign }),
        Option::None(_) => Option::None(())
    }
}

fn _u32_try_from_fp(x: FixedType) -> Option<u32> {
    let unscaled: Option<u32> = (x.mag / ONE).try_into();

    match unscaled {
        Option::Some(val) => Option::Some(unscaled.unwrap()),
        Option::None(_) => Option::None(())
    }
}

fn _i8_try_from_fp(x: FixedType) -> Option<i8> {
    let unscaled_mag: Option<u8> = (x.mag / ONE).try_into();

    match unscaled_mag {
        Option::Some(val) => Option::Some(i8 { mag: unscaled_mag.unwrap(), sign: x.sign }),
        Option::None(_) => Option::None(())
    }
}

