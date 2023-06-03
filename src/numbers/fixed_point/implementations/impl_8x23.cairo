use option::OptionTrait;
use debug::PrintTrait;
use traits::Into;

use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
use orion::numbers::fixed_point::math::math_8x23;

#[derive(Copy, Drop)]
extern type fp8x23;

const PRIME: felt252 = 3618502788666131213697322783095070105623107215331596699973092056135872020480;
const HALF_PRIME: felt252 =
    1809251394333065606848661391547535052811553607665798349986546028067936010240;
const ONE: u128 = 8388608; // 2 ** 23
const ONE_u64: u64 = 8388608; // 2 ** 23
const HALF: u128 = 4194304; // 2 ** 22
const MAX: u128 = 2147483647; // 2 ** 31 - 1

/// IMPLS

impl FP8x23Impl of FixedTrait<fp8x23> {
    fn new(mag: u128, sign: bool) -> FixedType<fp8x23> {
        if sign == true {
            assert(mag <= MAX, 'fixed type: out of range');
        } else {
            assert(mag <= MAX - 1_u128, 'fixed type: out of range');
        }
        return FixedType { mag: mag, sign: sign };
    }

    fn new_unscaled(mag: u128, sign: bool) -> FixedType<fp8x23> {
        return FixedTrait::new(mag * ONE, sign);
    }

    fn from_felt(val: felt252) -> FixedType<fp8x23> {
        let mag = integer::u128_try_from_felt252(_felt_abs(val)).unwrap();
        return FixedTrait::new(mag, _felt_sign(val));
    }

    fn from_unscaled_felt(val: felt252) -> FixedType<fp8x23> {
        return FixedTrait::from_felt(val * ONE.into());
    }

    fn abs(self: FixedType<fp8x23>) -> FixedType<fp8x23> {
        return math_8x23::abs(self);
    }


    fn ceil(self: FixedType<fp8x23>) -> FixedType<fp8x23> {
        return math_8x23::ceil(self);
    }


    fn floor(self: FixedType<fp8x23>) -> FixedType<fp8x23> {
        return math_8x23::floor(self);
    }

    fn exp(self: FixedType<fp8x23>) -> FixedType<fp8x23> {
        return math_8x23::exp(self);
    }

    fn exp2(self: FixedType<fp8x23>) -> FixedType<fp8x23> {
        return math_8x23::exp2(self);
    }

    fn ln(self: FixedType<fp8x23>) -> FixedType<fp8x23> {
        return math_8x23::ln(self);
    }

    fn log2(self: FixedType<fp8x23>) -> FixedType<fp8x23> {
        return math_8x23::log2(self);
    }

    fn log10(self: FixedType<fp8x23>) -> FixedType<fp8x23> {
        return math_8x23::log10(self);
    }

    fn pow(self: FixedType<fp8x23>, b: FixedType<fp8x23>) -> FixedType<fp8x23> {
        return math_8x23::pow(self, b);
    }

    fn round(self: FixedType<fp8x23>) -> FixedType<fp8x23> {
        return math_8x23::round(self);
    }


    fn sqrt(self: FixedType<fp8x23>) -> FixedType<fp8x23> {
        return math_8x23::sqrt(self);
    }
}

impl FP8x23Print of PrintTrait<FixedType<fp8x23>> {
    fn print(self: FixedType<fp8x23>) {
        self.sign.print();
        self.mag.print();
    }
}

impl FP8x23Into of Into<FixedType<fp8x23>, felt252> {
    fn into(self: FixedType<fp8x23>) -> felt252 {
        let mag_felt = self.mag.into();

        if (self.sign == true) {
            return mag_felt * -1;
        } else {
            return mag_felt;
        }
    }
}

impl FP8x23PartialEq of PartialEq<FixedType<fp8x23>> {
    #[inline(always)]
    fn eq(lhs: FixedType<fp8x23>, rhs: FixedType<fp8x23>) -> bool {
        return math_8x23::eq(lhs, rhs);
    }

    #[inline(always)]
    fn ne(lhs: FixedType<fp8x23>, rhs: FixedType<fp8x23>) -> bool {
        return math_8x23::ne(lhs, rhs);
    }
}

impl FP8x23Add of Add<FixedType<fp8x23>> {
    fn add(lhs: FixedType<fp8x23>, rhs: FixedType<fp8x23>) -> FixedType<fp8x23> {
        return math_8x23::add(lhs, rhs);
    }
}

impl FP8x23AddEq of AddEq<FixedType<fp8x23>> {
    #[inline(always)]
    fn add_eq(ref self: FixedType<fp8x23>, other: FixedType<fp8x23>) {
        self = Add::add(self, other);
    }
}

impl FP8x23Sub of Sub<FixedType<fp8x23>> {
    fn sub(lhs: FixedType<fp8x23>, rhs: FixedType<fp8x23>) -> FixedType<fp8x23> {
        return math_8x23::sub(lhs, rhs);
    }
}

impl FP8x23SubEq of SubEq<FixedType<fp8x23>> {
    #[inline(always)]
    fn sub_eq(ref self: FixedType<fp8x23>, other: FixedType<fp8x23>) {
        self = Sub::sub(self, other);
    }
}

impl FP8x23Mul of Mul<FixedType<fp8x23>> {
    fn mul(lhs: FixedType<fp8x23>, rhs: FixedType<fp8x23>) -> FixedType<fp8x23> {
        return math_8x23::mul(lhs, rhs);
    }
}

impl FP8x23MulEq of MulEq<FixedType<fp8x23>> {
    #[inline(always)]
    fn mul_eq(ref self: FixedType<fp8x23>, other: FixedType<fp8x23>) {
        self = Mul::mul(self, other);
    }
}

impl FP8x23Div of Div<FixedType<fp8x23>> {
    fn div(lhs: FixedType<fp8x23>, rhs: FixedType<fp8x23>) -> FixedType<fp8x23> {
        return math_8x23::div(lhs, rhs);
    }
}

impl FP8x23DivEq of DivEq<FixedType<fp8x23>> {
    #[inline(always)]
    fn div_eq(ref self: FixedType<fp8x23>, other: FixedType<fp8x23>) {
        self = Div::div(self, other);
    }
}

impl FP8x23PartialOrd of PartialOrd<FixedType<fp8x23>> {
    #[inline(always)]
    fn ge(lhs: FixedType<fp8x23>, rhs: FixedType<fp8x23>) -> bool {
        return math_8x23::ge(lhs, rhs);
    }

    #[inline(always)]
    fn gt(lhs: FixedType<fp8x23>, rhs: FixedType<fp8x23>) -> bool {
        return math_8x23::gt(lhs, rhs);
    }

    #[inline(always)]
    fn le(lhs: FixedType<fp8x23>, rhs: FixedType<fp8x23>) -> bool {
        return math_8x23::le(lhs, rhs);
    }

    #[inline(always)]
    fn lt(lhs: FixedType<fp8x23>, rhs: FixedType<fp8x23>) -> bool {
        return math_8x23::lt(lhs, rhs);
    }
}

impl FP8x23Neg of Neg<FixedType<fp8x23>> {
    #[inline(always)]
    fn neg(a: FixedType<fp8x23>) -> FixedType<fp8x23> {
        return math_8x23::neg(a);
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
