use debug::PrintTrait;
use option::OptionTrait;
use traits::Into;

use onnx_cairo::numbers::fixed_point::core;

/// CONSTANTS

const PRIME: felt252 = 3618502788666131213697322783095070105623107215331596699973092056135872020480;
const HALF_PRIME: felt252 =
    1809251394333065606848661391547535052811553607665798349986546028067936010240;
const ONE: felt252 = 67108864; // 2 ** 26
const ONE_u128: u128 = 67108864_u128; // 2 ** 26
const ONE_u64: u64 = 67108864_u64; // 2 ** 26
const HALF: felt252 = 33554432; // 2 ** 25
const HALF_u128: u128 = 33554432_u128; // 2 ** 25
const MAX_u128: u128 = 2147483647_u128; // 2 ** 31 - 1

/// STRUCTS

/// A structure representing a fixed point number.
#[derive(Copy, Drop)]
struct FixedType {
    mag: u128,
    sign: bool
}

/// TRAITS

/// A trait for fixed point numbers.
///
/// # Functions
/// * `new` - Creates a new FixedType instance with the specified magnitude and sign.
/// * `new_unscaled` - Creates a new FixedType instance with the specified unscaled magnitude and sign.
/// * `from_felt` - Creates a new FixedType instance from a felt252 value.
/// * `from_unscaled_felt` - Creates a new FixedType instance from an unscaled felt252 value.
/// * `abs` - Returns the absolute value of the fixed point number.
/// * `ceil` - Returns the smallest integer greater than or equal to the fixed point number.
/// * `exp` - Returns the value of e raised to the power of the fixed point number.
/// * `exp2` - Returns the value of 2 raised to the power of the fixed point number.
/// * `floor` - Returns the largest integer less than or equal to the fixed point number.
/// * `ln` - Returns the natural logarithm of the fixed point number.
/// * `log2` - Returns the base-2 logarithm of the fixed point number.
/// * `log10` - Returns the base-10 logarithm of the fixed point number.
/// * `pow` - Returns the result of raising the fixed point number to the power of another fixed point number.
/// * `round` - Rounds the fixed point number to the nearest whole number.
/// * `sqrt` - Returns the square root of the fixed point number.
trait Fixed {
    fn new(mag: u128, sign: bool) -> FixedType;
    fn new_unscaled(mag: u128, sign: bool) -> FixedType;
    fn from_felt(val: felt252) -> FixedType;
    fn from_unscaled_felt(val: felt252) -> FixedType;
    fn abs(self: FixedType) -> FixedType;
    fn ceil(self: FixedType) -> FixedType;
    fn exp(self: FixedType) -> FixedType;
    fn exp2(self: FixedType) -> FixedType;
    fn floor(self: FixedType) -> FixedType;
    fn ln(self: FixedType) -> FixedType;
    fn log2(self: FixedType) -> FixedType;
    fn log10(self: FixedType) -> FixedType;
    fn pow(self: FixedType, b: FixedType) -> FixedType;
    fn round(self: FixedType) -> FixedType;
    fn sqrt(self: FixedType) -> FixedType;
}

/// IMPLS

impl FixedImpl of Fixed {
    fn new(mag: u128, sign: bool) -> FixedType {
        return FixedType { mag: mag, sign: sign };
    }

    fn new_unscaled(mag: u128, sign: bool) -> FixedType {
        return Fixed::new(mag * ONE_u128, sign);
    }

    fn from_felt(val: felt252) -> FixedType {
        let mag = integer::u128_try_from_felt252(_felt_abs(val)).unwrap();
        return Fixed::new(mag, _felt_sign(val));
    }

    fn from_unscaled_felt(val: felt252) -> FixedType {
        return Fixed::from_felt(val * ONE);
    }

    fn abs(self: FixedType) -> FixedType {
        return core::abs(self);
    }


    fn ceil(self: FixedType) -> FixedType {
        return core::ceil(self);
    }


    fn floor(self: FixedType) -> FixedType {
        return core::floor(self);
    }

    fn exp(self: FixedType) -> FixedType {
        return core::exp(self);
    }

    fn exp2(self: FixedType) -> FixedType {
        return core::exp2(self);
    }

    fn ln(self: FixedType) -> FixedType {
        return core::ln(self);
    }

    fn log2(self: FixedType) -> FixedType {
        return core::log2(self);
    }

    fn log10(self: FixedType) -> FixedType {
        return core::log10(self);
    }

    fn pow(self: FixedType, b: FixedType) -> FixedType {
        return core::pow(self, b);
    }

    fn round(self: FixedType) -> FixedType {
        return core::round(self);
    }


    fn sqrt(self: FixedType) -> FixedType {
        return core::sqrt(self);
    }
}

impl FixedPrint of PrintTrait<FixedType> {
    fn print(self: FixedType) {
        self.sign.print();
        self.mag.into().print();
    }
}

impl FixedInto of Into<FixedType, felt252> {
    fn into(self: FixedType) -> felt252 {
        let mag_felt = self.mag.into();

        if (self.sign == true) {
            return mag_felt * -1;
        } else {
            return mag_felt;
        }
    }
}

impl FixedPartialEq of PartialEq<FixedType> {
    #[inline(always)]
    fn eq(lhs: FixedType, rhs: FixedType) -> bool {
        return core::eq(lhs, rhs);
    }

    #[inline(always)]
    fn ne(lhs: FixedType, rhs: FixedType) -> bool {
        return core::ne(lhs, rhs);
    }
}

impl FixedAdd of Add<FixedType> {
    fn add(lhs: FixedType, rhs: FixedType) -> FixedType {
        return core::add(lhs, rhs);
    }
}

impl FixedAddEq of AddEq<FixedType> {
    #[inline(always)]
    fn add_eq(ref self: FixedType, other: FixedType) {
        self = Add::add(self, other);
    }
}

impl FixedSub of Sub<FixedType> {
    fn sub(lhs: FixedType, rhs: FixedType) -> FixedType {
        return core::sub(lhs, rhs);
    }
}

impl FixedSubEq of SubEq<FixedType> {
    #[inline(always)]
    fn sub_eq(ref self: FixedType, other: FixedType) {
        self = Sub::sub(self, other);
    }
}

impl FixedMul of Mul<FixedType> {
    fn mul(lhs: FixedType, rhs: FixedType) -> FixedType {
        return core::mul(lhs, rhs);
    }
}

impl FixedMulEq of MulEq<FixedType> {
    #[inline(always)]
    fn mul_eq(ref self: FixedType, other: FixedType) {
        self = Mul::mul(self, other);
    }
}

impl FixedDiv of Div<FixedType> {
    fn div(lhs: FixedType, rhs: FixedType) -> FixedType {
        return core::div(lhs, rhs);
    }
}

impl FixedDivEq of DivEq<FixedType> {
    #[inline(always)]
    fn div_eq(ref self: FixedType, other: FixedType) {
        self = Div::div(self, other);
    }
}

impl FixedPartialOrd of PartialOrd<FixedType> {
    #[inline(always)]
    fn ge(lhs: FixedType, rhs: FixedType) -> bool {
        return core::ge(lhs, rhs);
    }

    #[inline(always)]
    fn gt(lhs: FixedType, rhs: FixedType) -> bool {
        return core::gt(lhs, rhs);
    }

    #[inline(always)]
    fn le(lhs: FixedType, rhs: FixedType) -> bool {
        return core::le(lhs, rhs);
    }

    #[inline(always)]
    fn lt(lhs: FixedType, rhs: FixedType) -> bool {
        return core::lt(lhs, rhs);
    }
}

impl FixedNeg of Neg<FixedType> {
    #[inline(always)]
    fn neg(a: FixedType) -> FixedType {
        return core::neg(a);
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

fn _split_unsigned(a: FixedType) -> (u128, u128) {
    return integer::u128_safe_divmod(a.mag, integer::u128_as_non_zero(ONE_u128));
}
