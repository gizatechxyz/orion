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

/// Trait
///
/// new - Constructs a new `FixedType` instance.
/// new_unscaled - Creates a new `FixedType` instance with the specified unscaled magnitude and sign.
/// from_felt - Creates a new `FixedType` instance from a `felt252` value.
/// from_unscaled_felt - Creates a new `FixedType` instance from an unscaled `felt252` value.
/// abs - Returns the absolute value of the fixed point number.
/// ceil - Returns the smallest integer greater than or equal to the fixed point number.
/// floor - Returns the largest integer less than or equal to the fixed point number.
/// exp - Returns the value of e raised to the power of the fixed point number. 
/// exp2 - Returns the value of 2 raised to the power of the fixed point number.
/// ln - Returns the natural logarithm of the fixed point number.
/// log2 - Returns the base-2 logarithm of the fixed point number.
/// log10 - Returns the base-10 logarithm of the fixed point number.
/// pow - Returns the result of raising the fixed point number to the power of another fixed point number
/// round - Rounds the fixed point number to the nearest whole number.
/// sqrt - Returns the square root of the fixed point number.
trait Fixed {
    /// # Fixed::new
    /// 
    /// ```rust
    /// fn new(mag: u128, sign: bool) -> FixedType;
    /// ```
    /// 
    /// Constructs a new FixedType instance.
    ///
    /// ## Args
    /// 
    /// * `mag`(`u128`) - The magnitude of the fixed point.
    /// * `sign`(`bool`) - The sign of the fixed point, where `true` represents a negative number.
    ///
    /// ## Returns
    ///
    /// A new `FixedType` instance.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// fn new_fp_example() -> FixedType {
    ///     // We can call `new` function as follows. 
    ///     Fixed::new(67108864, false)
    /// }
    /// >>> {mag: 67108864, sign: false} // = 1
    /// ```
    ///
    fn new(mag: u128, sign: bool) -> FixedType;
    /// # Fixed::new\_unscaled
    /// 
    /// ```rust
    /// fn new_unscaled(mag: u128, sign: bool) -> FixedType;
    /// ```
    ///
    /// Creates a new FixedType instance with the specified unscaled magnitude and sign.
    /// 
    /// ## Args
    ///
    /// `mag`(`u128`) - The unscaled magnitude of the fixed point.
    /// `sign`(`bool`) - The sign of the fixed point, where `true` represents a negative number.
    ///
    /// ## Returns
    /// 
    /// A new `FixedType` instance.
    /// 
    /// ## Examples
    /// 
    /// ```rust
    /// fn new_unscaled_example() -> FixedType {
    ///     // We can call `new_unscaled` function as follows. 
    ///     Fixed::new_unscaled(1);
    /// }
    /// >>> {mag: 67108864, sign: false}
    /// ```
    ///
    fn new_unscaled(mag: u128, sign: bool) -> FixedType;
    /// # Fixed::from\_felt
    ///
    /// 
    /// ```rust
    /// fn from_felt(val: felt252) -> FixedType;
    /// ```
    /// 
    /// Creates a new FixedType instance from a felt252 value.
    ///
    /// ## Args
    /// 
    /// * `val`(`felt252`) - `felt252` value to convert in FixedType
    ///
    /// ## Returns 
    ///
    /// A new `FixedType` instance.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// fn from_felt_example() -> FixedType {
    ///     // We can call `from_felt` function as follows . 
    ///     Fixed::from_felt(194615706);
    /// }
    /// >>> {mag: 194615706, sign: false} // = 2.9
    /// ```
    ///
    fn from_felt(val: felt252) -> FixedType;
    ///# Fixed::from\_unscaled\_felt
    ///
    ///```rust
    ///fn from_unscaled_felt(val: felt252) -> FixedType;
    ///```
    ///
    ///Creates a new FixedType instance from an unscaled felt252 value.
    ///
    /// ## Args
    /// 
    /// `val`(`felt252`) - `felt252` value to convert in FixedType
    ///
    /// ## Returns - A new `FixedType` instance.
    ///
    /// ## Examples
    ///
    ///```rust
    ///fn from_unscaled_felt_example() -> FixedType {
    ///    // We can call `from_unscaled_felt` function as follows . 
    ///    Fixed::from_unscaled_felt(1);
    ///}
    ///>>> {mag: 67108864, sign: false}
    ///```
    /// 
    fn from_unscaled_felt(val: felt252) -> FixedType;
    /// # fp.abs
    /// 
    /// ```rust
    /// fn abs(self: FixedType) -> FixedType;
    /// ```
    /// 
    /// Returns the absolute value of the fixed point number.
    ///
    /// ## Args
    /// 
    /// * `self`(`FixedType`) - The input fixed point
    ///
    /// ## Returns
    ///
    /// The absolute value of the input fixed point number.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// fn abs_fp_example() -> FixedType {
    ///     // We instantiate fixed point here.
    ///     let fp = Fixed::from_unscaled_felt(-1);
    ///     
    ///     // We can call `abs` function as follows.
    ///     fp.abs()
    /// }
    /// >>> {mag: 67108864, sign: false} // = 1
    /// ```
    /// 
    fn abs(self: FixedType) -> FixedType;
    /// # fp.ceil
    /// 
    /// ```rust
    /// fn ceil(self: FixedType) -> FixedType;
    /// ```
    /// 
    /// Returns the smallest integer greater than or equal to the fixed point number.
    ///
    /// ## Args
    ///
    /// *`self`(`FixedType`) - The input fixed point
    ///
    /// ## Returns
    ///
    /// The smallest integer greater than or equal to the input fixed point number.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// fn ceil_fp_example() -> FixedType {
    ///     // We instantiate fixed point here.
    ///     let fp = Fixed::from_felt(194615506); // 2.9
    ///     
    ///     // We can call `ceil` function as follows.
    ///     fp.ceil()
    /// }
    /// >>> {mag: 201326592, sign: false} // = 3
    /// ```
    ///
    fn ceil(self: FixedType) -> FixedType;
    /// # fp.exp
    /// 
    /// ```rust
    /// fn exp(self: FixedType) -> FixedType;
    /// ```
    /// 
    /// Returns the value of e raised to the power of the fixed point number.
    ///
    /// ## Args
    ///
    /// * `self`(`FixedType`) - The input fixed point
    ///
    /// ## Returns
    ///
    /// The natural exponent of the input fixed point number.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// fn exp_fp_example() -> FixedType {
    ///     // We instantiate fixed point here.
    ///     let fp = Fixed::from_unscaled_felt(2);
    ///     
    ///     // We can call `exp` function as follows.
    ///     fp.exp()
    /// }
    /// >>> {mag: 495871144, sign: false} // = 7.389056317241236
    /// ``` 
    ///
    fn exp(self: FixedType) -> FixedType;
    /// # fp.exp2
    /// 
    /// ```rust
    /// fn exp2(self: FixedType) -> FixedType;
    /// ```
    /// 
    /// Returns the value of 2 raised to the power of the fixed point number.
    ///
    /// ## Args
    /// 
    /// * `self`(`FixedType`) - The input fixed point
    ///
    /// ## Returns
    ///
    /// The binary exponent of the input fixed point number.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// fn exp2_fp_example() -> FixedType {
    ///     // We instantiate fixed point here.
    ///     let fp = Fixed::from_unscaled_felt(2);
    ///     
    ///     // We can call `exp2` function as follows.
    ///     fp.exp2()
    /// }
    /// >>> {mag: 268435456, sign: false} // = 3.99999957248
    /// ``` 
    ///
    fn exp2(self: FixedType) -> FixedType;
    /// # fp.floor
    /// 
    /// ```rust
    /// fn floor(self: FixedType) -> FixedType;
    /// ```
    /// 
    /// Returns the largest integer less than or equal to the fixed point number.
    ///
    /// ## Args
    /// 
    /// * `self`(`FixedType`) - The input fixed point
    ///
    /// ## Returns
    ///
    /// Returns the largest integer less than or equal to the input fixed point number.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// fn floor_fp_example() -> FixedType {
    ///     // We instantiate fixed point here.
    ///     let fp = Fixed::from_felt(194615506); // 2.9
    ///     
    ///     // We can call `floor` function as follows.
    ///     fp.floor()
    /// }
    /// >>> {mag: 134217728, sign: false} // = 2
    /// ```
    /// 
    fn floor(self: FixedType) -> FixedType;
    /// # fp.ln
    ///
    /// 
    /// ```rust
    /// fn ln(self: FixedType) -> FixedType;
    /// ```
    /// 
    /// Returns the natural logarithm of the fixed point number.
    /// 
    /// ## Args
    ///
    /// * `self`(`FixedType`) - The input fixed point
    ///
    /// ## Returns 
    ///
    /// A FixedType value representing the natural logarithm of the input number.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// fn ln_fp_example() -> FixedType {
    ///     // We instantiate fixed point here.
    ///     let fp = Fixed::from_unscaled_felt(1);
    ///     
    ///     // We can call `ln` function as follows.
    ///     fp.ln()
    /// }
    /// >>> {mag: 0, sign: false}
    /// ```
    ///
    fn ln(self: FixedType) -> FixedType;
    /// # fp.log2
    /// 
    /// ```rust
    /// fn log2(self: FixedType) -> FixedType;
    /// ```
    /// 
    /// Returns the base-2 logarithm of the fixed point number.
    ///
    /// ## Args
    ///
    /// * `self`(`FixedType`) - The input fixed point
    ///
    /// ## Panics
    ///
    /// * Panics if the input is negative.
    ///
    /// ## Returns
    ///
    /// A FixedType value representing the binary logarithm of the input number.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// fn log2_fp_example() -> FixedType {
    ///     // We instantiate fixed point here.
    ///     let fp = Fixed::from_unscaled_felt(32);
    ///     
    ///     // We can call `log2` function as follows.
    ///     fp.log2()
    /// }
    /// >>> {mag: 335544129, sign: false} // = 4.99999995767848
    /// ```
    /// 
    fn log2(self: FixedType) -> FixedType;
    /// # fp.log10
    /// 
    /// ```rust
    /// fn log10(self: FixedType) -> FixedType;
    /// ```
    /// 
    /// Returns the base-10 logarithm of the fixed point number.
    ///
    /// ## Args
    ///
    /// * `self`(`FixedType`) - The input fixed point
    ///
    /// ## Returns
    ///
    /// A FixedType value representing the base 10 logarithm of the input number.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// fn log10_fp_example() -> FixedType {
    ///     // We instantiate fixed point here.
    ///     let fp = Fixed::from_unscaled_felt(100);
    ///     
    ///     // We can call `log10` function as follows.
    ///     fp.log10()
    /// }
    /// >>> {mag: 134217717, sign: false} // = 1.9999999873985543
    /// ```
    ///
    fn log10(self: FixedType) -> FixedType;
    /// # fp.pow
    /// 
    /// ```rust
    /// fn pow(self: FixedType, b: FixedType) -> FixedType;
    /// ```
    /// 
    /// Returns the result of raising the fixed point number to the power of another fixed point number.
    ///
    /// ## Args
    ///
    /// * `self`(`FixedType`) - The input fixed point.
    /// * `b`(`FixedType`) - The exponent fixed point number.
    ///
    /// ## Returns
    ///
    /// A fixed point number representing the result of x^y.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// fn pow_fp_example() -> FixedType {
    ///     // We instantiate fixed points here.
    ///     let a = Fixed::from_unscaled_felt(3); 
    ///     let b = Fixed::from_unscaled_felt(4);
    ///     
    ///     // We can call `pow` function as follows.
    ///     a.pow(b)
    /// }
    /// >>> {mag: 5435817984, sign: false} // = 81
    /// ```
    ///
    fn pow(self: FixedType, b: FixedType) -> FixedType;
    /// # fp.round
    /// 
    /// ```rust
    /// fn round(self: FixedType) -> FixedType;
    /// ```
    /// 
    /// Rounds the fixed point number to the nearest whole number.
    ///
    /// ## Args
    ///
    /// * `self`(`FixedType`) - The input fixed point
    ///
    /// ## Returns
    ///
    /// A fixed point number representing the rounded value.
    ///
    /// ## Examples
    ///
    /// 
    /// ```rust
    /// fn round_fp_example() -> FixedType {
    ///     // We instantiate fixed points here.
    ///     let a = Fixed::from_felt(194615506); // 2.9
    ///     
    ///     // We can call `round` function as follows.
    ///     a.round(b)
    /// }
    /// >>> {mag: 201326592, sign: false} // = 3
    /// ```
    /// 
    fn round(self: FixedType) -> FixedType;
    /// # fp.sqrt
    /// 
    /// ```rust
    /// fn sqrt(self: FixedType) -> FixedType;
    /// ```
    /// 
    /// Returns the square root of the fixed point number.
    ///
    /// ## Args
    ///
    /// `self`(`FixedType`) - The input fixed point
    ///
    /// ## Panics
    ///
    /// * Panics if the input is negative.
    ///
    /// ## Returns
    /// 
    /// A fixed point number representing the square root of the input value.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// fn sqrt_fp_example() -> FixedType {
    ///     // We instantiate fixed points here.
    ///     let a = Fixed::from_unscaled_felt(25);
    ///     
    ///     // We can call `round` function as follows.
    ///     a.sqrt()
    /// }
    /// >>> {mag: 1677721600, sign: false} // = 5
    /// ```
    ///
    fn sqrt(self: FixedType) -> FixedType;
}

/// IMPLS

impl FixedImpl of Fixed {
    fn new(mag: u128, sign: bool) -> FixedType {
        if sign == true {
            assert(mag <= MAX_u128, 'fixed type: out of range');
        } else {
            assert(mag <= MAX_u128 - 1_u128, 'fixed type: out of range');
        }
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
        self.mag.print();
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
