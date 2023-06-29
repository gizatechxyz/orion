/// A struct representing a fixed point number.
#[derive(Copy, Drop)]
struct FixedType {
    mag: u128,
    sign: bool
}

/// A struct listing fixed point implementations.
#[derive(Copy, Drop)]
enum FixedImpl {
    FP8x23: (),
    FP16x16: ()
}

/// Trait
///
/// new - Constructs a new fixed point instance.
/// new_unscaled - Creates a new fixed point instance with the specified unscaled magnitude and sign.
/// from_felt - Creates a new fixed point instance from a `felt252` value.
/// from_unscaled_felt - Creates a new fixed point instance from an unscaled `felt252` value.
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
trait FixedTrait {
    /// # FixedTrait::new
    /// 
    /// ```rust
    /// fn new(mag: u128, sign: bool) -> FixedType;
    /// ```
    /// 
    /// Constructs a new fixed point instance.
    ///
    /// ## Args
    /// 
    /// * `mag`(`u128`) - The magnitude of the fixed point.
    /// * `sign`(`bool`) - The sign of the fixed point, where `true` represents a negative number.
    ///
    /// ## Returns
    ///
    /// A new fixed point instance.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// fn new_fp_example() -> FixedType {
    ///     // We can call `new` function as follows. 
    ///     FixedTrait::new(67108864, false)
    /// }
    /// >>> {mag: 67108864, sign: false} // = 1
    /// ```
    ///
    fn new(mag: u128, sign: bool) -> FixedType;
    /// # FixedTrait::new\_unscaled
    /// 
    /// ```rust
    /// fn new_unscaled(mag: u128, sign: bool) -> FixedType;
    /// ```
    ///
    /// Creates a new fixed point instance with the specified unscaled magnitude and sign.
    /// 
    /// ## Args
    ///
    /// `mag`(`u128`) - The unscaled magnitude of the fixed point.
    /// `sign`(`bool`) - The sign of the fixed point, where `true` represents a negative number.
    ///
    /// ## Returns
    /// 
    /// A new fixed point instance.
    /// 
    /// ## Examples
    /// 
    /// ```rust
    /// fn new_unscaled_example() -> FixedType {
    ///     // We can call `new_unscaled` function as follows. 
    ///     FixedTrait::new_unscaled(1);
    /// }
    /// >>> {mag: 67108864, sign: false}
    /// ```
    ///
    fn new_unscaled(mag: u128, sign: bool) -> FixedType;
    /// # FixedTrait::from\_felt
    ///
    /// 
    /// ```rust
    /// fn from_felt(val: felt252) -> FixedType;
    /// ```
    /// 
    /// Creates a new fixed point instance from a felt252 value.
    ///
    /// ## Args
    /// 
    /// * `val`(`felt252`) - `felt252` value to convert in FixedType
    ///
    /// ## Returns 
    ///
    /// A new fixed point instance.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// fn from_felt_example() -> FixedType {
    ///     // We can call `from_felt` function as follows . 
    ///     FixedTrait::from_felt(194615706);
    /// }
    /// >>> {mag: 194615706, sign: false} // = 2.9
    /// ```
    ///
    fn from_felt(val: felt252) -> FixedType;
    ///# FixedTrait::from\_unscaled\_felt
    ///
    ///```rust
    ///fn from_unscaled_felt(val: felt252) -> FixedType;
    ///```
    ///
    ///Creates a new fixed point instance from an unscaled felt252 value.
    ///
    /// ## Args
    /// 
    /// `val`(`felt252`) - `felt252` value to convert in FixedType
    ///
    /// ## Returns - A new fixed point instance.
    ///
    /// ## Examples
    ///
    ///```rust
    ///fn from_unscaled_felt_example() -> FixedType {
    ///    // We can call `from_unscaled_felt` function as follows . 
    ///    FixedTrait::from_unscaled_felt(1);
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
    ///     let fp = FixedTrait::from_unscaled_felt(-1);
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
    ///     let fp = FixedTrait::from_felt(194615506); // 2.9
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
    ///     let fp = FixedTrait::from_unscaled_felt(2);
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
    ///     let fp = FixedTrait::from_unscaled_felt(2);
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
    ///     let fp = FixedTrait::from_felt(194615506); // 2.9
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
    /// A fixed point representing the natural logarithm of the input number.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// fn ln_fp_example() -> FixedType {
    ///     // We instantiate fixed point here.
    ///     let fp = FixedTrait::from_unscaled_felt(1);
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
    /// A fixed point representing the binary logarithm of the input number.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// fn log2_fp_example() -> FixedType {
    ///     // We instantiate fixed point here.
    ///     let fp = FixedTrait::from_unscaled_felt(32);
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
    /// A fixed point representing the base 10 logarithm of the input number.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// fn log10_fp_example() -> FixedType {
    ///     // We instantiate fixed point here.
    ///     let fp = FixedTrait::from_unscaled_felt(100);
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
    ///     // We instantiate FixedTrait points here.
    ///     let a = FixedTrait::from_unscaled_felt(3); 
    ///     let b = FixedTrait::from_unscaled_felt(4);
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
    ///     // We instantiate FixedTrait points here.
    ///     let a = FixedTrait::from_felt(194615506); // 2.9
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
    ///     // We instantiate FixedTrait points here.
    ///     let a = FixedTrait::from_unscaled_felt(25);
    ///     
    ///     // We can call `round` function as follows.
    ///     a.sqrt()
    /// }
    /// >>> {mag: 1677721600, sign: false} // = 5
    /// ```
    ///
    fn sqrt(self: FixedType) -> FixedType;
    /// # fp.sinh
    /// 
    /// ```rust
    /// fn sinh(self: FixedType) -> FixedType;
    /// ```
    /// 
    /// Returns the value of the hyperbolic sine of the fixed point number.
    ///
    /// ## Args
    ///
    /// * `self`(`FixedType`) - The input fixed point
    ///
    /// ## Returns
    ///
    /// The hyperbolic sine of the input fixed point number.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// fn sinh_fp_example() -> FixedType {
    ///     // We instantiate fixed point here.
    ///     let fp = FixedTrait::from_unscaled_felt(2);
    ///     
    ///     // We can call `sinh` function as follows.
    ///     fp.sinh()
    /// }
    /// >>> {mag: 30424311, sign: false} // = 3.6268604
    /// ``` 
    ///
    fn sinh(self: FixedType) -> FixedType;
    /// # fp.tanh
    /// 
    /// ```rust
    /// fn tanh(self: FixedType) -> FixedType;
    /// ```
    /// 
    /// Returns the value of the hyperbolic tangent of the fixed point number.
    ///
    /// ## Args
    ///
    /// * `self`(`FixedType`) - The input fixed point
    ///
    /// ## Returns
    ///
    /// The hyperbolic tangent of the input fixed point number.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// fn tanh_fp_example() -> FixedType {
    ///     // We instantiate fixed point here.
    ///     let fp = FixedTrait::from_unscaled_felt(2);
    ///     
    ///     // We can call `tanh` function as follows.
    ///     fp.tanh()
    /// }
    /// >>> {mag: 8086850, sign: false} // = 0.964027...
    /// ``` 
    ///
    fn tanh(self: FixedType) -> FixedType;
}
