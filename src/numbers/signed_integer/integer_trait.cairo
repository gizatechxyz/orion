/// Trait
///
/// new - Constructs a new `signed_integer
/// div_rem - Computes `signed_integer` division and modulus simultaneously
/// abs - Computes the absolute value of the given `signed_integer`
/// max - Returns the maximum between two `signed_integer`
/// min - Returns the minimum between two `signed_integer`
trait IntegerTrait<T, U> {
    /// # IntegerTrait::new
    /// 
    /// ```rust
    /// fn new(mag: U, sign: bool) -> T;
    /// ```
    /// 
    /// Returns a new signed integer.
    ///
    /// ## Args
    ///
    /// * `mag`(`U`) - The magnitude of the integer.
    /// * `sign`(`bool`) - The sign of the integer, where `true` represents a negative number.
    ///
    /// > _`<U>` generic type depends on the uint type (u8, u16, u32, u64, u128)._
    ///
    /// ## Panics
    ///
    /// Panics if `mag` is out of range.
    ///
    /// ## Returns
    /// 
    /// A new signed integer.
    /// 
    /// ## Examples
    /// 
    /// ```rust
    /// fn new_i8_example() -> i8 {
    ///     IntegerTrait::<i8>::new(42_u8, true)
    /// }
    /// >>> {mag: 42, sign: true} // = -42
    /// ```
    /// 
    /// ```rust
    /// fn panic_i8_example() -> i8 {
    ///     IntegerTrait::<i8>::new(129_u8, true)
    /// }
    /// >>> panics with "int: out of range"
    /// ```
    /// 
    fn new(mag: U, sign: bool) -> T;
    /// # int.div_rem
    /// 
    /// ```rust
    /// fn div_rem(self: T, other: T) -> (T, T);
    /// ```
    /// 
    /// Computes signed\_integer division and modulus simultaneously
    ///
    /// ## Args
    /// 
    /// * `self`(`T`) - The dividend
    /// * `other`(`T`) - The divisor
    ///
    /// ## Panics
    ///
    /// Panics if the divisor is zero.
    ///
    /// ## Returns
    ///
    /// A tuple of signed integer `<T>`, containing the quotient and the remainder of the division.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// fn div_rem_example() -> (i32, i32) {
    ///     // We instantiate signed integers here.
    ///     let a = IntegerTrait::<i32>::new(13, false);
    ///     let b = IntegerTrait::<i32>::new(5, false);
    ///     
    ///     // We can call `div_rem` function as follows.
    ///     a.div_rem(b)
    /// }
    /// >>> ({mag: 2, sign: false}, {mag: 3, sign: false}) // = (2, 3)
    /// ```
    ///
    fn div_rem(self: T, other: T) -> (T, T);
    /// # int.abs 
    /// 
    /// ```rust
    /// fn abs(self: T) -> T;
    /// ```
    /// 
    /// Computes the absolute value of a signed\_integer.
    ///
    /// ## Args
    ///
    /// `self`(`T`) - The signed integer to which the absolute value is applied
    ///
    /// ## Returns
    ///
    /// A signed integer `<T>`, representing the absolute value of `self` .
    ///
    /// ## Examples
    ///
    /// ```rust
    /// fn abs_example() -> i32 {
    ///     // We instantiate signed integers here.
    ///     let int = IntegerTrait::<i32>::new(42, true);
    ///     
    ///     // We can call `abs` function as follows.
    ///     a.abs()
    /// }
    /// >>> {mag: 42, sign: false} // = 42
    /// ```
    ///
    fn abs(self: T) -> T;
    /// # int.max
    /// 
    /// ```rust
    /// fn max(self: T, other: T) -> T;
    /// ```
    /// 
    /// Returns the maximum between two signed\_integer.
    ///
    /// ## Args
    ///
    /// *`self`(`T`) - The first signed integer to compare.
    /// * `other`(`T`) - The second signed integer to compare.
    ///
    /// ## Returns
    ///
    /// A signed integer `<T>`, The maximum between `self` and `other`.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// fn max_example() -> i32 {
    ///     // We instantiate signed integer here.
    ///     let a = IntegerTrait::<i32>::new(42, true);
    ///     let b = IntegerTrait::<i32>::new(13, false);
    ///     
    ///     // We can call `max` function as follows.
    ///     a.max(b)
    /// }
    /// >>> {mag: 13, sign: false} // as 13 > -42
    /// ```
    ///
    fn max(self: T, other: T) -> T;
    /// # int.min
    /// 
    /// ```rust
    /// fn min(self: T, other: T) -> T;
    /// ```
    /// 
    /// Returns the minimum between two signed\_integer.
    ///
    /// ## Args
    ///
    /// `self`(`T`) - The first signed integer to compare.
    /// `other`(`T`) - The second signed integer to compare.
    ///
    /// ## Returns
    ///
    /// A signed integer `<T>`, The minimum between `self` and `other`.
    ///
    /// ## Examples
    /// 
    /// 
    /// ```rust
    /// fn min_example() -> i32 {
    ///     // We instantiate signed integer here.
    ///     let a = IntegerTrait::<i32>::new(42, true);
    ///     let b = IntegerTrait::<i32>::new(13, false);
    ///     
    ///     // We can call `max` function as follows.
    ///     a.min(b)
    /// }
    /// >>> {mag: 42, sign: true} // as -42 < 13
    /// ```
    /// 
    fn min(self: T, other: T) -> T;
}

