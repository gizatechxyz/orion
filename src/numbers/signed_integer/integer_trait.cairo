/// | function                                   | description                                                   |
/// | ------------------------------------------ | ------------------------------------------------------------- |
/// | [`IntegerTrait::new`](integertrait-new.md) | Constructs a new `signed_integer`                             |
/// | [`int.div_rem`](int.div\_rem.md)           | Computes `signed_integer` division and modulus simultaneously |
/// | [`int.abs`](int.abs.md)                    | Computes the absolute value of the given `signed_integer`     |
/// | [`int.max`](int.max.md)                    | Returns the maximum between two `signed_integer`              |
/// | [`int.min`](int.min.md)                    | Returns the minimum between two `signed_integer`              |
trait IntegerTrait<T, U> {
    /// # IntegerTrait::new
    /// 
    /// Returns a new signed integer.
    /// 
    /// ```rust
    /// fn new(mag: U, sign: bool) -> T;
    /// ```
    /// 
    /// #### Args
    /// 
    /// | Name   | Type   | Description                                                         |
    /// | ------ | ------ | ------------------------------------------------------------------- |
    /// | `mag`  | `U`    | The magnitude of the integer.                                       |
    /// | `sign` | `bool` | The sign of the integer, where `true` represents a negative number. |
    /// 
    /// > _`<T>` generic type depends on `signed_integer` dtype._
    /// >
    /// > _`<U>` generic type depends on the uint type (u8, u16, u32, u64, u128)._
    /// 
    /// #### Panics
    /// 
    /// | TypeError                        |
    /// | -------------------------------- |
    /// | Panics if `mag` is out of range. |
    /// 
    /// #### Returns
    /// 
    /// A new signed integer.
    /// 
    /// #### Examples
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
    fn new(mag: U, sign: bool) -> T;
    /// # int.div\_rem
    /// 
    /// Computes signed\_integer division and modulus simultaneously
    /// 
    /// ```rust
    /// fn div_rem(self: T, other: T) -> (T, T);
    /// ```
    /// 
    /// #### Args
    /// 
    /// | Name    | Type | Description  |
    /// | ------- | ---- | ------------ |
    /// | `self`  | `T`  | The dividend |
    /// | `other` | `T`  | The divisor  |
    /// 
    /// > _`<T>` generic type depends on `signed_integer` dtype._
    /// 
    /// #### Panics
    /// 
    /// | TypeError                      |
    /// | ------------------------------ |
    /// | Panics if the divisor is zero. |
    /// 
    /// #### Returns
    /// 
    /// A tuple of signed integer `<T>`, containing the quotient and the remainder of the division.
    /// 
    /// > _`<T>` generic type depends on `signed_integer` dtype._
    /// 
    /// #### Examples
    /// 
    /// ```rust
    /// fn div_rem_example() -> (i32, i32) {
    ///     // We instantiate signed integers here.
    ///     let a = IntegerTrait::<i32>::new(13_u32, false);
    ///     let b = IntegerTrait::<i32>::new(5_u32, false);
    ///     
    ///     // We can call `div_rem` function as follows.
    ///     a.div_rem(b)
    /// }
    /// >>> ({mag: 2, sign: false}, {mag: 3, sign: false}) // = (2, 3)
    /// ```
    fn div_rem(self: T, other: T) -> (T, T);
    /// # int.abs
    /// 
    /// Computes the absolute value of a signed\_integer.
    /// 
    /// ```rust
    /// fn abs(self: T) -> T;
    /// ```
    /// 
    /// #### Args
    /// 
    /// | Name   | Type | Description                                               |
    /// | ------ | ---- | --------------------------------------------------------- |
    /// | `self` | `T`  | The signed integer to which the absolute value is applied |
    /// 
    /// > _`<T>` generic type depends on `signed_integer` dtype._
    /// 
    /// #### Returns
    /// 
    /// A signed integer `<T>`, representing the absolute value of `self` .
    /// 
    /// > _`<T>` generic type depends on `signed_integer` dtype._
    /// 
    /// #### Examples
    /// 
    /// ```rust
    /// fn abs_example() -> i32 {
    ///     // We instantiate signed integers here.
    ///     let int = IntegerTrait::<i32>::new(42_u32, true);
    ///     
    ///     // We can call `abs` function as follows.
    ///     a.abs()
    /// }
    /// >>> {mag: 42, sign: false} // = 42
    /// ```
    fn abs(self: T) -> T;
    /// # int.max
    /// 
    /// Returns the maximum between two signed\_integer.
    /// 
    /// ```rust
    /// fn max(self: T, other: T) -> T;
    /// ```
    /// 
    /// #### Args
    /// 
    /// | Name    | Type | Description                           |
    /// | ------- | ---- | ------------------------------------- |
    /// | `self`  | `T`  | The first signed integer to compare.  |
    /// | `other` | `T`  | The second signed integer to compare. |
    /// 
    /// > _`<T>` generic type depends on `signed_integer` dtype._
    /// 
    /// #### Returns
    /// 
    /// A signed integer `<T>`, The maximum between `self` and `other`.
    /// 
    /// > _`<T>` generic type depends on `signed_integer` dtype._
    /// 
    /// #### Examples
    /// 
    /// ```rust
    /// fn max_example() -> i32 {
    ///     // We instantiate signed integer here.
    ///     let a = IntegerTrait::<i32>::new(42_u32, true);
    ///     let b = IntegerTrait::<i32>::new(13_u32, false);
    ///     
    ///     // We can call `max` function as follows.
    ///     a.max(b)
    /// }
    /// >>> {mag: 13, sign: false} // as 13 > -42
    /// ```
    fn max(self: T, other: T) -> T;
    /// # int.min
    /// 
    /// Returns the minimum between two signed\_integer.
    /// 
    /// ```rust
    /// fn min(self: T, other: T) -> T;
    /// ```
    /// 
    /// #### Args
    /// 
    /// | Name    | Type | Description                           |
    /// | ------- | ---- | ------------------------------------- |
    /// | `self`  | `T`  | The first signed integer to compare.  |
    /// | `other` | `T`  | The second signed integer to compare. |
    /// 
    /// > _`<T>` generic type depends on `signed_integer` dtype._
    /// 
    /// #### Returns
    /// 
    /// A signed integer `<T>`, The minimum between `self` and `other`.
    /// 
    /// > _`<T>` generic type depends on `signed_integer` dtype._
    /// 
    /// #### Examples
    /// 
    /// ```rust
    /// fn min_example() -> i32 {
    ///     // We instantiate signed integer here.
    ///     let a = IntegerTrait::<i32>::new(42_u32, true);
    ///     let b = IntegerTrait::<i32>::new(13_u32, false);
    ///     
    ///     // We can call `max` function as follows.
    ///     a.min(b)
    /// }
    /// >>> {mag: 42, sign: true} // as -42 < 13
    /// ```
    fn min(self: T, other: T) -> T;
}

